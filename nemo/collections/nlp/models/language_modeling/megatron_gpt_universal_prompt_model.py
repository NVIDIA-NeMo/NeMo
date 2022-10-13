# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_learning_dataset import GPTPromptLearningDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_universal_prompt_learning_dataset import (
    GPTUniversalPromptLearningDataset,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_base_prompt_learning_model import (
    MegatronBasePromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    init_method_normal,
    scaled_init_method_normal,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, SamplingParam, TextGeneration
from nemo.collections.nlp.modules.common.universal_prompt_encoder import UniversalPromptEncoder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.utils import logging

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ['MegatronGPTUniversalPromptLearningModel']


class MegatronGPTUniversalPromptLearningModel(MegatronBaseModel, TextGeneration):
    """
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        if self.trainer.precision == 32:
            self.autocast_dtype = torch.float
        elif self.trainer.precision == 16:
            self.autocast_dtype = torch.half
        elif self.trainer.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        self.cfg = cfg
        # TODO: Enable amp_o2 training
        self.megatron_amp_o2 = False
        self.load_frozen_model(self.cfg, trainer)

        self.pipeline_parallel = self.cfg.get('pipeline_model_parallel_size', 1) > 1
        self.tokenizer = self.frozen_model.tokenizer
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

        # Load templates for assigning virtual prompt token positions
        self.load_task_templates(self.cfg.task_templates)

        # make sure the default pytorch lightning gradient clipping in the basemodel
        self.grad_clip_pl_default = True
        self.lowest_val_loss = None

        self._reduced_loss_buffer = []
        self._inference_config = None
        self.hidden_size = self.frozen_model.cfg.hidden_size
        self.padded_vocab_size = self.frozen_model.padded_vocab_size
        if self.frozen_model.model.pre_process:
            self.word_embeddings = self.frozen_model.model.language_model.embedding.word_embeddings
        self._prompt_encoder_key = 'prompt_encoder'

    def add_virtual_prompt_params_to_param_group(self):
        """
        Passes only prompt table and prompt encoder params to the optimizer.
        """
        virtual_prompt_params = {'params': []}
        virtual_prompt_params['params'].extend([param for param in self.prompt_encoder.parameters()])
        self._optimizer_param_groups = (virtual_prompt_params,)

    def load_task_templates(self, task_templates):
        self.task_templates = OmegaConf.to_container(self.cfg.task_templates)

    def load_frozen_model(self, cfg, trainer):
        save_restore_connector = NLPSaveRestoreConnector()

        # Load frozen model from unpacked directory
        if os.path.isdir(cfg.get('language_model_path')):
            save_restore_connector.model_extracted_dir = cfg.get('language_model_path')

        frozen_model_cfg = MegatronGPTModel.restore_from(
            cfg.get('language_model_path'),
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )

        # Need to overwrite some params in frozen model's config before restoring
        with open_dict(frozen_model_cfg):
            frozen_model_cfg.megatron_amp_O2 = False
            frozen_model_cfg.optim.name = "fused_adam"
            frozen_model_cfg.micro_batch_size = self.cfg.micro_batch_size
            frozen_model_cfg.global_batch_size = self.cfg.global_batch_size
            frozen_model_cfg.precision = trainer.precision
            frozen_model_cfg.sequence_parallel = self.cfg.get("sequence_parallel", False)
            frozen_model_cfg.activations_checkpoint_granularity = self.cfg.get(
                "activations_checkpoint_granularity", None
            )
            frozen_model_cfg.activations_checkpoint_num_layers = self.cfg.get(
                "activations_checkpoint_num_layers", None
            )
            frozen_model_cfg.activations_checkpoint_method = self.cfg.get("activations_checkpoint_method", None)

        if cfg.get('language_model_path', None):
            self.frozen_model = MegatronGPTModel.restore_from(
                cfg.get('language_model_path'),
                trainer=trainer,
                save_restore_connector=save_restore_connector,
                override_config_path=frozen_model_cfg,
            ).to(dtype=self.autocast_dtype)

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Custom state dict that only contains prompt table and prompt encoder parameters. 
        No frozen model parameters are stored in the state dict. Prompt encoder parameters 
        are only in state dict for intermediate checkpoints saved during training. Final
        nemo checkpoints at the end of training will contain prompt table parameters only. 
        """
        if self.frozen_model.model.pre_process:
            state_dict_ = {}
            state_dict_[self._prompt_encoder_key] = self.prompt_encoder.state_dict()
            return state_dict_
        else:
            state_dict_ = {}

            return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Custom load state dict method that only loads prompt table and prompt encoder
        parameters. Matching load method for this class' custom state dict method. 
        """
        if self.frozen_model.model.pre_process and self._prompt_encoder_key in state_dict:
            state_dict_ = state_dict[self._prompt_encoder_key]

            if not hasattr(self, "prompt_encoder"):
                self.init_prompt_encoder()
            self.prompt_encoder.load_state_dict(state_dict_, strict)

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups. 
        Only want virtual prompt params to be passed to the optimizer.
        """
        ## Freeze frozen model
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        if self.frozen_model.model.pre_process:
            self.add_virtual_prompt_params_to_param_group()
        else:
            self._optimizer_param_groups = ({'params': []},)

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.
        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""

        self.frozen_model.model.set_input_tensor(input_tensor)

    def embed_input_train(self, input_ids: Tensor, prompt_input_mask):
        # Replace virtual token ids with padding for forward pass through vocab embeddings
        discrete_token_ids = input_ids.clone()
        discrete_token_embeds = self.word_embeddings(discrete_token_ids).clone()
        # [b, s, d] -> [s, b, d]
        prompt_input_emb = discrete_token_embeds.transpose(0, 1).contiguous()
        virtual_token_embeds = self.prompt_encoder(prompt_input_emb, prompt_input_mask)
        # [b, s, d] -> [s, b, d]
        virtual_token_embeds = virtual_token_embeds.transpose(0, 1).contiguous()
        return virtual_token_embeds, discrete_token_embeds

    def training_step(self, batch, batch_idx):
        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()
        loss_mean = self.fwd_bwd_step(batch, forward_only=False)
        self.allreduce_gradients()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        return loss_mean

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        prompt_input_mask,
        labels=None,
        inference=True,
        set_inference_key_value_memory=False,
        inference_max_sequence_len=None,
    ):
        """
        Special forward method for p-tuning/prompt-tuning pretrained
        GPT style models. Bypasses the vocab token preprocessing done
        in the MegatronGPT class.
        """
        # Get embeddings for text tokens and insert virtual token embeddings
        if self.frozen_model.model.pre_process:
            if inference and set_inference_key_value_memory:
                all_input_ids = input_ids[0]
                input_ids = input_ids[1]
                virtual_token_emb, _ = self.embed_input_train(all_input_ids, prompt_input_mask)
                input_embeds = self.word_embeddings(input_ids).clone()
            elif inference and not set_inference_key_value_memory:
                all_input_ids = input_ids[0]
                input_ids = input_ids[1]
                discrete_token_ids = input_ids.clone()
                input_embeds = self.word_embeddings(discrete_token_ids).clone()
                virtual_token_emb = None
            else:
                virtual_token_emb, input_embeds = self.embed_input_train(input_ids, prompt_input_mask)

            position_embeddings = self.frozen_model.model.language_model.embedding.position_embeddings(position_ids)
            encoder_input = input_embeds + position_embeddings
            if virtual_token_emb is not None:
                encoder_input = torch.concat([virtual_token_emb, encoder_input], axis=1)
            encoder_input = encoder_input.transpose(0, 1).contiguous()
            if self.cfg.get("sequence_parallel", False):
                encoder_input = tensor_parallel.mappings.scatter_to_sequence_parallel_region(encoder_input)
        else:
            encoder_input = None

        # Call forward on GPT model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.model(
                input_ids=None,
                position_ids=None,
                encoder_input=encoder_input,
                attention_mask=attention_mask,
                labels=labels,
                set_inference_key_value_memory=set_inference_key_value_memory,
                inference_max_sequence_len=inference_max_sequence_len,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.model(
                    input_ids=None,
                    position_ids=None,
                    encoder_input=encoder_input,
                    attention_mask=attention_mask,
                    labels=labels,
                    set_inference_key_value_memory=set_inference_key_value_memory,
                    inference_max_sequence_len=inference_max_sequence_len,
                )

        return output

    def fwd_bwd_step(self, batch, forward_only):
        """
            Dataloader produces a global batch which is turned into a list of microbatches.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        disable_autocast = False
        sequence_parallel_enabled = self.cfg.get("sequence_parallel", False)
        # Get seq length of batch
        _, seq_length = batch[0].shape
        tensor_shape = [seq_length + self.cfg.perceiver.hidden_steps, self.cfg.micro_batch_size, self.hidden_size]

        if self.pipeline_parallel:
            losses_reduced_per_micro_batch = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch,
                model=self,
                forward_only=forward_only,
                tensor_shape=tensor_shape,
                dtype=self.autocast_dtype,
                disable_autocast=disable_autocast,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
                sequence_parallel_enabled=sequence_parallel_enabled,
            )
        else:
            losses_reduced_per_micro_batch = forward_backward_no_pipelining(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch,
                model=self,
                forward_only=forward_only,
                tensor_shape=tensor_shape,
                dtype=self.autocast_dtype,
                disable_autocast=disable_autocast,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            input_ids, labels, loss_mask, position_ids, attention_mask, prompt_input_mask = batch
            labels = F.pad(labels, (self.cfg.perceiver.hidden_steps, 0, 0, 0), value=0)
            loss_mask = F.pad(loss_mask, (self.cfg.perceiver.hidden_steps, 0, 0, 0), value=0)
            output_tensor = model(input_ids, position_ids, attention_mask, prompt_input_mask, labels, inference=False)

            if isinstance(output_tensor, tuple):
                output_tensor, _ = output_tensor

            def loss_func(output_tensor):
                loss = self.frozen_model.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def init_prompt_encoder(self):
        """
        Init the prompt encoder needed for universal prompt encoder
        """
        perceiver_conf = OmegaConf.to_container(self.cfg.perceiver)
        init_method_std = self.cfg.perceiver.init_method_std
        del perceiver_conf['init_method_std']
        encoder_init = init_method_normal(init_method_std)
        output_init = scaled_init_method_normal(init_method_std, self.cfg.perceiver.num_layers)
        perceiver_conf['init_method'] = encoder_init
        perceiver_conf['output_layer_init_method'] = output_init
        self.prompt_encoder = UniversalPromptEncoder(perceiver_conf, output_dim=self.frozen_model.cfg.hidden_size)

    def setup(self, stage=None):
        self.setup_test_data()
        if stage == 'test' or stage == 'predict':
            return

        if self.frozen_model.model.pre_process:
            self.init_prompt_encoder()

        self.setup_training_data()
        self.setup_validation_data()

    def setup_training_data(self, training_data_config=None):
        if self.cfg.data.get('train_ds', None):
            self._train_ds, self._train_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.train_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('train_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def setup_validation_data(self, validation_data_config=None):
        if self.cfg.data.get('validation_ds', None):
            self._validation_ds, self._validation_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.validation_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=True,
                drop_last=True,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('validation_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def setup_test_data(self, test_data_config=None):
        if self.cfg.data.get('test_ds', None):
            self._test_ds, self._test_dl = self.build_virtual_prompt_dataset(
                data=self.cfg.data.test_ds,
                batch_size=self.cfg.global_batch_size,
                max_seq_length=self.frozen_model.cfg.encoder_seq_length,
                min_seq_length=self.cfg.data.get('min_seq_length', 1),
                add_bos=self.cfg.data.get('add_bos', False),
                add_eos=self.cfg.data.get('add_eos', True),
                for_train=False,
                drop_last=False,
                shuffle=False,
                num_workers=self.cfg.data.num_workers,
                pin_memory=True,
                cache_data_path=self.cfg.data.get('test_cache_data_path', None),
                load_cache=self.cfg.data.get('load_cache', False),
            )

    def build_virtual_prompt_dataset(
        self,
        data,
        batch_size=None,
        max_seq_length=2048,
        min_seq_length=1,
        add_bos=False,
        add_eos=False,
        for_train=True,
        drop_last=False,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        tokens_to_generate=None,
        get_dataset_only=False,
        cache_data_path=None,
        load_cache=False,
    ):
        dataset = GPTUniversalPromptLearningDataset(
            data=data,
            tokenizer=self.tokenizer,
            virtual_token_len=self.cfg.perceiver.hidden_steps,
            task_templates=self.task_templates,
            pad_token_id=self.pad_token_id,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            for_train=for_train,
            tokens_to_generate=tokens_to_generate,
            cache_data_path=cache_data_path,
            load_cache=load_cache,
        )

        if get_dataset_only:
            return dataset

        # Make distributed dataloader
        rank = parallel_state.get_data_parallel_rank()
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=data_parallel_size, rank=rank, shuffle=shuffle
        )

        assert batch_size % data_parallel_size == 0, "Global batch size must be evenly divisible by data parallel size"

        if for_train:
            if self.cfg.get("sequence_parallel", False):
                collate_fn = partial(
                    dataset.collate_fn, tp_workers=parallel_state.get_tensor_model_parallel_world_size()
                )
            else:
                collate_fn = partial(dataset.collate_fn, tp_workers=0)
        else:
            collate_fn = dataset.inference_collate_fn

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            batch_size=batch_size // data_parallel_size,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return dataset, dataloader

    def validation_step(self, batch, batch_idx):
        loss_mean = self.fwd_bwd_step(batch, forward_only=True)
        if loss_mean.item == 0.0:
            loss_mean = []

        return loss_mean

    def validation_epoch_end(self, outputs):
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss
            if len(outputs) == 0:
                return
            averaged_loss = torch.stack(outputs).mean()
        else:
            averaged_loss = torch.tensor(0.0).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)
        logging.info(f'val_loss: {averaged_loss}')

        # Save inference ready .nemo checkpoint version
        # if self.cfg.get("save_intermediate_nemo_file", True):
        #     if self.lowest_val_loss is None or averaged_loss < self.lowest_val_loss:
        #         self.save_checkpoint_as_nemo_file()
        #         self.lowest_val_loss = averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def on_train_end(self):
        pass

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ):

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()
            sampling_params["add_BOS"] = self.cfg.data.get("add_bos", False)

        if length_params is None:
            length_params = get_default_length_params()

        max_input_length = self.frozen_model.cfg.encoder_seq_length - length_params["max_length"]

        # input dicts are either dataset paths or already loaded example dicts
        if "taskname" not in inputs[0].keys():
            data = [path["data_path"] for path in inputs]
        else:
            data = inputs

        dataset = self.build_virtual_prompt_dataset(
            data=data,
            max_seq_length=max_input_length,
            min_seq_length=self.cfg.data.get('min_seq_length', 1),
            add_bos=sampling_params["add_BOS"],
            add_eos=False,
            for_train=False,
            tokens_to_generate=length_params["max_length"],
            get_dataset_only=True,
        )

        full_dataset = [dataset[i] for i in range(len(dataset))]
        task_ids, processed_inputs = dataset.inference_collate_fn(full_dataset)
        self.frozen_model.model.parallel_output = False

        # Call same generate code as in MegatronGPT
        return megatron_gpt_generate(
            self.cuda(), processed_inputs, self.tokenizer, length_params, sampling_params, task_ids
        )

    def get_forward_output_only_func(self):
        """
        Used for generate method only for now.
        """

        def fwd_output_only_func(batch, model):
            extra_arg = {}
            (
                all_tokens,
                tokens,
                position_ids,
                attention_mask,
                prompt_input_mask,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch

            all_tokens = all_tokens.cuda()
            tokens = tokens.cuda()
            position_ids = position_ids.cuda()
            attention_mask = attention_mask.cuda()
            prompt_input_mask = prompt_input_mask.cuda()
            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            output_tensor = model((all_tokens, tokens), position_ids, attention_mask, prompt_input_mask, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        else:
            length_params: LengthParam = {
                "max_length": inference_config["tokens_to_generate"],
                "min_length": inference_config["min_tokens_to_generate"],
            }

            sampling_params: SamplingParam = {
                "use_greedy": inference_config["greedy"],
                "temperature": inference_config["temperature"],
                "top_k": inference_config["top_k"],
                "top_p": inference_config["top_p"],
                "repetition_penalty": inference_config["repetition_penalty"],
                "add_BOS": inference_config["add_BOS"],
                "all_probs": inference_config["all_probs"],
                "compute_logprob": inference_config["compute_logprob"],
            }

            input_tokens, length_tensor = batch
            self.frozen_model.model.parallel_output = False

            # Call same generate code as in MegatronGPT
            return megatron_gpt_generate(
                self.cuda(), (input_tokens, length_tensor), self.tokenizer, length_params, sampling_params
            )

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from apex.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    @classmethod
    def list_available_models(cls):
        pass
