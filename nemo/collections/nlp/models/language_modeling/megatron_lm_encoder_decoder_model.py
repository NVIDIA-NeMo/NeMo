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
import re
from operator import itemgetter
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.clip_grads import clip_grad_norm_fp32
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import (
    MegatronTokenLevelEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core.optim import MainParamsOptimizerWrapper, prepare_lr_scheduler
from nemo.utils import AppState, logging

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.common import _get_params_for_weight_decay_optimization

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


__all__ = ["MegatronLMEncoderDecoderModel"]


class MegatronLMEncoderDecoderModel(MegatronBaseModel):
    """
    Megatron encoder-decoder base class
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        # build tokenizer (defaults to nemo supported tokenizers)
        self._build_tokenizer()

        # manipulate vocabulary (e.g., pad vocabulary for better efficiency)
        self._build_vocab()

        # TODO: create get_encoder_decoder_model()here for different losses (e..g, nll, vae, mim)
        self.enc_dec_model = MegatronTokenLevelEncoderDecoderModule(
            encoder_arch=cfg.encoder_arch,
            decoder_arch=cfg.decoder_arch,
            vocab_size=self.padded_vocab_size,
            hidden_size=cfg.hidden_size,
            max_position_embeddings=cfg.max_position_embeddings,
            num_layers=cfg.num_layers,
            num_attention_heads=cfg.num_attention_heads,
            apply_query_key_layer_scaling=cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=cfg.get('kv_channels', None),
            ffn_hidden_size=cfg.ffn_hidden_size,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=cfg.get('pre_process', True),
            post_process=cfg.get('post_process', True),
            init_method_std=cfg.get('init_method_std', 0.02),
            fp16_cross_entropy=cfg.get('fp16_lm_cross_entropy', False),
            use_cpu_initialization=cfg.get('use_cpu_initialization', False),
            hidden_dropout=cfg.get('hidden_dropout', 0.1),
            attention_dropout=cfg.get('attention_dropout', 0.1),
            precision=cfg.get('precision', 16),
            fp32_residual_connection=cfg.get('fp32_residual_connection', False),
            activations_checkpoint_method=cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=cfg.get('activations_checkpoint_num_layers', 1),
            layernorm_epsilon=cfg.get('layernorm_epsilon', 1e-5),
            persist_layer_norm=cfg.get('persist_layer_norm', False),
            bias_gelu_fusion=cfg.get('bias_gelu_fusion', True),
            masked_softmax_fusion=cfg.get('masked_softmax_fusion', True),
            onnx_safe=cfg.get('onnx_safe', False),
            activation=cfg.get('activation', 'gelu'),
        )

        self.setup_optimizer_param_groups()

        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)

        if self.megatron_amp_o2:

            # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
            self.enc_dec_model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            self.enc_dec_model = Float16Module(module=self.enc_dec_model, precision=cfg.precision)

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        Override default Encoder-decoder tokenizer to use legacy=True for sentencepiece.
        """
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=self._cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer_model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("vocab_file", self._cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("merges_file", self._cfg.tokenizer.merge_file),
            legacy=True if self._cfg.tokenizer.library == 'sentencepiece' else False,
        )

    def _build_vocab(self):
        """
        Manipulate vocabulary (e.g., pad vocabulary for increased performance)/
        """
        # TODO: add config to allow to disable it?
        self.padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=self._cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=self._cfg.get('tensor_model_parallel_size', 1),
        )

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        tokentype_ids=None,
        lm_labels=None,
        enc_hidden_states=None,
        enc_output_mask=None,
        output_enc_hidden_only=False,
        enc_input=None,
    ):
        ret_dict = self.enc_dec_model(
            enc_input_ids=encoder_input_ids,
            dec_input_ids=decoder_input_ids,
            enc_attn_mask=encoder_attn_mask,
            dec_attn_mask=decoder_attn_mask,
            tokentype_ids=tokentype_ids,
            labels=lm_labels,
            enc_hidden_states=enc_hidden_states,
            enc_output_mask=enc_output_mask,
            output_enc_hidden_only=output_enc_hidden_only,
            enc_input=enc_input,
        )

        return ret_dict

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = _get_params_for_weight_decay_optimization([self.enc_dec_model])

    def configure_optimizers(self):
        self.setup_optimization()

        # Wrap the baseline optimizer with the optimizer class with master parameters
        if self.megatron_amp_o2 and self._optimizer is not None:
            if self.cfg.precision == 'bf16':
                fp32_grad_accum = True
                contiguous_grad_bucket = True
                async_grad_allreduce = True

            elif self.cfg.precision == 16:
                fp32_grad_accum = False
                # TODO: contiguous grad bucket for fp16 is also planned to be supported
                contiguous_grad_bucket = False
                async_grad_allreduce = False

            self._optimizer = MainParamsOptimizerWrapper(
                self._optimizer,
                fp32_grad_accum=fp32_grad_accum,
                contiguous_grad_bucket=contiguous_grad_bucket,
                async_grad_allreduce=async_grad_allreduce,
            )
            assert self._trainer.max_steps is not None, "'max_steps' is missing in trainer config."
            if hasattr(self._cfg.optim, 'sched'):
                sched_config = self._cfg.optim.sched
                sched_config['max_steps'] = self._trainer.max_steps
                self._scheduler = prepare_lr_scheduler(
                    optimizer=self._optimizer, scheduler_config=sched_config, train_dataloader=self._train_dl
                )

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def get_parameters(self):
        params = []
        for param_group in self._optimizer_param_groups:
            for param in param_group['params']:
                params.append(param)
        return params

    def training_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = self.process_batch(batch)

        tokens_loss = itemgetter("tokens_loss")(
            self(tokens_enc, tokens_dec, enc_mask, dec_mask, tokentype_ids=None, lm_labels=labels,)
        )

        loss = self.loss_func(loss_mask, tokens_loss)
        self.log('train_loss', loss)
        # Reduced loss for logging. This averages the loss across all workers unlike "loss" above which is specific to a DDP rank.
        reduced_loss = average_losses_across_data_parallel_group([loss])
        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self.log(
                'consumed_samples',
                self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
                prog_bar=True,
            )
            self._reduced_loss_buffer = []

        return loss

    def validation_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask = self.process_batch(batch)

        tokens_loss = itemgetter("tokens_loss")(
            self(tokens_enc, tokens_dec, enc_mask, dec_mask, tokentype_ids=None, lm_labels=labels,)
        )
        loss = self.loss_func(loss_mask, tokens_loss)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        return reduced_loss

    def validation_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        self.log('val_loss', averaged_loss[0], prog_bar=True)
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step - self.init_global_step))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, tokens_loss):
        """
        This function takes as input per-token loss and masks non-required values.
        """
        losses = tokens_loss.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_batch(self, batch):
        """Build the batch."""

        keys = ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask']
        datatype = torch.int64
        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask']
        dec_mask = data_b['dec_mask']

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask

    def build_train_valid_test_datasets(self):
        raise NotImplementedError("Please implement this method in child-class")

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if self._cfg.data.dataloader_type == 'single':
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self._cfg.micro_batch_size,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        elif self._cfg.data.dataloader_type == 'cyclic':
            batch_sampler = MegatronPretrainingRandomSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self._cfg.micro_batch_size,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            )
        else:
            raise Exception('{} dataloader type is not supported.'.format(self._cfg.dataloader_type))

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self._cfg.data.num_workers, pin_memory=True,
        )

    def setup(self, stage=None):
        resume_checkpoint_path = self.trainer.checkpoint_connector.resume_checkpoint_path
        if resume_checkpoint_path:
            try:
                init_consumed_samples = int(
                    float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", resume_checkpoint_path)[0])
                )
            except (ValueError, TypeError):
                logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
                init_consumed_samples = 0
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples

        """A PTL method to setup the training, validation and test datasets."""
        if stage == 'predict':
            return
        if self._train_dl is not None and self._validation_dl is not None:
            return
        self.build_train_valid_test_datasets()
        self.setup_training_data(self._cfg.data)
        self.setup_validation_data(self._cfg.data)
        self.setup_test_data(self._cfg.data)

    def on_pretrain_routine_start(self) -> None:
        # keep a copy of init_global_step
        self.init_global_step = self.trainer.global_step
        return super().on_pretrain_routine_start()

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def compute_consumed_samples(self, steps_since_resume=0):
        app_state = AppState()
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume
            * app_state.data_parallel_size
            * self._cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def configure_gradient_clipping(self, *args, **kwargs):
        """PTL hook to configure gradients.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if self.megatron_amp_o2:
            # grep fp32 master parameters for gradient clipping
            parameters = self._optimizer.get_parameters()
        else:
            parameters = self.get_parameters()

        grad_norm = clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)

        self.log('grad_norm', grad_norm, rank_zero_only=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        request = batch
        response = self.complete(request)
        logging.info(f"response: {response}")
        return response

    def decode(self, tokens_enc, enc_mask, num_tokens_to_generate, enc_input=None):
        # TODO: move method into a class inside MegatronTokenLevelEncoderDecoderModule (?)
        encoder_hidden_states, enc_output_mask = itemgetter("enc_output", "enc_output_mask")(
            self(
                encoder_input_ids=tokens_enc,
                decoder_input_ids=None,
                encoder_attn_mask=enc_mask,
                decoder_attn_mask=None,
                tokentype_ids=None,
                lm_labels=None,
                enc_hidden_states=None,
                enc_output_mask=None,
                output_enc_hidden_only=True,
                enc_input=enc_input,
            )
        )
        predicted_tokens_dec = (
            torch.LongTensor([self.tokenizer.bos_id] * tokens_enc.size(0)).unsqueeze(1).to(tokens_enc.device)
        )
        for _ in range(num_tokens_to_generate):
            dec_mask = predicted_tokens_dec != self.tokenizer.pad_id
            token_logits = itemgetter("token_logits")(
                self(
                    encoder_input_ids=tokens_enc,
                    decoder_input_ids=predicted_tokens_dec,
                    encoder_attn_mask=enc_mask,
                    decoder_attn_mask=dec_mask,
                    tokentype_ids=None,
                    lm_labels=None,
                    enc_hidden_states=encoder_hidden_states,
                    enc_output_mask=enc_output_mask,
                    output_enc_hidden_only=False,
                    enc_input=enc_input,
                )
            )
            token_logits = tensor_parallel.gather_from_tensor_model_parallel_region(token_logits)
            log_probs, token_ids = torch.max(nn.functional.log_softmax(token_logits, dim=-1), dim=-1)
            predicted_tokens_dec = torch.cat([predicted_tokens_dec, token_ids[:, -1].unsqueeze(1)], 1)

        return predicted_tokens_dec, log_probs

    def complete(self, request: Dict):
        """
            Autoregressively invokes language model in the inference mode
        Args:
            request: Dictionary with the following fields
                * prompt: a string which text the model should complete.
                * tokens_to_generate: how many tokens to generate while doing prompt completion.
        Returns:
            response: A python dictionary with the following fields
                * prompt: original text of the prompt
                * tokenized_prompt: list of (str) tokens from prompt
                * completion: a python dictionary with the following subfields:
                    * tokens: a list of triples (token, token_id, log_prob) comprising completion
                    * text: completion text (as a single string)

        """
        response = {}
        self.freeze()
        # naive greedy slow loop
        # TODO: add option for BeamSearchDecoder

        response['prompt'] = request['prompt'][0]
        response['completion'] = {}
        tokens_enc = request['masked_sample']

        response['masked_input'] = ' '.join(self.tokenizer.ids_to_tokens(tokens_enc[0]))
        enc_mask = tokens_enc != self.tokenizer.pad_id
        enc_mask = enc_mask < 0.5

        predicted_tokens_ids, log_probs = self.decode(tokens_enc, enc_mask, int(request['tokens_to_generate']))
        predicted_tokens_ids = predicted_tokens_ids.cpu().numpy()[0].tolist()
        log_probs = log_probs.cpu().numpy()[0].tolist()
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        else:
            predicted_tokens_ids = [id for id in predicted_tokens_ids if id != self.tokenizer.pad_id]
        predicted_tokens_dec = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        response['completion']['text'] = self.tokenizer.tokens_to_text(predicted_tokens_dec)
        response['completion']['tokens'] = list(zip(predicted_tokens_ids, predicted_tokens_dec, log_probs))
        self.unfreeze()
        return response

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        """Pad vocab size so it is divisible by model parallel size and
        still having GPU friendly size."""

        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        logging.info(
            f'Padded vocab_size: {after}, original vocab_size: {orig_vocab_size}, dummy tokens: {after - orig_vocab_size}.'
        )
        return after

    def _enable_nvidia_optimizations(self):
        "These optimizations are present in NVIDIA NGC PyTorch Containers"

        # Version check
        nvidia_torch_version = os.getenv('NVIDIA_PYTORCH_VERSION', None)
        if nvidia_torch_version is not None:
            NVIDIA_TORCH_MAJOR = int(nvidia_torch_version.split('.')[0])
            NVIDIA_TORCH_MINOR = int(nvidia_torch_version.split('.')[1])

            # Apex Persistent layer norm is supported from Nvidia PyTorch container v21.11
            if NVIDIA_TORCH_MAJOR < 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR < 11):
                self._cfg.persist_layer_norm = False

            if NVIDIA_TORCH_MAJOR >= 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR >= 11):
                # NVFUSER
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                torch._C._jit_override_can_fuse_on_cpu(False)
                torch._C._jit_override_can_fuse_on_gpu(False)
                torch._C._jit_set_texpr_fuser_enabled(False)
                torch._C._jit_set_nvfuser_enabled(True)
                torch._C._debug_set_autodiff_subgraph_inlining(False)

        else:
            # Not a Nvidia container. Dependency check is on users
            pass

    def list_available_models(self):
        pass
