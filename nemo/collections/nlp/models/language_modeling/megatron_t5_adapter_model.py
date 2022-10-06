# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# This code has been adapted from the following private repo: https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/tree/prompt-learning/prefix_tuning_v2
# Adapted by: @adithyare


from typing import Any

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.data.language_modeling.megatron.t5_prompt_learning_dataset import T5Sentinel
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.models.language_modeling.megatron_t5_prompt_learning_model import (
    MegatronT5PromptLearningModel,
)
from nemo.collections.nlp.modules.common import VirtualPromptStyle
from nemo.collections.nlp.modules.common.megatron.parallel_adapters import (
    InfusedAdapterConfig,
    ParallelLinearAdapterConfig,
)
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.core.classes.mixins import adapter_mixins
from nemo.utils import logging


class MegatronT5BaseAdapterModel(MegatronT5PromptLearningModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        self.adapter_name_keys = []

    def forward(
        self, input_ids, dec_input, enc_mask, dec_mask, position_ids, taskname_ids, labels=None, inference=False,
    ):
        # Call forward on T5 model with preprocessed embeddings
        if self.autocast_dtype == torch.float32:
            output = self.frozen_model.enc_dec_model(
                enc_input_ids=input_ids,
                enc_attn_mask=enc_mask,
                dec_input_ids=dec_input,
                dec_attn_mask=dec_mask,
                token_type_ids=None,
                labels=labels,
                output_enc_hidden_only=False,
                enc_input=None,
            )
        else:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                output = self.frozen_model.enc_dec_model(
                    enc_input_ids=input_ids,
                    enc_attn_mask=enc_mask,
                    dec_input_ids=dec_input,
                    dec_attn_mask=dec_mask,
                    token_type_ids=None,
                    labels=labels,
                    output_enc_hidden_only=False,
                    enc_input=None,
                )

        return output, None

    def setup(self, stage=None):
        if stage == 'predict' or self.virtual_prompt_style == VirtualPromptStyle.INFERENCE:
            self.frozen_model.freeze()
            return

        self.setup_test_data()
        if stage == 'test':
            return

        self.setup_training_data()
        self.setup_validation_data()
        logging.info(f'setup completed:\n{self.frozen_model.summarize()}')

    def on_train_end(self):
        # Save the best nemo model
        self.save_to(save_path=self.cfg.nemo_path)

    def inference_step(self, batch, batch_idx, inference=False):
        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        mode = self.training
        self.eval()

        loss_mean = self.fwd_bwd_step(batch, batch_idx, forward_only=True)

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=enc_input,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=None,
        )

        processed_inputs, processed_preds, processed_labels = [], [], []
        preds = predicted_token_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        enc_inputs = enc_input.cpu().numpy().tolist()

        for i, (enc_input, pred, label) in enumerate(zip(enc_inputs, preds, labels)):
            if self.tokenizer.eos_id in pred:
                idx = pred.index(self.tokenizer.eos_id)
                pred = pred[:idx]

            pred = [id for id in pred if id not in self.tokenizer.tokenizer.additional_special_tokens_ids]
            label = [id for id in label if id not in self.tokenizer.tokenizer.additional_special_tokens_ids]
            enc_input = [id for id in enc_input if id not in self.tokenizer.tokenizer.additional_special_tokens_ids]

            pred = self.tokenizer.ids_to_text(pred)
            label = self.tokenizer.ids_to_text(label)
            enc_input = self.tokenizer.ids_to_text(enc_input)

            processed_preds.append(pred)
            processed_labels.append(label)
            processed_inputs.append(enc_input)

        self.train(mode=mode)
        return {
            'loss': loss_mean,
            'predicted_token_ids': processed_preds,
            'labels': processed_labels,
            'enc_inputs': processed_inputs,
        }

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:

        enc_input, dec_input, labels, loss_mask, enc_mask, dec_mask, position_ids, taskname_ids = batch

        predicted_token_ids, log_probs = self.frozen_model.decode(
            tokens_enc=enc_input,
            enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length,
            encoder_input=None,
        )

        processed_preds = []
        processed_labels = []
        processed_inputs = []

        preds = predicted_token_ids.cpu().numpy().tolist()
        enc_inputs = enc_input.cpu().numpy().tolist()

        if labels is not None:
            labels = labels.cpu().numpy().tolist()
        else:
            labels = [None] * len(preds)

        for i, (enc_input, pred, label) in enumerate(zip(enc_inputs, preds, labels)):
            if self.tokenizer.eos_id in pred:
                idx = pred.index(self.tokenizer.eos_id)
                pred = pred[:idx]

            pred = [
                id
                for id in pred
                if id not in self.tokenizer.tokenizer.additional_special_tokens_ids
                and id not in self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
            ]  # delete the sentinel token at the beginning of prediction

            pred = self.tokenizer.ids_to_text(pred)
            processed_preds.append(pred)

            enc_input = [
                id for id in enc_input if id not in self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
            ]  # delete the sentinel token added to the end of input

            input = self.tokenizer.ids_to_text(enc_input)
            processed_inputs.append(input)

            if label:
                label = [
                    id
                    for id in label
                    if id not in self.tokenizer.tokenizer.additional_special_tokens_ids
                    and id not in self.tokenizer.text_to_ids(T5Sentinel.FIRST.value)
                ]  # delete the sentinel token at the beginning of label

                label = self.tokenizer.ids_to_text(label)
            processed_labels.append(label)

        return {
            'enc_input': processed_inputs,
            'predicted_token_ids': processed_preds,
            'log_probs': log_probs,
            'labels': processed_labels,
        }

    def setup_optimizer_param_groups(self):
        """
        ModelPT override. Optimizer will get self._optimizer_param_groups. 
        Makes two optimizer param groups, one for the frozen model params
        and one for the prompt-table/prompt-encoder params. The learning 
        rate for the frozen model's params will always be zero effectively
        freezing the model's params but still allowing for the needed gradients
        to be passed around in pipeline parallel models. The prompt-encoder 
        and/or prompt table will use the learning rate set by the user. 
        """
        self.frozen_model.freeze()  # Freeze the entire model
        opt_params = []
        for _, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                module.set_enabled_adapters(enabled=True)
                module.unfreeze_enabled_adapters()  # selectively unfreeze the adapter modules.
                opt_params += [p for p in module.parameters()]

        self._optimizer_param_groups = [{'params': opt_params}]
        logging.info(f'Optimizer groups set:\n{self.frozen_model.summarize()}')

    def get_forward_output_only_func(self):
        """
        Used for generate method only for now.
        """

        def fwd_output_only_func(batch, model):
            extra_arg = {}
            (
                tokens,
                attention_mask,
                position_ids,
                task_ids,
                set_inference_key_value_memory,
                inference_max_sequence_len,
            ) = batch

            tokens = tokens.cuda()
            attention_mask = attention_mask.cuda()
            position_ids = position_ids.cuda()
            task_ids = task_ids.cuda()
            extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
            extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()

            output_tensor = model(tokens, position_ids, attention_mask, task_ids, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        state_dict_ = {}
        for name, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.adapter_layer[adapter_key]
                    state_adapter_key = ':'.join([name, adapter_key])
                    state_dict_[state_adapter_key] = adapter_module.state_dict()

                module.set_enabled_adapters(enabled=True)
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values 
        only for the adapter parameters.
        """
        for name, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    adapter_module = module.adapter_layer[adapter_key]
                    state_adapter_key = ':'.join([name, adapter_key])
                    adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)


class MegatronT5AdapterLearningModel(MegatronT5BaseAdapterModel):
    """
    TODO  (@adithyare)
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        assert cfg.adapter_tuning.get('adapter_dim', 0) > 0, "adapter_dim has not been set."
        assert (
            cfg.adapter_tuning.adapter_dim % cfg.tensor_model_parallel_size == 0
        ), "The adapter dim should be divisible by tensor_model_parallel_size."
        assert cfg.adapter_tuning.type in [
            'linear_adapter',
            'parallel_adapter',
        ], "Adapter type should be 'linear_adapter' or 'parallel_adapter'"

        self.adapter_name_keys = ['adapter_1', 'adapter_2']
        frozen_model_cfg = MegatronT5Model.restore_from(
            cfg.get('language_model_path'), trainer=trainer, return_config=True
        )
        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )

        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')

        if cfg.adapter_tuning.type == "parallel_adapter":
            adapter_cfg = ParallelLinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.adapter_dim,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                norm_type=cfg.adapter_tuning.get('norm_type', 'mixedfusedlayernorm'),
                column_init_method=cfg.adapter_tuning.get('column_init_method', 'xavier'),
                row_init_method=cfg.adapter_tuning.get('row_init_method', 'zero'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )
        else:
            adapter_cfg = LinearAdapterConfig(
                in_features=frozen_model_cfg.hidden_size,
                dim=cfg.adapter_tuning.adapter_dim,
                norm_position=cfg.adapter_tuning.get('norm_position', 'pre'),
                dropout=cfg.adapter_tuning.adapter_dropout,
            )

        self.frozen_model.freeze()
        for _, module in self.frozen_model.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in self.adapter_name_keys:
                    module.add_adapter(
                        name=adapter_key, cfg=adapter_cfg,
                    )

        logging.info(f'After adding adapters:\n{self.frozen_model.summarize()}')

    @classmethod
    def list_available_models(cls):
        pass


class MegatronT5InfusedAdapterModel(MegatronT5BaseAdapterModel):
    """
    MegatronGPTInfusedAdapterModel is a model that combines a base model (GPTModel) with a "Infused Adapter that can Inhibiting and Amplify Inner Activations", known as IA3.
    This class supports the addition of IA3 into a transformer based LM as described in Liu et al. (https://arxiv.org/pdf/2205.05638.pdf)

    Three adapter's are inserted into each Transformer layer in the base GPT Model. Each adapter is basically a vector that simply scales the key, value or ffn hidden representations.

    It is assumed that these set of adapters will then be trained for a specific task.
    Once trained, the adapter weights will be saved and can be re-loaded 
    and infused into the same GPT Model for inference. 
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer)
        frozen_model_cfg = MegatronT5Model.restore_from(
            cfg.get('pretrained_language_model_path'), trainer=trainer, return_config=True
        )
        for _, layer in self.frozen_model.named_modules():
            if hasattr(layer, 'activations_checkpoint_method'):
                layer.activations_checkpoint_method = (
                    None  # (@adithyare) adapter learning does not support activations checkpointing atm.
                )

        self.encoder_adapter_name_keys = ['mlp_infused_adapter', 'key_infused_adapter', 'value_infused_adapter']
        self.decoder_adapter_name_keys = self.encoder_adapter_name_keys + [
            'inter_key_infused_adapter',
            'inter_value_infused_adapter',
        ]
        self.frozen_model.freeze()
        logging.info(f'Before adding adapters:\n{self.frozen_model.summarize()}')
        encoder = self.frozen_model.enc_dec_model.enc_dec_model.encoder
        self._add_adapters_to_component(encoder, frozen_model_cfg, self.encoder_adapter_name_keys)
        logging.info(f'After adding encoder adapters:\n{self.frozen_model.summarize()}')
        decoder = self.frozen_model.enc_dec_model.enc_dec_model.decoder
        self._add_adapters_to_component(decoder, frozen_model_cfg, self.decoder_adapter_name_keys)
        logging.info(f'After adding all adapters:\n{self.frozen_model.summarize()}')

    def _add_adapters_to_component(self, component, layer_cfg, adapter_name_keys):
        for _, module in component.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in adapter_name_keys:
                    if adapter_key == 'mlp_infused_adapter':
                        cfg = InfusedAdapterConfig(
                            in_features=layer_cfg.ffn_hidden_size // layer_cfg.tensor_model_parallel_size
                        )
                    else:
                        if layer_cfg.get('kv_channels', None):
                            cfg = InfusedAdapterConfig(
                                in_features=layer_cfg.kv_channels
                                * layer_cfg.num_attention_heads
                                // layer_cfg.tensor_model_parallel_size
                            )
                        else:
                            cfg = InfusedAdapterConfig(
                                in_features=layer_cfg.hidden_size // layer_cfg.tensor_model_parallel_size
                            )
                    module.add_adapter(name=adapter_key, cfg=cfg)

    def _component_state_dict(self, component_name, component, adapter_name_keys):
        state_dict_ = {}
        for name, module in component.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in adapter_name_keys:
                    adapter_module = module.adapter_layer[adapter_key]
                    state_adapter_key = ':'.join([component_name, name, adapter_key])
                    state_dict_[state_adapter_key] = adapter_module.state_dict()

                module.set_enabled_adapters(enabled=True)
        return state_dict_

    def _load_component_state_dict(
        self, component_name, component, adapter_name_keys, state_dict, strict: bool = True
    ):
        for name, module in component.named_modules():
            if isinstance(module, adapter_mixins.AdapterModuleMixin):
                for adapter_key in adapter_name_keys:
                    adapter_module = module.adapter_layer[adapter_key]
                    state_adapter_key = ':'.join([component_name, name, adapter_key])
                    adapter_module.load_state_dict(state_dict[state_adapter_key], strict)
                module.set_enabled_adapters(enabled=True)

    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        """
        Creates a state_dict using only the adapter parameters.
        This ensures that this wrapper class will only checkpoint the adapter
        weights and not the rest of the base GPT Model.
        """
        encoder = self.frozen_model.enc_dec_model.enc_dec_model.encoder
        encoder_state_dict = self._component_state_dict('encoder', encoder, self.encoder_adapter_name_keys)
        decoder = self.frozen_model.enc_dec_model.enc_dec_model.decoder
        decoder_state_dict = self._component_state_dict('decoder', decoder, self.decoder_adapter_name_keys)
        state_dict_ = {
            **encoder_state_dict,
            **decoder_state_dict,
        }  # merge the two state dicts (does not check for collisions in keys)
        return state_dict_

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Loads a state_dict expecting the state_dict to contain key,values 
        only for the adapter parameters.
        """
        encoder = self.frozen_model.enc_dec_model.enc_dec_model.encoder
        self._load_component_state_dict('encoder', encoder, self.encoder_adapter_name_keys, state_dict, strict)
        decoder = self.frozen_model.enc_dec_model.enc_dec_model.decoder
        self._load_component_state_dict('decoder', decoder, self.decoder_adapter_name_keys, state_dict, strict)

    @classmethod
    def list_available_models(cls):
        pass
