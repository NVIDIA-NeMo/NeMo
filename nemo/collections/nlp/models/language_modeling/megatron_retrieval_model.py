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

import re
from typing import Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.retro_dataset import (
    build_mock_train_valid_test_datasets,
    build_train_valid_test_datasets,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.retrieval_token_level_encoder_decoder import (
    MegatronRetrievalTokenLevelEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import GradScaler
from nemo.utils import AppState, logging

try:
    from apex.transformer.enums import ModelType
    from apex.transformer import parallel_state

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    ModelType = ApexGuardDefaults()
    HAVE_APEX = False


__all__ = ["MegatronRetrievalModel"]


class MegatronRetrievalModel(MegatronBaseModel):
    """
    Megatron Retrieval enhanced language model
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)

        # TODO does not support PP yet
        self.model = self.model_provider_func(pre_process=True, post_process=True, add_encoder=True, add_decoder=True)

        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)

        if self.megatron_amp_o2:

            # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
            self.model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            self.model = Float16Module(module=self.model, precision=self.cfg.precision)

        # self.setup_optimizer_param_groups()
        if self.cfg.precision == 32:
            self.autocast_dtype = torch.float
        elif self.cfg.precision == 16:
            self.autocast_dtype = torch.half
        elif self.cfg.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')
        self.model.model_type = ModelType.encoder_and_decoder
        # self.grad_clip_pl_default = True

    def _build_tokenizer(self):
        self.tokenizer = get_nmt_tokenizer(
            library=self._cfg.tokenizer.library,
            model_name=self._cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer.model", self._cfg.tokenizer.model),
            vocab_file=self.register_artifact("tokenizer.vocab_file", self._cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("tokenizer.merge_file", self._cfg.tokenizer.merge_file),
            delimiter=self.cfg.tokenizer.get('delimiter', None),
            legacy=False,
        )

        # add pad special token
        if not hasattr(self.tokenizer, "pad_id"):
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        elif hasattr(self.tokenizer, "pad_id") and (self.tokenizer.pad_id is None or self.tokenizer.pad_id < 0):
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

    def model_provider_func(self, pre_process, post_process, add_encoder, add_decoder):
        # TODO: create get_encoder_decoder_model()here for different losses (e..g, nll, vae, mim)

        model = MegatronRetrievalTokenLevelEncoderDecoderModule(
            vocab_size=self.padded_vocab_size,
            hidden_size=self.cfg.hidden_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_attention_heads=self.cfg.num_attention_heads,
            ffn_hidden_size=self.cfg.ffn_hidden_size,
            apply_query_key_layer_scaling=self.cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=self.cfg.get('kv_channels', None),
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            init_method_std=self.cfg.get('init_method_std', 0.02),
            fp16_cross_entropy=self.cfg.get('fp16_lm_cross_entropy', False),
            use_cpu_initialization=self.cfg.get('use_cpu_initialization', False),
            hidden_dropout=self.cfg.get('hidden_dropout', 0.1),
            attention_dropout=self.cfg.get('attention_dropout', 0.1),
            precision=self.cfg.get('precision', 16),
            fp32_residual_connection=self.cfg.get('fp32_residual_connection', False),
            activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
            activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
            layernorm_epsilon=self.cfg.get('layernorm_epsilon', 1e-5),
            persist_layer_norm=self.cfg.get('persist_layer_norm', False),
            bias_gelu_fusion=self.cfg.get('bias_gelu_fusion', True),
            bias_dropout_add_fusion=self.cfg.get('bias_dropout_add_fusion', True),
            masked_softmax_fusion=self.cfg.get('masked_softmax_fusion', True),
            onnx_safe=self.cfg.get('onnx_safe', False),
            activation=self.cfg.get('activation', 'gelu'),
            bias=self.cfg.get('bias', True),
            normalization=self.cfg.get('normalization', 'layernorm'),
            headscale=self.cfg.get('headscale', False),
            transformer_block_type=self.cfg.get('transformer_block_type', 'pre_ln'),
            add_encoder=add_encoder,
            add_decoder=add_decoder,
            chunk_size=self.cfg.get('chunk_size', 64),  # the chunk size used to retrive
            enc_num_layers=self.cfg.get('enc_num_layers', 4),  # total number of encoder layers
            dec_num_layers=self.cfg.get('dec_num_layers', 6),  # total number of decoder layers
            enc_cross_attention=self.cfg.get('enc_cross_attention', [3]),  # layer numbers for cross attention
            dec_cross_attention=self.cfg.get(
                'dec_cross_attention', [3, 5]
            ),  # layer numbers for chunked cross attention
            add_position_embedding=self.cfg.get(
                'add_position_embedding', False
            ),  # whether use the absolute postion encoding
            tokenizer=self.tokenizer,
        )
        return model

    def forward(
        self,
        input_ids,
        input_attn_mask,
        retrieved_ids,
        retrieved_attn_mask,
        token_type_ids=None,
        labels=None,
        input_emb=None,
    ):
        output_tensor = self.model(
            input_ids=input_ids,
            input_attn_mask=input_attn_mask,
            retrieved_ids=retrieved_ids,
            retrieved_attn_mask=retrieved_attn_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            input_emb=input_emb,
        )
        return output_tensor

    def on_pretrain_routine_start(self) -> None:
        # keep a copy of init_global_step
        self.init_global_step = self.trainer.global_step
        return super().on_pretrain_routine_start()

    def training_step(self, batch, batch_idx):
        input_tokens_id = batch['tokens']
        input_attn_mask = batch['tokens_mask']
        loss_mask = batch['loss_mask']
        retrieved_ids = batch['retrieved_ids']
        retrieved_attn_mask = batch['retrieved_emb_mask']
        labels = batch['labels']

        loss = self(input_tokens_id, input_attn_mask, retrieved_ids, retrieved_attn_mask, labels=labels)
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(loss.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
        reduced_loss = average_losses_across_data_parallel_group([lm_loss])
        self._reduced_loss_buffer.append(reduced_loss[0])

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)
        # while async grad allreduce is enabled, bprop will keep moving forward without waiting for
        # the finish of async grad AR works. Hence, to guarantee the correctness of grads reduction,
        # we cannot start weight update until all async grad AR works are done.
        if self.megatron_amp_o2 and self.cfg.get('pipeline_model_parallel_size', 1) == 1:
            torch.cuda.synchronize()

        if self.megatron_amp_o2:
            # when using pipeline parallelism grads must be reduced after the pipeline (not asynchronously)
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # no pipeline, so use the default pytorch lightning way of doing all_reduce
            # self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)
            pass

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
        return lm_loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int, unused: Optional[int] = 0) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        # TODO: Replace with newer override for scheduler.step() instead of
        # search for plugins for fp16 GradScalar
        if self.trainer.precision_plugin is not None and isinstance(
            self.trainer.precision_plugin, NativeMixedPrecisionPlugin
        ):
            precision_plugin = self.trainer.precision_plugin

            if (
                hasattr(precision_plugin, 'scaler')
                and precision_plugin.scaler is not None
                and isinstance(precision_plugin.scaler, GradScaler)
            ):
                grad_scaler = precision_plugin.scaler

                # If the grad scaler skipped its optimizer step due to infs/nans,
                # decrement the step of all schedulers.
                if grad_scaler.optimizer_update_skipped is not None and grad_scaler.optimizer_update_skipped is True:
                    schedulers = self.trainer.lr_schedulers

                    if not schedulers or not self.trainer.lightning_module.automatic_optimization:
                        return

                    for scheduler in schedulers:
                        # Decrement the counter by 2, then perform a scheduler.step() to perform a no-up
                        # as well as update the optimizer lr in all param groups
                        scheduler['scheduler'].last_epoch -= 2
                        scheduler['scheduler'].step()

                    # Increase the max step count by 1

                    # Reset the optimizer update skipped to `None` - this is to prevent scheduler no-ops during
                    # accumulated gradient updates.
                    grad_scaler.optimizer_update_skipped = None

    def validation_step(self, batch, batch_idx):
        input_tokens_id = batch['tokens']
        input_attn_mask = batch['tokens_mask']
        loss_mask = batch['loss_mask']
        retrieved_ids = batch['retrieved_ids']
        retrieved_attn_mask = batch['retrieved_emb_mask']
        labels = batch['labels']
        loss = self(input_tokens_id, input_attn_mask, retrieved_ids, retrieved_attn_mask, labels=labels)
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(loss.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
        reduced_loss = average_losses_across_data_parallel_group([lm_loss])
        return reduced_loss

    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        averaged_loss = torch.stack(outputs).mean()
        self.log('val_loss', averaged_loss, prog_bar=True)
        # formula to compute the perplexity
        # https://towardsdatascience.com/the-relationship-between-perplexity-and-entropy-in-nlp-f81888775ccc
        self.log('perplexity', torch.exp(averaged_loss), prog_bar=True)
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step - self.init_global_step))
        return averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')
        self.log(
            'consumed_samples', self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
        )
        return averaged_loss

    def build_train_valid_test_datasets(self):
        logging.info('Building RETRO datasets.')
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size // self.cfg.tensor_model_parallel_size
        # Compute trianing micro-batch steps: total_global_batch_steps x grad_acumms_per_global_batch
        max_train_steps = self.trainer.max_steps * self.trainer.accumulate_grad_batches
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        if self.cfg.data.get('mock', False):
            self._train_ds, self._validation_ds, self._test_ds = build_mock_train_valid_test_datasets(
                cfg=self.cfg,
                trainer=self.trainer,
                splits_string=self.cfg.data.splits_string,
                tokenizer=self.tokenizer,
                mock_data_size=self.cfg.data.get('mock_data_size', 10000),
            )
        else:
            self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
                cfg=self.cfg,
                trainer=self.trainer,
                data_prefix=self.cfg.data.data_prefix,
                data_impl=self.cfg.data.data_impl,
                splits_string=self.cfg.data.splits_string,
                train_valid_test_num_samples=train_valid_test_num_samples,
                seq_length=self.cfg.data.seq_length,
                seed=self.cfg.seed,
                skip_warmup=self.cfg.data.get('skip_warmup', True),
                tokenizer=self.tokenizer,
                retrieval_prefix=self.cfg.data.retrieval_prefix,
                knn_map_path=self.cfg.data.knn_index,
            )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building RETRO datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def setup(self, stage=None):
        resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path
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
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return int(consumed_samples)

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = get_params_for_weight_decay_optimization([self.model])

    def list_available_models(self):
        pass
