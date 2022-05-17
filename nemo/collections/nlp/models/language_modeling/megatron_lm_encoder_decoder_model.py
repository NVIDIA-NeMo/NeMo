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
from typing import Any, Dict, Optional

import torch
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.token_level_encoder_decoder import (
    MegatronTokenLevelEncoderDecoderModule,
)
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.parts.nlp_overrides import GradScaler
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.optim import MainParamsOptimizerWrapper, prepare_lr_scheduler
from nemo.utils import AppState, logging

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.enums import ModelType
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.pipeline_parallel.schedules.common import build_model
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.utils import (
        get_num_microbatches,
        _reconfigure_microbatch_calculator,
        get_micro_batch_size,
    )

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    ModelType = ApexGuardDefaults()
    HAVE_APEX = False


__all__ = ["MegatronLMEncoderDecoderModel"]


class MegatronLMEncoderDecoderModel(MegatronBaseModel):
    """
    Megatron encoder-decoder base class
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        if cfg.get('pipeline_model_parallel_size', 1) > 1:
            if cfg.get('pipeline_model_parallel_split_rank', 0) <= 0:
                raise ValueError(
                    f"pipeline_model_parallel_split_rank must be > 0 when using pipeline_model_parallel_size > 1"
                )

        # Make sure trainer.accumulate_grad_batches is 1.
        self._validate_trainer()

        # TODO: Not sure how to use lists of modules with PTL.
        # This means we can only use pipeline parallelism without the interleaved schedule.
        self.enc_dec_model = build_model(
            model_provider_func=self.model_provider_func,
            wrap_with_ddp=False,
            model_type=ModelType.encoder_and_decoder,
        )[0]

        # We don't need to call it explicitly? Since it is a pytorch lightning hook function
        # self.setup_optimizer_param_groups()

        self.megatron_amp_o2 = cfg.get('megatron_amp_O2', False)

        if self.megatron_amp_o2:

            # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
            self.enc_dec_model.cuda(torch.cuda.current_device())

            # Model wrapper to convert both model and inputs to half precision
            self.enc_dec_model = Float16Module(module=self.enc_dec_model, precision=cfg.precision)

        if self.cfg.precision == 32:
            self.autocast_dtype = torch.float
        elif self.cfg.precision == 16:
            self.autocast_dtype = torch.half
        elif self.cfg.precision == 'bf16':
            self.autocast_dtype = torch.bfloat16
        else:
            raise ValueError('precision must be in [32, 16, "bf16"]')

        self.enc_dec_model.model_type = ModelType.encoder_and_decoder

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        self._optimizer_param_groups = get_params_for_weight_decay_optimization([self.enc_dec_model])

    def model_provider_func(self, pre_process, post_process, add_encoder, add_decoder):
        # TODO: create get_encoder_decoder_model()here for different losses (e..g, nll, vae, mim)

        model = MegatronTokenLevelEncoderDecoderModule(
            encoder_arch=self.cfg.encoder_arch,
            decoder_arch=self.cfg.decoder_arch,
            vocab_size=self.padded_vocab_size,
            hidden_size=self.cfg.hidden_size,
            max_position_embeddings=self.cfg.max_position_embeddings,
            num_layers=self.cfg.num_layers,
            num_attention_heads=self.cfg.num_attention_heads,
            apply_query_key_layer_scaling=self.cfg.get('apply_query_key_layer_scaling', True),
            kv_channels=self.cfg.get('kv_channels', None),
            ffn_hidden_size=self.cfg.ffn_hidden_size,
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
            transformer_block_type=self.cfg.get('transformer_block_type', 'pre_ln'),
            headscale=self.cfg.get('headscale', False),
            add_encoder=add_encoder,
            add_decoder=add_decoder,
        )
        return model

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attn_mask,
        decoder_attn_mask,
        token_type_ids=None,
        lm_labels=None,
        enc_hidden_states=None,
        enc_output_mask=None,
        output_enc_hidden_only=False,
        enc_input=None,
    ):
        output_tensor = self.enc_dec_model(
            enc_input_ids=encoder_input_ids,
            dec_input_ids=decoder_input_ids,
            enc_attn_mask=encoder_attn_mask,
            dec_attn_mask=decoder_attn_mask,
            token_type_ids=token_type_ids,
            labels=lm_labels,
            enc_hidden_states=enc_hidden_states,
            enc_output_mask=enc_output_mask,
            output_enc_hidden_only=output_enc_hidden_only,
            enc_input=enc_input,
        )

        return output_tensor

    def training_step(self, batch, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            Batch should be a list of microbatches and those microbatches should on CPU.
            Microbatches are then moved to GPU during the pipeline.
            The list of microbatches is then piped through the pipeline using Apex fwd/bwd functions.
        """
        # we zero grads here because we also call backward in the apex fwd/bwd functions
        self._optimizer.zero_grad()

        # we prepare the micro batches for the apex fwd/bwd function
        batch_for_pipeline = self.process_global_batch(batch)
        encoder_seq_length = batch_for_pipeline[0].size(1)
        decoder_seq_length = batch_for_pipeline[1].size(1)
        tensor_shape = [encoder_seq_length, get_micro_batch_size(), self.cfg.hidden_size]

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            losses_reduced_per_micro_batch = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch_for_pipeline,
                model=self.enc_dec_model,
                forward_only=False,
                tensor_shape=tensor_shape,
                decoder_sequence_length=decoder_seq_length,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            )
        else:
            # no pipeline parallelism so we reduce grads asynchronously
            if self.megatron_amp_o2:
                custom_sync_context_handler = self._optimizer.no_sync
            else:
                # TODO: enable async grad all reduce for O1/autocast mixed precision training
                custom_sync_context_handler = None
            losses_reduced_per_micro_batch = forward_backward_no_pipelining(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch_for_pipeline,
                model=self.enc_dec_model,
                forward_only=False,
                tensor_shape=tensor_shape,
                decoder_sequence_length=decoder_seq_length,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
                custom_sync_context_handler=custom_sync_context_handler,
            )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            loss_mean = torch.tensor(0.0).cuda()

        if self.megatron_amp_o2:
            # when using pipeline parallelism grads must be reduced after the pipeline (not asynchronously)
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we allreduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # when using pipeline parallelism, we need keep the word and position embeddings in sync
            self.allreduce_word_and_position_embeddings()

        # while async grad allreduce is enabled, bprop will keep moving forward without waiting for
        # the finish of async grad AR works. Hence, to guarantee the correctness of grads reduction,
        # we cannot start weight update until all async grad AR works are done.
        if self.megatron_amp_o2 and self.cfg.get('pipeline_model_parallel_size', 1) == 1:
            torch.cuda.synchronize()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.cfg.precision == 16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True)
        self.log('global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True)
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
            prog_bar=True,
            rank_zero_only=True,
        )

        return loss_mean

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
                    self.trainer.fit_loop.max_steps = self.trainer.fit_loop.max_steps + 1

                    # Reset the optimizer update skipped to `None` - this is to prevent scheduler no-ops during
                    # accumulated gradient updates.
                    grad_scaler.optimizer_update_skipped = None

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

    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks.
           Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/model/distributed.py#L188
        """
        # Bucketize and all-reduce
        buckets = {}
        # Pack the buckets.
        for param in self.parameters():
            if param.requires_grad and param.grad is not None:
                tp = param.data.type()
                if tp not in buckets:
                    buckets[tp] = []
                buckets[tp].append(param)
                # param.main_grad = param.grad

        # For each bucket, all-reduce and copy all-reduced grads.
        for tp in buckets:
            bucket = buckets[tp]
            grads = [param.grad.data for param in bucket]
            coalesced = torch._utils._flatten_dense_tensors(grads)
            coalesced /= parallel_state.get_data_parallel_world_size()
            torch.distributed.all_reduce(coalesced, group=parallel_state.get_data_parallel_group())
            for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    def allreduce_word_and_position_embeddings(self):

        # Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/training.py#L407
        # All-reduce word_embeddings' grad across first and last stages to ensure
        # that word_embeddings parameters stay in sync.
        # This should only run for models that support pipelined model parallelism
        # (BERT and GPT-2).
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
            parallel_state.is_rank_in_embedding_group()
        ):
            if self.enc_dec_model.share_word_embeddings:
                word_embeddings_weight = self.enc_dec_model.word_embeddings_weight()
                if self.megatron_amp_o2:
                    # O2 recipe stores a "main" copy of weights and grads
                    grad = word_embeddings_weight.main_grad
                else:
                    grad = word_embeddings_weight.grad
                torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())

                # All reduce position embeddings for T5.
                if (
                    parallel_state.is_rank_in_position_embedding_group()
                    and parallel_state.get_pipeline_model_parallel_world_size() > 1
                    and parallel_state.get_pipeline_model_parallel_split_rank() is not None
                ):
                    position_embeddings_weight = self.enc_dec_model.position_embeddings_weight()
                    if self.megatron_amp_o2:
                        grad = position_embeddings_weight.main_grad
                    else:
                        grad = position_embeddings_weight.grad
                    torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())

    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            encoder_input_ids, decoder_input_ids, loss_mask, lm_labels, encoder_attn_mask, decoder_attn_mask = batch
            output = model(
                encoder_input_ids,  # enc_input_ids
                encoder_attn_mask,  # enc_attn_mask
                decoder_input_ids,  # dec_input_ids
                decoder_attn_mask,  # dec_attn_mask
                None,  # token_type_ids
                lm_labels,  # labels
                None,  # enc_hidden_states
            )

            def loss_func(output_tensor):
                loss = self.loss_func(loss_mask, output_tensor)
                reduced_loss = average_losses_across_data_parallel_group([loss])
                return loss, {'avg': reduced_loss}

            return output, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(batch, model):
            batch = [x.cuda(non_blocking=True) for x in batch]
            if len(batch) == 4:
                encoder_input_ids, decoder_input_ids, encoder_attn_mask, decoder_attn_mask = batch
                enc_input = None
            elif len(batch) == 5:
                encoder_input_ids, decoder_input_ids, encoder_attn_mask, decoder_attn_mask, enc_input = batch
            else:
                raise ValueError("wrong number of items in the batch")
            output = model(
                encoder_input_ids,  # enc_input_ids
                encoder_attn_mask,  # enc_attn_mask
                decoder_input_ids,  # dec_input_ids
                decoder_attn_mask,  # dec_attn_mask
                None,  # token_type_ids
                None,  # labels
                enc_input,  # enc_hidden_states
            )

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output, id_func

        return fwd_output_only_func

    def validation_step(self, batch, batch_idx):
        batch_for_pipeline = self.process_global_batch(batch)
        encoder_seq_length = batch_for_pipeline[0].size(1)
        decoder_seq_length = batch_for_pipeline[1].size(1)

        tensor_shape = [encoder_seq_length, get_micro_batch_size(), self.cfg.hidden_size]

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            losses_reduced_per_micro_batch = forward_backward_pipelining_without_interleaving(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch_for_pipeline,
                model=self.enc_dec_model,
                forward_only=True,
                tensor_shape=tensor_shape,
                decoder_sequence_length=decoder_seq_length,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            )
        else:
            losses_reduced_per_micro_batch = forward_backward_no_pipelining(
                forward_step_func=self.get_forward_output_and_loss_func(),
                batch=batch_for_pipeline,
                model=self.enc_dec_model,
                forward_only=True,
                tensor_shape=tensor_shape,
                decoder_sequence_length=decoder_seq_length,
                dtype=self.autocast_dtype,
                grad_scaler=self.trainer.precision_plugin.scaler if self.cfg.precision == 16 else None,
            )

        if losses_reduced_per_micro_batch:
            # average loss across micro batches
            loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.concat(loss_tensors_list)
            loss_mean = loss_tensor.mean()
        else:
            # we're not on the last pipeline stage so no losses
            loss_mean = []

        return loss_mean

    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss
            averaged_loss = torch.stack(outputs).mean()
        else:
            averaged_loss = torch.tensor(0.0).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())
        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
            rank_zero_only=True,
        )

        return averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        if not outputs:
            return
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss
            averaged_loss = torch.stack(outputs).mean()
        else:
            averaged_loss = torch.tensor(0.0).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())
        self.log('test_loss', averaged_loss, prog_bar=True, rank_zero_only=True)
        self.log(
            'consumed_samples',
            self.compute_consumed_samples(self.trainer.global_step - self.init_global_step),
            rank_zero_only=True,
        )

        return averaged_loss

    def loss_func(self, loss_mask, tokens_loss):
        """
        This function takes as input per-token loss and masks non-required values.
        """
        losses = tokens_loss.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_micro_batch(self, micro_batch):
        """ Micro batch returned by MegatronT5 dataloader"""

        data_b = micro_batch

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask']
        dec_mask = data_b['dec_mask']

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask

    def _process_global_batch_without_megatron_batch_sampler(self, global_batch, tokenizer=None):
        """ Prepares the global batch for apex fwd/bwd functions.
            Global batch is a list of micro batches.
        """
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        text_enc_list = []
        text_dec_list = []
        labels_list = []
        loss_mask_list = []
        enc_mask_list = []
        dec_mask_list = []

        # Determine the maximum encoder and decoder sequence lengths amongst microbatches and pad each microbatch to the max seq length.
        # NOTE: This should only happen for model finetuning where we pad dynamically. Training uses fixed training shapes.

        max_enc_seq_lenth = max([micro_batch['text_enc'].shape[1] for micro_batch in global_batch])
        max_dec_seq_lenth = max([micro_batch['text_dec'].shape[1] for micro_batch in global_batch])

        for micro_batch in global_batch:
            text_enc, text_dec, loss_mask, labels, enc_mask, dec_mask = self.process_micro_batch(micro_batch)
            # Check if encoder sequence length < max encoder sequence length of the global batch and pad.
            if text_enc.shape[1] < max_enc_seq_lenth:
                text_enc = torch.nn.functional.pad(
                    text_enc, (0, max_enc_seq_lenth - text_enc.shape[1], 0, 0), 'constant', tokenizer.pad_id
                )
                enc_mask = torch.nn.functional.pad(
                    enc_mask, (0, max_enc_seq_lenth - enc_mask.shape[1], 0, 0), 'constant', 0
                )
            if text_dec.shape[1] < max_dec_seq_lenth:
                text_dec = torch.nn.functional.pad(
                    text_dec, (0, max_dec_seq_lenth - text_dec.shape[1], 0, 0), 'constant', tokenizer.pad_id
                )
                dec_mask = torch.nn.functional.pad(
                    dec_mask, (0, max_dec_seq_lenth - dec_mask.shape[1], 0, 0), 'constant', 0
                )
                labels = torch.nn.functional.pad(
                    labels, (0, max_dec_seq_lenth - labels.shape[1], 0, 0), 'constant', tokenizer.pad_id
                )
                loss_mask = torch.nn.functional.pad(
                    loss_mask, (0, max_dec_seq_lenth - loss_mask.shape[1], 0, 0), 'constant', 0
                )
            text_enc_list.append(text_enc)
            text_dec_list.append(text_dec)
            labels_list.append(labels)
            loss_mask_list.append(loss_mask)
            enc_mask_list.append(enc_mask)
            dec_mask_list.append(dec_mask)

        # Concatenate to (num_microbatches x micro_batch_size x seq_len)
        tokens_enc_tensor = torch.concat(text_enc_list, dim=0)
        tokens_dec_tensor = torch.concat(text_dec_list, dim=0)
        labels_tensor = torch.concat(labels_list, dim=0)
        loss_mask_tensor = torch.concat(loss_mask_list, dim=0)
        enc_mask_tensor = torch.concat(enc_mask_list, dim=0)
        dec_mask_tensor = torch.concat(dec_mask_list, dim=0)

        return {
            'text_enc': tokens_enc_tensor,
            'text_dec': tokens_dec_tensor,
            'loss_mask': loss_mask_tensor,
            'labels': labels_tensor,
            'enc_mask': enc_mask_tensor,
            'dec_mask': dec_mask_tensor,
        }

    def process_global_batch(self, global_batch):
        return [
            global_batch["text_enc"],
            global_batch["text_dec"],
            global_batch["loss_mask"],
            global_batch["labels"],
            global_batch["enc_mask"],
            global_batch["dec_mask"],
        ]

    def build_train_valid_test_datasets(self):
        raise NotImplementedError("Please implement this method in child-class")

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self._cfg.data, 'dataloader_type') and self._cfg.data.dataloader_type is not None:
            if self._cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self._cfg.micro_batch_size,
                    global_batch_size=self._cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self._cfg.get('drop_last', True),
                )
            elif self._cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self._cfg.micro_batch_size,
                    global_batch_size=self._cffg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self._cfg.get('drop_last', True),
                )
            else:
                raise Exception(f'{self._cfg.dataloader_type} dataloader type is not supported.')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self._cfg.data.num_workers, pin_memory=True,
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

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            self.enc_dec_model.sync_initial_word_embeddings()
            self.enc_dec_model.sync_initial_position_embeddings()

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            consumed_samples = self.compute_consumed_samples(0)
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def on_pretrain_routine_start(self) -> None:
        # keep a copy of init_global_step
        self.init_global_step = self.trainer.global_step
        return super().on_pretrain_routine_start()

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def configure_optimizers(self):
        self.setup_optimization()

        # Wrap the baseline optimizer with the optimizer class with master parameters
        if self.megatron_amp_o2 and self._optimizer is not None:
            if self.cfg.precision == 'bf16':
                fp32_grad_accum = True
                contiguous_grad_bucket = True

            elif self.cfg.precision == 16:
                fp32_grad_accum = False
                # TODO: contiguous grad bucket for fp16 is also planned to be supported
                contiguous_grad_bucket = False

            # if using tensor parallel only, we can use async grad all-reduce
            if self.cfg.get('pipeline_model_parallel_size', 1) == 1:
                async_grad_allreduce = True
            else:
                async_grad_allreduce = False

            self._optimizer = MainParamsOptimizerWrapper(
                self._optimizer,
                fp32_grad_accum=fp32_grad_accum,
                contiguous_grad_bucket=contiguous_grad_bucket,
                async_grad_allreduce=async_grad_allreduce,
                grad_allreduce_chunk_size_mb=self.cfg.get('grad_allreduce_chunk_size_mb', 125),
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

    def compute_consumed_samples(self, steps_since_resume=0):
        app_state = AppState()
        consumed_samples = (
            self.init_consumed_samples
            + steps_since_resume * app_state.data_parallel_size * self.cfg.micro_batch_size * get_num_microbatches()
        )
        return int(consumed_samples)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        request = batch
        response = self.complete(request)
        logging.info(f"response: {response}")
        return response

    def decode(self, tokens_enc, enc_mask, num_tokens_to_generate, encoder_input=None, tokenizer=None):
        # If classes that inherit from this class are using a different tokenizer,
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        app_state = AppState()
        global_batch_per_gpu = tokens_enc.size(0)

        num_micro_batches_before_decode = get_num_microbatches()
        # Reconfigure microbatch calculator here to set num microbatches to 1 while decoding since its not clear how to decode with "grad acc".
        # TODO: reconfigure back to how things were before decode?
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_per_gpu,  # Make sure that there is no "grad acc" while decoding.
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
        predicted_tokens_dec = (
            torch.LongTensor([tokenizer.bos_id] * global_batch_per_gpu).unsqueeze(1).to(tokens_enc.device)
        )
        encoder_seq_length = tokens_enc.size(1)
        tensor_shape = [encoder_seq_length, global_batch_per_gpu, self.cfg.hidden_size]
        assert predicted_tokens_dec.size(0) == global_batch_per_gpu

        for i in range(num_tokens_to_generate):
            # No microbatches in decoding. Just the global batch.
            decoder_seq_length = predicted_tokens_dec.size(1)
            dec_mask = predicted_tokens_dec != tokenizer.pad_id

            if encoder_input is not None:
                batch_for_pipeline = [tokens_enc, predicted_tokens_dec, enc_mask, dec_mask, encoder_input]
            else:
                batch_for_pipeline = [tokens_enc, predicted_tokens_dec, enc_mask, dec_mask]
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                output_tensor = forward_backward_pipelining_without_interleaving(
                    forward_step_func=self.get_forward_output_only_func(),
                    batch=batch_for_pipeline,
                    model=self.enc_dec_model,
                    forward_only=True,
                    tensor_shape=tensor_shape,
                    decoder_sequence_length=decoder_seq_length,
                    dtype=self.autocast_dtype,
                )
            else:
                output_tensor = forward_backward_no_pipelining(
                    forward_step_func=self.get_forward_output_only_func(),
                    batch=batch_for_pipeline,
                    model=self.enc_dec_model,
                    forward_only=True,
                    tensor_shape=tensor_shape,
                    decoder_sequence_length=decoder_seq_length,
                    dtype=self.autocast_dtype,
                )

            # get output tensor
            if parallel_state.is_pipeline_last_stage():
                output_tensor = output_tensor[0]['logits']
                output_tensor = tensor_parallel.gather_from_tensor_model_parallel_region(output_tensor)
                log_probs, token_ids = torch.max(torch.nn.functional.log_softmax(output_tensor, dim=-1), dim=-1)
                predicted_tokens_dec = torch.cat(
                    [predicted_tokens_dec.to(token_ids.device), token_ids[:, -1].unsqueeze(1)], dim=1
                )
            else:
                log_probs = torch.zeros(
                    (predicted_tokens_dec.shape[0], predicted_tokens_dec.shape[1]), dtype=self.autocast_dtype
                ).cuda()
                predicted_tokens_dec = torch.zeros(
                    (predicted_tokens_dec.shape[0], predicted_tokens_dec.shape[1] + 1),
                    dtype=predicted_tokens_dec.dtype,
                ).cuda()

            if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
                # Broadcast from the last pipeline stage to all other model-parallel ranks.
                torch.distributed.broadcast(
                    predicted_tokens_dec,
                    parallel_state.get_pipeline_model_parallel_last_rank(),
                    group=parallel_state.get_model_parallel_group(),
                )
                torch.distributed.broadcast(
                    log_probs,
                    parallel_state.get_pipeline_model_parallel_last_rank(),
                    group=parallel_state.get_model_parallel_group(),
                )

        # Reset microbatch calculator to what it was before decoding.
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=global_batch_per_gpu * parallel_state.get_data_parallel_world_size(),
            micro_batch_size=global_batch_per_gpu // num_micro_batches_before_decode,
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
        )
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
        app_state = AppState()

        # The complete method only works with global batch = micro batch size = data parallel size = 1.
        _reconfigure_microbatch_calculator(
            rank=app_state.global_rank,
            rampup_batch_size=None,
            global_batch_size=1,
            micro_batch_size=1,
            data_parallel_size=1,
        )
        app_state = AppState()

        response = {}
        self.freeze()
        # naive greedy slow loop
        # TODO: add option for BeamSearchDecoder

        response['prompt'] = request['prompt'][0]
        response['completion'] = {}
        tokens_enc = request['masked_sample']

        response['masked_input'] = ' '.join(self.tokenizer.ids_to_tokens(tokens_enc[0].cpu().numpy().tolist()))
        enc_mask = tokens_enc != self.tokenizer.pad_id

        predicted_tokens_ids, log_probs = self.decode(tokens_enc, enc_mask, int(request['tokens_to_generate']))
        predicted_tokens_ids = predicted_tokens_ids.cpu().numpy()[0].tolist()
        log_probs = log_probs.cpu().numpy()[0].tolist()
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        else:
            predicted_tokens_ids = [id for id in predicted_tokens_ids if id != self.tokenizer.pad_id]
        if self.tokenizer.eos_id in predicted_tokens_ids:
            idx = predicted_tokens_ids.index(self.tokenizer.eos_id)
            predicted_tokens_ids = predicted_tokens_ids[:idx]
        # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
        if hasattr(self.tokenizer, 'special_token_to_id'):
            predicted_tokens_ids = [
                id for id in predicted_tokens_ids if id not in self.tokenizer.special_token_to_id.values()
            ]

        predicted_tokens_dec = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        response['completion']['text'] = self.tokenizer.tokens_to_text(predicted_tokens_dec)
        response['completion']['tokens'] = list(zip(predicted_tokens_ids, predicted_tokens_dec, log_probs))
        self.unfreeze()
        return response

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """ PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
            When using pipeline parallelism, we need the global batch to remain on the CPU,
            since the memory overhead will be too high when using a large number of microbatches.
            Microbatches are transferred from CPU to GPU inside the pipeline. 
        """
        return batch

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
            Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    def list_available_models(self):
        pass
