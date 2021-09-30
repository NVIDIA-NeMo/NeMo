# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from megatron.optimizer.clip_grads import clip_grad_norm_fp32
import torch
from megatron import fused_kernels
from apex import mpu
from megatron.global_vars import get_args
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import initialize_megatron_for_nemo
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
)
from nemo.utils import AppState, logging


class MegatronGPTModel(NLPModel):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg, trainer=trainer)
        self.cfg = cfg

        if self.cfg.get('use_cpu_initialization', False) is False:
            torch.cuda.set_device(trainer.local_rank)

        app_state = AppState()
        app_state.global_rank = trainer.global_rank
        app_state.world_size = trainer.world_size
        app_state.model_parallel_size = cfg.get('tensor_model_parallel_size', 1)
        app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

        initialize_megatron_for_nemo(
            world_size=app_state.world_size,
            global_rank=app_state.global_rank,
            micro_batch_size=cfg.get('micro_batch_size', 1),
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
            tensor_model_parallel_rank=app_state.model_parallel_rank,
            encoder_seq_length=cfg.get('encoder_seq_length', 512),
            num_layers=cfg.get('num_layers', 1),
            hidden_size=cfg.get('hidden_size', 16),
            num_attention_heads=cfg.get('num_attention_heads', 1),
            max_position_embeddings=cfg.get('max_position_embeddings', 512),
            init_method_std=cfg.get('init_method_std', 0.02),
            tokenizer_type='GPT2BPETokenizer',
            vocab_file=cfg.tokenizer.vocab_file,
            merge_file=cfg.tokenizer.merge_file,
            fp16=cfg.get('fp16', True),
            bf16=cfg.get('bf16', False),
            use_cpu_initialization=cfg.get('use_cpu_initialization', False),
        )
        args = get_args()

        fused_kernels.load(args)

        self.model = GPTModel(
            num_tokentypes=0, parallel_output=True, pre_process=cfg.pre_process, post_process=cfg.post_process
        )

        self.tokenizer = get_nmt_tokenizer(
            library=self.cfg.tokenizer.library,
            model_name=self.cfg.tokenizer.type,
            tokenizer_model=self.register_artifact("tokenizer_model", self.cfg.tokenizer.model),
            vocab_file=self.register_artifact("vocab_file", self.cfg.tokenizer.vocab_file),
            merges_file=self.register_artifact("merges_file", self.cfg.tokenizer.merge_file),
        )

    def forward(self, tokens, position_ids, attention_mask, labels):
        output_tensor = self.model(tokens, position_ids, attention_mask, labels=labels)
        return output_tensor

    def training_step(self, batch, batch_idx):
        tokens, labels, loss_mask, attention_mask, position_ids = self.process_batch(batch)
        output_tensor = self(tokens, position_ids, attention_mask, labels)
        loss = self.loss_func(loss_mask, output_tensor)
        self.log('train_loss', loss)
        # Reduced loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])
        self.log('reduced_train_loss', averaged_loss[0], prog_bar=True)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr)
        self.log('global_step', self.trainer.global_step, prog_bar=True)
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, labels, loss_mask, attention_mask, position_ids = self.process_batch(batch)
        output_tensor = self(tokens, position_ids, attention_mask, labels)
        loss = self.loss_func(loss_mask, output_tensor)
        # TODO: add text generation
        # take the first k tokens and then generate text - compare with ground truth (won't look similar in general)
        """
        k = num_context
        n = max_generate_length
        context_tokens = tokens[0:k]
        while k < n:
            output_tensor = self(context_tokens)
            next_token = sample(output_tensor)
            context_tokens.append(next_token)
            k += 1
        """
        return loss

    def validation_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        self.log('val_loss', averaged_loss[0], prog_bar=True)
        self.log('consumed_samples', self.compute_consumed_samples(self.trainer.global_step))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        averaged_loss = average_losses_across_data_parallel_group(outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def process_batch(self, batch):
        args = get_args()

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        data = batch
        data_b = mpu.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_ = data_b['text'].long()
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()

        if self.cfg.debug:
            logging.info('debugging')
            tokens_list = tokens.detach().tolist()[0]
            labels_list = labels.detach().tolist()[0]
            logging.info(f'detokenize tokens: {self.tokenizer.ids_to_text(tokens_list)}')
            logging.info(f'detokenize labels: {self.tokenizer.ids_to_text(labels_list)}')

        # Get the masks and postition ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens, self.tokenizer.eos_id, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss
        )

        return tokens, labels, loss_mask, attention_mask, position_ids

    def build_train_valid_test_datasets(self):
        logging.info('Building GPT datasets.')
        global_batch_size = self.trainer.world_size * self.cfg.micro_batch_size / self.cfg.tensor_model_parallel_size
        eval_iters = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches
        train_valid_test_num_samples = [
            self.trainer.max_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]
        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            data_prefix=self.cfg.data.data_prefix,
            data_impl=self.cfg.data.data_impl,
            splits_string=self.cfg.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.cfg.data.seq_length,
            seed=self.cfg.seed,
            skip_warmup=self.cfg.data.skip_warmup,
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building GPT datasets.')
        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if self.cfg.data.dataloader_type == 'single':
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
        elif self.cfg.data.dataloader_type == 'cyclic':
            batch_sampler = MegatronPretrainingRandomSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=self.cfg.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
        else:
            raise Exception('{} dataloader type is not supported.'.format(self.cfg.dataloader_type))

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=self.cfg.data.num_workers, pin_memory=True,
        )

    def setup(self, stage=None):
        self.build_train_valid_test_datasets()
        self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)
        self.setup_test_data(self.cfg.data)

    def setup_training_data(self, cfg):
        if hasattr(self, '_train_ds'):
            if self.trainer.checkpoint_connector._loaded_checkpoint:
                global_step = self.trainer.checkpoint_connector._loaded_checkpoint['global_step']
                consumed_samples = self.compute_consumed_samples(global_step)
            else:
                consumed_samples = 0
            self._train_dl = self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def setup_validation_data(self, cfg):
        if hasattr(self, '_validation_ds'):
            consumed_samples = 0
            self._validation_dl = self.build_pretraining_data_loader(self._validation_ds, consumed_samples)

    def setup_test_data(self, cfg):
        if hasattr(self, '_test_ds'):
            consumed_samples = 0
            self._test_dl = self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def compute_consumed_samples(self, global_step):
        app_state = AppState()
        consumed_samples = (
            global_step
            * app_state.data_parallel_size
            * self.cfg.micro_batch_size
            * self.trainer.accumulate_grad_batches
        )
        return consumed_samples

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        """PTL hook that is called after unscaling gradients when using native amp.
           We use gradient clipping implementation from megatron-lm.
        """
        clip_val = self.trainer.gradient_clip_val
        if clip_val is None:
            return

        clip_val = float(clip_val)
        if clip_val <= 0:
            return

        if self.trainer.amp_backend == 'native':
            parameters = self.model.parameters()
            clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)
        else:
            return

    def list_available_models():
        pass
