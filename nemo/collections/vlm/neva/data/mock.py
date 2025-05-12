# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.collections.vlm.neva.data.multimodal_tokens import IMAGE_TOKEN_INDEX
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


# pylint: disable=C0115, C0116
class MockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_length: int = 2048,
        decoder_seq_length: Optional[int] = None,
        tokenizer: Optional = None,
        image_processor: Optional = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000_000,
        num_val_samples: int = 10_000_000,
        num_test_samples: int = 10_000_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.decoder_seq_len = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.packed_sequence = packed_sequence

        if tokenizer is None or image_processor is None:
            logging.warning("Processor or tokenizer are not provided! Fall back to `llava-hf/llava-1.5-7b-hf`.")
            from transformers import AutoProcessor

            from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.tokenizer = tokenizer or AutoTokenizer("llava-hf/llava-1.5-7b-hf", use_fast=False)
            self.image_processor = image_processor or processor.image_processor
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        seq_length = self.seq_length
        if self.packed_sequence and self.micro_batch_size > 1:
            seq_length = seq_length // self.micro_batch_size
            logging.warning(
                f"Packed sequence is used with mock dataset. Sequence length for each "
                f"sample is update to `seq_length // self.micro_batch_size = {seq_length}`!"
            )
        self._train_ds = _MockNevaDataset(
            self.tokenizer,
            self.image_processor,
            "train",
            self.num_train_samples,
            seq_length,
            packed_sequence=self.packed_sequence,
        )
        self._validation_ds = _MockNevaDataset(
            self.tokenizer,
            self.image_processor,
            "valid",
            self.num_val_samples,
            seq_length,
            packed_sequence=self.packed_sequence,
        )
        self._test_ds = _MockNevaDataset(
            self.tokenizer,
            self.image_processor,
            "test",
            self.num_test_samples,
            seq_length,
            packed_sequence=self.packed_sequence,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )


class _MockNevaDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        image_processor,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
        packed_sequence: bool = False,
        num_image_embeddings_per_tile=576,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length

        self.vocab_size = tokenizer.vocab_size

        crop_size = image_processor.crop_size
        self.image_height, self.image_width = crop_size["height"], crop_size["width"]

        self.length = num_samples
        self.seed = seed
        self.packed_sequence = packed_sequence
        self.num_image_embeddings_per_tile = num_image_embeddings_per_tile

        self.loss_mask = torch.ones(self.seq_length + 1 - num_image_embeddings_per_tile, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self) -> int:
        return self.length

    def _get_text(self, idx: int) -> np.ndarray:
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        return np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Generate data of the expected size and datatype (based on GPTDataset).
        np_gen = np.random.default_rng(seed=(self.seed + idx))
        tokens = torch.from_numpy(
            np_gen.integers(
                self.vocab_size, size=[self.seq_length + 2 - self.num_image_embeddings_per_tile], dtype=np.int64
            )
        )
        tokens[2] = IMAGE_TOKEN_INDEX  # ImageToken token index
        labels = tokens.clone()
        images = torch.from_numpy(np_gen.random(size=[3, self.image_height, self.image_width], dtype=np.float32))
        tokens = tokens[:-1]
        labels = labels[1:]
        return {
            "media": images,
            "tokens": tokens,
            "labels": labels,
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        collated_batch = data.dataloader.default_collate(batch)
        collated_batch["attention_mask"] = None
        if self.packed_sequence:
            from megatron.core.packed_seq_params import PackedSeqParams

            tokens = collated_batch["tokens"]
            batch_size = tokens.shape[0]
            valid_seqlen = self.seq_length
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * (valid_seqlen), step=(valid_seqlen), dtype=torch.int32, device=tokens.device
            )
            cu_seqlens_padded = torch.arange(
                0, (batch_size + 1) * (valid_seqlen), step=(valid_seqlen), dtype=torch.int32, device=tokens.device
            )
            qkv_format = 'thd'
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
                max_seqlen_q=valid_seqlen,
                max_seqlen_kv=valid_seqlen,
                qkv_format=qkv_format,
            )
            collated_batch["packed_seq_params"] = packed_seq_params

            for key in ["tokens", "labels", "loss_mask", "position_ids"]:
                collated_batch[key] = collated_batch[key].reshape(1, -1)

        return collated_batch

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)
