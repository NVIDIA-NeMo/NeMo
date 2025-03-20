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

from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


class MockDataModule(pl.LightningDataModule):
    """
    Mock data module with data sampling and preprocessing configurations.
    """

    def __init__(
        self,
        seq_length: int = 77,
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
        task_encoder: Optional[Any] = None,
    ):
        """
        Initializes the mock data module with data sampling and preprocessing configurations.
        task_encoder: This Mock data module uses Energon Task encoder if provided.

        Args:
            seq_length (int): Maximum sequence length for tokens.
            decoder_seq_length (Optional[int]): Sequence length for the decoder. Used by Megatron Sampler.
            tokenizer: Tokenizer for text processing.
            image_processor: Processor for image preprocessing.
            micro_batch_size (int): Batch size for training and validation.
            global_batch_size (int): Total batch size across GPUs.
            rampup_batch_size (Optional[List[int]]): Batch size ramp-up schedule. Used by Megatron Sampler.
            num_train_samples (int): Number of training samples.
            num_val_samples (int): Number of validation samples.
            num_test_samples (int): Number of testing samples.
            num_workers (int): Number of workers for data loading.
            pin_memory (bool): Whether to pin memory for data loading.
            persistent_workers (bool): Whether workers should remain persistent.
            task_encoder: Task encoder for Energon tasks.
        """
        super().__init__()
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.task_encoder = task_encoder
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        if tokenizer is None or image_processor is None:
            logging.warning("Processor or tokenizer are not provided! Fall back to `openai/clip-vit-large-patch14`.")
            from transformers import AutoProcessor

            from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.tokenizer = tokenizer or AutoTokenizer("openai/clip-vit-large-patch14")
            self.image_processor = image_processor or processor.image_processor
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        # pylint: disable=C0116
        self._train_ds = _MockClipDataset(
            self.tokenizer,
            self.image_processor,
            "train",
            self.num_train_samples,
            self.seq_length,
            task_encoder=self.task_encoder,
        )
        self._validation_ds = _MockClipDataset(
            self.tokenizer, self.image_processor, "valid", self.num_val_samples, self.seq_length
        )
        self._test_ds = _MockClipDataset(
            self.tokenizer, self.image_processor, "test", self.num_test_samples, self.seq_length
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # pylint: disable=C0116
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0116
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        # pylint: disable=C0116
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        # pylint: disable=C0116
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            batch_size=self.micro_batch_size,
            collate_fn=dataset.collate_fn,
            **kwargs,
        )

    def state_dict(self) -> Dict[str, Any]:
        """
        Save the state of the data module.

        This method is called when saving a checkpoint. It generates and saves the state of the data module,
        including the state of the dataloader and the number of consumed samples.

        Returns:
        Dict[str, Any]: A dictionary containing the state of the data module.
        """

        logging.warning("trainer object not connected to data module object returning empty state")
        return {}


class _MockClipDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        image_processor,
        name: str,
        num_samples: int,
        seq_length: int,
        seed: int = 42,
        task_encoder=None,
    ) -> None:
        super().__init__()
        self.name = name
        self.seq_length = seq_length

        self.vocab_size = tokenizer.vocab_size

        crop_size = image_processor.crop_size
        self.image_height, self.image_width = crop_size["height"], crop_size["width"]

        self.length = num_samples
        self.seed = seed
        self.task_encoder = task_encoder

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # Generate data of the expected size and datatype (based on GPTDataset).

        np_gen = np.random.default_rng(seed=(self.seed + idx))
        tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length], dtype=np.int64))
        images = torch.from_numpy(np_gen.random(size=[3, self.image_height, self.image_width], dtype=np.float32))

        if self.task_encoder is not None:
            # Use energon task encoder if provided
            return self.task_encoder.encode_sample({"image": images, "txt": "This is Random Mock Text"})

        return {
            "images": images,
            "captions": tokens,
        }

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        collated_batch = data.dataloader.default_collate(batch)
        collated_batch["attention_mask"] = None
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
