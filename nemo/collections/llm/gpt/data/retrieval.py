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

import json
import os.path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from datasets import Dataset, concatenate_datasets

from nemo.collections.llm.bert.data.fine_tuning import FineTuningDataModule
from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


class CustomRetrievalDataModule(FineTuningDataModule):
    """Custom Retrieval Data Module loaded with json file"""

    def __init__(
        self,
        data_root: Union[str, List[str]],
        val_root: Optional[str] = None,
        test_root: Optional[str] = None,
        val_ratio: Optional[float] = 0.04,
        test_ratio: Optional[float] = 0.01,
        dataset_identifier: str = "custom_retrieval_dataset",
        seq_length: int = 2048,
        tokenizer: Optional["TokenizerSpec"] = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        force_redownload: bool = False,
        delete_raw: bool = True,
        seed: int = 1234,
        memmap_workers: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        packed_sequence_specs: Optional["PackedSequenceSpecs"] = None,
        query_key: str = "question",
        pos_doc_key: str = "pos_doc",
        neg_doc_key: str = "neg_doc",
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Custom DataModule for Finetuning retrieval Dataset for Embedding model.

        Args:
            data_root (Union[str, List[str]]): The JSONL data file(s) used for training/validation/test.
                if val_root/test_root is not present, data_root will be split to training and val/test based on
                val_ratio/test_ratio.
            val_root (Optional[str]): The JSONL data file used for validation. If not provided, validation set
                will be split from data_root.
            test_root (Optional[str]): The JSONL data file used for test. If not provided, test set
                will be split from data_root.
            val_ratio (Optional[float]): The ratio of validation set when splitting from data_root.
            test_ratio (Optional[float]): The ratio of test set when splitting from data_root.
            dataset_identifier (str): Dataset identifier when saving the dataset to NEMO_HOME.
            seq_length (int, optional): The maximum sequence length for the input and output text. Defaults to 2048.
            tokenizer (Optional[TokenizerSpec], optional): The tokenizer to use for preprocessing the text.
                If not provided, a Megatron GPT2 BPE tokenizer will be used.
            micro_batch_size (int, optional): The micro batch size for training. Defaults to 4.
            global_batch_size (int, optional): The global batch size for training. Defaults to 8.
            rampup_batch_size (Optional[List[int]], optional): A list of batch sizes for ramping up during training.
                Defaults to None.
            seed (int, optional): The random seed for data shuffling. Defaults to 1234.
            memmap_workers (int, optional): The number of worker processes for loading data using TextMemMapDataset.
                Defaults to 1.
            num_workers (int, optional): The number of worker processes for data loading. Defaults to 8.
            pin_memory (bool, optional): Whether to pin memory during data loading for faster GPU training.
                Defaults to True.
            persistent_workers (bool, optional): Whether to keep data loading workers persistent across epochs.
                Defaults to False.
            dataset_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments to pass into the GPTSFTDataset class
        """
        self.force_redownload = force_redownload
        self.delete_raw = delete_raw

        assert packed_sequence_specs is None, "RetrievalDataModule does not support packed sequences."
        if isinstance(data_root, str):
            data_root = [data_root]
        for directory in data_root:
            assert os.path.exists(directory), f"Data root {directory} does not exist."

        if val_root is not None:
            assert os.path.exists(val_root), f"Validation root {val_root} does not exist."
            self.val_ratio = 0  # Use val_root for all validation
        else:
            self.val_ratio = val_ratio

        if test_root is not None:
            assert os.path.exists(test_root), f"Test root {test_root} does not exist."
            self.test_ratio = 0
        else:
            self.test_ratio = test_ratio

        self.train_ratio = 1 - self.val_ratio - self.test_ratio
        self.val_root = val_root
        self.test_root = test_root
        self.query_key = query_key
        self.pos_doc_key = pos_doc_key
        self.neg_doc_key = neg_doc_key
        self.unprocessed_root = data_root

        log_info = (
            f"data_root: {data_root} will be split to "
            f"{self.train_ratio}:{self.val_ratio}:{self.test_ratio} used for train/val/test"
        )
        if self.val_root is not None:
            log_info += f", separate validation root: {self.val_root}"
        if self.test_root is not None:
            log_info += f", separate test root: {self.test_root}"
        logging.info(log_info)

        super().__init__(
            dataset_root=get_dataset_root(dataset_identifier),
            seq_length=seq_length,
            tokenizer=tokenizer,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
            seed=seed,
            memmap_workers=memmap_workers,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            dataset_kwargs=dataset_kwargs,
        )

    def prepare_data(self) -> None:
        """Prepare data if not split already."""
        if not self.train_path.exists() or self.force_redownload:
            self._preprocess_and_split_data()
        super().prepare_data()

    def _preprocess_and_split_data(self):
        logging.info(f"Preprocessing {self.__class__.__name__} to jsonl format and splitting...")

        save_splits = {}

        train_datasets = []
        for data_dir in self.unprocessed_root:
            train_datasets.append(Dataset.from_list(json.load(open(data_dir, 'r'))))
        train_dataset = concatenate_datasets(train_datasets)

        if self.val_ratio + self.test_ratio != 0:
            split_dataset = train_dataset.train_test_split(test_size=self.val_ratio + self.test_ratio, seed=self.seed)
            save_splits['training'] = split_dataset['train']
        else:
            # No split
            save_splits['training'] = train_dataset

        split_dataset2 = {}
        if self.val_ratio != 0 and self.test_ratio != 0:
            # split_dataset2['train'] is the actual validation set
            # split_dataset2['test'] is the actual test set
            split_dataset2 = split_dataset['test'].train_test_split(
                test_size=self.test_ratio / (self.val_ratio + self.test_ratio), seed=self.seed
            )
        elif self.val_ratio == 0 and self.test_ratio != 0:
            split_dataset2['test'] = split_dataset['test']
        elif self.test_ratio == 0 and self.val_ratio != 0:
            split_dataset2['train'] = split_dataset['test']

        if self.val_root is not None:
            save_splits['validation'] = Dataset.from_list(json.load(open(self.val_root, 'r')))
        else:
            save_splits['validation'] = split_dataset2['train']

        if self.test_root is not None:
            save_splits['test'] = Dataset.from_list(json.load(open(self.test_root, 'r')))
        else:
            save_splits['test'] = split_dataset2['test']

        for split_name, dataset in save_splits.items():
            output_file = self.dataset_root / f"{split_name}.jsonl"
            with output_file.open("w", encoding="utf-8") as f:
                for o in dataset:
                    # We only write one positive document for now
                    # All negative document are written
                    pos_doc = o[self.pos_doc_key][0] if isinstance(o[self.pos_doc_key], list) else o[self.pos_doc_key]
                    neg_doc = o[self.neg_doc_key] if isinstance(o[self.pos_doc_key], list) else [o[self.neg_doc_key]]
                    f.write(json.dumps({"query": o[self.query_key], "pos_doc": pos_doc, "neg_doc": neg_doc}) + "\n")

            logging.info(f"{split_name} split saved to {output_file}")
