# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import numpy as np

from nemo.collections.nlp.data.text_normalization_as_tagging.bert_example import BertExampleBuilder, read_input_file
from nemo.core.classes.dataset import Dataset
from nemo.core.neural_types import ChannelType, IntType, LabelsType, MaskType, NeuralType

__all__ = ["ThutmoseTaggerDataset", "ThutmoseTaggerTestDataset"]


class ThutmoseTaggerDataset(Dataset):
    """
    Dataset as used by the ThutmoseTaggerModel for training, validation, and inference
    pipelines.

    Args:
        input_file (str): path to tsv-file with data
        example_builder: instance of BertExampleBuilder
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), MaskType()),
            "segment_ids": NeuralType(('B', 'T'), ChannelType()),
            "labels_mask": NeuralType(('B', 'T'), MaskType()),
            "tag_labels": NeuralType(('B', 'T'), LabelsType()),
            "semiotic_labels": NeuralType(('B', 'T'), LabelsType()),
            "semiotic_spans": NeuralType(('B', 'T', 'C'), IntType()),
        }

    def __init__(self, input_file: str, example_builder: BertExampleBuilder) -> None:
        self.examples = read_input_file(example_builder, input_file, infer=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        input_ids = np.array(self.examples[idx].features["input_ids"])
        input_mask = np.array(self.examples[idx].features["input_mask"])
        segment_ids = np.array(self.examples[idx].features["segment_ids"])
        labels_mask = np.array(self.examples[idx].features["labels_mask"])
        tag_labels = np.array(self.examples[idx].features["tag_labels"])
        semiotic_labels = np.array(self.examples[idx].features["semiotic_labels"])
        semiotic_spans = np.array(self.examples[idx].features["semiotic_spans"])
        return input_ids, input_mask, segment_ids, labels_mask, tag_labels, semiotic_labels, semiotic_spans


class ThutmoseTaggerTestDataset(Dataset):
    """
    Dataset for inference pipeline.

    Args:
        sents: list of strings
        example_builder: instance of BertExampleBuilder
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), MaskType()),
            "segment_ids": NeuralType(('B', 'T'), ChannelType()),
        }

    def __init__(self, sents: List[str], example_builder: BertExampleBuilder) -> None:
        self.examples = []
        for source in sents:
            example = example_builder.build_bert_example(source, infer=True)
            if example is None:
                raise ValueError("Cannot build example from: " + source)
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        input_ids = np.array(self.examples[idx].features["input_ids"])
        input_mask = np.array(self.examples[idx].features["input_mask"])
        segment_ids = np.array(self.examples[idx].features["segment_ids"])
        return input_ids, input_mask, segment_ids
