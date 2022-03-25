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

import os
import pickle
from typing import Dict, List, Optional

import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils.data_preprocessing import (
    DataProcessor,
    fill_class_weights,
    get_freq_weights,
    get_label_stats,
)
from nemo.collections.nlp.data.glue_benchmark.data_processors import InputExample
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import GLUEDataset
from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['ZeroShotIntentProcessor', 'ZeroShotIntentDataset', 'ZeroShotIntentInferenceDataset']


class ZeroShotIntentProcessor(DataProcessor):
    """
    Processor for entailment data sets used to train NLI models for zero shot intent classification.
    """

    def __init__(self, sent1_col: int, sent2_col: int, label_col: int, num_classes: int):
        """
        Args:
            sent1_col: the index of the column containing the premise (or sentence 1)
            sent2_col: the index of the column containing the hypothesis (or sentence 2)
            label_col: the index of the column containing the label
            num_classes: number of classes in the data (should be either 2 or 3, corresponding to
            labels ['entailment', 'not_entailment'] or ["contradiction", "entailment", "neutral"])
        """
        self.sent1_col = sent1_col
        self.sent2_col = sent2_col
        self.label_col = label_col
        self.num_classes = num_classes

    def get_train_examples(self, file_path: str):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(self._read_tsv(file_path), "train")

    def get_dev_examples(self, file_path: str):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self._read_tsv(file_path), "dev")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        if self.num_classes == 2:
            return ['not_entailment', 'entailment']
        elif self.num_classes == 3:
            return ["contradiction", "entailment", "neutral"]
        else:
            raise ValueError("num_classes must be either 2 or 3!")

    def _create_examples(self, lines: List[str], set_type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[self.sent1_col]
            text_b = line[self.sent2_col]
            label = line[self.label_col]
            if label == "-":
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ZeroShotIntentDataset(GLUEDataset):
    """
    Dataset for training a NLI model for zero shot intent recognition. Similar to GLUE/MNLI
    dataset, but allows the user to specify which columns in the data files contain the
    premise, hypothesis, and gold label.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'labels': NeuralType(tuple('B'), CategoricalValuesType()),
        }

    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: str,
        sent1_col: int,
        sent2_col: int,
        label_col: int,
        num_classes: int,
        use_cache: bool = True,
    ):
        """
        Args:
            file_path: path to file
            tokenizer: such as AutoTokenizer
            max_seq_length: max sequence length including [CLS] and [SEP]
            sent1_col: the index of the column containing the premise (or sentence 1)
            sent2_col: the index of the column containing the hypothesis (or sentence 2)
            label_col: the index of the column containing the label
            num_classes: number of classes in the data (should be either 2 or 3, corresponding to
            labels ['entailment', 'not_entailment'] or ["contradiction", "entailment", "neutral"])
            use_cache: whether to use data cache
        """
        self.task_name = "mnli"  # for compatibility with parent class
        data_dir, file_name = os.path.split(file_path)
        logging.info(f'Processing {file_name}')
        self.tokenizer = tokenizer
        evaluate = False if 'train' in file_name else True
        processor = ZeroShotIntentProcessor(sent1_col, sent2_col, label_col, num_classes)
        self.label_list = processor.get_labels()
        if not evaluate:
            self.examples = processor.get_train_examples(file_path)

            # check the labels found in the training set
            all_train_labels = [example.label for example in self.examples]
            unique_labels = set(all_train_labels)
            if len(unique_labels) != num_classes:
                raise ValueError(
                    "Number of classes specified in config doesn't match the number found in the training data!"
                )
            elif len(unique_labels) == 2:
                if not unique_labels == set(self.label_list):
                    raise ValueError(
                        f"Found unexpected labels! For a two-class model, labels are expected to be {self.label_list}"
                    )
            elif len(unique_labels) == 3:
                if not unique_labels == set(self.label_list):
                    raise ValueError(
                        f"Found unexpected labels! For a three-class model, labels are expected to be {self.label_list}"
                    )

            # save the label map for reference
            label_file = os.path.join(data_dir, "label_ids.csv")
            with open(label_file, "w") as out:
                out.write('\n'.join(self.label_list))
            logging.info(f'Labels: {self.label_list}')
            logging.info(f'Label mapping saved to : {label_file}')

        else:
            self.examples = processor.get_dev_examples(file_path)

        processor_name = type(processor).__name__
        vocab_size = getattr(tokenizer, "vocab_size", 0)
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}_{}".format(
                processor_name, file_name, tokenizer.name, str(max_seq_length), str(vocab_size)
            ),
        )

        if use_cache and os.path.exists(cached_features_file):
            logging.info(f"loading from {cached_features_file}")
            with open(cached_features_file, "rb") as reader:
                self.features = pickle.load(reader)
        else:
            token_params = {
                'bos_token': None,
                'eos_token': tokenizer.eos_token,
                'pad_token': tokenizer.pad_token,
                'cls_token': tokenizer.cls_token,
                'sep_token_extra': tokenizer.eos_token if 'roberta' in tokenizer.name.lower() else None,
            }

            self.features = self.convert_examples_to_features(
                self.examples, self.label_list, max_seq_length, tokenizer, output_mode="classification", **token_params
            )
            master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if master_device:
                logging.info(f'Saving train features into {cached_features_file}')
                with open(cached_features_file, "wb") as writer:
                    pickle.dump(self.features, writer)


class ZeroShotIntentInferenceDataset(GLUEDataset):
    """
    Similar to ZeroShotIntentDataset, but gets utterances and candidate labels from lists
    rather than sentence pairs and labels from a file.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'labels': NeuralType(tuple('B'), CategoricalValuesType()),
        }

    def __init__(
        self,
        queries: List[str],
        candidate_labels: List[str],
        tokenizer: TokenizerSpec,
        max_seq_length: int,
        hypothesis_template: str,
    ):
        """
        Args:
            queries: list of utterances to classify
            candidate_labels: list of candidate labels
            tokenizer: such as AutoTokenizer
            max_seq_length: max sequence length including [CLS] and [SEP]
            hypothesis_template: template used to turn each candidate label into a NLI-style hypothesis
        """

        logging.info(f'Processing queries for inference')
        self.tokenizer = tokenizer
        token_params = {
            'bos_token': None,
            'eos_token': tokenizer.eos_token,
            'pad_token': tokenizer.pad_token,
            'cls_token': tokenizer.cls_token,
            'sep_token_extra': tokenizer.eos_token if 'roberta' in tokenizer.name.lower() else None,
        }
        self.examples = []
        for i, query in enumerate(queries):
            for j, candidate_label in enumerate(candidate_labels):
                guid = "query-%s-label-%s" % (i, j)
                text_a = query
                text_b = hypothesis_template.format(candidate_label)
                label = 3  # dummy label for inference; training labels are 0, 1, 2 or 0, 1
                self.examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        self.features = self.convert_examples_to_features(
            self.examples, [0, 1, 2, 3], max_seq_length, tokenizer, output_mode="classification", **token_params
        )


def calc_class_weights_from_dataloader(
    dataloader: 'torch.utils.data.DataLoader', num_classes: int, data_dir: str
) -> List[float]:
    """
    Calculate the weights of each class to be used for weighted loss. This is similar to the function calc_class_weights
    in text_classification_dataset, but it gets the labels from a dataloader rather than from a file.
    Args:
        dataloader: the dataloader for the training set
        num_classes: number of classes in the dataset
    """
    labels = []
    for batch in dataloader:
        labels.extend(tensor2list(batch[-1]))
    logging.info(f'Calculating label frequency stats...')
    total_sents, sent_label_freq, max_id = get_label_stats(
        labels, os.path.join(data_dir, 'sentence_stats.tsv'), verbose=False
    )
    if max_id >= num_classes:
        raise ValueError(f'Found an invalid label! Labels should be from [0, num_classes-1].')

    class_weights_dict = get_freq_weights(sent_label_freq)

    logging.info(f'Total Sentence Pairs: {total_sents}')
    logging.info(f'Class Frequencies: {sent_label_freq}')
    logging.info(f'Class Weights: {class_weights_dict}')
    class_weights = fill_class_weights(weights=class_weights_dict, max_id=num_classes - 1)
    return class_weights
