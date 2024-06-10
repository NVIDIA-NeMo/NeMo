# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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


from typing import Dict, List, Optional, Union

import numpy as np

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.glue_benchmark.data_processors import InputExample
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import GLUEDataset
from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import deprecated_warning

__all__ = ['DialogueZeroShotIntentDataset']


class DialogueZeroShotIntentDataset(GLUEDataset):
    """
    Dataset for training a NLI model for zero shot intent recognition. Similar to GLUE/MNLI
    dataset, but allows the user to specify which columns in the data files contain the
    premise, hypothesis, and gold label.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'input_ids': NeuralType(('B', 'T'), ChannelType()),
            'segment_ids': NeuralType(('B', 'T'), ChannelType()),
            'input_mask': NeuralType(('B', 'T'), MaskType()),
            'labels': NeuralType(tuple('B'), CategoricalValuesType()),
        }

    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, cfg):
        """
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for dialogues
            tokenizer: tokenizer to split text into sub-word tokens
            cfg: config dict for dataset
                num_classes: number of classes in the data (should be either 2 or 3, corresponding to
                labels ['entailment', 'not_entailment'] or ["contradiction", "entailment", "neutral"])
        """
        # deprecation warning
        deprecated_warning("DialogueZeroShotIntentDataset")

        self.cfg = cfg
        self.tokenizer = tokenizer
        if self.cfg.num_classes not in [2, 3]:
            raise ValueError("num_classes must be either 2 or 3!")
        self.label_list = (
            ["contradiction", "entailment", "neutral"]
            if self.cfg.num_classes == 3
            else ['not_entailment', 'entailment']
        )
        token_params = {
            'bos_token': None,
            'eos_token': tokenizer.eos_token,
            'pad_token': tokenizer.pad_token,
            'cls_token': tokenizer.cls_token,
            'sep_token_extra': (
                tokenizer.eos_token if hasattr(tokenizer, 'name') and 'roberta' in tokenizer.name.lower() else None
            ),
        }

        self.raw_features = dialogues_processor.get_dialog_examples(dataset_split)
        self.examples = self._create_examples(self.raw_features, dataset_split)
        self.features = self.convert_examples_to_features(
            self.examples,
            [0, 1, 2, 3],
            self.cfg.max_seq_length,
            tokenizer,
            output_mode="classification",
            **token_params,
        )

    def _create_examples(self, raw_features, dataset_split: str):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx in range(len(raw_features)):
            ex = self.raw_features[idx].data
            user_utterance = ex["utterance"]
            intent = ex["labels"]["intent"]
            for candidate_idx, candidate_intent in enumerate(ex["possible_labels"]["intent"]):
                guid = "{}-{}-{}".format(dataset_split, idx, candidate_idx)
                text_a = user_utterance
                text_b = "{} {}".format(self.cfg.prompt_template, candidate_intent)
                label = 1 if candidate_intent == intent else 0
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def convert_examples_to_features(
        self,
        examples: List[str],
        label_list: List[int],
        max_seq_length: int,
        tokenizer: TokenizerSpec,
        output_mode: str,
        bos_token: str = None,
        eos_token: str = '[SEP]',
        pad_token: str = '[PAD]',
        cls_token: str = '[CLS]',
        sep_token_extra: str = None,
        cls_token_at_end: bool = False,
        cls_token_segment_id: int = 0,
        pad_token_segment_id: int = 0,
        pad_on_left: bool = False,
        mask_padding_with_zero: bool = True,
        sequence_a_segment_id: int = 0,
        sequence_b_segment_id: int = 1,
    ):
        """
        Loads a data file into a list of `InputBatch`s.
        The `cls_token_at_end` defines the location of the CLS token:

            * False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            * True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]

        The `cls_token_segment_id` defines the segment id associated to the CLS token (0 for BERT, 2 for XLNet)

        The convention in BERT is:

            a. For sequence pairs:
                * tokens:   [CLS] is this jack ##ville ? [SEP] no it is not . [SEP]
                * type_ids:   0   0  0    0    0       0   0   1  1  1  1   1   1
            b. For single sequences:
                * tokens:   [CLS] the dog is hairy . [SEP]
                * type_ids:   0   0   0   0  0     0   0

        Where "type_ids" are used to indicate whether this is the first
        sequence or the second sequence. The embedding vectors for `type=0`
        and `type=1` were learned during pre-training and are added to the
        wordpiece embedding vector (and position vector). This is
        not *strictly* necessarysince the [SEP] token unambiguously separates
        the sequences, but it makes it easier for the model to learn
        the concept of sequences.
        For classification tasks, the first vector (corresponding to [CLS])
        is used as as the "sentence vector". Note that this only makes sense
        because the entire model is fine-tuned.

        The convention for NMT is:

            a. For sequence pairs:
                * tokens:<BOS> is this jack ##ville ? <EOS> <BOS> no it is not . <EOS>
                * type_ids:0   0  0    0    0       0   0     1   1  1  1  1   1   1
            b. For single sequences:
                * tokens:   <BOS> the dog is hairy . <EOS>
                * type_ids:   0   0   0   0  0     0   0

        """
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for ex_index, example in enumerate(examples):
            if example.label == "-":  # skip examples without a consensus label (e.g. in SNLI data set)
                continue
            if ex_index % 10000 == 0:
                logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            if hasattr(tokenizer, 'text_to_tokens'):
                tokens_a = tokenizer.text_to_tokens(example.text_a)
            else:
                tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                if hasattr(tokenizer, 'text_to_tokens'):
                    tokens_b = tokenizer.text_to_tokens(example.text_b)
                else:
                    tokens_b = tokenizer.tokenize(example.text_b)

                special_tokens_count = 2 if eos_token else 0
                special_tokens_count += 1 if sep_token_extra else 0
                special_tokens_count += 2 if bos_token else 0
                special_tokens_count += 1 if cls_token else 0
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
            else:
                special_tokens_count = 1 if eos_token else 0
                special_tokens_count += 1 if sep_token_extra else 0
                special_tokens_count += 1 if bos_token else 0
                if len(tokens_a) > max_seq_length - special_tokens_count:
                    tokens_a = tokens_a[: max_seq_length - special_tokens_count]
            # Add special tokens to sequence_a
            tokens = tokens_a
            if bos_token:
                tokens = [bos_token] + tokens
            if eos_token:
                tokens += [eos_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            # Add sequence separator between sequences
            if tokens_b and sep_token_extra:
                tokens += [sep_token_extra]
                segment_ids += [sequence_a_segment_id]

            # Add special tokens to sequence_b
            if tokens_b:
                if bos_token:
                    tokens += [bos_token]
                    segment_ids += [sequence_b_segment_id]
                tokens += tokens_b
                segment_ids += [sequence_b_segment_id] * (len(tokens_b))
                if eos_token:
                    tokens += [eos_token]
                    segment_ids += [sequence_b_segment_id]

            # Add classification token - for BERT models
            if cls_token:
                if cls_token_at_end:
                    tokens += [cls_token]
                    segment_ids += [cls_token_segment_id]
                else:
                    tokens = [cls_token] + tokens
                    segment_ids = [cls_token_segment_id] + segment_ids
            if hasattr(tokenizer, 'tokens_to_ids'):
                input_ids = tokenizer.tokens_to_ids(tokens)
            else:
                input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)

            if hasattr(tokenizer, 'tokens_to_ids'):
                pad_token_id = tokenizer.tokens_to_ids([pad_token])[0]
            else:
                pad_token_id = tokenizer.convert_tokens_to_ids([pad_token])[0]

            if pad_on_left:
                input_ids = ([pad_token_id] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token_id] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            if len(input_ids) != max_seq_length:
                raise ValueError("input_ids must be of length max_seq_length")
            if len(input_mask) != max_seq_length:
                raise ValueError("input_mask must be of length max_seq_length")
            if len(segment_ids) != max_seq_length:
                raise ValueError("segment_ids must be of length max_seq_length")
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = np.float32(example.label)
            else:
                raise KeyError(output_mode)

            if ex_index < 5:
                logging.info("*** Example ***")
                logging.info("guid: %s" % (example.guid))
                logging.info("tokens: %s" % " ".join(list(map(str, tokens))))
                logging.info("input_ids: %s" % " ".join(list(map(str, input_ids))))
                logging.info("input_mask: %s" % " ".join(list(map(str, input_mask))))
                logging.info("segment_ids: %s" % " ".join(list(map(str, segment_ids))))
                logging.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label_id)
            )

        return features


class InputFeatures(object):
    """A single set of features of data.

    Args:
        input_ids: input/token ids
        input_mask: masks out subword tokens
        segment_ids: distinguish one sentence from the other one (if present)
        label_ids: label for the current example
    """

    def __init__(
        self, input_ids: List[int], input_mask: List[int], segment_ids: List[int], label_id: Union[float, int]
    ):
        """Initialized InputFeatures."""
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
