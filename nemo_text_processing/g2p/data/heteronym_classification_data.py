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


import json
import os
from typing import Dict, List

import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['HeteronymClassificationDataset']


class HeteronymClassificationDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        tokenizer: TokenizerSpec,
        wiki_homograph_dict: Dict[str, Dict[str, str]],
        wordid_to_idx: Dict[str, int],
        max_seq_len: int = 512,
        grapheme_field: str = "text_graphemes",
        with_labels: bool = True,
    ):
        """
        Creates dataset to use to run training and inference on G2PClassificationModel.
        Processes WikiHomograph raw data files:
        https://github.com/google-research-datasets/WikipediaHomographData/tree/master/data

        Args:
            manifest: path to manifest with "homograph_span", "start_end", "text_graphemes"
                and (optional) "word_id" fields. "word_id" is required for model training.
            tokenizer: pretrained tokenizer
            wiki_homograph_dict: a dictionary where each grapheme contains word_id to ipa_form mappings, e.g.,
                {'use': {'use_nou': "'juːs", 'use_vrb': "'juːz"}}
            wordid_to_idx: mapping from word id to index
            max_seq_len: maximum input sequence length
            grapheme_field: name of the field in the .json manifest with grapheme input
            with_labels: indicates whether labels are provided in the manifest. False for inference, True for training
        """
        super().__init__()

        if not os.path.exists(manifest):
            raise ValueError(f"{manifest} not found")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        self.pad_token = 0
        self.with_labels = with_labels
        self.wiki_homograph_dict = wiki_homograph_dict
        self.wordid_to_idx = wordid_to_idx

        sentences, start_end_indices, homographs, word_ids = [], [], [], []
        num_skipped = 0
        with open(manifest, "r") as f:
            for line in f:
                line = json.loads(line)
                cur_start_end, cur_homographs, cur_word_ids = (
                    line["start_end"],
                    line["homograph_span"],
                    line["word_id"],
                )
                if isinstance(cur_homographs, str):
                    cur_start_end, cur_homographs, cur_word_ids = [cur_start_end], [cur_homographs], [cur_word_ids]

                for se, h, w in zip(cur_start_end, cur_homographs, cur_word_ids):
                    grapheme_sent = line[grapheme_field]

                    # +2 for bos and eos tokens
                    if (
                        grapheme_sent[se[0] : se[1]] != h
                        or len(self.tokenizer.text_to_tokens(grapheme_sent)) + 2 > max_seq_len
                    ):
                        num_skipped += 1
                    else:
                        sentences.append(grapheme_sent)
                        start_end_indices.append(se)
                        homographs.append(h)
                        word_ids.append(w)

        for sentence, start_end_index, homograph, word_id in zip(sentences, start_end_indices, homographs, word_ids):
            start, end = start_end_index
            if self.with_labels:
                target_and_negatives, subword_mask, target = self._prepare_sample(
                    sentence, start, end, homograph, word_id
                )
                self.data.append(
                    {
                        "input": sentence,
                        "target": target,
                        "target_and_negatives": target_and_negatives,
                        "subword_mask": subword_mask,
                    }
                )
            else:
                target_and_negatives, subword_mask = self._prepare_sample(sentence, start, end, homograph)
                self.data.append(
                    {"input": sentence, "subword_mask": subword_mask, "target_and_negatives": target_and_negatives}
                )

        logging.info(f"Number of samples in {manifest}: {len(self.data)}, remove {num_skipped} lines")

    def _prepare_sample(self, sentence: str, start: int, end: int, homograph: str, word_id=None):
        """
        Prepares a single training sample

        Args:
            sentence: input sentence in grapheme form
            start: start of the homograph span
            end: end of the homograph span
            homograph: homograph
            word_id: [Optional] target word_id, use None for inference
        """
        l_context = self.tokenizer.text_to_ids(sentence[:start])
        r_context = self.tokenizer.text_to_ids(sentence[end:])
        l_context_len = len(l_context)
        r_context_len = len(r_context)
        sentence_tokenized = self.tokenizer.text_to_ids(sentence)

        grapheme_ipa_forms = self.wiki_homograph_dict[homograph.lower()]
        homograph_len = len(sentence_tokenized[l_context_len : len(sentence_tokenized) - r_context_len])
        # use prediction of the first subword of the homograph
        homograph_mask = [1] + [0] * (homograph_len - 1)
        subword_mask = [0] * (l_context_len + 1) + homograph_mask + [0] * (r_context_len + 1)
        target_and_negatives = [self.wordid_to_idx[wordid_] for wordid_ in grapheme_ipa_forms]
        output = [target_and_negatives, subword_mask]

        if self.with_labels:
            if word_id is None:
                raise ValueError(f"word_id must be provided when self.with_labels==True, i.e., training mode")

            target_word_id = self.wordid_to_idx[word_id]
            # add extra -100 tokens at the begging and end for [CLS] and [SEP] bert tokens
            target_word_id = (
                [-100] * (l_context_len + 1)
                + [target_word_id]
                + [-100] * (homograph_len - 1)
                + [-100] * (r_context_len + 1)
            )
            assert len(target_word_id) == len(self.tokenizer.tokenizer([sentence])["input_ids"][0])
            output.append(target_word_id)

        return output  # [target_and_negatives, subword_mask] and (optional) target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def map(self, text: str) -> List[int]:
        """ Creates a mapping from target labels to ids."""
        tokens = []
        for word_id, word in enumerate(text.split()):
            tokens.append(self.labels_tkn2id[word])
        return tokens

    def _collate_fn(self, batch):
        # Encode inputs (graphemes)
        input_batch = [entry["input"] for entry in batch]
        input_encoding = self.tokenizer.tokenizer(
            input_batch, padding='longest', max_length=self.max_seq_len, truncation=True, return_tensors='pt',
        )

        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        # create a mask 1 for target_and_negatives and 0 for the rest of the entries since they're not relevant to the target word
        target_and_negatives = [entry["target_and_negatives"] for entry in batch]
        batch_size = input_ids.shape[0]
        num_classes = len(self.wordid_to_idx)
        target_and_negatives_mask = torch.zeros(batch_size, num_classes)
        for i, values in enumerate(target_and_negatives):
            for v in values:
                target_and_negatives_mask[i][v] = 1

        # pad prediction mask (masks out irrelevant subwords)
        subword_mask = [entry["subword_mask"] for entry in batch]
        pred_mask_len = [len(entry) for entry in subword_mask]
        max_pred_mask_len = max(pred_mask_len)
        subword_mask = [
            entry + [0] * (max_pred_mask_len - entry_len) for entry, entry_len in zip(subword_mask, pred_mask_len)
        ]
        subword_mask = torch.tensor(subword_mask)
        output = [input_ids, attention_mask, target_and_negatives_mask, subword_mask]

        if self.with_labels:
            # Encode targets
            targets = [entry["target"] for entry in batch]
            targets_len = [len(entry) for entry in targets]
            max_target_len = max(targets_len)
            targets = [entry + [-100] * (max_target_len - entry_len) for entry, entry_len in zip(targets, targets_len)]
            targets = torch.tensor(targets)
            assert input_ids.shape == targets.shape
            output.append(targets)
        return output
