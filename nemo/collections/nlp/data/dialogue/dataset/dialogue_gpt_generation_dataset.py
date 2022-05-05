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

import copy

import torch

from nemo.collections.nlp.data.dialogue.dataset.dialogue_dataset import DialogueDataset


class DialogueGPTGenerationDataset(DialogueDataset):
    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, cfg):
        """ Constructor
        Designed for free form generation tasks such as Dialogue Response Generation 

        Args:
            dataset_split: dataset split
            dialogues_processor: dialogues processor
            tokenizer: tokenizer
            cfg: cfg container for dataset
        """
        self.cfg = cfg
        self.input_label_type = self.cfg.input_field
        self.output_label_type = self.cfg.output_field
        self.tokenizer = tokenizer
        self.tokenizer.tokenizer.padding_side = "right"
        if not isinstance(dataset_split, str):
            dataset_split = dataset_split[0]

        self.features = dialogues_processor.get_dialog_examples(dataset_split)
        self.features = self.remove_invalid_samples(self.features)

        if self.cfg.debug_mode:
            self.features = self.features[:16]

    def remove_invalid_samples(self, features):
        valid_idxs = []
        all_fields = self.input_label_type.split('+') + self.output_label_type.split('+')
        for i in range(len(features)):
            features[i].data["labels"]["utterance"] = features[i].data["utterance"]
            all_fields_non_empty = True
            for field in all_fields:
                if not features[i].data["labels"][field] or not features[i].data["labels"][field].strip():
                    all_fields_non_empty = False
            if all_fields_non_empty:
                valid_idxs.append(i)
        return [features[i] for i in valid_idxs]

    def __len__(self):
        return len(self.features)

    def get_n_tokens_in_sentence(self, sentence):
        encodings_dict = self.tokenizer.tokenizer(
            sentence, truncation=True, max_length=self.cfg.max_seq_length, padding=False, return_tensors="pt"
        )
        output = torch.squeeze(encodings_dict['input_ids'])
        return len(output) if len(output.size()) > 0 else 0

    def default_encode(self, sentence):
        encodings_dict = self.tokenizer.tokenizer(
            sentence, truncation=True, max_length=self.cfg.max_seq_length, padding="max_length", return_tensors="pt"
        )
        input_ids = torch.squeeze(encodings_dict['input_ids'])
        attn_masks = torch.squeeze(encodings_dict['attention_mask'])
        return encodings_dict, input_ids, attn_masks

    def format_prompt(self, ex):
        '''
        Formats training prompt based on self.input_field_type

        Training example: 
            e.g. response: <response> # input_label_type = response
            e.g. utterance: <utterance> # input_label_type = utterance
            e.g. passage: <passage> utterance: <utterance> # input_label_type = passage+utterance
        '''
        ex["labels"]["utterance"] = ex["utterance"]
        parts = self.input_label_type.split('+')
        input_sentence = ' '.join([part + ': ' + ex["labels"][part] for part in parts])
        return input_sentence

    def __getitem__(self, idx: int):

        '''
        For each example, this function determines the format of input and output sequences based on user-specified conguration.
        This is controlled by model.dataset.input_field and model.dataset.output_field
        For instance:
            If model.dataset.input_field == response and model.dataset.output_field == fluent_response:
                Input = "response: <response>" and output = "response: <response> fluent_response: <fluent_response>" (with loss calculated from <fluent_response> only)
            If model.dataset.input_field == utterance and model.dataset.output_field == response:
                Input = "utterance: <utterance>" and output = "utterance: <utterance> response: <response>" (with loss calculated from <response> only) 
            If model.dataset.input_field == passage+utterance and model.dataset.output_field == response:
                Input = "passage: <passage> utterance: <utterance>" and output="passage: <passage> utterance: <utterance> response: <response>" (with loss calculated from <response> only) 
        '''
        ex = self.features[idx].data

        input_sentence = self.format_prompt(ex)

        utterance_length = self.get_n_tokens_in_sentence(input_sentence)

        output_sentence = ex["labels"][self.output_label_type]

        base_template = input_sentence

        sentence_without_answer = base_template + ' ' + self.output_label_type + ':'

        sentence = sentence_without_answer + ' ' + output_sentence

        encodings_dict, input_ids, attn_masks = self.default_encode(sentence)

        labels = copy.copy(torch.squeeze(encodings_dict['input_ids']))

        training_mask_end = self.get_n_tokens_in_sentence(sentence_without_answer)

        labels.data = torch.tensor(
            [-100 if i < training_mask_end else labels.data[i] for i in range(len(labels.data))]
        )

        return (input_ids, attn_masks, labels, training_mask_end, utterance_length)
