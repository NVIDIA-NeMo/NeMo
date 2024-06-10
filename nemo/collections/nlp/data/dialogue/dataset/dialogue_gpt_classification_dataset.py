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
import random
from collections import defaultdict

import torch

from nemo.collections.nlp.data.dialogue.dataset.dialogue_dataset import DialogueDataset
from nemo.utils import logging
from nemo.utils.decorators import deprecated_warning


class DialogueGPTClassificationDataset(DialogueDataset):
    '''
    Designed for classification tasks such as intent/domain classification as well as slot tagging

    Dataset Class
        1. Performs Model-dependent (but Data-independent) operations (tokenization etc)
        2. This can allow the same model preprocessing for multiple datasources
        3. Users can configurate which labels to use for modelling
            (e.g. intent classification, slot filling or both together etc)
    '''

    def __init__(self, dataset_split: str, dialogues_processor: object, tokenizer, cfg):
        """Constructor
        Args:
            dataset_split: dataset split
            dialogues_processor: Data generator for SGD dialogues
            tokenizer: tokenizer
            cfg: cfg container for dataset
        """
        # deprecation warning
        deprecated_warning("DialogueGPTClassificationDataset")

        self.cfg = cfg

        if self.cfg.target_template == "with_slots" and self.cfg.eval_mode != "generation":
            raise ValueError(
                "slot-filling is not supported by eval_mode {}, please set model.dataset.eval_mode=generation instead".format(
                    self.cfg.eval_mode
                )
            )
        if self.cfg.target_template != "with_slots" and self.cfg.field == "slots":
            raise ValueError("please set model.dataset.target_template='with_slots' if model.dataset.field='slots'")
        self.label_type = self.cfg.field
        if self.cfg.target_template == "with_description":
            self.label_to_description = defaultdict(str)
        self.all_possible_labels = set()
        self.tokenizer = tokenizer
        self.tokenizer.tokenizer.padding_side = "right"
        self.max_candidates = 2
        if not isinstance(dataset_split, str):
            dataset_split = dataset_split[0]
        self.features = dialogues_processor.get_dialog_examples(dataset_split)
        for idx in range(len(self.features)):
            self.preprocess_feature(idx)
        if self.cfg.debug_mode:
            self.features = self.features[:16]
        # for few shot learning to append in the prompt
        self.lm_features = self.get_lm_samples()

    def transform(self, label):
        """
        Normalize labels by replacing underscore with space

        Args:
            label: str
        Returns:
            normalized_label: str
        """
        if self.cfg.task == "assistant" and self.cfg.prompt_template != "prompt_tuning":
            label = label.replace('_', ' ')
        return label

    def __len__(self):
        return len(self.features)

    def get_n_tokens_in_sentence(self, sentence):
        encodings_dict = self.tokenizer.tokenizer(
            sentence, truncation=True, max_length=self.cfg.max_seq_length, padding=False, return_tensors="pt"
        )
        output = torch.squeeze(encodings_dict['input_ids'])
        return len(output) if len(output.size()) > 0 else 0

    def preprocess_feature(self, idx):
        ex = self.features[idx].data
        label = ex["labels"][self.label_type]
        candidates = ex["possible_labels"][self.label_type]

        if self.label_type in ["service", "intent"]:
            label = self.transform(label)
            candidates = [self.transform(candidate) for candidate in candidates]

        self.features[idx].data["labels"][self.label_type] = label
        self.features[idx].data["possible_labels"][self.label_type] = candidates
        if self.cfg.target_template == "with_description":
            description = ex["description"][self.label_type]
            self.label_to_description[label] = description
        for candidate in candidates:
            self.all_possible_labels.add(candidate)
        self.max_candidates = max(self.max_candidates, len(candidates))

    def default_encode(self, sentence):
        encodings_dict = self.tokenizer.tokenizer(
            sentence, truncation=True, max_length=self.cfg.max_seq_length, padding="max_length", return_tensors="pt"
        )
        input_ids = torch.squeeze(encodings_dict['input_ids'])
        attn_masks = torch.squeeze(encodings_dict['attention_mask'])
        return encodings_dict, input_ids, attn_masks

    @staticmethod
    def linearize_slots(slots):
        """
        Serialize slots into a linear text

        Args:
            slots: dict with each slot_name as key and possible slot values as value
        Returns:
            linear_slots: text based representation of slot names and values
        """
        if not slots:
            return "None"
        return ", ".join(
            ["{}({})".format(slot, value if isinstance(value, str) else value[0]) for slot, value in slots.items()]
        )

    def format_target(self, target, slots=None):
        """
        Formats the back part of the training example, after the base_template
        for instance, "restaurant" in  "<utterance> service: restaurant"
        or "set alarm\nslots: <slot_name1>(<slot_value1>), <slot_name1>(<slot_value1>)" in \
        "<utterance>\nintent: set alarm\nslots: <slot_name1>(<slot_value1>), <slot_name1>(<slot_value1>)"
        """
        if self.cfg.target_template == "with_description":
            return target + ' (' + self.label_to_description[target] + ')'
        elif self.cfg.target_template == "default":
            return target
        elif self.cfg.target_template == "with_slots" and slots is not None and self.cfg.field == "intent":
            return target + '\nslots: ' + DialogueGPTClassificationDataset.linearize_slots(slots)
        elif self.cfg.target_template == "with_slots" and slots is not None and self.cfg.field == "slots":
            return DialogueGPTClassificationDataset.linearize_slots(slots)
        else:
            raise ValueError("Please choose a target format from {default, with_description, with_slots}")

    def get_lm_samples(self):
        max_sample_length = 0
        lm_features = []
        for idx in range(len(self.features)):
            ex = self.features[idx].data
            utterance = ex["utterance"]
            label = ex["labels"][self.label_type]
            slots = ex["labels"]["slots"] if self.cfg.target_template == "with_slots" else None
            lm_feature = self.format_prompt(utterance) + ' ' + self.format_target(label, slots=slots)
            feature_len = self.get_n_tokens_in_sentence(lm_feature)
            max_sample_length = max(max_sample_length, feature_len)
            lm_features.append(lm_feature)
        logging.info("max feature length per sample with label: ".format(max_sample_length))
        logging.info(
            "please adjust max seq len to at least {} * ({} + 1) = {} but not too much more for efficiency".format(
                max_sample_length, self.cfg.few_shot, max_sample_length * (1 + self.cfg.few_shot)
            )
        )
        return lm_features

    def format_prompt(self, utterance, few_shot=0, idx=None):
        if self.cfg.prompt_template == "default":
            base_template = utterance + ' ' + self.label_type + ':'
        elif self.cfg.prompt_template == "i_want_to":
            base_template = utterance + ' ' + 'I want to'
        elif self.cfg.prompt_template == "prompt_tuning":
            base_template = utterance + '\n' + self.label_type + ':'
        elif self.cfg.prompt_template == "prompt_tuning_with_options":
            base_template = (
                'possible intents: '
                + ', '.join(sorted(list(self.all_possible_labels)))
                + '\n\n'
                + utterance
                + '\n'
                + self.label_type
                + ':'
            )

        if few_shot > 0:
            few_shot_indices = random.sample(range(len(self.features)), few_shot + 1)
            few_shot_indices = [i for i in few_shot_indices if i != idx][:few_shot]
            few_shot_samples = [self.lm_features[i] for i in few_shot_indices]
            base_template = (
                self.tokenizer.tokenizer.pad_token.join(few_shot_samples)
                + self.tokenizer.tokenizer.pad_token
                + base_template
            )
        return base_template

    def collate_fn(self, batch):
        """
        Truncates elements to max length in batch
        """
        _, _, _, _, candidate_attn_masks, _, _, _ = zip(*batch)
        # determine max length in batch
        batch_max_length = 0
        for candidate_attn_mask in candidate_attn_masks:
            for one_attn_mask in candidate_attn_mask:
                batch_max_length = max(batch_max_length, torch.sum(one_attn_mask).item())
        # padding for tp=2 situation
        if batch_max_length % 2:
            batch_max_length += 1

        all_items = []
        for item in zip(*batch):
            if isinstance(item[0], int):
                item = [torch.tensor(i) for i in item]
            item_stack = torch.stack(item)
            # if item_stack is 1d, elements refers to indexes and there is no need to truncate
            if len(item_stack.size()) == 1:
                all_items.append(item_stack)
            # otherwise, truncate last dimension to max length in batch
            else:
                all_items.append(item_stack[..., :batch_max_length])
        return all_items

    def __getitem__(self, idx: int):
        '''
        State how the input and output samples look like

        This template can be changed

        Training example:
            e.g. <utterance> service: restaurant
            e.g. <task description> <utterance> service: restaurant
            e.g. <utterance>\nintent: set alarm\nslots: <slot_name1>(<slot_value1>), <slot_name1>(<slot_value1>)

        Generation example:
            e.g. <utterance> service:

        '''
        ex = self.features[idx].data

        utterance = ex["utterance"]
        utterance_length = self.get_n_tokens_in_sentence(utterance)

        label = ex["labels"][self.label_type]
        candidates = ex["possible_labels"][self.label_type]

        slots = ex["labels"]["slots"] if self.cfg.target_template == "with_slots" else None

        base_template = self.format_prompt(utterance, few_shot=self.cfg.few_shot, idx=idx)

        sentence_without_answer = base_template

        sentence = base_template + ' ' + self.format_target(label, slots=slots)

        if self.cfg.eval_mode == "binary_score":
            candidate_sentences = []
            for candidate in candidates:
                positive_answer = base_template + ' ' + candidate + ' Answer: ' + 'yes'
                negative_answer = base_template + ' ' + candidate + ' Answer: ' + 'no'
                if candidate == label:
                    correct_candidate = len(candidate_sentences) // 2
                    candidate_sentences.append(positive_answer)
                    candidate_sentences.append(negative_answer)
                else:
                    candidate_sentences.append(negative_answer)
                    candidate_sentences.append(positive_answer)
        else:
            correct_candidate = 0
            candidate_sentences = [
                base_template + ' ' + self.format_target(candidate, slots=slots) for candidate in candidates
            ]

        encodings_dict, input_ids, attn_masks = self.default_encode(sentence)

        candidate_tokenized_sentences = [
            self.default_encode(candidate_sentence) for candidate_sentence in candidate_sentences
        ]

        # ensure all samples have the same number of candidates for collating into tensor
        while len(candidate_tokenized_sentences) < self.max_candidates:
            candidate_tokenized_sentences.append(candidate_tokenized_sentences[0])

        candidate_input_ids = torch.stack([i[1] for i in candidate_tokenized_sentences])
        candidate_attn_masks = torch.stack([i[2] for i in candidate_tokenized_sentences])

        labels = copy.copy(torch.squeeze(encodings_dict['input_ids']))

        training_mask_end = self.get_n_tokens_in_sentence(sentence_without_answer)

        labels.data = torch.tensor(
            [-100 if i < training_mask_end else labels.data[i] for i in range(len(labels.data))]
        )

        return (
            input_ids,
            attn_masks,
            labels,
            candidate_input_ids,
            candidate_attn_masks,
            training_mask_end,
            utterance_length,
            correct_candidate,
        )
