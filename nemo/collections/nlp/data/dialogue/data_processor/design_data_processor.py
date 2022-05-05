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

import os

import pandas as pd

from nemo.collections.nlp.data.dialogue.data_processor.data_processor import DialogueDataProcessor
from nemo.collections.nlp.data.dialogue.input_example.input_example import DialogueInputExample

__all__ = ['DialogueDesignDataProcessor']


class DialogueDesignDataProcessor(DialogueDataProcessor):
    """Data Processor for Design Dataset"""

    def __init__(self, data_dir: str, tokenizer: object, cfg=None):
        """
        Constructs DialogueDesignDataProcessor
        Args:
            data_dir: path to data directory
            tokenizer: tokenizer object
            cfg: cfg container for dataset
        """
        self.data_dir = data_dir
        self._tokenizer = tokenizer
        self.cfg = cfg

    def open_csv(self, filename):
        """
        Reads file into a list
        """
        filename = os.path.join(self.data_dir, filename)
        with open(filename, "r", encoding="UTF-8") as f:
            df = pd.read_csv(filename)
        return df.to_dict(orient='index')

    def get_dialog_examples(self, dataset_split: str):
        """
        Process raw files into DialogueInputExample
        Args: 
            dataset_split: {train, dev, test}
        Dev set contains self.cfg.dev_proportion % of samples with the rest going into the train set
        Test set contains the whole dataset (Dev + Train) as this dataset is small (~100) and primarily used in a zero shot setting
        """

        examples = []

        raw_examples = self.open_csv('mellon_design_OV.csv')
        # remove disabled examples
        raw_examples = [raw_examples[i] for i in range(len(raw_examples)) if raw_examples[i]['disabled'] != 'yes']

        n_samples = len(raw_examples)

        idxs = DialogueDataProcessor.get_relevant_idxs(dataset_split, n_samples, self.cfg.dev_proportion)

        all_intents = sorted(list(set(raw_examples[i]['intent labels'] for i in range(len(raw_examples)))))
        all_services = sorted(list(set(raw_examples[i]['domain'] for i in range(len(raw_examples)))))
        for i in idxs:
            raw_example = raw_examples[i]
            utterances = [raw_example['example_{}'.format(i)] for i in range(1, 4)]
            service = raw_example['domain']
            intent = raw_example['intent']
            intent_description = raw_example['intent labels']
            system_utterance = raw_example['response']

            slot_names = [raw_example['slot{}'.format(i)] for i in range(1, 3)]
            # these are possible slot values not ground truth slot values
            slot_values = [raw_example['slot{}_values'.format(i)] for i in range(1, 3)]
            slot_questions = [raw_example['slot{}_values'.format(i)] for i in range(1, 3)]

            for j in range(1, 3):
                value = raw_example['slot{}'.format(j)]
                if isinstance(value, str):
                    system_utterance = system_utterance.replace('slot{}'.format(j), value)

            valid_slots_ids = [i for i, slot in enumerate(slot_names) if isinstance(slot, str)]
            slot_names = [slot_names[i] for i in valid_slots_ids]
            slot_values = [slot_values[i] if isinstance(slot_values[i], str) else '' for i in valid_slots_ids]
            slot_questions = [slot_questions[i] if isinstance(slot_questions[i], str) else '' for i in valid_slots_ids]

            for utterance in utterances:
                if not isinstance(utterance, str):
                    continue
                input_example = {
                    "utterance": utterance,
                    "system_utterance": system_utterance,
                    "labels": {
                        "service": service,
                        "intent": intent_description,
                        "slots": {
                            slot: '' for slot in slot_names
                        },  # dataset does not contain ground truth slot values
                    },
                    "possible_labels": {
                        'intent': all_intents,
                        "service": all_services,
                        "slots": {slot: slot_values[i] for i, slot in enumerate(slot_names)},
                    },
                    "description": {
                        "service": service,
                        "intent": intent_description,
                        "slots": {slot: slot_questions[i] for i, slot in enumerate(slot_names)},
                    },
                }

                example = DialogueInputExample(input_example)
                examples.append(example)
        return examples

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.get_dialog_examples("train")

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_dialog_examples("dev")

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        return self.get_dialog_examples("test")
