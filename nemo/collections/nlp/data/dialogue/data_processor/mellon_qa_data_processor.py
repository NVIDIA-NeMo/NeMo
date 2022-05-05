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

__all__ = ['DialogueMellonQADataProcessor']


class DialogueMellonQADataProcessor(DialogueDataProcessor):
    """Data Processor for Mellon QA dialogues. 
    """

    def __init__(self, data_dir: str, tokenizer: object, cfg=None):
        """
        Constructs DialogueMSMarcoDataProcessor
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
        For the Mellon QA dataset, there is no explicit dev set (instead uses the test set as the dev set)
        Therefore, this function creates a dev set and a new train set from the train set.
        Dev set contains self.cfg.dev_proportion % of samples with the rest going into the train set
        Test set contains the whole dataset (Dev + Train) as this dataset is small (~100) and primarily used in a zero shot setting
        """

        examples = []

        raw_examples = self.open_csv('mellon_qa_data.csv')
        raw_examples = list(raw_examples.values())
        # filter out answers with no answer
        raw_examples = [
            example
            for example in raw_examples
            if isinstance(example['Non Generative Question Answering '], str)
            and isinstance(example['Generative Question Answering '], str)
        ]

        n_samples = len(raw_examples)
        idxs = DialogueDataProcessor.get_relevant_idxs(dataset_split, n_samples, self.cfg.dev_proportion)

        for i in idxs:
            utterance = str(raw_examples[i]['Question'])
            answer = str(raw_examples[i]['Non Generative Question Answering '])
            well_formed_answer = str(raw_examples[i]['Generative Question Answering '])
            passage = raw_examples[i]['Passage']
            input_example = {
                "utterance": utterance,
                "example_id": i,
                "labels": {"response": answer, "fluent_response": well_formed_answer, "passage": passage,},
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
