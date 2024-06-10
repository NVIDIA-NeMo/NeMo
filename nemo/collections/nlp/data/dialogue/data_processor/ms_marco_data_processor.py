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

import json
import os
from ast import literal_eval

from nemo.collections.nlp.data.dialogue.data_processor.data_processor import DialogueDataProcessor
from nemo.collections.nlp.data.dialogue.input_example.input_example import DialogueInputExample
from nemo.utils.decorators import deprecated_warning

__all__ = ['DialogueMSMarcoDataProcessor']


class DialogueMSMarcoDataProcessor(DialogueDataProcessor):
    """Data Processor for MS Marco dialogues. (https://github.com/microsoft/MSMARCO-Question-Answering)
    Please agree to the Terms of Use before downloading data at
    https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz
    https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz
    """

    def __init__(self, data_dir: str, tokenizer: object, cfg=None):
        """
        Constructs DialogueMSMarcoDataProcessor
        Args:
            data_dir: path to data directory
            tokenizer: tokenizer object
            debug_mode: reduce number of samples to load in order to increase speed of processing
            cfg: cfg container for dataset
        """
        # deprecation warning
        deprecated_warning("DialogueMSMarcoDataProcessor")

        self.data_dir = data_dir
        self._tokenizer = tokenizer
        self.cfg = cfg

    def open_json(self, filename):
        """
        Reads file into a list
        """
        filename = os.path.join(self.data_dir, filename)
        with open(filename, "r", encoding="UTF-8") as f:
            data = json.load(f)
        return data

    def get_dialog_examples(self, dataset_split: str):
        """
        Process raw files into DialogueInputExample
        Args:
            dataset_split: {train, dev, test}
        For the MS Marco dataset, there is no explicit dev set (instead uses the test set as the dev set)
        Therefore, this function creates a dev set and a new train set from the train set.
        Dev set contains self.cfg.dev_proportion % of samples with the rest going into the train set
        """

        examples = []

        dataset_split_print = {"train": "train", "dev": "train", "test": "dev"}

        raw_examples = self.open_json("{}_v2.1.json".format(dataset_split_print[dataset_split]))

        n_samples = len(raw_examples['answers'])

        idxs = DialogueDataProcessor.get_relevant_idxs(dataset_split, n_samples, self.cfg.dev_proportion)

        if self.cfg.debug_mode:
            idxs = idxs[:100]

        for i in idxs:
            utterance = raw_examples['query'][str(i)]
            # answer need not be extracted from passage
            # taking the first answer as the ground truth correct answer as only <1% has multiple answers
            answer = raw_examples['answers'][str(i)]
            answer = answer[0] if isinstance(answer, list) else answer

            well_formed_answer = raw_examples['wellFormedAnswers'][str(i)]
            well_formed_answer = (
                well_formed_answer if isinstance(well_formed_answer, list) else literal_eval(well_formed_answer)
            )
            well_formed_answer = well_formed_answer[0] if well_formed_answer else None
            query_type = raw_examples['query_type'][str(i)]
            candidate_passages = raw_examples['passages'][str(i)]
            passage = [
                candidate_passage["passage_text"]
                for candidate_passage in candidate_passages
                if int(candidate_passage["is_selected"])
            ]
            passage = passage[0] if passage else None

            possible_passages = [candidate_passage["passage_text"] for candidate_passage in candidate_passages]

            input_example = {
                "utterance": utterance,
                "example_id": i,
                "labels": {
                    "service": query_type,
                    "response": answer,
                    "fluent_response": well_formed_answer,
                    "passage": passage,
                },
                "possible_labels": {
                    "service": "LOCATION,NUMERIC,PERSON,DESCRIPTION,ENTITY".split(','),
                    "passage": possible_passages,
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
