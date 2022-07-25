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

import ijson
import numpy as np

from nemo.collections.nlp.data.data_utils import DataProcessor
from nemo.collections.nlp.data.question_answering.input_example.qa_input_example import QAExample
from nemo.utils import logging

TRAINING_MODE = "train"
EVALUATION_MODE = "eval"
INFERENCE_MODE = "infer"


class QAProcessor(DataProcessor):
    """
    Processor for a QA dataset, expected in SQuAD format.

    Args:
        data_file: data file path
        mode: TRAINING_MODE/EVALUATION_MODE/INFERENCE_MODE
            for creating training/evaluation/inference dataset
    """

    def __init__(self, data_file: str, mode: str):
        self.data_file = data_file
        self.mode = mode

        # Memoizes documents to reduce memory use (as the same document is often used for many questions)
        self.doc_id = 0
        self.context_text_to_doc_id = {}
        self.doc_id_to_context_text = {}

    def get_examples(self):
        """ Get examples from raw json file """

        if self.data_file is None:
            raise ValueError(f"{self.mode} data file is None.")

        # remove this line and the replace cache line below - which is a temp fix
        with open(self.data_file.replace('_cache', ''), "r", encoding="utf-8") as reader:
            input_data = ijson.items(reader, "data.item")

            examples = []
            for entry in input_data:
                len_docs = []
                title = entry["title"]
                for paragraph in entry["paragraphs"]:
                    context_text = paragraph["context"]
                    for qa in paragraph["qas"]:
                        qas_id = qa["id"]
                        question_text = qa["question"]
                        if not question_text:
                            continue
                        start_position_character = None
                        answer_text = None
                        answers = []
                        if "is_impossible" in qa:
                            is_impossible = qa["is_impossible"] or len(qa["answers"]) < 1
                        else:
                            is_impossible = False

                        if not is_impossible:
                            if self.mode in [TRAINING_MODE, EVALUATION_MODE]:
                                answer = qa["answers"][0]
                                answer_text = answer["text"]
                                start_position_character = answer["answer_start"]
                            if self.mode == EVALUATION_MODE:
                                answers = qa["answers"]
                        if context_text in self.context_text_to_doc_id:
                            doc_id = self.context_text_to_doc_id[context_text]
                        else:
                            doc_id = self.doc_id
                            self.context_text_to_doc_id[context_text] = doc_id
                            self.doc_id_to_context_text[doc_id] = context_text
                            self.doc_id += 1
                            len_docs.append(len(context_text))

                        example = QAExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            context_text=context_text,
                            context_id=doc_id,
                            answer_text=answer_text,
                            start_position_character=start_position_character,
                            title=title,
                            is_impossible=is_impossible,
                            answers=answers,
                        )

                        examples.append(example)

                logging.info('mean no. of chars in doc: {}'.format(np.mean(len_docs)))
                logging.info('max no. of chars in doc: {}'.format(np.max(len_docs)))

        return examples
