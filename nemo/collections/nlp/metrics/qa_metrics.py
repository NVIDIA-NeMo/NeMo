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

import collections
import json
import re
import string

import torch
from tqdm import tqdm

from nemo.collections.nlp.parts.utils_funcs import tensor2list
from nemo.utils import logging


class QAMetrics(object):
    @staticmethod
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    @staticmethod
    def white_space_fix(text):
        return " ".join(text.split())

    @staticmethod
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    @staticmethod
    def normalize_answer(s: str):
        """ Lower text and remove punctuation, articles and extra whitespace """

        return QAMetrics.white_space_fix(QAMetrics.remove_articles(QAMetrics.remove_punc(s.lower())))

    @staticmethod
    def _get_normalized_tokens(s: str):
        """ Get normalized tokens """
        if not s:
            return []
        return QAMetrics.normalize_answer(s).split()

    @staticmethod
    def get_one_f1(prediction: str, ground_truth: str):
        """ Computes f1 score between prediction and ground truth """

        prediction_tokens = QAMetrics._get_normalized_tokens(prediction)
        ground_truth_tokens = QAMetrics._get_normalized_tokens(ground_truth)
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())

        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
            return int(ground_truth_tokens == prediction_tokens)
        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    @staticmethod
    def get_one_exact_match(prediction: str, ground_truth: str):
        """ Computes exact match between prediction and ground truth """

        return int(QAMetrics.normalize_answer(prediction) == QAMetrics.normalize_answer(ground_truth))

    @staticmethod
    def convert_dict_outputs_to_lists(outputs, keys):
        output_lists = [[] for _ in range(len(keys))]
        for output in outputs:
            for i, key in enumerate(keys):
                if isinstance(output[key], torch.Tensor):
                    output_lists[i].extend(tensor2list(output[key]))
                else:
                    output_lists[i].extend(output[key])

        return output_lists

    @staticmethod
    def get_exact_match_and_f1(examples, preds, question_id_filter=[]):
        """
        Returns a dictionary of question id: exact match/f1 score
        Questions with ids *not* present in `question_id_filter` are excluded
        """
        exact_scores = {}
        f1_scores = {}

        for example in examples:
            question_id = example.qas_id
            if question_id not in question_id_filter:
                continue

            gold_answers = [answer["text"] for answer in example.answers if QAMetrics.normalize_answer(answer["text"])]

            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = [""]

            pred = preds[question_id]
            exact_scores[question_id] = max(QAMetrics.get_one_exact_match(pred, a) for a in gold_answers)
            f1_scores[question_id] = max(QAMetrics.get_one_f1(pred, a) for a in gold_answers)

        return exact_scores, f1_scores

    @staticmethod
    def make_eval_dict(exact_scores, f1_scores, prefix=""):
        """ Returns dictionary with formatted evaluation scores """

        total = len(exact_scores)
        return collections.OrderedDict(
            [
                (f"{prefix}exact", (100.0 * sum(exact_scores.values()) / total) if total != 0 else 0.0),
                (f"{prefix}f1", (100.0 * sum(f1_scores.values()) / total) if total != 0 else 0.0),
                (f"{prefix}total", float(total)),
            ]
        )

    @staticmethod
    def merge_eval_dicts(eval_dicts):
        """
        Combines multiple evaluation dict outputs into one dict
        Ex: combines eval dicts for HasAns F1, NoAnsF1, and Total F1
        """

        merged_dict = collections.OrderedDict()
        for eval_dict in eval_dicts:
            for key in eval_dict:
                merged_dict[key] = eval_dict[key]

        return merged_dict

    @staticmethod
    def evaluate_predictions(examples, all_predictions):
        """ 
        Calculates exact match and f1 scores for all predictions, 
            questions with answers, and no answer questions
        """

        qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples[: len(all_predictions)]}
        has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
        no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

        filters_and_prefixes = [
            (list(qas_id_to_has_answer), ""),
            (has_answer_qids, "HasAns_"),
            (no_answer_qids, "NoAns_"),
        ]

        eval_dicts = []
        for qas_id_filter, prefix in filters_and_prefixes:
            curr_exact, curr_f1 = QAMetrics.get_exact_match_and_f1(examples, all_predictions, qas_id_filter)
            curr_eval_dict = QAMetrics.make_eval_dict(curr_exact, curr_f1, prefix=prefix)
            eval_dicts.append(curr_eval_dict)

        merged_eval_dict = QAMetrics.merge_eval_dicts(eval_dicts)

        return merged_eval_dict

    @staticmethod
    def dump_predicted_answers_to_file(output_filename, examples, predictions):
        logging.info(f"Writing predictions to {output_filename}")

        with open(output_filename, "w") as writer:
            for ex in tqdm(examples):
                output_item = {
                    "id": ex.qas_id,
                    "context": ex.context_text,
                    "question": ex.question_text,
                    "predicted_answer": predictions[ex.qas_id],
                }
                writer.write(json.dumps(output_item) + "\n")

    @staticmethod
    def dump_nbest_predictions_to_file(output_filename, examples, nbest_predictions, keys_to_dump=[]):
        logging.info(f"Writing nbest predictions to {output_filename}")

        with open(output_filename, "w") as writer:
            for ex in tqdm(examples):
                output_item = {
                    "id": ex.qas_id,
                    "context": ex.context_text,
                    "question": ex.question_text,
                    "nbest_predictions": [],
                }
                for pred in nbest_predictions[ex.qas_id]:
                    output_item["nbest_predictions"].append({key: pred[key] for key in keys_to_dump})

                writer.write(json.dumps(output_item) + "\n")
