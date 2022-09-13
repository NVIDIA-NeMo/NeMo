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

import argparse
import json

import numpy as np
from rouge_score import rouge_scorer, scoring

"""
Example command for T5 Preds

    ```
    python compute_rouge.py \
        --ground-truth dialogsum_test_gt.jsonl \
        --preds dialogsum_preds_t5.txt \
        --answer-field "answer" 
    ```

Example command for GPT Preds

    ```
    python compute_rouge.py \
        --ground-truth dialogsum_test_gt.jsonl \
        --preds dialogsum_preds_gpt.txt \
        --answer-field "answer" \
        --split-string "summary:"
    ```
"""


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        ln_scores = []
        for possible_ln in reference_ln:
            scores = scorer.score(possible_ln, output_ln)
            ln_scores.append(scores)

        best_index = np.argmax([score_dict["rouge1"][-1] for score_dict in ln_scores])
        aggregator.add_scores(ln_scores[best_index])

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


def load_ref(filename, answer_field):
    lines = open(filename).readlines()
    all_answers = []
    for line in lines:
        line = line.strip()
        line = json.loads(line)
        answers = line[answer_field]

        if isinstance(answers, str):
            answers = [answers]

        all_answers.append(answers)

    return all_answers


def load_preds(filename, split_string):
    with open(filename) as f:
        lines = [line.split(split_string)[-1].strip() for line in f.readlines()]

    return lines


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ground-truth', type=str, help="ground truth .jsonl")
    parser.add_argument('--preds', type=str, help="Text file with test set prompts + model predictions.")
    parser.add_argument(
        '--answer-field',
        type=str,
        help="The key in the ground truth json object containing specifying the correct answer.",
        default="answer",
    )
    parser.add_argument(
        '--split-string',
        type=str,
        help="The text at the end of the prompt, write before the predicted answer. This will be used to find the model's predictions in pred files when the pred file containers both the prompt and prediction.",
        default=None,
    )  # If the pred file only has preditions, just pass none

    args = parser.parse_args()

    pred_file = args.preds
    ref_filename = args.ground_truth
    answer_field = args.answer_field  # The field in the ground truth json that contains the answer
    split_string = args.split_string  # The final few tokens of the prompt right before the generated answer

    output_lns = load_preds(pred_file, split_string)
    reference_lns = load_ref(ref_filename, answer_field)
    assert len(output_lns) == len(reference_lns)
    print("Calculating Rouge")

    scores = calculate_rouge(output_lns=output_lns, reference_lns=reference_lns)
    print(scores)
