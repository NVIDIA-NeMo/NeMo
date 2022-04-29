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
import re

import numpy as np

from nemo.collections.nlp.metrics.dialogue_metrics import DialogueGenerationMetrics


def read_jsonl(filename):
    with open(filename, 'r', encoding="UTF-8") as f:
        docs = [json.loads(line) for line in f.readlines()]
    return docs


def get_incorrect_labels(docs):
    incorrect_labels_docs = []
    for doc in docs:
        if doc["ground_truth_labels"] != doc["generated_labels"]:
            incorrect_labels_docs.append(
                {
                    "input": doc["input"],
                    "ground_truth_labels": doc["ground_truth_labels"],
                    "generated_labels": doc["generated_labels"],
                }
            )
    return incorrect_labels_docs


def get_incorrect_slots(docs):
    incorrect_slots_docs = []
    for doc in docs:
        if doc["ground_truth_slots"] != doc["generated_slots"]:
            incorrect_slots_docs.append(
                {
                    "input": doc["input"],
                    "ground_truth_slots": doc["ground_truth_slots"],
                    "generated_slots": doc["generated_slots"],
                }
            )
    return incorrect_slots_docs


def sort_by_f1(docs):
    for i in range(len(docs)):
        doc = docs[i]
        generated_field = doc["generated"]
        ground_truth_field = doc["ground_truth"]
        generated_field = remove_punctation(generated_field.lower())
        ground_truth_field = remove_punctation(ground_truth_field.lower())
        p, r, f1 = DialogueGenerationMetrics._get_one_f1(generated_field, ground_truth_field)
        docs[i]["f1"] = f1
        docs[i]["generated"] = generated_field
        docs[i]["ground_truth"] = ground_truth_field
    docs.sort(key=lambda x: x["f1"])
    return docs


def remove_punctation(sentence):
    return re.sub(r'[^\w\s]', '', sentence)


def generation_main(filename):
    docs = read_jsonl(filename)
    docs = sort_by_f1(docs)
    bleu = DialogueGenerationMetrics.get_bleu(
        [doc["generated"] for doc in docs], [doc["ground_truth"] for doc in docs]
    )
    acc = np.mean([int(doc["generated"] == doc["ground_truth"]) for doc in docs]) * 100
    f1 = np.mean([doc["f1"] for doc in docs])
    print("Token level F1 is {:.3}".format(f1))
    print("BLEU is {:.3}".format(bleu))
    print("Exact match accuracy is {:.3}".format(acc))
    for i in range(0):
        print(docs[i])


def classification_main(filename):
    docs = read_jsonl(filename)
    incorrect_labels_docs = get_incorrect_labels(docs)
    incorrect_slots_docs = get_incorrect_slots(docs)

    print("{} / {} have incorrect labels".format(len(incorrect_labels_docs), len(docs)))
    print("{} / {} have incorrect slots".format(len(incorrect_slots_docs), len(docs)))

    for doc in incorrect_labels_docs:
        print(doc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_filename")
    parser.add_argument("--mode", choices=['generation', 'classification'], default='classification')
    args = parser.parse_args()
    if args.mode == 'classification':
        classification_main(args.prediction_filename)
    else:
        generation_main(args.prediction_filename)
