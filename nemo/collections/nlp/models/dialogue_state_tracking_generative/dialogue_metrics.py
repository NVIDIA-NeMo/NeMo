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

import numpy as np


class IntentSlotMetrics(object):
    @staticmethod
    def save_predictions(
        filename,
        generated_labels,
        generated_slots,
        ground_truth_labels,
        ground_truth_slots,
        generated_field,
        ground_truth_field,
        inputs,
    ):
        """
        Save predictions as a jsonl file

        Args:
            Each arg is a list of strings (all args have the same length)
        """
        docs = []
        for i in range(len(generated_labels)):
            docs.append(
                {
                    "input": inputs[i],
                    "ground_truth": ground_truth_field[i],
                    "ground_truth_slots": ground_truth_slots[i],
                    "ground_truth_labels": ground_truth_labels[i],
                    "generated": generated_field[i],
                    "generated_slots": generated_slots[i],
                    "generated_labels": generated_labels[i],
                }
            )
        with open(filename, 'w', encoding="UTF-8") as f:
            for item in docs:
                f.write(json.dumps(item) + "\n")

    @staticmethod
    def split_label_and_slots(fields, with_slots=False):
        """
        Split target into label and slots when doing joint label (i.e. intent) classificaiton and slot filling
        For instance, split "reserve_restaurant\nslots: time_of_day(7pm) number_of_people(3)" into 
        label = "reserve_restaurant" and slots = ["time_of_day(7pm)", "number_of_people(3)"]
        Args:
            fields: list of strings 
        """
        # self.cfg.dataset.target_template == "with_slots":
        labels = []
        slots_list = []
        for field in fields:
            if with_slots:
                combo = [i.strip() for i in field.split('slots:', 1)]
                label = 'none'
                if len(combo) == 2:
                    label, slots = combo
                elif len(combo) == 1:
                    slots = combo
                    label = 'none'
                if isinstance(slots, str):
                    # temporary patch for purnendu model output
                    if 'possible intents:' in slots:
                        slots = slots.split('possible intents:')[0]
                    slots = slots.split(', ')
                else:
                    slots = ['None']
            else:
                label = field
                slots = []
            slots_list.append(slots)
            labels.append(label)

        return labels, slots_list

    @staticmethod
    def get_slot_filling_metrics(generated_slots, ground_truth_slots):
        """
        Args:
            generated_slots: list of list of strings. 
                Each string is slot-name and slot-value pair e.g. location(Seattle)
            ground_truth_slots: list of list of strings
        """
        all_recall = []
        all_precision = []
        all_joint_goal_accuracy = []

        for i in range(len(generated_slots)):
            # depulicate and sort
            ground_truth = sorted(list(set(ground_truth_slots[i])))
            predicted = sorted(list(set(generated_slots[i])))
            correct = [item for item in predicted if item in ground_truth]
            recall = len(correct) / len(ground_truth) if len(ground_truth) > 0 else 0
            precision = len(correct) / len(predicted) if len(predicted) > 0 else 0
            joint_goal_accuracy = int(ground_truth == predicted)
            all_recall.append(recall)
            all_precision.append(precision)
            all_joint_goal_accuracy.append(joint_goal_accuracy)

        avg_joint_goal_accuracy = np.mean(all_joint_goal_accuracy) * 100
        avg_precision = np.mean(all_precision) * 100
        avg_recall = np.mean(all_recall) * 100
        avg_f1 = 2 * (avg_recall * avg_precision) / (avg_recall + avg_precision + 1e-20)

        return avg_precision, avg_recall, avg_f1, avg_joint_goal_accuracy
