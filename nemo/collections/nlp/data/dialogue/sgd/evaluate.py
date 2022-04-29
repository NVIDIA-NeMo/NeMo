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

"""
Evaluate predictions JSON file, w.r.t. ground truth file.
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/google-research/blob/master/schema_guided_dst/evaluate.py
"""

import collections
import glob
import json
import os

import numpy as np

from nemo.collections.nlp.metrics.sgd_metrics import (
    ACTIVE_INTENT_ACCURACY,
    JOINT_CAT_ACCURACY,
    JOINT_GOAL_ACCURACY,
    JOINT_NONCAT_ACCURACY,
    NAN_VAL,
    REQUESTED_SLOTS_F1,
    REQUESTED_SLOTS_PRECISION,
    REQUESTED_SLOTS_RECALL,
    SLOT_TAGGING_F1,
    SLOT_TAGGING_PRECISION,
    SLOT_TAGGING_RECALL,
    get_active_intent_accuracy,
    get_average_and_joint_goal_accuracy,
    get_requested_slots_f1,
    get_slot_tagging_f1,
)
from nemo.utils import logging

__all__ = ['get_in_domain_services']

ALL_SERVICES = "#ALL_SERVICES"
SEEN_SERVICES = "#SEEN_SERVICES"
UNSEEN_SERVICES = "#UNSEEN_SERVICES"

# Name of the file containing all predictions and their corresponding frame metrics.
PER_FRAME_OUTPUT_FILENAME = "dialogues_and_metrics.json"


def get_service_set(schema_path: str) -> set:
    """
    Get the set of all services present in a schema.
    Args:
        schema_path: schema file path
    Returns:
        service_set: set of services in file
    """
    service_set = set()
    with open(schema_path, encoding="UTF-8") as f:
        schema = json.load(f)
        for service in schema:
            service_set.add(service["service_name"])
        f.close()
    return service_set


def get_in_domain_services(schema_path: str, service_set: set) -> set:
    """Get the set of common services between a schema and set of services.
    Args:
        schema_path: path to schema file
        service_set: set of services
    Returns: 
        joint_services: joint services between schema path file and service set
    """
    joint_services = get_service_set(schema_path) & service_set
    return joint_services


def get_dataset_as_dict(file_path_patterns) -> dict:
    """Read the DSTC8/SGD json dialogue data as dictionary with dialog ID as keys.
    Args:
        file_path_patterns: list or directory of files 
    Returns:
        dataset_dict: dataset dictionary with dialog ID as keys
    """
    dataset_dict = {}
    if isinstance(file_path_patterns, list):
        list_fp = file_path_patterns
    else:
        list_fp = sorted(glob.glob(file_path_patterns))
    for fp in list_fp:
        if PER_FRAME_OUTPUT_FILENAME in fp:
            continue
        logging.debug("Loading file: %s", fp)
        with open(fp, encoding="UTF-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for dial in data:
                    dataset_dict[dial["dialogue_id"]] = dial
            elif isinstance(data, dict):
                dataset_dict.update(data)
            f.close()
    return dataset_dict


def get_metrics(
    dataset_ref: dict,
    dataset_hyp: dict,
    service_schemas: dict,
    in_domain_services: set,
    joint_acc_across_turn: bool,
    use_fuzzy_match: bool,
):
    """Calculate the DSTC8/SGD metrics.
    Args:
        dataset_ref: The ground truth dataset represented as a dict mapping dialogue id to the corresponding dialogue.
        dataset_hyp: The predictions in the same format as `dataset_ref`.
        service_schemas: A dict mapping service name to the schema for the service.
        in_domain_services: The set of services which are present in the training set.
        joint_acc_across_turn: Whether to compute joint accuracy across turn instead of across service. Should be set to True when conducting multiwoz style evaluation.
        use_fuzzy_match: Whether to use fuzzy string matching when comparing non-categorical slot values. Should be set to False when conducting multiwoz style evaluation.

    Returns:
        all_metric_aggregate: A dict mapping a metric collection name to a dict containing the values
            for various metrics. Each metric collection aggregates the metrics across a specific set of frames in the dialogues.
        per_frame_metric: metrics aggregated for each frame
    """
    # Metrics can be aggregated in various ways, eg over all dialogues, only for
    # dialogues containing unseen services or for dialogues corresponding to a
    # single service. This aggregation is done through metric_collections, which
    # is a dict mapping a collection name to a dict, which maps a metric to a list
    # of values for that metric. Each value in this list is the value taken by
    # the metric on a frame.
    metric_collections = collections.defaultdict(lambda: collections.defaultdict(list))

    # Ensure the dialogs in dataset_hyp also occur in dataset_ref.
    assert set(dataset_hyp.keys()).issubset(set(dataset_ref.keys()))
    logging.debug("len(dataset_hyp)=%d, len(dataset_ref)=%d", len(dataset_hyp), len(dataset_ref))

    # Store metrics for every frame for debugging.
    per_frame_metric = {}

    for dial_id, dial_hyp in dataset_hyp.items():
        dial_ref = dataset_ref[dial_id]

        if set(dial_ref["services"]) != set(dial_hyp["services"]):
            raise ValueError(
                "Set of services present in ground truth and predictions don't match "
                "for dialogue with id {}".format(dial_id)
            )

        joint_metrics = [JOINT_GOAL_ACCURACY, JOINT_CAT_ACCURACY, JOINT_NONCAT_ACCURACY]
        for turn_id, (turn_ref, turn_hyp) in enumerate(zip(dial_ref["turns"], dial_hyp["turns"])):
            metric_collections_per_turn = collections.defaultdict(lambda: collections.defaultdict(lambda: 1.0))
            if turn_ref["speaker"] != turn_hyp["speaker"]:
                raise ValueError("Speakers don't match in dialogue with id {}".format(dial_id))

            # Skip system turns because metrics are only computed for user turns.
            if turn_ref["speaker"] != "USER":
                continue

            if turn_ref["utterance"] != turn_hyp["utterance"]:
                logging.error("Ref utt: %s", turn_ref["utterance"])
                logging.error("Hyp utt: %s", turn_hyp["utterance"])
                raise ValueError("Utterances don't match for dialogue with id {}".format(dial_id))

            hyp_frames_by_service = {frame["service"]: frame for frame in turn_hyp["frames"]}

            # Calculate metrics for each frame in each user turn.
            for frame_ref in turn_ref["frames"]:
                service_name = frame_ref["service"]
                if service_name not in hyp_frames_by_service:
                    raise ValueError(
                        "Frame for service {} not found in dialogue with id {}".format(service_name, dial_id)
                    )
                service = service_schemas[service_name]
                frame_hyp = hyp_frames_by_service[service_name]

                active_intent_acc = get_active_intent_accuracy(frame_ref, frame_hyp)
                slot_tagging_f1_scores = get_slot_tagging_f1(frame_ref, frame_hyp, turn_ref["utterance"], service)
                requested_slots_f1_scores = get_requested_slots_f1(frame_ref, frame_hyp)
                goal_accuracy_dict = get_average_and_joint_goal_accuracy(
                    frame_ref, frame_hyp, service, use_fuzzy_match
                )

                frame_metric = {
                    ACTIVE_INTENT_ACCURACY: active_intent_acc,
                    REQUESTED_SLOTS_F1: requested_slots_f1_scores.f1,
                    REQUESTED_SLOTS_PRECISION: requested_slots_f1_scores.precision,
                    REQUESTED_SLOTS_RECALL: requested_slots_f1_scores.recall,
                }
                if slot_tagging_f1_scores is not None:
                    frame_metric[SLOT_TAGGING_F1] = slot_tagging_f1_scores.f1
                    frame_metric[SLOT_TAGGING_PRECISION] = slot_tagging_f1_scores.precision
                    frame_metric[SLOT_TAGGING_RECALL] = slot_tagging_f1_scores.recall
                frame_metric.update(goal_accuracy_dict)

                frame_id = "{:s}-{:03d}-{:s}".format(dial_id, turn_id, frame_hyp["service"])
                per_frame_metric[frame_id] = frame_metric
                # Add the frame-level metric result back to dialogues.
                frame_hyp["metrics"] = frame_metric

                # Get the domain name of the service.
                domain_name = frame_hyp["service"].split("_")[0]
                domain_keys = [ALL_SERVICES, frame_hyp["service"], domain_name]
                if frame_hyp["service"] in in_domain_services:
                    domain_keys.append(SEEN_SERVICES)

                else:
                    domain_keys.append(UNSEEN_SERVICES)
                for domain_key in domain_keys:
                    for metric_key, metric_value in frame_metric.items():
                        if metric_value != NAN_VAL:
                            if joint_acc_across_turn and metric_key in joint_metrics:
                                metric_collections_per_turn[domain_key][metric_key] *= metric_value
                            else:
                                metric_collections[domain_key][metric_key].append(metric_value)
            if joint_acc_across_turn:
                # Conduct multiwoz style evaluation that computes joint goal accuracy
                # across all the slot values of all the domains for each turn.
                for domain_key in metric_collections_per_turn:
                    for metric_key, metric_value in metric_collections_per_turn[domain_key].items():
                        metric_collections[domain_key][metric_key].append(metric_value)

    all_metric_aggregate = {}
    for domain_key, domain_metric_vals in metric_collections.items():
        domain_metric_aggregate = {}
        for metric_key, value_list in domain_metric_vals.items():
            if value_list:
                # Metrics are macro-averaged across all frames.
                domain_metric_aggregate[metric_key] = round(float(np.mean(value_list)) * 100.0, 2)
            else:
                domain_metric_aggregate[metric_key] = NAN_VAL
        all_metric_aggregate[domain_key] = domain_metric_aggregate
    return all_metric_aggregate, per_frame_metric


def evaluate(
    prediction_dir: str,
    data_dir: str,
    eval_dataset: str,
    in_domain_services: set,
    joint_acc_across_turn: bool,
    use_fuzzy_match: bool,
) -> dict:
    """Calculate the DSTC8/SGD metrics for given data.

    Args:
        prediction_dir: prediction location
        data_dir: ground truth data location.
        eval_dataset: evaluation data split
        in_domain_services: The set of services which are present in the training set.
        joint_acc_across_turn: Whether to compute joint goal accuracy across turn instead of across service. Should be set to True when conducting multiwoz style evaluation.
        use_fuzzy_match: Whether to use fuzzy string matching when comparing non-categorical slot values. Should be set to False when conducting multiwoz style evaluation.

    Returns:
        A dict mapping a metric collection name to a dict containing the values
        for various metrics for all dialogues and all services
    """

    with open(os.path.join(data_dir, eval_dataset, "schema.json"), encoding="UTF-8") as f:
        eval_services = {}
        list_services = json.load(f)
        for service in list_services:
            eval_services[service["service_name"]] = service
        f.close()

    dataset_ref = get_dataset_as_dict(os.path.join(data_dir, eval_dataset, "dialogues_*.json"))
    dataset_hyp = get_dataset_as_dict(os.path.join(prediction_dir, "*.json"))

    # has ALLSERVICE, SEEN_SERVICES, UNSEEN_SERVICES, SERVICE, DOMAIN
    all_metric_aggregate, _ = get_metrics(
        dataset_ref, dataset_hyp, eval_services, in_domain_services, joint_acc_across_turn, use_fuzzy_match
    )
    if SEEN_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {SEEN_SERVICES}  : {sorted(all_metric_aggregate[SEEN_SERVICES].items())}')
    if UNSEEN_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {UNSEEN_SERVICES}: {sorted(all_metric_aggregate[UNSEEN_SERVICES].items())}')
    if ALL_SERVICES in all_metric_aggregate:
        logging.info(f'Dialog metrics for {ALL_SERVICES}   : {sorted(all_metric_aggregate[ALL_SERVICES].items())}')

    # Write the per-frame metrics values with the corrresponding dialogue frames.
    with open(os.path.join(prediction_dir, PER_FRAME_OUTPUT_FILENAME), "w", encoding="UTF-8") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "))
        f.close()
    return all_metric_aggregate[ALL_SERVICES]
