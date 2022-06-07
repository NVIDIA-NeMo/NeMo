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

import os
import re
import shutil

from nemo.collections.nlp.data.data_utils.data_preprocessing import DATABASE_EXISTS_TMP, if_exist, write_files
from nemo.utils import logging


def copy_input_files(infold):
    """ 
    Put training files in convenient place for conversion to our format. 
    
    Args:
        infold: location of an original fold of the dataset (in the sense of k-fold cross validation)
    """
    our_infold = infold + "/dataset"

    if os.path.exists(our_infold + "/trainset") and os.path.exists(our_infold + "/testset"):
        logging.info("Input folders exists")
        return

    logging.info(f"Copying files to input folder: {our_infold}")
    os.makedirs(infold, exist_ok=True)

    old_infold = (
        infold + '/CrossValidation/autoGeneFromRealAnno/autoGene_2018_03_22-13_01_25_169/CrossValidation/KFold_1'
    )
    if not os.path.exists(our_infold + "/trainset"):
        shutil.copytree(old_infold + '/trainset', our_infold + '/trainset')

    if not os.path.exists(our_infold + "/testset"):
        shutil.copytree(old_infold + '/testset/csv', our_infold + '/testset')


def get_intents(infold):
    """ Get list of intents from file names. """
    intents = [f[:-4] for f in os.listdir(infold)]
    intents.sort()
    logging.info(f'Found {len(intents)} intents')

    return intents


def get_intent_queries(infold, intent_names, mode):
    """ Get list of queries with their corresponding intent number. """
    intent_queries = ['sentence\tlabel\n']

    for index, intent in enumerate(intent_names):
        queries = open(f'{infold}/{mode}set/{intent}.csv', 'r', encoding='utf-8').readlines()
        for query in queries[1:]:
            phrases = query.split(";")
            intent_query = phrases[4][1:-1] + "\t" + str(index)
            intent_queries.append(intent_query)

    return intent_queries


def get_slots(infold, modes):
    """
    Find a list of unique slot types in training and testing data.
    We use a single slot type name both for starting and continuation tokens (not using B-, I- notation).
    """
    slots = set()

    for mode in modes:
        path = f'{infold}/{mode}set'
        for filename in os.listdir(path):
            lines = open(f'{path}/{filename}', 'r', encoding='utf-8').readlines()
            for line in lines[1:]:
                query = line.split(";")[3]
                slot_phrases = re.findall('\[.*?\]', query)
                for slot_phrase in slot_phrases:
                    slot = slot_phrase.split(" : ")[0][1:]
                    slots.add(slot)

    slots = sorted(slots)
    slots.append("O")
    logging.info(f'Found {len(slots)} slot types')

    return slots


def get_slot_queries(infold, slot_dict, mode, intent_names):
    """ 
    Convert each word in a query to corresponding slot number. 
    Args:
        infold: fold of the data
        slot_dict: dict containing slot-names to positions 
        mode: train, validation or test
        intent_names: list of intents
    """
    slot_queries = []
    outside_slot = len(slot_dict) - 1

    # keep the same order of files/queries as for intents
    for intent in intent_names:
        lines = open(f'{infold}/{mode}set/{intent}.csv', 'r', encoding='utf-8').readlines()
        for line in lines[1:]:
            slot_query = ""
            query = line.split(";")[3]
            words = query.split(" ")
            current_slot = outside_slot
            for word in words:
                if word[0] == "[":
                    current_slot = slot_dict[word[1:]]
                elif word[0] == ":":
                    continue
                else:
                    slot_query += str(current_slot) + " "
                    if word[-1] == ']':
                        current_slot = outside_slot

            slot_queries.append(slot_query.strip())

    return slot_queries


def process_assistant(infold, outfold, modes=['train', 'test']):
    """
    https://github.com/xliuhw/NLU-Evaluation-Data - this dataset includes
    about 25 thousand examples with 66 various multi-domain intents and 57 entity types.
    """
    if if_exist(outfold, [f'{mode}_slots.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('robot', outfold))
        return outfold

    logging.info(f'Processing assistant commands dataset and store at {outfold}')
    os.makedirs(outfold, exist_ok=True)

    # copy train/test files to the convenient directory to work with
    copy_input_files(infold)
    infold += "/dataset"

    # get list of intents from train folder (test folder supposed to be the same)
    intent_names = get_intents(infold + "/trainset")
    write_files(intent_names, f'{outfold}/dict.intents.csv')

    # get all train and test queries with their intent
    for mode in modes:
        intent_queries = get_intent_queries(infold, intent_names, mode)
        write_files(intent_queries, f'{outfold}/{mode}.tsv')

    # get list of all unique slots in training and testing files
    slot_types = get_slots(infold, modes)
    write_files(slot_types, f'{outfold}/dict.slots.csv')

    # create files of slot queries
    slot_dict = {k: v for v, k in enumerate(slot_types)}
    for mode in modes:
        slot_queries = get_slot_queries(infold, slot_dict, mode, intent_names)
        write_files(slot_queries, f'{outfold}/{mode}_slots.tsv')
