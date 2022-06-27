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
import os
import shutil
from os.path import exists

from assistant_utils import process_assistant

from nemo.collections.nlp.data.data_utils.data_preprocessing import (
    DATABASE_EXISTS_TMP,
    MODE_EXISTS_TMP,
    create_dataset,
    get_dataset,
    get_vocab,
    if_exist,
)
from nemo.utils import logging


def ids2text(ids, vocab):
    """
    Map list of ids of words in utterance to utterance
    """
    return ' '.join([vocab[int(id_)] for id_ in ids])


def process_atis(infold, outfold, modes=['train', 'test'], do_lower_case=False):
    """ 
    Process ATIS dataset found at https://www.kaggle.com/siddhadev/atis-dataset-from-ms-cntk
    Args:
        infold: location for input fold of data
        outfold: location for output fold of data
        modes: dataset splits to process
        do_lowercase: whether to lowercase the input utterances
    """
    vocab = get_vocab(f'{infold}/atis.dict.vocab.csv')

    if if_exist(outfold, [f'{mode}.tsv' for mode in modes]):
        logging.info(DATABASE_EXISTS_TMP.format('ATIS', outfold))
        return outfold
    logging.info(f'Processing ATIS dataset and storing at {outfold}.')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    for mode in modes:
        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w', encoding='utf-8')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w', encoding='utf-8')

        queries = open(f'{infold}/atis.{mode}.query.csv', 'r', encoding='utf-8').readlines()
        intents = open(f'{infold}/atis.{mode}.intent.csv', 'r', encoding='utf-8').readlines()
        slots = open(f'{infold}/atis.{mode}.slots.csv', 'r', encoding='utf-8').readlines()

        for i, query in enumerate(queries):
            sentence = ids2text(query.strip().split()[1:-1], vocab)
            if do_lower_case:
                sentence = sentence.lower()
            outfiles[mode].write(f'{sentence}\t{intents[i].strip()}\n')
            slot = ' '.join(slots[i].strip().split()[1:-1])
            outfiles[mode + '_slots'].write(slot + '\n')

    shutil.copyfile(f'{infold}/atis.dict.intent.csv', f'{outfold}/dict.intents.csv')
    shutil.copyfile(f'{infold}/atis.dict.slots.csv', f'{outfold}/dict.slots.csv')
    for mode in modes:
        outfiles[mode].close()


def process_snips(infold, outfold, do_lower_case, modes=['train', 'test'], dev_split=0.1):
    """
    Process snips dataset
    Args:
        infold: location for input fold of data
        outfold: location for output fold of data
        do_lowercase: whether to lowercase the input utterances
        modes: dataset splits to process
        dev_split: proportion of train samples to put into dev set
    """
    if not os.path.exists(infold):
        link = 'https://github.com/snipsco/spoken-language-understanding-research-datasets'
        raise ValueError(f'Data not found at {infold}. ' f'You may request to download the SNIPS dataset from {link}.')

    exist = True
    for dataset in ['light', 'speak', 'all']:
        if if_exist(f'{outfold}/{dataset}', [f'{mode}.tsv' for mode in modes]):
            logging.info(DATABASE_EXISTS_TMP.format('SNIPS-' + dataset, outfold))
        else:
            exist = False
    if exist:
        return outfold

    logging.info(f'Processing SNIPS dataset and storing at folders "speak", "light" and "all" under {outfold}.')
    logging.info(
        f'Processing and importing "smart-speaker-en-close-field" -> "speak" and "smart-speaker-en-close-field" -> "light".'
    )

    os.makedirs(outfold, exist_ok=True)

    speak_dir = 'smart-speaker-en-close-field'
    light_dir = 'smart-lights-en-close-field'

    light_files = [f'{infold}/{light_dir}/dataset.json']
    speak_files = [f'{infold}/{speak_dir}/training_dataset.json']
    speak_files.append(f'{infold}/{speak_dir}/test_dataset.json')

    light_train, light_dev, light_slots, light_intents = get_dataset(light_files, dev_split)
    speak_train, speak_dev, speak_slots, speak_intents = get_dataset(speak_files)

    create_dataset(light_train, light_dev, light_slots, light_intents, do_lower_case, f'{outfold}/light')
    create_dataset(speak_train, speak_dev, speak_slots, speak_intents, do_lower_case, f'{outfold}/speak')
    create_dataset(
        light_train + speak_train,
        light_dev + speak_dev,
        light_slots | speak_slots,
        light_intents | speak_intents,
        do_lower_case,
        f'{outfold}/all',
    )


def process_jarvis_datasets(
    infold, outfold, modes=['train', 'test', 'dev'], do_lower_case=False, ignore_prev_intent=False
):
    """ 
    Process and convert Jarvis datasets into NeMo's BIO format
    Args:
        infold: location for input fold of data
        outfold: location for output fold of data
        modes: dataset splits to process
        do_lowercase: whether to lowercase the input utterances
        ignore_prev_intent: whether to include intent from previous turn in predicting intent of current turn
    """
    dataset_name = "jarvis"
    if if_exist(outfold, ['dict.intents.csv', 'dict.slots.csv']):
        logging.info(DATABASE_EXISTS_TMP.format(dataset_name, outfold))
        return outfold

    logging.info(f'Processing {dataset_name} dataset and storing at {outfold}')

    os.makedirs(outfold, exist_ok=True)

    outfiles = {}
    intents_list = {}
    slots_list = {}
    slots_list_all = {}

    outfiles['dict_intents'] = open(f'{outfold}/dict.intents.csv', 'w', encoding='utf-8')
    outfiles['dict_slots'] = open(f'{outfold}/dict.slots.csv', 'w', encoding='utf-8')

    outfiles['dict_slots'].write('O\n')
    slots_list["O"] = 0
    slots_list_all["O"] = 0

    for mode in modes:
        if if_exist(outfold, [f'{mode}.tsv']):
            logging.info(MODE_EXISTS_TMP.format(mode, dataset_name, outfold, mode))
            continue

        if not if_exist(infold, [f'{mode}.tsv']):
            logging.info(f'{mode} mode of {dataset_name}' f' is skipped as it was not found.')
            continue

        outfiles[mode] = open(os.path.join(outfold, mode + '.tsv'), 'w', encoding='utf-8')
        outfiles[mode].write('sentence\tlabel\n')
        outfiles[mode + '_slots'] = open(f'{outfold}/{mode}_slots.tsv', 'w', encoding='utf-8')

        queries = open(f'{infold}/{mode}.tsv', 'r', encoding='utf-8').readlines()

        for i, query in enumerate(queries):
            line_splits = query.strip().split("\t")
            if len(line_splits) == 3:
                intent_str, slot_tags_str, sentence = line_splits
            else:
                intent_str, sentence = line_splits
                slot_tags_str = ""

            if intent_str not in intents_list:
                intents_list[intent_str] = len(intents_list)
                outfiles['dict_intents'].write(f'{intent_str}\n')

            if ignore_prev_intent:
                start_token = 2
            else:
                start_token = 1

            if do_lower_case:
                sentence = sentence.lower()
            sentence_cld = " ".join(sentence.strip().split()[start_token:-1])
            outfiles[mode].write(f'{sentence_cld}\t' f'{str(intents_list[intent_str])}\n')

            slot_tags_list = []
            if slot_tags_str.strip():
                slot_tags = slot_tags_str.strip().split(",")
                for st in slot_tags:
                    if not st.strip():
                        continue
                    [start_i, end_i, slot_name] = st.strip().split(":")
                    slot_tags_list.append([int(start_i), int(end_i), slot_name])
                    if slot_name not in slots_list:
                        slots_list[slot_name] = len(slots_list)
                        slots_list_all[f'B-{slot_name}'] = len(slots_list_all)
                        slots_list_all[f'I-{slot_name}'] = len(slots_list_all)
                        outfiles['dict_slots'].write(f'B-{slot_name}\n')
                        outfiles['dict_slots'].write(f'I-{slot_name}\n')

            slot_tags_list.sort(key=lambda x: x[0])
            slots = []
            processed_index = 0
            for tag_start, tag_end, tag_str in slot_tags_list:
                if tag_start > processed_index:
                    words_list = sentence[processed_index:tag_start].strip().split()
                    slots.extend([str(slots_list_all['O'])] * len(words_list))
                words_list = sentence[tag_start:tag_end].strip().split()
                slots.append(str(slots_list_all[f'B-{tag_str}']))
                slots.extend([str(slots_list_all[f'I-{tag_str}'])] * (len(words_list) - 1))
                processed_index = tag_end

            if processed_index < len(sentence):
                words_list = sentence[processed_index:].strip().split()
                slots.extend([str(slots_list_all['O'])] * len(words_list))

            slots = slots[1:-1]
            slot = ' '.join(slots)
            outfiles[mode + '_slots'].write(slot + '\n')

        outfiles[mode + '_slots'].close()
        outfiles[mode].close()

    outfiles['dict_slots'].close()
    outfiles['dict_intents'].close()

    return outfold


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=['atis', 'snips', 'jarvis', 'assistant'],
    )
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help='path to the folder containing the dataset files'
    )
    parser.add_argument("--target_data_dir", required=True, type=str, help='path to save the processed dataset')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument(
        "--ignore_prev_intent",
        action='store_true',
        help='ignores previous intent while importing datasets in jarvis\'s format',
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    source_dir = args.source_data_dir
    target_dir = args.target_data_dir

    if not exists(source_dir):
        raise FileNotFoundError(f"{source_dir} does not exist.")

    if dataset_name == 'atis':
        process_atis(infold=source_dir, outfold=target_dir, do_lower_case=args.do_lower_case)
    elif dataset_name == 'snips':
        process_snips(infold=source_dir, outfold=target_dir, do_lower_case=args.do_lower_case)
    elif dataset_name == 'jarvis':
        process_jarvis_datasets(
            infold=source_dir,
            outfold=target_dir,
            modes=["train", "test", "dev"],
            do_lower_case=args.do_lower_case,
            ignore_prev_intent=args.ignore_prev_intent,
        )
    elif dataset_name == 'assistant':
        process_assistant(infold=source_dir, outfold=target_dir)
    else:
        raise ValueError(f'Dataset {dataset_name} is not supported.')
