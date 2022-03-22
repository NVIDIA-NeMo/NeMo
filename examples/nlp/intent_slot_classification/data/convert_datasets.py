# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import pandas as pd


def convert_atis_multi_label(source_dir: str, target_dir: str, mode: str) -> None:
    """
    Converts single label atis nemo data to multi-label data. Previous 
    labels in atis mapped multi-labels to a single index rather than two separate indicies.

    Args:
        source_dir: directory that stored original nemo files
        target_dir: directory to store multi-label nemo files
        mode: specifies the name of the dataset i.e, train, test, dev

    Returns:
        None
    """
    data = pd.read_csv(f'{source_dir}/{mode}.tsv', sep='\t')
    # Get the original intent dictionary
    old_intents_file = f'{source_dir}/dict.intents.csv'
    new_intents_file = f'{target_dir}/dict.intents.csv'
    intent_labels = []

    with open(old_intents_file, "r") as input_file:
        old_intents = input_file.read().splitlines()

    with open(new_intents_file, "r") as input_file:
        new_intents = input_file.read().splitlines()

    for index, intent in data.iterrows():
        temp_dict = {}
        temp_dict['sentence'] = intent['sentence']
        old_label = old_intents[int(intent['label'])]

        values = [old_label]

        if '+' in old_label:
            values = old_label.split('+')

        for index, label in enumerate(new_intents):
            if label in values:
                if 'label' not in temp_dict:
                    temp_dict['label'] = f"{index}"
                else:
                    temp_dict['label'] = f"{temp_dict['label']},{index}"

        intent_labels.append(temp_dict)

    multi_intent_df = pd.DataFrame(intent_labels)
    multi_intent_df.to_csv(f'{target_dir}/{mode}.tsv', sep='\t', index=False)


def convert_intent_dictionary(source_dir: str, target_dir: str) -> None:
    """
    Converts original intent dictionary containing labels that represented multiple labels into 
    dictionary with only single labels. Example: if index 5 was referring to label "a+b", it is no longer 
    a label in the new intent dictionary. Only labels "a" and "b" are included within the new dictionary

    Args:
        source_dir: directory that stored original nemo files
        target_dir: directory to store multi-label nemo files

    Returns:
        None
    """
    os.makedirs(target_dir, exist_ok=True)
    source_file = os.path.join(source_dir, "dict.intents.csv")
    target_file = os.path.join(target_dir, "dict.intents.csv")

    with open(source_file, "r") as input_file:
        orig_intents = input_file.read().splitlines()

    with open(target_file, "w") as output_file:
        for line in orig_intents:
            if "+" not in line:
                output_file.write(f"{line}\n")


if __name__ == "__main__":
    # Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Process and convert datasets into NeMo\'s format.")
    parser.add_argument(
        "--source_data_dir", required=True, type=str, help='path to the folder containing the dataset files'
    )
    parser.add_argument("--target_data_dir", required=True, type=str, help='path to save the processed dataset')

    args = parser.parse_args()

    source_dir = args.source_data_dir
    target_dir = args.target_data_dir

    shutil.copyfile(f'{source_dir}/test.tsv', f'{source_dir}/dev.tsv')

    convert_intent_dictionary(f'{source_dir}', f'{target_dir}')
    convert_atis_multi_label(f'{source_dir}', f'{target_dir}', 'train')
    convert_atis_multi_label(f'{source_dir}', f'{target_dir}', 'dev')
    shutil.copyfile(f'{source_dir}/dict.slots.csv', f'{target_dir}/dict.slots.csv')
    shutil.copyfile(f'{source_dir}/train_slots.tsv', f'{target_dir}/train_slots.tsv')
    shutil.copyfile(f'{source_dir}/test_slots.tsv', f'{target_dir}/dev_slots.tsv')
