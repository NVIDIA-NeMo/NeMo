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
import json
import random

from tqdm import tqdm


"""
Dataset preprocessing script for the Financial Phrase Bank Sentiement dataset: 
https://www.researchgate.net/profile/Pekka_Malo/publication/251231364_FinancialPhraseBank-v10/data/0c96051eee4fb1d56e000000/FinancialPhraseBank-v10.zip

Converts the dataset into a jsonl format that can be used for p-tuning/prompt tuning in NeMo. 

Inputs:
    data-dir: (str) The unziped directory where the Financial PhraseBank dataset was downloaded, files will be saved here
    file-name: (str) Name of the input file you want to process
    save-name-base: (str) The base name for each of the train, val, and test files. If save-name-base were 'financial_phrase_bank' for
                    example, the files would be saved as financial_phrase_bank_train.jsonl, financial_phrase_bank_val.jsonl, and 
                    financial_phrase_bank_test.jsonl
    make-ground-truth: (bool) If true, test files will include labels, if false, test files will not include labels
    random-seed: (int) Random seed for repeatable shuffling of train/val/test splits. 
    train-percent: (float) Precentage of data that should be used for the train split. The val and test splits will be made
                    by splitting the remaining data evenly. 

Saves train, val, and test files for the Financial PhraseBank dataset.

An example of the processed output written to file:
    
    {"taskname": "sentiment", "sentence": "In the Baltic countries , sales fell by 42.6 % .", "label": " negative"}
    {"taskname": "sentiment", "sentence": "Danske Bank is Denmark 's largest bank with 3.5 million customers .", "label": " neutral"}
    {"taskname": "sentiment", "sentence": "The total value of the deliveries is some EUR65m .", "label": " neutral"}
    {"taskname": "sentiment", "sentence": "Operating profit margin increased from 11.2 % to 11.7 % .", "label": " positive"}
    {"taskname": "sentiment", "sentence": "It will also strengthen Ruukki 's offshore business .", "label": " positive"}
    {"taskname": "sentiment", "sentence": "Sanoma News ' advertising sales decreased by 22 % during the year .", "label": " negative"}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/FinancialPhraseBank-v1.0")
    parser.add_argument("--file-name", type=str, default="Sentences_AllAgree.txt")
    parser.add_argument("--save-name-base", type=str, default="financial_phrase_bank")
    parser.add_argument("--make-ground-truth", action='store_true')
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--train-percent", type=float, default=0.8)
    args = parser.parse_args()

    data = open(f"{args.data_dir}/{args.file_name}", "r", encoding="ISO-8859-1").readlines()
    save_name_base = f"{args.data_dir}/{args.save_name_base}"

    process_data(data, save_name_base, args.train_percent, args.random_seed, args.make_ground_truth)


def process_data(data, save_name_base, train_percent, random_seed, make_ground_truth=False):
    random.seed(random_seed)
    random.shuffle(data)

    data_total = len(data)
    train_total = int(data_total * train_percent)
    val_total = (data_total - train_total) // 2

    train_set = data[0:train_total]
    val_set = data[train_total : train_total + val_total]
    test_set = data[train_total + val_total :]

    gen_file(train_set, save_name_base, 'train')
    gen_file(val_set, save_name_base, 'val')
    gen_file(test_set, save_name_base, 'test', make_ground_truth)


def gen_file(data, save_name_base, split_type, make_ground_truth=False):
    save_path = f"{save_name_base}_{split_type}.jsonl"
    print(f"Saving {split_type} split to {save_path}")

    with open(save_path, 'w') as save_file:
        for line in tqdm(data):
            example_json = {"taskname": "sentiment"}
            sent, label = line.split('@')
            sent = sent.strip()
            label = label.strip()
            example_json["sentence"] = sent

            # Dont want labels in the test set
            if split_type != "test" or make_ground_truth:
                example_json["label"] = " " + label

            save_file.write(json.dumps(example_json) + '\n')


if __name__ == "__main__":
    main()
