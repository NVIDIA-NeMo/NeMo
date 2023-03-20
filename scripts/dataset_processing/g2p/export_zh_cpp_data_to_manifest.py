# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from argparse import ArgumentParser
from collections import defaultdict


"""
Converts Chinese Polyphones with Pinyin (CPP) data to .json manifest format for Chinese HeteronymClassificationModel training.
Chinese dataset could be found here:
    https://github.com/kakaobrain/g2pM#the-cpp-dataset

Usage
# prepare manifest
mkdir -p ./cpp_manifest
git clone https://github.com/kakaobrain/g2pM.git
python3 export_zh_cpp_data_to_manifest.py --data_folder g2pM/data/ --output_folder ./cpp_manifest

"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_folder', help="Path to data folder with the CPP data files", type=str, required=True)
    parser.add_argument(
        "--output_folder", help="Path to data folder with output .json file to store the data", type=str, required=True
    )
    return parser.parse_args()


def convert_cpp_data_to_manifest(data_folder: str, output_folder: str):
    """
    Convert CPP data to .json manifest

    Args:
        data_folder: data_folder that contains data files
        output_folder: path to output folder
    """
    wordid_dict = defaultdict(set)
    for key in ['train', 'dev', 'test']:
        output_manifest = f"{output_folder}/{key}.json"
        sent_file = f"{data_folder}/{key}.sent"
        label_file = f"{data_folder}/{key}.lb"

        with open(output_manifest, "w") as f_out, open(sent_file, 'r') as f_sent, open(label_file, 'r') as f_label:
            lines_sent, lines_label = f_sent.readlines(), f_label.readlines()
            lines_label = [line.strip('\n') for line in lines_label]
            lines_idx = [line.index('\u2581') for line in lines_sent]
            lines_sent = [line.strip('\n').replace('\u2581', '') for line in lines_sent]

            for i, index in enumerate(lines_idx):
                wordid_dict[lines_sent[i][index]].add(lines_label[i])

            for i, sent in enumerate(lines_sent):
                start, end = lines_idx[i], lines_idx[i] + 1
                heteronym_span = sent[start:end]
                entry = {
                    "text_graphemes": sent,
                    "start_end": [start, start + 1],
                    "heteronym_span": heteronym_span,
                    "word_id": f"{heteronym_span}_{lines_label[i]}",
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Data saved at {output_manifest}")

    word_id_file = f"{output_folder}/wordid.tsv"
    with open(word_id_file, 'w') as f_wordid:
        f_wordid.write(f"homograph\twordid\tlabel\tpronunciation\n")
        for key, pronunciations in wordid_dict.items():
            for value in pronunciations:
                f_wordid.write(f"{key}\t{key}_{value}\tNone\t{value}\n")


if __name__ == '__main__':
    args = parse_args()
    convert_cpp_data_to_manifest(args.data_folder, args.output_folder)
