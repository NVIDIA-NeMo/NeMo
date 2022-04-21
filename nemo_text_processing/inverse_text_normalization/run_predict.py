# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from argparse import ArgumentParser
from typing import List

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer


'''
Runs inverse text normalization on text data
'''


def load_file(file_path: str) -> List[str]:
    """
    Load given text file into list of string.

    Args: 
        file_path: file path

    Returns: flat list of string
    """
    res = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            res.append(line)
    return res


def write_file(file_path: str, data: List[str]):
    """
    Writes out list of string to file.

    Args:
        file_path: file path
        data: list of string
        
    """
    with open(file_path, 'w', encoding='utf-8') as fp:
        for line in data:
            fp.write(line + '\n')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", help="input file path", required=True, type=str)
    parser.add_argument(
        "--language", help="language", choices=['en', 'de', 'es', 'ru', 'fr', 'vi'], default="en", type=str
    )
    parser.add_argument("--output", help="output file path", required=True, type=str)
    parser.add_argument("--verbose", help="print denormalization info. For debugging", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.input
    inverse_normalizer = InverseNormalizer(lang=args.language)

    print("Loading data: " + file_path)
    data = load_file(file_path)

    print("- Data: " + str(len(data)) + " sentences")
    inverse_normalizer_prediction = inverse_normalizer.inverse_normalize_list(data, verbose=args.verbose)
    write_file(args.output, inverse_normalizer_prediction)
    print(f"- Denormalized. Writing out to {args.output}")
