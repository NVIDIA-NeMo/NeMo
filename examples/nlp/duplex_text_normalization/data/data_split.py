# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script creates data splits of the Google Text Normalization dataset
of the format mentioned in the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.

USAGE Example:
1. Download the Google TN dataset from https://www.kaggle.com/google-nlu/text-normalization
2. Unzip the English subset (e.g., by running `tar zxvf  en_with_types.tgz`). Then there will a folder named `en_with_types`.
3. Run this script
# python data_split.py       \
        --data_dir=en_with_types/           \
        --output_dir=data_split/          \
        --lang=en

In this example, the split files will be stored in the `data_split` folder.
The folder should contain three subfolders `train`, 'dev', and `test` with `.tsv` files.
"""

from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import isdir, isfile, join

from tqdm import tqdm

from nemo.collections.nlp.data.text_normalization import constants

# Local Constants
TEST_SIZE_EN = 100002
TEST_SIZE_RUS = 100007


def read_google_data(data_file: str, lang: str, split: str, add_test_full=False):
    """
    The function can be used to read the raw data files of the Google Text Normalization
    dataset (which can be downloaded from https://www.kaggle.com/google-nlu/text-normalization)

    Args:
        data_file: Path to the data file. Should be of the form output-xxxxx-of-00100
        lang: Selected language.
        split: data split
        add_test_full: do not truncate test data i.e. take the whole test file not #num of lines
    Return:
        data: list of examples
    """
    data = []
    cur_classes, cur_tokens, cur_outputs = [], [], []
    with open(data_file, 'r', encoding='utf-8') as f:
        for linectx, line in tqdm(enumerate(f)):
            es = line.strip().split('\t')
            if split == "test" and not add_test_full:
                # For the results reported in the paper "RNN Approaches to Text Normalization: A Challenge":
                # + For English, the first 100,002 lines of output-00099-of-00100 are used for the test set
                # + For Russian, the first 100,007 lines of output-00099-of-00100 are used for the test set
                if lang == constants.ENGLISH and linectx == TEST_SIZE_EN:
                    break
                if lang == constants.RUSSIAN and linectx == TEST_SIZE_RUS:
                    break
            if len(es) == 2 and es[0] == '<eos>':
                data.append((cur_classes, cur_tokens, cur_outputs))
                # Reset
                cur_classes, cur_tokens, cur_outputs = [], [], []
                continue

            # Remove _trans (for Russian)
            if lang == constants.RUSSIAN:
                es[2] = es[2].replace('_trans', '')
            # Update the current example
            assert len(es) == 3
            cur_classes.append(es[0])
            cur_tokens.append(es[1])
            cur_outputs.append(es[2])
    return data


if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess Google text normalization dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder with data')
    parser.add_argument('--output_dir', type=str, default='preprocessed', help='Path to folder with preprocessed data')
    parser.add_argument(
        '--lang', type=str, default=constants.ENGLISH, choices=constants.SUPPORTED_LANGS, help='Language'
    )
    parser.add_argument(
        '--add_test_full',
        action='store_true',
        help='If True, additional folder test_full will be created without truncation of files',
    )
    args = parser.parse_args()

    # Create the output dir (if not exist)
    if not isdir(args.output_dir):
        mkdir(args.output_dir)
        mkdir(args.output_dir + '/train')
        mkdir(args.output_dir + '/dev')
        mkdir(args.output_dir + '/test')
        if args.add_test_full:
            mkdir(args.output_dir + '/test_full')

    for fn in sorted(listdir(args.data_dir))[::-1]:
        fp = join(args.data_dir, fn)
        if not isfile(fp):
            continue
        if not fn.startswith('output'):
            continue

        # Determine the current split
        split_nb = int(fn.split('-')[1])
        if split_nb < 90:
            cur_split = "train"
        elif split_nb < 95:
            cur_split = "dev"
        elif split_nb == 99:
            cur_split = "test"
        data = read_google_data(data_file=fp, lang=args.lang, split=cur_split)
        # write out
        output_file = join(args.output_dir, f'{cur_split}', f'{fn}.tsv')
        print(fp)
        print(output_file)
        output_f = open(output_file, 'w', encoding='utf-8')
        for inst in data:
            cur_classes, cur_tokens, cur_outputs = inst
            for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
                output_f.write(f'{c}\t{t}\t{o}\n')
            output_f.write('<eos>\t<eos>\n')

        print(f'{cur_split}_sentences: {len(data)}')

        # additionally generate full test files if needed
        if cur_split == "test" and args.add_test_full:
            data = read_google_data(data_file=fp, lang=args.lang, split=cur_split, add_test_full=True)
            # write out
            output_file = join(args.output_dir, 'test_full', f'{fn}.tsv')
            output_f = open(output_file, 'w', encoding='utf-8')
            for inst in data:
                cur_classes, cur_tokens, cur_outputs = inst
                for c, t, o in zip(cur_classes, cur_tokens, cur_outputs):
                    output_f.write(f'{c}\t{t}\t{o}\n')
                output_f.write('<eos>\t<eos>\n')

            print(f'{cur_split}_sentences: {len(data)}')
