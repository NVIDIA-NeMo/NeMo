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
"""
In order to build a regexp tokenizer model use the following command.
The script will create:

.vocab file - with learned vocabulary
.model file - with provided regex
To build vocabulary from text files:

python -- scripts/nlp_language_modeling/build_regex_tokenizer.py \
  --regex '\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]' \
  --input_type text \
  --output_file regex_tokenizer -- \
  data_file1.txt data_file2.txt

To build vocabulary from CSV files ("smiles" column):

python -- scripts/nlp_language_modeling/build_regex_tokenizer.py \
  --regex '\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]' \
  --input_type csv \
  --input_csv_col smiles \
  --output_file regex_tokenizer -- \
  data_file1.csv data_file2.csv
"""
import argparse

from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer
from nemo.utils import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builds vocabulary from regex tokenizer. Outputs .model (regular expression) and .vocab (learned vocabualry)",
    )
    parser.add_argument(
        'input_files', type=str, nargs='+', help='Input text/csv file',
    )
    parser.add_argument(
        '--regex', type=str, required=True, help='Regular expression to split text',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output base file name. Two files will be created: .vocab (learned vocabulary), .model (the regex)',
    )
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        choices=["text", "csv"],
        default="text",
        help='Type of input file: text, csv',
    )
    parser.add_argument(
        '--input_csv_col', type=str, required=False, default="smiles", help='Column of data in CSV file',
    )
    args = parser.parse_args()

    tokenizer = RegExTokenizer(regex=args.regex)

    # build vocabulary from all files
    for input_file in args.input_files:
        if args.input_type == "csv":
            tokenizer.build_vocab_from_csv(data_csv_file=input_file, col=args.input_csv_col)
        elif args.input_type == "text":
            tokenizer.build_vocab_from_text(data_text_file=input_file)
        else:
            raise ValueError(f"Unknown input_type = {args.input_type}")

    # save model
    if not args.output_file.endswith(".model"):
        args.output_file += ".model"
        logging.info("Adding .model to output file")

    tokenizer.save_tokenizer(args.output_file)
