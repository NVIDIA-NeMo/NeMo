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


"""
This script is used to filter sentences containing bad examples from Google TN Dataset.
"""

from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import exists, isfile, join
from typing import Set

parser = ArgumentParser(description="Filter Google TN Dataset by error vocabulary")
parser.add_argument(
    "--data_dir", required=True, type=str, help='Path to data directory with files like output-00000-of-00100.tsv'
)
parser.add_argument(
    "--out_dir", required=True, type=str, help='Output data directory, same files (with some sentences filtered)'
)
parser.add_argument("--errors_vocab_filename", required=True, type=str, help='File with error vocabulary')
parser.add_argument("--lang", required=True, type=str, help="Language")
args = parser.parse_args()


def filter_file(inp_filename: str, out_filename: str, error_vcb: Set) -> None:
    """Filter out whole sentences containing bad itn conversions. The output format is the same as input.

    Args:
        inp_filename: Name of input file in Google TN Dataset format.
        out_filename: Name of output file in Google TN Dataset format.
        error_vcb: Set of tuples with erroneous conversion, e.g. ("CARDINAL", "two", "132")
    """
    out = open(out_filename, "w", encoding="utf-8")
    sent_lines = []
    sent_is_ok = True
    with open(inp_filename, "r", encoding="utf-8") as f:
        for line in f:
            sent_lines.append(line.strip())
            if line.startswith("<eos>"):
                if sent_is_ok and len(sent_lines) > 1:  # there should be at least one line except <eos>
                    out.write("\n".join(sent_lines) + "\n")
                sent_lines = []
                sent_is_ok = True
            else:
                cls, written, spoken = line.strip().split("\t")
                k = (cls, spoken.casefold(), written.casefold())
                if k in error_vcb:
                    sent_is_ok = False
    out.close()


def main() -> None:
    if not exists(args.data_dir):
        raise ValueError(f"Data dir {args.data_dir} does not exist")

    # load errors vocabulary
    error_vcb = set()
    with open(args.errors_vocab_filename, "r", encoding="utf-8") as f:
        for line in f:
            cls, spoken, written = line.strip().split("\t")
            k = (cls, spoken, written)
            error_vcb.add(k)

    for subdir in listdir(args.data_dir):
        mkdir(join(args.out_dir, subdir))
        for filename in listdir(join(args.data_dir, subdir)):
            if not filename.startswith('output'):
                continue
            inp_filename = join(args.data_dir, subdir, filename)
            out_filename = join(args.out_dir, subdir, filename)
            if not isfile(inp_filename):
                continue
            filter_file(inp_filename, out_filename, error_vcb)


if __name__ == "__main__":
    main()
