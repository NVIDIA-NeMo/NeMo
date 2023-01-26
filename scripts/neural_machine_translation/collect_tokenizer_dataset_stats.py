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

import argparse
import json
import multiprocessing as mp
import os

import numpy as np
from matplotlib import pyplot as plt

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

# =============================================================================#
# Auxiliary methods
# =============================================================================#

worker_data = {
    "tokenizer": None,
}


def init_tokenizer(library, tokenizer_model):
    tokenizer = get_nmt_tokenizer(library=library, tokenizer_model=tokenizer_model)
    worker_data["tokenizer"] = tokenizer


def read_batch(fh, batch_size):
    """
    Reads a batch (or smaller) chunk of lines.
    """
    lines = []
    for i in range(batch_size):
        l = fh.readline()
        if not l:
            break
        else:
            lines.append(l.strip())

    return lines


def tokenize_line(line, tokenizer):
    """
    Returns a tokenized line
    """
    line = line.rstrip("\n")
    tokens = tokenizer.text_to_ids(line)

    return tokens


def line_len(line, tokenizer=None):
    """
    Returns a tokenized length of a text line
    """
    if tokenizer is None:
        tokenizer = worker_data["tokenizer"]

    tokens = tokenize_line(line, tokenizer)

    return len(tokens)


# =============================================================================#
# Main script
# =============================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collects statistics over tokenized dataset')
    parser.add_argument('input_files', metavar='N', type=str, nargs='+', help='Input files to parse')
    parser.add_argument(
        '--tokenizer_library', type=str, required=True, help='Path to pre-trained nemo-supported tokenizer model'
    )
    parser.add_argument(
        '--tokenizer_model', type=str, required=True, help='Path to pre-trained nemo-supported tokenizer model'
    )
    parser.add_argument(
        '--num_workers', type=int, default=mp.cpu_count(), help='Number of workers (default to number of CPUs)'
    )
    parser.add_argument('--max_lines', type=int, default=-1, help='Max number of lines to parse')
    parser.add_argument('--batch_size', type=int, default=10000000, help='Batch size to parse in parallel')
    parser.add_argument('--out_dir', type=str, default="", help='Path to store data and plots')

    args = parser.parse_args()

    tokenizer = get_nmt_tokenizer(library=args.tokenizer_library, tokenizer_model=args.tokenizer_model,)

    all_len = []

    for fn in args.input_files:
        print(f"Parsing fn = {fn}")
        # read file
        fh = open(fn)

        # read all batches
        while True:
            lines = read_batch(fh, args.batch_size)

            # move to next file when no lines are read
            if not lines:
                break

            # tokenize lines
            with mp.Pool(
                args.num_workers, initializer=init_tokenizer, initargs=(args.tokenizer_library, args.tokenizer_model)
            ) as p:
                all_len.extend(p.map(line_len, lines))

            print(f"{fn}: Parsed {len(all_len)} lines")

            # early stop, if required
            if (args.max_lines > 0) and (len(all_len) >= args.max_lines):
                lines = lines[: args.max_lines]
                break

        # early stop, if required
        if (args.max_lines > 0) and (len(all_len) >= args.max_lines):
            lines = lines[: args.max_lines]
            break

    # compute stats

    # save all results
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    stats = {
        "samples": int(len(all_len)),
        "mean": float(np.mean(all_len)),
        "stdev": float(np.std(all_len)),
        "min": float(np.min(all_len)),
        "max": float(np.max(all_len)),
        "median": float(np.median(all_len)),
    }

    print(f"stats = \n{stats}")

    # save all results
    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)

        fh = open(os.path.join(args.out_dir, "lengths.txt"), "w")
        fh.writelines(["{l}\n".format(l=l) for l in all_len])

        json.dump(stats, open(os.path.join(args.out_dir, "stats.json"), "w"))

        fig = plt.hist(all_len)
        plt.savefig(os.path.join(args.out_dir, "lengths_hist.pdf"))
