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
This script can be used to get label vocab from train and dev labeled files.
"""

import sys
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser(description="Get label vocab")
parser.add_argument("--train_filename", required=True, type=str, help='File with training data')
parser.add_argument("--dev_filename", required=True, type=str, help='File with development data')
parser.add_argument("--out_filename", required=True, type=str, help='Output file')
args = parser.parse_args()

vocab = Counter()

n = 0
for fn in [args.train_filename, args.dev_filename]:
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                print("Warning: bad format in line: " + str(n) + ": " + line, file=sys.stderr)
                continue
            tags = parts[1].split(" ")
            for t in tags:
                if t == "<SELF>":
                    vocab["KEEP"] += 1
                elif t == "<DELETE>":
                    vocab["DELETE"] += 1
                else:
                    vocab["DELETE|" + t] += 1
            n += 1

print("len(vocab)=", len(vocab))
with open(args.out_filename, "w", encoding="utf-8") as out:
    out.write("KEEP\n")
    out.write("DELETE\n")
    for t, freq in vocab.most_common(10000000):
        if t == "KEEP":
            continue
        if t == "DELETE":
            continue
        out.write(t + "\n")
