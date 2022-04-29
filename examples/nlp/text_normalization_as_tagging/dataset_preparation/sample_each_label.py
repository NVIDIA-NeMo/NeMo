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
This script can be used to sample each label from the labeled files.
"""

import sys
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser(description="Sample labels")
parser.add_argument("--filename", required=True, type=str, help='File with input data')
parser.add_argument("--max_count", required=True, type=int, help='Count')
args = parser.parse_args()


vocab = Counter()

out_sample = open(args.filename + ".sample_" + str(args.max_count), "w", encoding="utf-8")
out_rest = open(args.filename + ".rest_" + str(args.max_count), "w", encoding="utf-8")

n = 0
with open(args.filename, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            print("Warning: bad format in line: " + str(n) + ": " + line, file=sys.stderr)
            continue

        tags = parts[1].split(" ")
        ok = False
        for t in tags:
            if t not in vocab:
                vocab[t] = 0
            if vocab[t] < args.max_count:
                ok = True
                vocab[t] += 1
        if ok:
            out_sample.write(line)
        else:
            out_rest.write(line)
        n += 1

out_sample.close()
out_rest.close()
