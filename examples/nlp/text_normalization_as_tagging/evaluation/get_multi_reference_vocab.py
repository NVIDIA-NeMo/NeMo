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
This script can be used to construct a vocabulary of multiple references
"""
from argparse import ArgumentParser
from collections import Counter
from os import listdir

from nemo.collections.nlp.data.text_normalization_as_tagging.utils import spoken_preprocessing

parser = ArgumentParser(description="Get reference vocabulary from corpus (it will be used in testing)")
parser.add_argument("--data_dir", type=str, required=True, help="Path to folder with data")
parser.add_argument("--out_filename", type=str, required=True, help="Path to output file")
args = parser.parse_args()

if __name__ == "__main__":

    vcb = {}
    filenames = []
    for fn in listdir(args.data_dir + "/train"):
        filenames.append(args.data_dir + "/train/" + fn)
    for fn in listdir(args.data_dir + "/dev"):
        filenames.append(args.data_dir + "/dev/" + fn)
    for fn in filenames:
        print("Processing ", fn)
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                if len(parts) != 3:
                    raise ValueError("Expect 3 parts, got " + str(len(parts)))
                semiotic_class, written, spoken = parts[0], parts[1].strip().casefold(), parts[2].strip().casefold()
                spoken = spoken_preprocessing(spoken)
                if spoken == "<self>":
                    continue
                if spoken == "" or written == "":
                    continue
                if len(spoken.split(" ")) >= 100:
                    continue
                k = (semiotic_class, spoken)
                if k not in vcb:
                    vcb[k] = Counter()
                vcb[k][written] += 1

    with open(args.out_filename, "w", encoding="utf-8") as out:
        for sem, spoken in vcb:
            for written in vcb[(sem, spoken)]:
                out.write(sem + "\t" + spoken + "\t" + written + "\t" + str(vcb[(sem, spoken)][written]) + "\n")
            out.write(sem + "\t" + spoken + "\t" + spoken + "\t1\n")
