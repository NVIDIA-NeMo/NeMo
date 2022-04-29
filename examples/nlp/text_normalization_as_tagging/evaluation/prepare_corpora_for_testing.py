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
This script can be used to prepare test corpus for the ThutmoseTaggerModel from Google Text Normalization dataset.
"""

import os
import re
from argparse import ArgumentParser
from collections import Counter
from typing import Dict, TextIO, Tuple

from nemo.collections.nlp.data.text_normalization_as_tagging.utils import spoken_preprocessing

parser = ArgumentParser(description="Text Normalization Data Preprocessing for English")
parser.add_argument(
    "--data_dir", required=True, type=str, help="Path to data directory with files like output-00000-of-00100.tsv"
)
parser.add_argument("--reference_vocab", required=True, type=str, help="Multi Reference vocabulary")
parser.add_argument("--output_file", required=True, type=str, help="Output file")
parser.add_argument(
    "--sampling_count", required=True, type=int, help="Number of examples per class, you want, use -1 for all examples"
)
args = parser.parse_args()


def process_file(
    inputname: str,
    out: TextIO,
    out_raw: TextIO,
    reference_vcb: Dict[Tuple[str, str], Dict[str, int]],
    sampling_vcb: Dict[str, int],
) -> None:
    words = []
    reference_words = []  # size may be different
    semiotic_info = []
    raw_lines = []
    sent_ok = True if args.sampling_count == -1 else False
    with open(inputname, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("<eos>"):
                if len(words) > 0 and sent_ok:
                    out.write(
                        " ".join(words) + "\t" + " ".join(reference_words) + "\t" + ";".join(semiotic_info) + "\n"
                    )
                    out_raw.write("\n".join(raw_lines) + "\n" + line)
                words = []
                reference_words = []
                semiotic_info = []
                raw_lines = []
                sent_ok = True if args.sampling_count == -1 else False
            else:
                raw_lines.append(line.strip())
                cls, written, spoken = line.strip().split("\t")
                spoken = spoken_preprocessing(spoken)
                written = written.casefold()
                references = set()
                if spoken == "sil":
                    continue
                if spoken == "<self>":
                    words.append(written)
                    reference_words.append(written)
                    # if reference is <self>, but the word has itn conversions in our dictionary, add them
                    for cls in ["CARDINAL", "ORDINAL", "DATE"]:  # date, ex sixties -> 60s
                        k = (cls, written)
                        if k in reference_vcb:
                            for tr_variant in reference_vcb[k]:
                                references.add(tr_variant)
                            semiotic_info.append(
                                cls
                                + " "
                                + str(len(words) - 1)
                                + " "
                                + str(len(words))
                                + " | "
                                + " | ".join(references)
                            )
                            break
                    continue

                spoken_words = spoken.split()
                words.extend(spoken_words)

                k = (cls, spoken)
                if k in reference_vcb:
                    for tr_variant in reference_vcb[k]:
                        references.add(tr_variant)
                references.add(spoken)
                references.add(written)
                for tr_variant in list(references):
                    # 6,51 km² => 6,51 km 2
                    (tr_variant2, n2) = re.subn(r"²", " 2", tr_variant)
                    (tr_variant3, n3) = re.subn(r"³", " 3", tr_variant)
                    if n2 > 0:
                        references.add(tr_variant2)
                    if n3 > 0:
                        references.add(tr_variant3)

                semiotic_info.append(
                    cls
                    + " "
                    + str(len(words) - len(spoken_words))
                    + " "
                    + str(len(words))
                    + " | "
                    + " | ".join(list(references))
                )
                reference_words.append(written.casefold())

                if cls not in sampling_vcb:
                    sampling_vcb[cls] = 0
                if sampling_vcb[cls] < args.sampling_count:
                    sent_ok = True
                    sampling_vcb[cls] += 1


def main() -> None:
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data dir {args.data_dir} does not exist")
    reference_vcb = {}
    with open(args.reference_vocab, "r", encoding="utf-8") as f:
        for line in f:
            sem, spoken, written, freq = line.strip().split("\t")
            k = (sem, spoken)
            if k not in reference_vcb:
                reference_vcb[k] = {}
            reference_vcb[k][written] = int(freq)
    sampling_vcb = Counter()
    out = open(args.output_file, "w", encoding="utf-8")
    out_raw = open(args.output_file + ".raw", "w", encoding="utf-8")
    input_paths = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)])
    for inputname in input_paths:
        process_file(inputname, out, out_raw, reference_vcb, sampling_vcb)
    out.close()
    out_raw.close()


if __name__ == "__main__":
    main()
