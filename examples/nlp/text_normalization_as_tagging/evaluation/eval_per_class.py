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
This script can be used to compare the inference output of Thutmose tagger with multi_reference file.
The additional report is stored to a separate file for each semiotic class.

USAGE Example:
  python eval_per_class.py \
     --inference_file= \
     --reference_file= \
     --output_file=

The inference file is a tsv file in which the first column contains the predicted sentence text.
The reference file is a tsv file in which
    the first column contains the input sentence text,
    the second column contains the reference sentence text (taken from Google TN dataset)
    the third column (optional) contains additional acceptable references for semiotic spans in this sentence.
    E.g.
        mizoguchi akiko september twenty ten    mizoguchi akiko september 2010     DATE 2 5 | sept 2010 | sep. 2010 ...

The script generates:
    a file with report on accuracy per semiotiotic class (output_file).
    files (<output_file>.<semiotic_class>) with sentences, containing errors in this semiotic span.

"""
import glob
import os
from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser(description="Compare inference output with multi-reference, print report per class")
parser.add_argument("--inference_file", type=str, required=True, help="Path to inference file 1")
parser.add_argument("--reference_file", type=str, required=True, help="Path to reference file")
parser.add_argument("--output_file", type=str, required=True, help="Path to output file")
args = parser.parse_args()

# Main code
if __name__ == '__main__':

    # delete all class-specific reports, as they are created in the append mode
    for f in glob.glob(args.output_file + ".*"):
        os.remove(f)

    total_count = Counter()
    correct_count = Counter()

    f_ref = open(args.reference_file, "r", encoding="utf-8")
    f_infer = open(args.inference_file, "r", encoding="utf-8")
    f_out = open(args.output_file, "w", encoding="utf-8")
    lines_ref = f_ref.readlines()
    lines_infer = f_infer.readlines()
    f_ref.close()
    f_infer.close()
    if len(lines_ref) != len(lines_infer):
        raise ValueError(
            "Number of lines doesn't match: len(lines_ref)="
            + str(len(lines_ref))
            + "; len(lines_infer)="
            + str(len(lines_infer))
        )
    for i in range(len(lines_infer)):
        _, inp_str, _, tag_with_swap_str, semiotic = lines_infer[i].strip().split("\t")
        input_words = inp_str.split(" ")
        predicted_tags = tag_with_swap_str.split(" ")
        predicted_words = predicted_tags[:]
        for k in range(len(predicted_tags)):
            t = predicted_tags[k]
            if t == "<SELF>":
                predicted_words[k] = input_words[k]
            elif t == "<DELETE>":
                predicted_words[k] = ""
            else:
                predicted_words[k] = predicted_words[k].replace(">", "").replace("<", "")

        parts = lines_ref[i].strip().split("\t")
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError("Bad format: " + lines_ref[i])
        if len(parts) == 3:  # there are non-trivial semiotic spans
            spans = parts[2].split(";")
            for span in spans:
                span_parts = span.split(" | ")
                try:
                    sem, begin, end = span_parts[0].split(" ")
                except Exception:
                    print("error: ", lines_ref[i])
                    continue
                begin = int(begin)
                end = int(end)

                ok = False
                predicted_span = " ".join(predicted_words[begin:end]).replace("_", " ").replace(" ", "").casefold()
                input_span = " ".join(input_words[begin:end])
                total_count[sem] += 1
                for tr_variant in span_parts[1:]:
                    ref_span = tr_variant.replace("_", " ").replace(" ", "").casefold()
                    if ref_span == predicted_span:
                        ok = True
                        correct_count[sem] += 1
                        break
                if not ok:
                    out_sem = open(args.output_file + "." + sem, "a", encoding="utf-8")
                    out_sem.write(
                        "error: pred="
                        + " ".join(predicted_words[begin:end])
                        + "; inp="
                        + input_span
                        + "; ref="
                        + span
                        + "\n"
                    )
                    out_sem.write("\tinput=" + " ".join(input_words) + "\n")
                    out_sem.write("\ttags=" + " ".join(predicted_tags) + "\n")
                    out_sem.write("\tpred=" + " ".join(predicted_words) + "\n")
                    out_sem.write("\tsemiotic=" + semiotic + "\n")
                    out_sem.write("\tref=" + parts[1] + "\n")
                    out_sem.close()

    f_out.write("class\ttotal\tcorrect\terrors\taccuracy\n")
    for sem in total_count:
        f_out.write(
            sem
            + "\t"
            + str(total_count[sem])
            + "\t"
            + str(correct_count[sem])
            + "\t"
            + str(total_count[sem] - correct_count[sem])
            + "\t"
            + str(correct_count[sem] / total_count[sem])
            + "\n"
        )
    f_out.close()
