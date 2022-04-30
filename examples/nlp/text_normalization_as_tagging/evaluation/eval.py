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
This script can be used to compare the inference output of Thutmose tagger with multi_reference file

USAGE Example:
  python eval.py \
     --inference_file= \
     --reference_file= \
     --print_other_errors

The inference file is a tsv file in which the first column contains the predicted sentence text.
The reference file is a tsv file in which
    the first column contains the input sentence text,
    the second column contains the reference sentence text (taken from Google TN dataset)
    the third column (optional) contains additional acceptable references for semiotic spans in this sentence.
    E.g.
        mizoguchi akiko september twenty ten    mizoguchi akiko september 2010     DATE 2 5 | sept 2010 | sep. 2010 ...
 (to get a reference file see the last steps in examples/nlp/text_normalization_as_tagging/prepare_dataset_en.sh,
   starting from ".../examples/nlp/text_normalization_as_tagging/evaluation/get_multi_reference_vocab.py"
 )

The script outputs the following metrics:
    Word Error Rate (WER) - an automatic metric commonly used in ASR.
       It does not take into account additional references.
    Sentence accuracy:
       The sentence is regarded as correct if its characters (without spaces) match to the reference,
       It takes into account additional references.

       If at least one digital character doesn't match this sentence is regarded as containing Digit Error.
       If all digital character match, but at least one non-digital character doesn't match
          this sentence is regarded as containing Other Error.
"""


import re
from argparse import ArgumentParser

from nemo.collections.asr.metrics.wer import word_error_rate

parser = ArgumentParser(description="Compare inference output with multi-reference")
parser.add_argument("--inference_file", type=str, required=True, help="Path to inference file")
parser.add_argument(
    "--print_other_errors",
    action='store_true',
    help="Whether to print other errors, if false only digit errors will be printed",
)
parser.add_argument("--reference_file", type=str, required=True, help="Path to reference file")
args = parser.parse_args()

# Main code
if __name__ == "__main__":
    inputs = []
    references = []  # list(size=len(inputs)) of lists
    skip_ids = set()  # sentences ids to be skipped during evaluation
    with open(args.reference_file, "r", encoding="utf-8") as f:
        for line in f:
            multi_references = []
            parts = line.strip().split("\t")
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError("Bad format: " + line)
            words = parts[0].split()
            inputs.append(words)
            if len(parts) == 3:  # there are non-trivial semiotic spans
                multi_references.append("")
                input_position = 0
                if "TELEPHONE" in parts[2] or "ELECTRONIC" in parts[2]:
                    skip_ids.add(len(references))
                spans = parts[2].split(";")
                multi_references_updated = []
                for span in spans:
                    span_parts = span.split(" | ")
                    try:
                        sem, begin, end = span_parts[0].split(" ")
                    except Exception:
                        print("error: ", line)
                        continue
                    begin = int(begin)
                    end = int(end)
                    for ref in multi_references:
                        if len(span_parts) > 20 or len(multi_references_updated) > 20000:
                            print("warning: too many references: ", inputs[-1])
                            break
                        for tr_variant in span_parts[1:]:
                            multi_references_updated.append(
                                ref
                                + " "
                                + " ".join(inputs[-1][input_position:begin])  # copy needed words from input
                                + " "
                                + tr_variant
                            )
                    multi_references = multi_references_updated[:]  # copy
                    multi_references_updated = []
                    input_position = end
                for i in range(len(multi_references)):  # copy needed words from the input end
                    multi_references[i] += " " + " ".join(inputs[-1][input_position : len(inputs[-1])])
            # the last reference added is the actual one
            multi_references.append(parts[1])
            references.append(multi_references)

    predictions = []
    predicted_tags = []
    predicted_semiotic = []
    # load predictions
    with open(args.inference_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 1:
                predictions.append(parts[0].casefold())
                predicted_tags.append([])
                continue
            if len(parts) != 5:
                raise ValueError("Bad format: " + line)
            prediction, inp_str, tag_str, tags_with_swap_str, semiotic = parts
            predictions.append(prediction.casefold())
            tags = tag_str.split(" ")
            predicted_tags.append(tags)
            predicted_semiotic.append(semiotic)

    sentences_with_errors_on_digits = 0
    correct_sentences_disregarding_space = 0

    if len(inputs) != len(predictions) or len(inputs) != len(references):
        raise ValueError(
            "Length mismatch: len(inputs)="
            + str(len(inputs))
            + "; len(predictions)="
            + str(len(predictions))
            + "; len(references)="
            + str(len(references))
        )

    refs_for_wer = []
    preds_for_wer = []
    for i in range(len(inputs)):
        ok_digit = False
        ok_all = False
        if i in skip_ids:
            continue
        refs_for_wer.append(references[i][-1])
        preds_for_wer.append(predictions[i])
        for ref in references[i]:
            ref_digit_fragments = re.findall(r"\d+", ref)
            pred_digit_fragments = re.findall(r"\d+", predictions[i])
            if "".join(pred_digit_fragments) == "".join(ref_digit_fragments):
                ok_digit = True
            if predictions[i].replace("_", "").replace(" ", "") == ref.replace("_", "").replace(" ", ""):
                ok_all = True
        if not ok_digit:
            print("digit error:")
            print("\tinput=", " ".join(inputs[i]))
            print("\ttags=", " ".join(predicted_tags[i]))
            print("\tpred=", predictions[i])
            print("\tsemiotic=", predicted_semiotic[i])
            print("\tref=", references[i][-1])  # last reference is actual reference
            sentences_with_errors_on_digits += 1
        elif ok_all:
            correct_sentences_disregarding_space += 1
        elif args.print_other_errors:
            print("other error:")
            print("\tinput=", " ".join(inputs[i]))
            print("\ttags=", " ".join(predicted_tags[i]))
            print("\tpred=", predictions[i])
            print("\tsemiotic=", predicted_semiotic[i])
            print("\tref=", references[i][-1])  # last reference is actual reference

    wer = word_error_rate(refs_for_wer, preds_for_wer)
    print("WER: ", wer)
    print(
        "Sentence accuracy: ",
        correct_sentences_disregarding_space / (len(inputs) - len(skip_ids)),
        correct_sentences_disregarding_space,
    )
    print(
        "digit errors: ",
        sentences_with_errors_on_digits / (len(inputs) - len(skip_ids)),
        sentences_with_errors_on_digits,
    )
    print(
        "other errors: ",
        (len(inputs) - len(skip_ids) - correct_sentences_disregarding_space - sentences_with_errors_on_digits)
        / (len(inputs) - len(skip_ids)),
        len(inputs) - len(skip_ids) - correct_sentences_disregarding_space - sentences_with_errors_on_digits,
    )
