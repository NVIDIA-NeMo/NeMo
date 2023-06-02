# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script contains an example on how to prepare input for SpellMapper inference from a nemo ASR manifest.
It splits sentences to shorter fragments, runs candidate retrieval and generates input in the required format.
It produces two output files:
    1. File with correspondence between sentence fragments and full sentences.
    2. File that will serve as input for SpellMapper inference.

See "examples/nlp/spellchecking_asr_customization/run_infer.sh" for the whole inference pipeline.
"""

from argparse import ArgumentParser

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    extract_and_split_text_from_manifest,
    get_candidates,
    load_index,
)

parser = ArgumentParser(description="Prepare input for SpellMapper inference from a nemo ASR manifest")
parser.add_argument("--manifest", required=True, type=str, help="Path to input manifest file")
parser.add_argument(
    "--custom_vocab_index", required=True, type=str, help="Path to input file with custom vocabulary index"
)
parser.add_argument(
    "--big_sample",
    required=True,
    type=str,
    help="Path to input file with big sample of phrases to sample dummy candidates if there less than 10 are found by retrieval",
)
parser.add_argument(
    "--short2full_name",
    required=True,
    type=str,
    help="Path to output file with correspondence between sentence fragments and full sentences",
)
parser.add_argument(
    "--output_name",
    required=True,
    type=str,
    help="Path to output file that will serve as input for SpellMapper inference",
)
parser.add_argument("--field_name", default="pred_text", type=str, help="Name of json field with ASR hypothesis text")
parser.add_argument("--len_in_words", default=16, type=int, help="Maximum fragment length in words")
parser.add_argument(
    "--step_in_words",
    default=8,
    type=int,
    help="Step in words for moving to next fragment. If less than len_in_words, fragments will intersect",
)

args = parser.parse_args()

# Split ASR hypotheses to shorter fragments, because SpellMapper can't handle arbitrarily long sequences.
# The correspondence between short and original fragments is saved to a file and will be used at post-processing.
extract_and_split_text_from_manifest(
    input_name=args.manifest,
    output_name=args.short2full_name,
    field_name=args.field_name,
    len_in_words=args.len_in_words,
    step_in_words=args.step_in_words,
)

# Load index of custom vocabulary from file
phrases, ngram2phrases = load_index(args.custom_vocab_index)

# Load big sample of phrases to sample dummy candidates if there less than 10 are found by retrieval
big_sample_of_phrases = set()
with open(args.big_sample, "r", encoding="utf-8") as f:
    for line in f:
        phrase, freq = line.strip().split("\t")
        if int(freq) > 50:  # do not want to use frequent phrases as dummy candidates
            continue
        if len(phrase) < 6 or len(phrase) > 15:  # do not want to use too short or too long phrases as dummy candidates
            continue
        big_sample_of_phrases.add(phrase)

big_sample_of_phrases = list(big_sample_of_phrases)

# Generate input for SpellMapper inference
out = open(args.output_name, "w", encoding="utf-8")
with open(args.short2full_name, "r", encoding="utf-8") as f:
    for line in f:
        short_sent, _ = line.strip().split("\t")
        sent = "_".join(short_sent.split())
        letters = list(sent)
        candidates = get_candidates(ngram2phrases, phrases, letters, big_sample_of_phrases)
        if len(candidates) == 0:
            continue
        if len(candidates) != 10:
            raise ValueError("expect 10 candidates, got: ", len(candidates))

        # We add two columns with targets and span_info.
        # They have same format as during training, but start and end positions are APPROXIMATE, they will be adjusted when constructing BertExample.
        targets = []
        span_info = []
        for idx, c in enumerate(candidates):
            if c[1] == -1:
                continue
            targets.append(str(idx + 1))  # targets are 1-based
            start = c[1]
            # ensure that end is not outside sentence length (it can happen because c[2] is candidate length used as approximation)
            end = min(c[1] + c[2], len(letters))
            span_info.append("CUSTOM " + str(start) + " " + str(end))
        out.write(
            " ".join(letters)
            + "\t"
            + ";".join([x[0] for x in candidates])
            + "\t"
            + " ".join(targets)
            + "\t"
            + ";".join(span_info)
            + "\n"
        )
out.close()
