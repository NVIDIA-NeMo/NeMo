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
This script is used to postprocess SpellMapper results and generate an updated nemo ASR manifest.
See "examples/nlp/spellchecking_asr_customization/run_infer.sh" for the whole inference pipeline.
"""

from argparse import ArgumentParser

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    update_manifest_with_spellmapper_corrections,
)

parser = ArgumentParser(description="Postprocess SpellMapper results and generate an updated nemo ASR manifest")

parser.add_argument("--input_manifest", required=True, type=str, help="Path to input nemo ASR manifest")
parser.add_argument(
    "--field_name", default="pred_text", type=str, help="Name of json field with original ASR hypothesis text"
)
parser.add_argument(
    "--short2full_name",
    required=True,
    type=str,
    help="Path to input file with correspondence between sentence fragments and full sentences",
)
parser.add_argument(
    "--spellmapper_results", required=True, type=str, help="Path to input file with SpellMapper inference results"
)
parser.add_argument("--output_manifest", required=True, type=str, help="Path to output nemo ASR manifest")
parser.add_argument("--min_prob", default=0.5, type=float, help="Threshold on replacement probability")
parser.add_argument(
    "--use_dp",
    action="store_true",
    help="Whether to use additional replacement filtering by using dynamic programming",
)
parser.add_argument(
    "--replace_hyphen_to_space",
    action="store_true",
    help="Whether to use space instead of hyphen in replaced fragments",
)
parser.add_argument(
    "--ngram_mappings", type=str, required=True, help="File with ngram mappings, only needed if use_dp=true"
)
parser.add_argument(
    "--min_dp_score_per_symbol",
    default=-1.5,
    type=float,
    help="Minimum dynamic programming sum score averaged by hypothesis length",
)

args = parser.parse_args()

update_manifest_with_spellmapper_corrections(
    input_manifest_name=args.input_manifest,
    short2full_name=args.short2full_name,
    output_manifest_name=args.output_manifest,
    spellmapper_results_name=args.spellmapper_results,
    min_prob=args.min_prob,
    replace_hyphen_to_space=args.replace_hyphen_to_space,
    field_name=args.field_name,
    use_dp=args.use_dp,
    ngram_mappings=args.ngram_mappings,
    min_dp_score_per_symbol=args.min_dp_score_per_symbol,
)

print("Resulting manifest saved to: ", args.output_manifest)
