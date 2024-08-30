#!/usr/bin/env
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
Convert nemo style (fused) lora checkpoint to canonical (unfused) lora checkpoint.
Currently supports TP=PP=1 only.

Example usage:
python scripts/checkpoint_converters/lora_converters/convert_nemo_to_canonical.py \
    --nemo_lora_path nemo_style_lora_model.nemo \
    --output_path ./canonical_style_lora_model.nemo 

Example usage to also convert into huggingface format (the script expects a adapter_config.json file which is standard in HF):
python scripts/checkpoint_converters/lora_converters/convert_nemo_to_canonical.py \
    --nemo_lora_path nemo_style_lora_model.nemo \
    --output_path ./canonical_style_lora_model.nemo \
    --hf_format --hf_config checkpoints/bin/adapter_config.json
"""

from argparse import ArgumentParser

from nemo.export.utils.lora_converter import convert_lora_nemo_to_canonical


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--nemo_lora_path",
        type=str,
        default=None,
        required=True,
        help="Path to NeMo style (fused) lora checkpoint in .nemo file format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the canonical (unfused) lora .nemo file.",
    )
    parser.add_argument("--hf_format", action='store_true', help="saves tensors in huggingface naming format.")
    parser.add_argument("--hf_config", type=str, help="the adapter config in HF-PEFT format.")
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    convert_lora_nemo_to_canonical(
        args.nemo_lora_path, args.output_path, args.hf_format, donor_hf_config=args.hf_config
    )
