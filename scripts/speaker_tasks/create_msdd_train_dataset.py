# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import random

from nemo.collections.asr.parts.utils.manifest_utils import create_segment_manifest

random.seed(42)


"""
This scipt converts a scp file where each line contains
<absolute path of wav file>
to a manifest json file.
Args:
--scp: scp file name
--id: index of speaker label in filename present in scp file that is separated by '/'
--out: output manifest file name
--split: True / False if you would want to split the  manifest file for training purposes
        you may not need this for test set. output file names is <out>_<train/dev>.json
        Defaults to False
--create_chunks: bool if you would want to chunk each manifest line to chunks of 3 sec or less
        you may not need this for test set, Defaults to False
"""


def main(input_manifest_path, output_manifest_path, window, shift, step_count, deci):
    create_segment_manifest(input_manifest_path, output_manifest_path, window, shift, step_count, deci)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_path", help="input json file name", type=str, required=True)
    parser.add_argument(
        "--output_manifest_path", help="output manifest_file name", type=str, default=None, required=False
    )
    parser.add_argument("--window", help="Window length for segmentation", type=float, required=True)
    parser.add_argument("--shift", help="Shift length for segmentation", type=float, required=True)
    parser.add_argument("--deci", help="Rounding decimals", type=int, default=3, required=False)
    parser.add_argument(
        "--step_count", help="Number of the unit segments you want to create per utterance", required=True,
    )
    args = parser.parse_args()

    main(args.input_manifest_path, args.output_manifest_path, args.window, args.shift, args.step_count, args.deci)
