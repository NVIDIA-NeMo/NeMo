#!/usr/bin/env python3
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Processing data for megatron pretraining."""

import argparse
import glob

from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import build_index_files


def main():
    parser = argparse.ArgumentParser(description="Builds index files for a list of text files",)
    parser.add_argument(
        'dataset_paths', type=str, nargs='+', help='Input text files (support glob)',
    )
    parser.add_argument(
        '--newline_int', type=int, default=10, help='Int value to split text (default: newline "\\n"',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of workers to parse files in parallel (default: max(cpu num // 2, 1)',
    )
    args = parser.parse_args()

    # expand all dataset_paths
    dataset_paths = []
    for ds in args.dataset_paths:
        dataset_paths.extend(glob.glob(ds))

    # build index files in parallel
    build_index_files(
        dataset_paths=dataset_paths, newline_int=args.newline_int, workers=args.workers,
    )


if __name__ == '__main__':
    main()
