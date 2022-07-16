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

import logging
from argparse import ArgumentParser

from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel

"""
python preprocess_tokenization_normalization.py --input-src train.en \
    --input-tgt train.zh \
    --output-src train.tok.norm.en \
    --output-tgt train.tok.norm.zh \
    --source-lang en \
    --target-lang zh 
"""

logging.basicConfig(level=logging.INFO)


def tokenize_normalize(file, wfile, processor):
    rptr = open(file)
    wptr = open(wfile, 'w')
    logging.info(f"Processing {file}")
    for line in rptr:
        txt = line.strip()
        if processor is not None:
            txt = processor.normalize(txt)
            txt = processor.tokenize(txt)
        wptr.write(txt + "\n")
    logging.info(f"Output written to {file}")
    rptr.close()
    wptr.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-src", type=str, required=True, help="Path to input file in src language")
    parser.add_argument("--input-tgt", type=str, required=True, help="Path to input file in tgt language")
    parser.add_argument("--output-src", type=str, required=True, help="Path to write the src language output file")
    parser.add_argument("--output-tgt", type=str, required=True, help="Path to write the tgt language output file")
    parser.add_argument("--source-lang", type=str, required=True, help="Language for the source file")
    parser.add_argument("--target-lang", type=str, required=True, help="Language for the target file")

    args = parser.parse_args()

    src_processor, tgt_processor = MTEncDecModel.setup_pre_and_post_processing_utils(
        args.source_lang, args.target_lang, "bpe-placeholder", "bpe-placeholder"
    )
    tokenize_normalize(args.input_src, args.output_src, src_processor)
    tokenize_normalize(args.input_tgt, args.output_tgt, tgt_processor)


if __name__ == '__main__':
    main()
