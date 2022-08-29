# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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


import os
import time
from argparse import ArgumentParser

import pynini
from nemo_text_processing.text_normalization.en.graph_utils import generator_main

# This script exports compiled grammars inside nemo_text_processing into OpenFst finite state archive files
# tokenize_and_classify.far and verbalize.far for production purposes


def itn_grammars(**kwargs):
    d = {}
    d['classify'] = {
        'TOKENIZE_AND_CLASSIFY': ITNClassifyFst(
            cache_dir=kwargs["cache_dir"], overwrite_cache=kwargs["overwrite_cache"]
        ).fst
    }
    d['verbalize'] = {'ALL': ITNVerbalizeFst().fst, 'REDUP': pynini.accep("REDUP")}
    return d


def tn_grammars(**kwargs):
    d = {}
    d['classify'] = {
        'TOKENIZE_AND_CLASSIFY': TNClassifyFst(
            input_case=kwargs["input_case"],
            deterministic=True,
            cache_dir=kwargs["cache_dir"],
            overwrite_cache=kwargs["overwrite_cache"],
        ).fst
    }
    d['verbalize'] = {'ALL': TNVerbalizeFst(deterministic=True).fst, 'REDUP': pynini.accep("REDUP")}
    return d


def export_grammars(output_dir, grammars):
    """
    Exports tokenizer_and_classify and verbalize Fsts as OpenFst finite state archive (FAR) files.

    Args:
        output_dir: directory to export FAR files to. Subdirectories will be created for tagger and verbalizer respectively.
        grammars: grammars to be exported
    """

    for category, graphs in grammars.items():
        out_dir = os.path.join(output_dir, category)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            time.sleep(1)
        if category == "classify":
            category = "tokenize_and_classify"
        generator_main(f"{out_dir}/{category}.far", graphs)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", help="output directory for grammars", required=True, type=str)
    parser.add_argument(
        "--language", help="language", choices=["en", "de", "es", "pt", "ru", 'fr', 'vi', 'zh'], type=str, default='en'
    )
    parser.add_argument(
        "--grammars", help="grammars to be exported", choices=["tn_grammars", "itn_grammars"], type=str, required=True
    )
    parser.add_argument(
        "--input_case", help="input capitalization", choices=["lower_cased", "cased"], default="cased", type=str
    )
    parser.add_argument("--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true")
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.language in ['pt', 'ru', 'fr', 'vi'] and args.grammars == 'tn_grammars':
        raise ValueError('Only ITN grammars could be deployed in Sparrowhawk for the selected languages.')

    if args.language == 'en':
        from nemo_text_processing.inverse_text_normalization.en.taggers.tokenize_and_classify import (
            ClassifyFst as ITNClassifyFst,
        )
        from nemo_text_processing.inverse_text_normalization.en.verbalizers.verbalize import (
            VerbalizeFst as ITNVerbalizeFst,
        )
        from nemo_text_processing.text_normalization.en.taggers.tokenize_and_classify import (
            ClassifyFst as TNClassifyFst,
        )
        from nemo_text_processing.text_normalization.en.verbalizers.verbalize import VerbalizeFst as TNVerbalizeFst
    elif args.language == 'de':
        from nemo_text_processing.inverse_text_normalization.de.taggers.tokenize_and_classify import (
            ClassifyFst as ITNClassifyFst,
        )
        from nemo_text_processing.inverse_text_normalization.de.verbalizers.verbalize import (
            VerbalizeFst as ITNVerbalizeFst,
        )
        from nemo_text_processing.text_normalization.de.taggers.tokenize_and_classify import (
            ClassifyFst as TNClassifyFst,
        )
        from nemo_text_processing.text_normalization.de.verbalizers.verbalize import VerbalizeFst as TNVerbalizeFst
    elif args.language == 'ru':
        from nemo_text_processing.inverse_text_normalization.ru.taggers.tokenize_and_classify import (
            ClassifyFst as ITNClassifyFst,
        )
        from nemo_text_processing.inverse_text_normalization.ru.verbalizers.verbalize import (
            VerbalizeFst as ITNVerbalizeFst,
        )
    elif args.language == 'es':
        from nemo_text_processing.inverse_text_normalization.es.taggers.tokenize_and_classify import (
            ClassifyFst as ITNClassifyFst,
        )
        from nemo_text_processing.inverse_text_normalization.es.verbalizers.verbalize import (
            VerbalizeFst as ITNVerbalizeFst,
        )
        from nemo_text_processing.text_normalization.es.taggers.tokenize_and_classify import (
            ClassifyFst as TNClassifyFst,
        )
        from nemo_text_processing.text_normalization.es.verbalizers.verbalize import VerbalizeFst as TNVerbalizeFst
    elif args.language == 'pt':
        from nemo_text_processing.inverse_text_normalization.pt.taggers.tokenize_and_classify import (
            ClassifyFst as ITNClassifyFst,
        )
        from nemo_text_processing.inverse_text_normalization.pt.verbalizers.verbalize import (
            VerbalizeFst as ITNVerbalizeFst,
        )
    elif args.language == 'fr':
        from nemo_text_processing.inverse_text_normalization.fr.taggers.tokenize_and_classify import (
            ClassifyFst as ITNClassifyFst,
        )
        from nemo_text_processing.inverse_text_normalization.fr.verbalizers.verbalize import (
            VerbalizeFst as ITNVerbalizeFst,
        )
    elif args.language == 'vi':
        from nemo_text_processing.inverse_text_normalization.vi.taggers.tokenize_and_classify import (
            ClassifyFst as ITNClassifyFst,
        )
        from nemo_text_processing.inverse_text_normalization.vi.verbalizers.verbalize import (
            VerbalizeFst as ITNVerbalizeFst,
        )
    elif args.language == 'zh':
        from nemo_text_processing.text_normalization.zh.taggers.tokenize_and_classify import (
            ClassifyFst as TNClassifyFst,
        )
        from nemo_text_processing.text_normalization.zh.verbalizers.verbalize import VerbalizeFst as TNVerbalizeFst

    output_dir = os.path.join(args.output_dir, args.language)
    export_grammars(
        output_dir=output_dir,
        grammars=locals()[args.grammars](
            input_case=args.input_case, cache_dir=args.cache_dir, overwrite_cache=args.overwrite_cache
        ),
    )
