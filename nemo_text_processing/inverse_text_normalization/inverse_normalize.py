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

from argparse import ArgumentParser
from time import perf_counter
from typing import List

from nemo_text_processing.text_normalization.data_loader_utils import (
    check_installation,
    get_installation_msg,
    load_file,
    write_file,
)
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.token_parser import TokenParser


class InverseNormalizer(Normalizer):
    """
    Inverse normalizer that converts text from spoken to written form. Useful for ASR postprocessing. 
    Input is expected to have no punctuation outside of approstrophe (') and dash (-) and be lower cased.

    Args:
        lang: language specifying the ITN
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, lang: str = 'en', cache_dir: str = None, overwrite_cache: bool = False):

        if not check_installation():
            raise ImportError(get_installation_msg())
        if lang == 'en':
            from nemo_text_processing.inverse_text_normalization.en.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.en.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == 'es':
            from nemo_text_processing.inverse_text_normalization.es.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.es.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == 'ru':
            from nemo_text_processing.inverse_text_normalization.ru.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.ru.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == 'de':
            from nemo_text_processing.inverse_text_normalization.de.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.de.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'fr':
            from nemo_text_processing.inverse_text_normalization.fr.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.fr.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == 'vi':
            from nemo_text_processing.inverse_text_normalization.vi.taggers.tokenize_and_classify import ClassifyFst
            from nemo_text_processing.inverse_text_normalization.vi.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        self.tagger = ClassifyFst(cache_dir=cache_dir, overwrite_cache=overwrite_cache)
        self.verbalizer = VerbalizeFinalFst()
        self.parser = TokenParser()
        self.lang = lang

    def inverse_normalize_list(self, texts: List[str], verbose=False) -> List[str]:
        """
        NeMo inverse text normalizer 

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information

        Returns converted list of input strings
        """
        return self.normalize_list(texts=texts, verbose=verbose)

    def inverse_normalize(self, text: str, verbose: bool) -> str:
        """
        Main function. Inverse normalizes tokens from spoken to written form
            e.g. twelve kilograms -> 12 kg

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information

        Returns: written form
        """
        return self.normalize(text=text, verbose=verbose)


def parse_args():
    parser = ArgumentParser()
    input = parser.add_mutually_exclusive_group()
    input.add_argument("--text", dest="input_string", help="input string", type=str)
    input.add_argument("--input_file", dest="input_file", help="input file path", type=str)
    parser.add_argument('--output_file', dest="output_file", help="output file path", type=str)
    parser.add_argument(
        "--language", help="language", choices=['en', 'de', 'es', 'ru', 'fr', 'vi'], default="en", type=str
    )
    parser.add_argument("--verbose", help="print info for debugging", action='store_true')
    parser.add_argument("--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true")
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = perf_counter()
    inverse_normalizer = InverseNormalizer(
        lang=args.language, cache_dir=args.cache_dir, overwrite_cache=args.overwrite_cache
    )
    print(f'Time to generate graph: {round(perf_counter() - start_time, 2)} sec')

    if args.input_string:
        print(inverse_normalizer.inverse_normalize(args.input_string, verbose=args.verbose))
    elif args.input_file:
        print("Loading data: " + args.input_file)
        data = load_file(args.input_file)

        print("- Data: " + str(len(data)) + " sentences")
        prediction = inverse_normalizer.inverse_normalize_list(data, verbose=args.verbose)
        if args.output_file:
            write_file(args.output_file, prediction)
            print(f"- Denormalized. Writing out to {args.output_file}")
        else:
            print(prediction)
