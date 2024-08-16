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

import argparse
import ast
import math
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from lhotse.cut import Cut
from omegaconf import OmegaConf

from nemo.collections.asr.data.audio_to_text_lhotse import TokenizerWrapper
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
from nemo.collections.common.data.lhotse.dataloader import (
    DurationFilter,
    FixedBucketBatchSizeConstraint2D,
    LhotseDataLoadingConfig,
    TokenPerSecondFilter,
    tokenize,
)
from nemo.collections.common.prompts.formatter import PromptFormatter
from nemo.collections.common.tokenizers import AggregateTokenizer, SentencePieceTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate duration bins for Lhotse dynamic bucketing using a sample of the input dataset. "
        "The dataset is read either from one or more manifest files and supports data weighting. "
        "Unlike estimate_duration_bins.py, this script prepares the setup for 2D bucketing. "
        "This means that each main bucket for audio duration is sub-divided into sub-buckets "
        "for the number of output tokens (supporting BPE and Aggregated tokenizers). "
        "2D bucketing is especially useful for encoder-decoder models where input audio duration is often "
        "not sufficient to stratify the sampling with an optimal GPU utilization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help='Data input. Options: '
        '1) "path.json" - any single NeMo manifest; '
        '2) "[[path1.json],[path2.json],...]" - any collection of NeMo manifests; '
        '3) "[[path1.json,weight1],[path2.json,weight2],...]" - any collection of weighted NeMo manifests; '
        '4) "input_cfg.yaml" - a new option supporting input configs, same as in model training \'input_cfg\' arg; '
        '5) "path/to/shar_data" - a path to Lhotse Shar data directory; '
        '6) "key=val" - in case none of the previous variants cover your case: "key" is the key you\'d use in NeMo training config with its corresponding value ',
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        nargs="+",
        required=True,
        help="Path to one or more SPE tokenizers. More than one means we'll use AggregateTokenizer and --langs argument must also be used. When provided, we'll estimate a 2D distribution for input and output sequence lengths.",
    )
    parser.add_argument(
        "-a", "--langs", nargs="+", help="Language names for each of AggregateTokenizer sub-tokenizers."
    )
    parser.add_argument(
        "-b",
        "--buckets",
        type=int,
        default=30,
        help="The desired number of buckets (dim0 => covers input sequence length / audio duration).",
    )
    parser.add_argument(
        "-s",
        "--sub-buckets",
        type=int,
        default=2,
        help="The desired number of sub-buckets (dim1 => covers output sequence length / num_tokens).",
    )
    parser.add_argument("--text-field", default="text", help="The key in manifests to read transcripts from.")
    parser.add_argument("--lang-field", default="lang", help="The key in manifests to read language from.")
    parser.add_argument(
        "-n",
        "--num_examples",
        type=int,
        default=-1,
        help="The number of examples (utterances) to estimate the bins. -1 means use all data "
        "(be careful: it could be iterated over infinitely).",
    )
    parser.add_argument(
        "-l",
        "--min_duration",
        type=float,
        default=-float("inf"),
        help="If specified, we'll filter out utterances shorter than this.",
    )
    parser.add_argument(
        "-u",
        "--max_duration",
        type=float,
        default=float("inf"),
        help="If specified, we'll filter out utterances longer than this.",
    )
    parser.add_argument(
        "--max_tps",
        type=float,
        default=float("inf"),
        help="If specified, we'll filter out utterances with more tokens/second than this. "
        "On regular utterances and BPE tokenizers with 1024 tokens 10-12tps is generally a reasonable limit.",
    )
    parser.add_argument(
        "-q", "--quiet", type=bool, default=False, help="When specified, only print the estimated duration bins."
    )
    parser.add_argument(
        "-f",
        "--prompt-format",
        type=str,
        help="When specified, we'll use a prompt formatter in addition to the tokenizer for the purpose of estimating token count bins. "
        "This is useful for accurate 2D bucket estimation with models such as EncDecMultiTaskModel (Canary-1B), "
        "or any model where the label sequence consists of a user prompt and a model's response.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Prompt slots provided as a Python list of dicts. It is used together with --prompt-format option."
        "For example, with Canary-1B you may use: [{'role':'user','slots':{'source_lang':'en','target_lang':'en','task':'asr','pnc':'yes'}]",
    )
    return parser.parse_args()


def estimate_duration_buckets(
    cuts: Iterable[Cut],
    num_buckets: int,
    num_subbuckets: int,
    max_tps: float,
    max_duration: float,
    quiet: bool,
) -> list[tuple[float, float]]:
    """
    This function is based on lhotse.dataset.sampling.dynamic_bucketing.estimate_duration_buckets.
    It extends it to a 2D bucketing case.
    """
    assert num_buckets > 1

    constraint = FixedBucketBatchSizeConstraint2D([(0.0, 0.0)], [0])

    # Gather the duration and token count statistics for the dataset.
    sizes = []
    num_tokens = []
    for c in cuts:
        dur, toks = constraint.measure_length(c)
        sizes.append(dur)
        num_tokens.append(toks)
    sizes = np.array(sizes, dtype=np.float32)
    num_tokens = np.array(num_tokens, dtype=np.int32)
    joint = np.rec.fromarrays([sizes, num_tokens])
    joint.sort()
    sizes = joint.f0
    num_tokens = joint.f1

    # We are building buckets with equal duration (empirically leads to more even bucket exhaustion over time).
    # We need to determine how much duration to allocate per bucket.
    size_per_bucket = sizes.sum() / num_buckets

    if not quiet:
        print("Duration distribution:")
        print(pd.Series(sizes).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    if math.isinf(max_duration):
        max_duration = sizes[-1]

    tps = num_tokens / sizes
    if not quiet:
        print("Token per second distribution:")
        print(pd.Series(tps).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
    if math.isinf(max_tps):
        max_tps = tps.max()
    del tps

    bins = []
    bin_indexes = [0]
    tot = 0.0

    def _estimate_token_buckets(max_bucket_duration):
        # Since this is 2D bucketing, apply the same bin creation logic
        # for the second dimension (i.e. token count) as for the first dimension (duration).
        # That means we aim to have each bucket contain roughly the same number of tokens.
        # Note that this estimation is biased towards more padding if you have
        # a lot of zero-token examples (e.g. non-speech).
        nonlocal bins
        num_tokens_bucket = num_tokens[bin_indexes[-1] : binidx]
        num_tokens_bucket.sort()
        tokens_per_subbucket = num_tokens_bucket.sum() / num_subbuckets
        tot_toks = 0
        # Iterate over token counts, and whenever we hit tokens_per_subbucket, create a new 2D bucket bin.
        for num_toks in num_tokens_bucket:
            # Threshold hit: we are creating a new (max_duration, max_num_tokens) bin.
            if tot_toks > tokens_per_subbucket:
                bins.append((max_bucket_duration, num_toks))
                tot_toks = 0
            tot_toks += num_toks
        bins.append((size, math.ceil(size * max_tps)))

    # Iterate over data, and whenever we hit size_per_bucket, create a new bucket bin.
    for binidx, size in enumerate(sizes):
        if tot > size_per_bucket:
            # Threshold hit: we are creating a new duration bin (multiplied by number of token bins).
            _estimate_token_buckets(max_bucket_duration=size)
            tot = 0.0
        tot += size

    # Estimate an extra 2D bin set for global max duration.
    _estimate_token_buckets(max_bucket_duration=max_duration)

    return bins


def load_tokenizer(paths: list[str], langs: list[str] = None) -> TokenizerWrapper:
    if len(paths) == 1:
        tok = SentencePieceTokenizer(paths[0])
    else:
        assert langs is not None and len(paths) == len(
            langs
        ), f"Cannot create AggregateTokenizer; each tokenizer must have assigned a language via --langs option (we got --tokenizers={paths} and --langs={langs})"
        tok = AggregateTokenizer({lang: SentencePieceTokenizer(p) for lang, p in zip(langs, paths)})
    return TokenizerWrapper(tok)


def apply_tokenizer(cut, tokenizer=None, prompt: PromptFormatter = None):
    if prompt is not None:
        turns = prompt.get_default_dialog_slots()
        last_turn = {"role": prompt.OUTPUT_ROLE, "slots": prompt.get_slots(prompt.OUTPUT_ROLE)}
        assert len(last_turn["slots"]) == 1  # TODO: not sure how to handle multi-slot for system output here
        for key in last_turn["slots"]:
            last_turn["slots"][key] = cut.supervisions[0].text
        last_turn["slots"][prompt.PROMPT_LANGUAGE_SLOT] = cut.supervisions[0].language
        turns.append(last_turn)
        ans = prompt.encode_dialog(turns)
        cut.supervisions[0].tokens = ans["input_ids"]

    elif tokenizer is not None:
        cut = tokenize(cut, tokenizer)

    return cut


class RejectionsCounter:
    def __init__(self, predicate: Callable, message: str):
        self.predicate = predicate
        self.message = message
        self.total = 0
        self.rejected = 0

    def __call__(self, example) -> bool:
        ans = self.predicate(example)
        self.total += 1
        if not ans:
            self.rejected += 1
        return ans

    def print_report(self) -> None:
        if self.rejected:
            print(f"{self.message} | Rejected {self.rejected}/{self.total} examples.")


def main():
    args = parse_args()

    if not args.quiet:
        pd.set_option('display.float_format', lambda x: '%.2f' % x)

    tokenizer = None
    prompt = None
    if args.tokenizer is not None:
        tokenizer = load_tokenizer(args.tokenizer, args.langs)
        if args.prompt_format is not None:
            prompt_defaults = None
            if args.prompt is not None:
                prompt_defaults = ast.literal_eval(args.prompt)
            prompt = PromptFormatter.resolve(args.prompt_format)(tokenizer._tokenizer, defaults=prompt_defaults)

    if '=' in args.input:
        inp_arg = args.input
    elif args.input.endswith(".yaml"):
        inp_arg = f"input_cfg={args.input}"
    elif Path(args.input).is_dir():
        inp_arg = f"shar_path={args.input}"
    else:
        inp_arg = f"manifest_filepath={args.input}"
    config = OmegaConf.merge(
        OmegaConf.structured(LhotseDataLoadingConfig),
        OmegaConf.from_dotlist(
            [inp_arg, "metadata_only=true", f"text_field={args.text_field}", f"lang_field={args.lang_field}"]
        ),
    )
    cuts, _ = read_cutset_from_config(config)
    duration_filter = RejectionsCounter(DurationFilter(args.min_duration, args.max_duration), "Duration filtering")
    cuts = cuts.filter(duration_filter)
    cuts = cuts.map(partial(apply_tokenizer, tokenizer=tokenizer, prompt=prompt))
    tps_filter = RejectionsCounter(TokenPerSecondFilter(-1, args.max_tps), "Token per second filtering")
    cuts = cuts.filter(tps_filter)
    if (N := args.num_examples) > 0:
        cuts = islice(cuts, N)

    duration_bins = estimate_duration_buckets(
        cuts,
        num_buckets=args.buckets,
        num_subbuckets=args.sub_buckets,
        max_tps=args.max_tps,
        max_duration=args.max_duration,
        quiet=args.quiet,
    )
    duration_bins = "[" + ','.join(f"[{b:.3f},{sb:d}]" for b, sb in duration_bins) + "]"
    if args.quiet:
        print(duration_bins)
        return
    duration_filter.print_report()
    tps_filter.print_report()
    print("Use the following options in your config:")
    print(f"\tnum_buckets={args.buckets}")
    print(f"\tbucket_duration_bins={duration_bins}")


if __name__ == "__main__":
    main()
