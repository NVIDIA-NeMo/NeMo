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


import argparse
import os
import pickle
import re
import shutil
from typing import Dict, List

import model_utils
import pandas as pd
import utils
from joblib import Parallel, delayed
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio
from tqdm import tqdm

from nemo.utils import logging

parser = argparse.ArgumentParser(description="Re-scoring")
parser.add_argument("--lang", default="en", type=str, choices=["en"])
parser.add_argument("--n_tagged", default=100, type=int, help="Number WFST options")
parser.add_argument("--context_len", default=-1, type=int, help="Context length, -1 to use full context")
parser.add_argument("--threshold", default=0.2, type=float, help="delta threshold value")
parser.add_argument("--overwrite_cache", action="store_true", help="overwrite cache")
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--cache_dir", default='cache', type=str, help="use cache dir")
parser.add_argument(
    "--data",
    default="text_normalization_dataset_files/EngConf.txt",
    help="For En only. Path to a file for evaluation.",
)
parser.add_argument("--n_jobs", default=-2, type=int, help="The maximum number of concurrently running jobs")
parser.add_argument(
    "--models", default="mlm_bert-base-uncased", type=str, help="Comma separated string of model names"
)
parser.add_argument(
    "--regenerate_pkl",
    action="store_true",
    help="Set to True to re-create pickle file with WFST normalization options",
)
parser.add_argument("--batch_size", default=200, type=int, help="Batch size for parallel processing")


def rank(sentences: List[str], labels: List[int], models: Dict[str, 'Model'], context_len=None, do_lower=True):
    """
    computes scores for each sentences using all provided models and returns summary in data frame
    """
    df = pd.DataFrame({"sent": sentences, "labels": labels})
    for model_name, model in models.items():
        scores = model_utils.score_options(
            sentences=sentences, context_len=context_len, model=model, do_lower=do_lower
        )
        df[model_name] = scores
    return df


def threshold_weights(norm_texts_weights, delta: float = 0.2):
    """
    norm_texts_weights: list of [ List[normalized options of input], list[weights] ]
    delta: delta to add to minimum weight in options to compose upper limit for threshhold

    returns:
        filter list of same format as input 
    """
    # threshold value is factor applied to lowest/first weight of all normalization options for every input
    res = []
    for i, options_weights in enumerate(norm_texts_weights):
        thresh = options_weights[1][0] + delta  # minimum weight plus delta
        item = [x for x in zip(*options_weights)]
        # filters out all options for every input that is larger than threshold
        res.append(list(filter(lambda x: x[1] < thresh, item)))

    return [list(map(list, zip(*item))) for item in res]


def _get_unchanged_count(text):
    """
    returns number of unchanged words in text
    """
    exclude = '#$%&<>'

    # remove normalized whitelist
    text = re.sub(r"\|norm_start\|[^|]+\|norm_end\|", "", text)
    # remove raw text boundaries
    text = text.replace("|raw_start|", "").replace("|raw_end|", "")

    start_pattern = "<"
    end_pattern = ">"

    text = utils.remove_punctuation(text, remove_spaces=False, do_lower=False, exclude=exclude)
    text_clean = ""
    for ch in text:
        if ch.isalpha() or ch.isspace() or ch in [start_pattern, end_pattern]:
            text_clean += ch
        else:
            text_clean += " " + ch + " "

    text = text_clean
    unchanged_count = 0
    skip = False

    for word in text.split():
        if start_pattern == word:
            skip = True
        elif end_pattern == word:
            skip = False
        elif not skip:
            unchanged_count += 1
    return unchanged_count


def _get_replacement_count(text):
    """
    returns number of token replacements
    """
    start_pattern = "<"
    end_pattern = ">"
    return min(text.count(start_pattern), text.count(end_pattern))


def threshold(norm_texts_weights, unchanged=True, replacement=True):
    """
    Reduces the number of WFST options based for LM rescoring.

    Args:
        :param norm_texts_weights: WFST options with associated weight
        :param unchanged: set to True to filter out examples based on number of words left unchanged
            (punct is not taken into account)
        :param replacement: set to True to filter out examples based on number of replacements made
            (Given A and B are WFST options, if the number of unchanged for A and B are the same,
            the option with a smaller number of replacements is preferable (i.e., larger span)).

    :return: WFST options with associated weight (reduced)
    """

    def __apply(norm_texts_weights, f, use_min=True):
        inputs_filtered = []
        for example in norm_texts_weights:
            texts = example[0]
            counts = [f(t) for t in texts]
            [logging.debug(f"{c} -- {t}") for t, c in zip(texts, counts)]
            target_count = min(counts) if use_min else max(counts)
            filtered_texts = []
            filtered_weights = []
            for i, c in enumerate(counts):
                if c == target_count:
                    filtered_texts.append(example[0][i])
                    filtered_weights.append(example[1][i])
            inputs_filtered.append([filtered_texts, filtered_weights])
        return inputs_filtered

    logging.debug("BASIC THRESHOLDING INPUT:")
    [logging.debug(x) for x in norm_texts_weights[0][0]]
    if unchanged:
        norm_texts_weights = __apply(norm_texts_weights, _get_unchanged_count)
        logging.debug("AFTER UNCHANGED FILTER:")
        [logging.debug(x) for x in norm_texts_weights[0][0]]

    if replacement:
        norm_texts_weights = __apply(norm_texts_weights, _get_replacement_count)
        logging.debug("AFTER REPLACEMENT FILTER:")
        [logging.debug(x) for x in norm_texts_weights[0][0]]

    return norm_texts_weights


def main():
    args = parser.parse_args()

    logging.setLevel(logging.INFO)
    lang = args.lang
    input_f = args.data

    if args.data == "text_normalization_dataset_files/LibriTTS.json":
        args.dataset = "libritts"
    elif args.data == "text_normalization_dataset_files/GoogleTN.json":
        args.dataset = "google"
    else:
        args.dataset = None
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"{args.data} file not found")

    print("Create Masked Language Model...")
    models = model_utils.init_models(model_name_list=args.model_name)
    input_fs = input_f.split(",")
    print("LOAD DATA...")
    inputs, targets, _, _ = utils.load_data(input_fs)
    pre_inputs, pre_targets = utils.clean_pre_norm(dataset=args.dataset, inputs=inputs, targets=targets)

    print("INIT WFST...")
    normalizer = NormalizerWithAudio(
        input_case="cased", lang=lang, cache_dir=args.cache_dir, lm=True, overwrite_cache=args.overwrite_cache
    )

    print("APPLYING NORMALIZATION RULES...")
    p_file = (
        f"norm_texts_weights_{args.n_tagged}_{os.path.basename(args.data)}_{args.context_len}_{args.threshold}.pkl"
    )

    if not os.path.exists(p_file) or args.regenerate_pkl:
        print(f"Creating WFST and saving to {p_file}")

        def __process_batch(batch_idx, batch, dir_name):
            normalized = []
            for x in tqdm(batch):
                ns, ws = normalizer.normalize(x, n_tagged=args.n_tagged, punct_post_process=False)
                ns = [re.sub(r"<(.+?)>", r"< \1 >", x) for x in ns]
                normalized.append((ns, ws))
            with open(f"{dir_name}/{batch_idx}.p", "wb") as handle:
                pickle.dump(normalized, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Batch -- {batch_idx} -- is complete")
            return batch_idx

        # to save intermediate results to a file
        batch = min(len(pre_inputs), args.batch_size)

        tmp_dir = f"/tmp/{os.path.basename(args.data)}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        batch_ids = Parallel(n_jobs=args.n_jobs)(
            delayed(__process_batch)(idx, pre_inputs[i : i + batch], tmp_dir)
            for idx, i in enumerate(range(0, len(pre_inputs), batch))
        )

        # aggregate all intermediate results
        norm_texts_weights = []
        for batch_id in batch_ids:
            batch_f = f"{tmp_dir}/{batch_id}.p"
            norm_texts_weights.extend(pickle.load(open(batch_f, "rb")))

        with open(p_file, "wb") as handle:
            pickle.dump(norm_texts_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Loading WFST from {p_file}")
        norm_texts_weights = pickle.load(open(p_file, "rb"))

    print("THRESHOLDING...")
    # apply weights threshold to reduce number of options

    if args.threshold > 0:
        norm_texts_weights = threshold_weights(norm_texts_weights, delta=args.threshold)
        logging.debug("AFTER WEIGHTS THRESHOLDING:")
        [logging.debug(x) for x in norm_texts_weights[0][0]]

    # reduce number of options by selecting options with the smallest number of unchanged words
    norm_texts_weights = threshold(norm_texts_weights)

    print("POST PROCESSING...")
    post_targets, post_norm_texts_weights = utils.clean_post_norm(
        dataset=args.dataset, inputs=pre_inputs, targets=pre_targets, norm_texts=norm_texts_weights
    )

    print("GETTING LABELS...")
    labels = utils.get_labels(targets=post_targets, norm_texts_weights=post_norm_texts_weights)

    examples_with_no_labels_among_wfst = [i for i, x in enumerate(labels) if 1 not in x]

    print("GATHERING STATS...")
    model_stats = {m: 0 for m in models}
    gt_in_options = 0
    for i, example in tqdm(enumerate(zip(post_norm_texts_weights, labels))):
        data, curr_labels = example
        assert len(data[0]) == len(curr_labels)
        df = rank(
            sentences=data[0],
            labels=curr_labels,
            models=models,
            context_len=args.context_len if args.context_len is not None and args.context_len >= 0 else None,
            do_lower=True,
        )
        df['sent'] = df['sent'].apply(lambda x: utils.remove_whitelist_boudaries(x))
        df["weights"] = data[1]

        do_print = False

        for model in models:
            # one hot vector for predictions, 1 for the best score option
            df[f"{model}_pred"] = (df[model] == min(df[model])).astype(int)
            # add constrain when multiple correct labels per example
            pred_is_correct = min(sum((df["labels"] == df[f"{model}_pred"]) & df["labels"] == 1), 1)

            if not pred_is_correct or logging.getEffectiveLevel() <= logging.DEBUG:
                do_print = True

            if do_print:
                print(f"{model} prediction is correct: {pred_is_correct == 1}")
            model_stats[model] += pred_is_correct
        gt_in_options += 1 in curr_labels

        if do_print:
            print(f"INPUT: {pre_inputs[i]}")
            print(f"GT   : {post_targets[i]}\n")
            utils.print_df(df)
            print("-" * 80 + "\n")

    if gt_in_options != len(post_norm_texts_weights):
        print("WFST options for some examples don't contain the ground truth:")
        for i in examples_with_no_labels_among_wfst:
            print(f"INPUT: {pre_inputs[i]}")
            print(f"GT   : {post_targets[i]}\n")
            print(f"WFST:")
            for x in post_norm_texts_weights[i]:
                print(x)
            print("=" * 40)

    all_correct = True
    for model, correct in model_stats.items():
        print(
            f"{model} -- correct: {correct}/{len(post_norm_texts_weights)} or ({round(correct/len(post_norm_texts_weights) * 100, 2)}%)"
        )
        all_correct = all_correct and (correct == len(post_norm_texts_weights))

    print(f"examples_with_no_labels_among_wfst: {len(examples_with_no_labels_among_wfst)}")
    return all_correct


if __name__ == "__main__":
    all_correct = main()
    print(f"all_correct: {all_correct}")
