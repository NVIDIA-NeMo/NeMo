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


import copy
import difflib
import json
import re
import string
from typing import List, Optional, Tuple, Union

import pandas as pd
import pynini
from nemo_text_processing.inverse_text_normalization.en.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from pynini.lib.rewrite import top_rewrite
from tqdm import tqdm

from nemo.utils import logging

DELIMITER = '~~'

cardinal_graph = CardinalFst().graph_no_exception
cardinal_graph = (
    pynini.closure(pynini.union("In ", "in ")) + cardinal_graph + pynini.closure(pynini.accep(" ") + cardinal_graph)
)

inverse_normalizer = InverseNormalizer()


def load_data(input_fs: List[str]):
    """
    loads data from list of abs file paths
    Returns:
        inputs: List[str] list of abs file paths
        targets: List[List[str]] list of targets, can contain multiple options for each target
        sentences: List[List[str]] list of sentence options
        labels: List[List[int]] list of labels (1,0)
    """
    inputs = []
    sentences = []
    cur_sentences = []
    labels = []
    cur_labels = []
    for input_f in input_fs:
        if input_f.endswith(".json"):
            with open(input_f, "r") as f:
                for line in f:
                    line = json.loads(line)
                    try:
                        inputs.append(line['text'].strip())
                        sentences.append([line['gt_normalized'].strip()])
                        labels.append([1])
                    except Exception as e:
                        print(e)
                        raise ValueError(f"Check format for line {line}")
        else:
            with open(input_f, "r") as f:
                for line in f:
                    if line != "\n":
                        try:
                            sent, label = line.strip().split(DELIMITER)
                        except Exception as e:
                            if line.startswith("#"):
                                continue
                            print(e)
                            raise ValueError(f"Check format for line {line}")
                        if label == "RAW":
                            inputs.append(sent)
                        elif label == "1":
                            cur_sentences.append(sent)
                            cur_labels.append(1)
                        elif label == "0":
                            cur_sentences.append(sent)
                            cur_labels.append(0)
                    else:
                        sentences.append(cur_sentences)
                        cur_sentences = []
                        labels.append(cur_labels)
                        cur_labels = []

    if len(cur_sentences) > 0:
        sentences.append(cur_sentences)
        labels.append(cur_labels)
    assert len(inputs) == len(sentences)
    targets = [[x for i, x in enumerate(sents) if ls[i]] for (sents, ls) in zip(sentences, labels)]
    return inputs, targets, sentences, labels


def remove_whitelist_boudaries(x):
    # remove raw whitelist
    x = re.sub(r"\|raw_start\|[^|]+\|raw_end\|", "", x)
    # remove norm text boundaries
    x = x.replace("|norm_start|", "").replace("|norm_end|", "")
    return x


def _clean_pre_norm_libritts(inputs: List[str], targets: List[List[str]]):
    """
    standardizes format of inputs and targets before being normalized, so more rules apply.
    This is specific for libritts.
    """
    for i in range(len(targets)):
        for j in range(len(targets[i])):
            targets[i][j] = clean_libri_tts(targets[i][j])

    for i in range(len(inputs)):
        for target in targets[i]:
            diffs = get_diff(a=inputs[i].lower(), b=target.lower())
            for diff in diffs[::-1]:
                in_diff = inputs[i][diff[0][0] : diff[0][1]].lower()
                tg_diff = target[diff[1][0] : diff[1][1]].lower()
                replacement = inputs[i][: diff[0][0]] + tg_diff + inputs[i][diff[0][1] :]
                if (in_diff == "s" and tg_diff == "z") or (in_diff == "z" and tg_diff == "s"):
                    inputs[i] = replacement
                elif (in_diff == "re" and tg_diff == "er") or (in_diff == "er" and tg_diff == "re"):
                    inputs[i] = replacement
                elif (in_diff == "me" and tg_diff == "") or (in_diff == "" and tg_diff == "me"):
                    inputs[i] = replacement
                elif (in_diff == "ue" and tg_diff == "") or (in_diff == "" and tg_diff == "ue"):
                    inputs[i] = replacement
    return inputs, targets


def _clean_pre_norm_google(inputs: List[str], targets: List[List[str]]):
    """
    standardizes format of inputs and targets before being normalized, so more rules apply.
    This is specific for google dataset.
    """
    for i in range(len(inputs)):

        inputs[i] = re.sub(r"\$\s([0-9]{1,})", r"$\1", inputs[i])
        inputs[i] = re.sub(r"\bmr ", r"Mr. ", inputs[i])
        inputs[i] = re.sub(r"\bdr ", r"Dr. ", inputs[i])
        inputs[i] = re.sub(r"\bdr$", r"Dr.", inputs[i])
        inputs[i] = re.sub(r"\bmrs ", r"Mrs. ", inputs[i])
        inputs[i] = re.sub(r"\bjr ", r"Jr. ", inputs[i])
        inputs[i] = re.sub(r"\bjr$", r"Jr.", inputs[i])
        inputs[i] = re.sub(r"\dsr ", r"Sr. ", inputs[i])
        inputs[i] = re.sub(r"\dsr$", r"Sr.", inputs[i])
        for target in targets[i]:
            diffs = get_diff(a=inputs[i].lower(), b=target.lower())
            for diff in diffs[::-1]:
                in_diff = inputs[i][diff[0][0] : diff[0][1]].lower()
                tg_diff = target[diff[1][0] : diff[1][1]].lower()
                replacement = inputs[i][: diff[0][0]] + tg_diff + inputs[i][diff[0][1] :]
                if (in_diff == "s" and tg_diff == "z") or (in_diff == "z" and tg_diff == "s"):
                    inputs[i] = replacement
                elif (in_diff == "re" and tg_diff == "er") or (in_diff == "er" and tg_diff == "re"):
                    inputs[i] = replacement
                elif (in_diff == "me" and tg_diff == "") or (in_diff == "" and tg_diff == "me"):
                    inputs[i] = replacement
                elif (in_diff == "" and tg_diff == "u") or (in_diff == "u" and tg_diff == ""):
                    inputs[i] = replacement
                elif (in_diff == "ue" and tg_diff == "") or (in_diff == "" and tg_diff == "ue"):
                    inputs[i] = replacement
                elif re.sub(r"\.", "", in_diff) == re.sub(r"( |\.)", "", tg_diff):
                    inputs[i] = replacement

    return inputs, targets


def clean_pre_norm(inputs: List[str], targets: List[List[str]], dataset: Optional[str] = None):
    """
    standardizes format of inputs and targets before being normalized, so more rules apply.
    """
    # deep copy
    pre_inputs = copy.deepcopy(inputs)
    pre_targets = copy.deepcopy(targets)

    # --- data specific pre cleaning ---
    if dataset == "libritts":
        pre_inputs, pre_targets = _clean_pre_norm_libritts(inputs=pre_inputs, targets=pre_targets)
    elif dataset == "google":
        pre_inputs, pre_targets = _clean_pre_norm_google(inputs=pre_inputs, targets=pre_targets)
    else:
        pass

    # --- general pre cleaning ---
    for i in range(len(pre_inputs)):
        pre_inputs[i] = re.sub("librivox.org", "librivox dot org", pre_inputs[i])
        pre_inputs[i] = re.sub(
            rf"([0-9]?[0-9](\.|:)[0-9][0-9]\s?)(a|A|p|P)(\.?)\s(M|m)(\.?)", rf"\1\3\4\5\6", pre_inputs[i]
        )
        # pre_inputs[i] =re.sub(rf"\b(S|s)t\.", rf"saint", pre_inputs[i])
    return pre_inputs, pre_targets


def _clean_post_norm_libritts(inputs: List[str], targets: List[List[str]], norm_texts):
    return targets, norm_texts


def _clean_post_norm_google(inputs: List[str], targets: List[List[str]], norm_texts):
    """
    standardizes format of inputs and targets, and predicted normalizations for easier evaluation.
    This is specific for google dataset.
    """
    for i in range(len(targets)):
        for target in targets[i]:
            for j, norm in enumerate(norm_texts[i][0]):
                diffs = get_diff(a=norm.lower(), b=target.lower())
                for diff in diffs[::-1]:
                    norm_diff = norm[diff[0][0] : diff[0][1]].lower()
                    tg_diff = target[diff[1][0] : diff[1][1]].lower()
                    replacement = norm[: diff[0][0]] + tg_diff + norm[diff[0][1] :]
                    if norm_diff == re.sub(r" ", "", tg_diff):
                        norm_texts[i][0][j] = replacement

    return targets, norm_texts


def _clean_post_general(str) -> str:
    """
    standardizes format of inputs and targets, and predicted normalizations for easier evaluation.
    """
    str = re.sub(rf" oh ", " zero ", str)
    str = re.sub(rf" oh$", " zero", str)
    str = re.sub(rf"^oh ", "zero ", str)
    # str = re.sub(rf" o ", " zero ", str)
    str = re.sub(rf"\sO\b", "zero", str)
    str = re.sub(rf" o$", " zero", str)
    str = re.sub(rf"^o ", "zero ", str)
    str = re.sub(rf"'o ", "'zero ", str)
    str = str.replace("mountain", "mount")
    return str


def _clean_targets(str) -> str:
    """Clean ground truth options."""
    str = re.sub(rf" o ", " zero ", str)
    return str


def adjust_pred(pred: str, gt: str, dataset: str, delim_present=True):
    """Standardize prediction format to make evaluation easier"""
    orig_pred = pred
    orig_gt = gt
    if delim_present and not re.search(rf"< (.*?) >", pred):
        return pred
    pred = re.sub(rf"< ", "", pred)
    pred = re.sub(rf" >", "", pred)
    pred = pred.lower().strip()
    gt = gt.lower().strip()
    can_be_adjusted = False

    if dataset in ["google", "libritts"] and pred != gt:
        if is_date(pred=pred, gt=gt, cardinal_graph=cardinal_graph):
            pred = gt
        elif contains_month(pred, gt):
            pred = re.sub(r",", "", pred)
            gt = re.sub(r",", "", gt)
            pred = re.sub(r" zero ", " o ", pred)
            gt = re.sub(r" zero ", " o ", gt)
            gt = re.sub(rf" +", " ", gt)
            pred = re.sub(rf" +", " ", pred)

        if pred != gt:
            gt_itn = inverse_normalizer.normalize(gt, verbose=False)
            pred_itn = inverse_normalizer.normalize(pred, verbose=False)
            if len(gt_itn) == len(pred_itn) and set(gt_itn) == set(pred_itn):
                can_be_adjusted = True
                pred = gt
            elif " of " in gt:
                gt = re.sub(r"(^the | of)", "", gt)
                idx = gt.index(" ")
                idx2 = (gt[idx + 1 :].index(" ") if " " in gt[idx + 1 :] else len(gt[idx + 1 :])) + idx + 1
                gt = gt[idx + 1 : idx2] + " " + gt[:idx] + gt[idx2:]
    if dataset == "libritts" and pred != gt:
        if "dollar" in gt:
            gt = re.sub(rf"\band\b", "", gt)
            pred = re.sub(rf"\band\b", "", pred)
            if re.search(r"\bus dollar", pred) and not re.search(r"\bus dollar", gt):
                pred = re.sub(rf"\bus dollar", "dollar", pred)
        else:
            gt = re.sub(rf"(\bthe\b|\.)", "", gt)
            pred = re.sub(rf"\bone\b", "a", pred)
            gt = re.sub(rf"\bmr\b", "mister", gt)
            gt = re.sub(rf"\bmrs\b", "misses", gt)
            gt = re.sub(rf"\bdr\b", "doctor", gt)
            gt = re.sub(rf"\bco\b", "company", gt)
    if gt != pd and dataset in ["google", "libritts"]:
        if gt.replace("/", "").replace("  ", " ") == pred.replace("slash", "").replace("  ", " "):
            pred = gt
        elif gt in ["s", "z"] and pred in ["s", "z"]:
            pred = gt
        elif gt == "hash tag" and pred == "hash":
            pred = "hash tag"
        elif gt[:-2] == pred[:-2] and gt[-2:] in ["er", "re"] and pred[-2:] in ["er", "re"]:
            pred = gt
        # elif gt.replace("-", " ").replace("  ", " ") == pred.replace("minus", "").replace("  ", " "):
        #     pred = gt
        elif gt.replace("to", "").replace("-", "") == pred.replace("to", "").replace("-", ""):
            pred = gt

    gt = re.sub(rf" +", " ", gt)
    pred = re.sub(rf"(\.)", "", pred)
    pred = re.sub(rf" +", " ", pred)
    if gt == pred:
        can_be_adjusted = True
    if can_be_adjusted:
        if delim_present:
            res = f" < {orig_gt} > "
        else:
            res = orig_gt
        return res
    else:
        return orig_pred


def clean_post_norm(
    inputs: List[str],
    targets: List[List[str]],
    norm_texts,
    dataset: Optional[str] = None,
    delim_present: Optional[bool] = True,
):
    """
    Args:
        inputs (List[str]): inputs
        targets (List[List[str]]): targets
        norm_texts (List[(List[str], List[float])]): List of normalization options, weights
        dataset (Optional[str], optional): _description_. Defaults to None.
        delim_present (Optional[str], optional): The flag indicates whether normalization output contain delimiters "<>".
            Set to False for NN baseline.
    """
    # deep copy
    post_norm_texts = copy.deepcopy(norm_texts)
    post_targets = copy.deepcopy(targets)

    # --- data specific pre cleaning ---
    if dataset == "libritts":
        post_targets, post_norm_texts = _clean_post_norm_libritts(
            inputs=inputs, targets=post_targets, norm_texts=post_norm_texts
        )
    elif dataset == "google":
        post_targets, post_norm_texts = _clean_post_norm_google(
            inputs=inputs, targets=post_targets, norm_texts=post_norm_texts
        )

    else:
        pass

    # --- general pre cleaning ---

    for i in range(len(targets)):
        for j, x in enumerate(post_targets[i]):
            post_targets[i][j] = _clean_post_general(x)
        for j, x in enumerate(post_norm_texts[i][0]):
            if x.count("< ") != x.count(" >"):
                x = x.replace("<", "< ").replace(">", " >").replace("  ", " ")
            post_norm_texts[i][0][j] = _clean_post_general(x)
    if dataset in ["libritts", "google"]:
        for i, _targets in enumerate(post_targets):
            for jj, option in enumerate(post_norm_texts[i][0]):
                for _, _target in enumerate(_targets):

                    if not delim_present:
                        # nn doesn't have punctuation marks that leads for diff_pred_gt mismatch
                        _target = remove_punctuation(_target, remove_spaces=False, do_lower=True)
                        option = remove_punctuation(option, remove_spaces=False, do_lower=True)

                    diffs = diff_pred_gt(pred=option, gt=_target)
                    for diff in diffs[::-1]:
                        if diff[0][1] - diff[0][0] == 0 and diff[1][1] - diff[1][0] == 0:
                            continue
                        pred = option[diff[0][0] : diff[0][1]]
                        gt = _target[diff[1][0] : diff[1][1]]
                        logging.debug(f"pred: |{pred}|\tgt: |{gt}|")
                        new_pred = adjust_pred(pred=pred, gt=gt, dataset=dataset, delim_present=delim_present)
                        new_pred = (
                            post_norm_texts[i][0][jj][: diff[0][0]]
                            + new_pred
                            + post_norm_texts[i][0][jj][diff[0][1] :]
                        )
                        logging.debug(f"|{post_norm_texts[i][0][jj]}| -> |{new_pred}|")
                        post_norm_texts[i][0][jj] = new_pred
    return post_targets, post_norm_texts


def clean_libri_tts(target: str):
    """
	Replace abbreviations in LibriTTS dataset
	"""

    # Normalized text in LibriTTS by Google which contains abbreviations from `libri_sometimes_converts_abbrs` sometimes wasn't converted.
    libri_sometimes_converts_abbrs = {"St.": "saint", "Rev.": "reverend"}

    # Normalized text in LibriTTS by Google which contains abbreviations from `libri_wo_changes_abbrs` wasn't converted.
    libri_wo_changes_abbrs = {"vs.": "versus"}

    google_abbr2expand = {
        "mr": "mister",
        "Mr": "Mister",
        "mrs": "misses",
        "Mrs": "Misses",
        "dr": "doctor",
        "Dr": "Doctor",
        "drs": "doctors",
        "Drs": "Doctors",
        "lt": "lieutenant",
        "Lt": "Lieutenant",
        "sgt": "sergeant",
        "Sgt": "Sergeant",
        "st": "saint",
        "St": "Saint",
        "jr": "junior",
        "Jr": "Junior",
        "maj": "major",
        "Maj": "Major",
        "hon": "honorable",
        "Hon": "Honorable",
        "gov": "governor",
        "Gov": "Governor",
        "capt": "captain",
        "Capt": "Captain",
        "esq": "esquire",
        "Esq": "Esquire",
        "gen": "general",
        "Gen": "General",
        "ltd": "limited",
        "Ltd": "Limited",
        "rev": "reverend",
        "Rev": "Reverend",
        "col": "colonel",
        "Col": "Colonel",
        "and co": "and Company",
        "and Co": "and Company",
        "mt": "mount",
        "Mt": "Mount",
        "ft": "fort",
        "Ft": "Fort",
        "tenn": "tennessee",
        "Tenn": "Tennessee",
        "vs": "versus",
        "Vs": "Versus",
        "&": "and",
        "§": "section",
        "#": "hash",
        "=": "equals",
    }

    # let's normalize `libri_only_remove_dot_abbrs` abbreviations, because google doesn't do it well
    for abbr in google_abbr2expand.keys():
        if abbr in target:
            # replace abbr in google text via regex and using \b to match only whole words, keep original 1 and 2 groups
            target = re.sub(rf'(^|\s|\W){abbr}($|\s)', rf"\1{google_abbr2expand[abbr]}\2", target)

    # let's normalize `libri_sometimes_converts_abbrs` abbreviations manually, google sometimes forgets to expand them
    for abbr, t in libri_sometimes_converts_abbrs.items():
        target = target.replace(abbr, t)

    # let's normalize `libri_wo_changes_abbrs` abbreviations manually, google doesn't change, but they should be
    for abbr, t in libri_wo_changes_abbrs.items():
        target = target.replace(abbr, t)

    return target


def remove_punctuation(text: str, remove_spaces=True, do_lower=True, lang="en", exclude=None):
    """Removes punctuation (and optionally spaces) in text for better evaluation"""
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    if lang == "en":
        # remove things like \x94 and \x93
        text = re.sub(r"[^\x00-\x7f]", r" ", text)

    text = re.sub(r" +", " ", text)
    if remove_spaces:
        text = text.replace(" ", "").replace("\u00A0", "").strip()

    if do_lower:
        text = text.lower()
    return text.strip()


def get_alternative_label(pred: str, targets: List[str]) -> bool:
    """Returns true if prediction matches target options"""

    def _relax_diff(text):
        text = text.replace("us dollars", "dollars")
        text = text.replace("etcetera", "").replace("etc", "")
        text = text.replace("one half ounce", "").replace("half an ounce", "")
        text = text.replace("television", "").replace("t v ", " ").replace("tv", "")
        text = text.replace("hundred", "")
        text = text.replace("forty two", "").replace("four two", "")
        text = text.replace("re", "").replace("er", "")
        text = text.replace("ou", "").replace("o", "")
        text = text.replace("  ", " ").strip()
        return text

    acceptable = False
    pred = remove_punctuation(pred, remove_spaces=False, do_lower=True)
    for target in targets:
        target = _clean_post_general(remove_punctuation(target, remove_spaces=False, do_lower=True))
        target = _clean_targets(remove_punctuation(target, remove_spaces=False, do_lower=True))
        if _relax_diff(target) == _relax_diff(pred):
            acceptable = True
            break
    return acceptable


def get_labels(targets: List[str], norm_texts_weights: List[Tuple[str, str]], lang="en",) -> List[List[str]]:
    """
    Assign labels to generated normalization options (1 - for ground truth, 0 - other options)
    Args:
        targets: ground truth normalization sentences
        norm_texts_weights: List of tuples: (normalization options, weights of normalization options)
    returns:
        List of labels [1, 0] for every normalization option
    """
    print("Assign labels to generated normalization options...")
    labels = []
    for i, cur_targets in tqdm(enumerate(targets)):
        curr_labels = []
        cur_targets = [_clean_targets(t) for t in cur_targets]
        for norm_option in norm_texts_weights[i][0]:
            norm_option = _clean_targets(norm_option)
            norm_option = remove_whitelist_boudaries(norm_option)

            if is_correct(pred=norm_option, targets=cur_targets, lang=lang):
                curr_labels.append(1)
            elif get_alternative_label(pred=norm_option, targets=cur_targets):
                curr_labels.append(1)
            else:
                curr_labels.append(0)
        labels.append(curr_labels)
    return labels


def contains_month(pred, gt):
    """Check is the pred/gt contain month in the span"""
    months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]

    for mon in months:
        if mon in gt and mon in pred:
            return True
    return False


def is_date(pred, gt, cardinal_graph):
    """Returns True is pred and gt are date format modifications and are equal."""
    is_date_case = False

    # for cases "1890" -> "one thousand eight hundred ninety" vs "eighteen ninety"
    if "thousand" in pred and "hundred" in pred and pred.strip().split()[-2:] == gt.strip().split()[-2:]:
        is_date_case = True
    elif "thousand" in gt and "hundred" in gt and gt.strip().split()[-2:] == pred.strip().split()[-2:]:
        is_date_case = True
    else:
        try:
            if top_rewrite(gt.replace(" oh ", " zero ").replace(" o ", " zero "), cardinal_graph).replace(
                " ", ""
            ) == top_rewrite(pred.replace(" oh ", " zero ").replace(" o ", " zero "), cardinal_graph).replace(" ", ""):
                is_date_case = True
        except:
            pass

    return is_date_case


def is_correct(pred: str, targets: Union[List[str], str], lang: str) -> bool:
    """
    returns True if prediction matches targets for language lang.
    """
    if isinstance(targets, List):
        targets = [remove_punctuation(x, remove_spaces=True, do_lower=True, lang=lang) for x in targets]
    else:
        targets = [remove_punctuation(targets, remove_spaces=True, do_lower=True, lang=lang)]

    pred = remove_punctuation(pred, remove_spaces=True, do_lower=True)
    return pred in targets


def print_df(df):
    """
    prints data frame
    """
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", 1000, "display.max_colwidth", 400,
    ):
        print(df)


def get_diff(a: str, b: str):
    """returns list of different substrings between and b

    Returns:
        list of Tuple(pred start and end, gt start and end) subsections
    """
    s = difflib.SequenceMatcher(None, a, b, autojunk=False)

    # s contains a list of triples. Each triple is of the form (i, j, n), and means that a[i:i+n] == b[j:j+n].
    # The triples are monotonically increasing in i and in j.
    s = s.get_matching_blocks()
    s = [x for x in s if x[2] != 1]
    # get not matching blocks
    matches = [[0, 0, 0]] + s
    unmatches_l = []
    unmatches_r = []
    for l, r in zip(matches[:-1], matches[1:]):
        unmatches_l.append([l[0] + l[2], r[0]])
        unmatches_r.append([l[1] + l[2], r[1]])

    result = list(zip(unmatches_l, unmatches_r))

    for item in list(zip(unmatches_l, unmatches_r)):
        logging.debug(f"a: {a[item[0][0]:item[0][1]]}")
        logging.debug(f"b: {b[item[1][0]:item[1][1]]}")
        logging.debug("=" * 20)
    return result[1:]


def diff_pred_gt(pred: str, gt: str):
    """returns list of different substrings between prediction and gt
    relies on that prediction uses '< '  ' >'  

    Args:
        pred (str): prediction
        gt (str): ground truth

    Returns:
        list of Tuple(pred start and end, gt start and end) subsections
    
    e.g. pred="< Edward third >., king Our own . loss had been < two thousand two hundred >"
         gt  ="Edward III., king Our own loss had been twenty two hundred"
         --> [([0, 16], [0, 10]),      ([32, 34], [26, 26]),      ([48, 76], [40, 58])]
    """
    s = difflib.SequenceMatcher(None, pred, gt, autojunk=False)

    # s contains a list of triples. Each triple is of the form (i, j, n), and means that a[i:i+n] == b[j:j+n].
    # The triples are monotonically increasing in i and in j.
    s = s.get_matching_blocks()

    left = list(re.finditer("< ", pred))
    left = [x.start() for x in left]
    right = list(re.finditer(" >", pred))
    right = [x.end() for x in right]
    left = [-1] + left + [len(pred)]
    right = [0] + right + [len(pred)]

    matches = []
    assert len(left) == len(right)
    idx = 1
    for i, seq in enumerate(s):
        if i == len(s) - 1 and seq[2] == 0:
            break
        while idx < len(left) - 1 and (seq[0] >= right[idx]):
            idx += 1

        if right[idx - 1] <= seq[0] < left[idx] and (seq[0] + seq[2]) <= left[idx]:
            matches.append(seq)

    # get not matching blocks
    matches = [[0, 0, 0]] + matches + [[len(pred), len(gt), 0]]
    unmatches_l = []
    unmatches_r = []
    for l, r in zip(matches[:-1], matches[1:]):
        unmatches_l.append([l[0] + l[2], r[0]])
        unmatches_r.append([l[1] + l[2], r[1]])

    result = list(zip(unmatches_l, unmatches_r))

    for item in list(zip(unmatches_l, unmatches_r)):
        logging.debug(f"pred: {pred[item[0][0]:item[0][1]]}")
        logging.debug(f"gt  : {gt[item[1][0]:item[1][1]]}")
        logging.debug("=" * 20)
    return result
