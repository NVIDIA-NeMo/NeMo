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


import re
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np

"""Utility functions for Thutmose Tagger."""


def get_token_list(text: str) -> List[str]:
    """Returns a list of tokens.

    This function expects that the tokens in the text are separated by space
    character(s). Example: "ca n't , touch". This is the case at least for the
    public DiscoFuse and WikiSplit datasets.

    Args:
        text: String to be split into tokens.
    """
    return text.split()


def yield_sources_and_targets(input_filename: str):
    """Reads and yields source lists and targets from the input file.

    Args:
        input_filename: Path to the input file.

    Yields:
        Tuple with (list of source texts, target text).
    """
    # The format expects a TSV file with the source on the first and the
    # target on the second column.
    with open(input_filename, 'r') as f:
        for line in f:
            source, target, semiotic_info = line.rstrip('\n').split('\t')
            yield source, target, semiotic_info


def read_label_map(path: str) -> Dict[str, int]:
    """Return label map read from the given path."""
    with open(path, 'r') as f:
        label_map = {}
        empty_line_encountered = False
        for tag in f:
            tag = tag.strip()
            if tag:
                label_map[tag] = len(label_map)
            else:
                if empty_line_encountered:
                    raise ValueError('There should be no empty lines in the middle of the label map ' 'file.')
                empty_line_encountered = True
        return label_map


def read_semiotic_classes(path: str) -> Dict[str, int]:
    """Return semiotic classes map read from the given path."""
    with open(path, 'r') as f:
        semiotic_classes = {}
        empty_line_encountered = False
        for tag in f:
            tag = tag.strip()
            if tag:
                semiotic_classes[tag] = len(semiotic_classes)
            else:
                if empty_line_encountered:
                    raise ValueError('There should be no empty lines in the middle of the label map ' 'file.')
                empty_line_encountered = True
        return semiotic_classes


def split_text_by_isalpha(s: str):
    """Split string into segments, so that alphabetic sequence is one segment"""
    for k, g in groupby(s, str.isalpha):
        yield ''.join(g)


def spoken_preprocessing(spoken: str) -> str:
    """Preprocess spoken input for Thuthmose tagger model.
    Attention!
    This function is used both during data preparation and during inference.
    If you change it, you should rerun data preparation and retrain the model.
    """
    spoken = spoken.casefold()
    spoken = spoken.replace('_trans', '').replace('_letter_latin', '').replace('_letter', '')

    #  "долларов сэ ш а"  => "долларов-сэ-ш-а"    #join into one token to simplify alignment
    spoken = re.sub(r" долларов сэ ш а", r" долларов-сэ-ш-а", spoken)
    spoken = re.sub(r" доллара сэ ш а", r" доллара-сэ-ш-а", spoken)
    spoken = re.sub(r" доллар сэ ш а", r" доллар-сэ-ш-а", spoken)
    spoken = re.sub(r" фунтов стерлингов", r" фунтов-стерлингов", spoken)
    spoken = re.sub(r" фунта стерлингов", r" фунта-стерлингов", spoken)
    spoken = re.sub(r" фунт стерлингов", r" фунт-стерлингов", spoken)
    spoken = re.sub(r" долларами сэ ш а", r" долларами-сэ-ш-а", spoken)
    spoken = re.sub(r" долларам сэ ш а", r" долларам-сэ-ш-а", spoken)
    spoken = re.sub(r" долларах сэ ш а", r" долларах-сэ-ш-а", spoken)
    spoken = re.sub(r" долларе сэ ш а", r" долларе-сэ-ш-а", spoken)
    spoken = re.sub(r" доллару сэ ш а", r" доллару-сэ-ш-а", spoken)
    spoken = re.sub(r" долларом сэ ш а", r" долларом-сэ-ш-а", spoken)
    spoken = re.sub(r" фунтами стерлингов", r" фунтами-стерлингов", spoken)
    spoken = re.sub(r" фунтам стерлингов", r" фунтам-стерлингов", spoken)
    spoken = re.sub(r" фунтах стерлингов", r" фунтах-стерлингов", spoken)
    spoken = re.sub(r" фунте стерлингов", r" фунте-стерлингов", spoken)
    spoken = re.sub(r" фунту стерлингов", r" фунту-стерлингов", spoken)
    spoken = re.sub(r" фунтом стерлингов", r" фунтом-стерлингов", spoken)

    return spoken


## This function is used only in data preparation (examples/nlp/normalisation_as_tagging/dataset_preparation)
def get_src_and_dst_for_alignment(
    semiotic_class: str, written: str, spoken: str, lang: str
) -> Tuple[str, str, str, str]:
    """Tokenize written and spoken span.
        Args:
            semiotic_class: str - lowercase semiotic class, ex. "cardinal"
            written: str - written form, ex. "2015 году"
            spoken: str - spoken form, ex. "две тысячи пятнадцатом году"
            lang: str - language
        Return:
            src: str - written part, where digits and foreign letters are tokenized by characters, ex. "2 0 1 5"
            dst: str - spoken part tokenized by space, ex. "две тысячи пятнадцатом"
            same_begin: str
            same_end: str
    """
    written = written.casefold()
    # ATTENTION!!! This is INPUT transformation! Need to do the same at inference time!
    spoken = spoken_preprocessing(spoken)

    # remove same fragments at the beginning or at the end of spoken and written form
    written_parts = written.split()
    spoken_parts = spoken.split()
    same_from_begin = 0
    same_from_end = 0
    for i in range(min(len(written_parts), len(spoken_parts))):
        if written_parts[i] == spoken_parts[i]:
            same_from_begin += 1
        else:
            break
    for i in range(min(len(written_parts), len(spoken_parts))):
        if written_parts[-i - 1] == spoken_parts[-i - 1]:
            same_from_end += 1
        else:
            break
    same_begin = written_parts[0:same_from_begin]
    same_end = []
    if same_from_end == 0:
        written = " ".join(written_parts[same_from_begin:])
        spoken = " ".join(spoken_parts[same_from_begin:])
    else:
        written = " ".join(written_parts[same_from_begin:-same_from_end])
        spoken = " ".join(spoken_parts[same_from_begin:-same_from_end])
        same_end = written_parts[-same_from_end:]

    fragments = list(split_text_by_isalpha(written))
    written_tokens = []
    for frag in fragments:
        if frag.isalpha():
            if semiotic_class == "plain" or semiotic_class == "letters" or semiotic_class == "electronic":
                chars = list(frag.strip())
                chars[0] = "_" + chars[0]  # prepend first symbol of a word with underscore
                chars[-1] = chars[-1] + "_"  # append underscore to the last symbol
                written_tokens += chars
            else:
                written_tokens.append("_" + frag + "_")
        else:
            chars = list(frag.strip().replace(" ", ""))
            if len(chars) > 0:
                chars[0] = "_" + chars[0]  # prepend first symbol of a non-alpha fragment with underscore
                chars[-1] = chars[-1] + "_"  # append underscore to the last symbol of a non-alpha fragment
                written_tokens += chars
    written_str = " ".join(written_tokens)

    # _н_ _._ _г_ _._ => _н._ _г._
    written_str = re.sub(
        r"([abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя])_ _\._", r"\g<1>._", written_str
    )
    # _тыс_ _. $ => _тыс._ _$
    written_str = re.sub(
        r"([abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя])_ _\. ([^_])]", r"\g<1>._ _\g<2>", written_str
    )

    if semiotic_class == "ordinal":
        #  _8 2 -_ _ом_  =>  _8 2-ом_
        written_str = re.sub(
            r"([\d]) -_ _([abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя]+)_",
            r"\g<1>-\g<2>_",
            written_str,
        )
        #  _8 8_ _й_       _8 8й_
        written_str = re.sub(
            r"([\d])_ _([abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя]+)_", r"\g<1>\g<2>_", written_str
        )

    if semiotic_class == "cardinal":
        #  _2 5 -_ _ти_ => _2 5-ти_
        written_str = re.sub(r"([\d]) -_ _(ти)_", r"\g<1>-\g<2>_", written_str)
        written_str = re.sub(r"([\d]) -_ _(и)_", r"\g<1>-\g<2>_", written_str)
        written_str = re.sub(r"([\d]) -_ _(мя)_", r"\g<1>-\g<2>_", written_str)
        written_str = re.sub(r"([\d]) -_ _(ех)_", r"\g<1>-\g<2>_", written_str)

    #  _i b m_ _'_ _s_ =>  _i b m's_
    if lang == "en":
        written_str = re.sub(r"_ _'_ _s_", r"'s_", written_str)

    if semiotic_class == "date" and lang == "en":
        #  _1 9 8 0_ _s_ =>  _1 9 8 0s_
        written_str = re.sub(r"([\d])_ _s_", r"\g<1>s_", written_str)
        #  _1 9 5 0 '_ _s_ =>  _1 9 5 0's_
        written_str = re.sub(r"([\d]) '_ _s_", r"\g<1>'s_", written_str)
        #  _wednesday_ _2 6_ _th_ _september_ _2 0 1 2_ =>  _wednesday_ _2 6th_ _september_ _2 0 1 2_
        written_str = re.sub(r"([\d])_ _th_", r"\g<1>th_", written_str)
        #  _wednesday_ _may_ _2 1_ _st_ _, 2 0 1 4_ => _wednesday_ _may_ _2 1st_ _, 2 0 1 4_
        written_str = re.sub(r"([\d])_ _st_", r"\g<1>st_", written_str)
        # _wednesday_ _2 3_ _rd_ _july_ _2 0 1 4_ => _wednesday_ _2 3rd_ _july_ _2 0 1 4_
        written_str = re.sub(r"([\d])_ _rd_", r"\g<1>rd_", written_str)
        # _wednesday_ _2 2_ _nd_ _july_ _2 0 1 4_ => _wednesday_ _2 2nd_ _july_ _2 0 1 4_
        written_str = re.sub(r"([\d])_ _nd_", r"\g<1>nd_", written_str)

        written_str = re.sub(r"_mon_ _\. ", r"_mon._ ", written_str)
        written_str = re.sub(r"_tue_ _\. ", r"_tue._ ", written_str)
        written_str = re.sub(r"_wen_ _\. ", r"_wen._ ", written_str)
        written_str = re.sub(r"_thu_ _\. ", r"_thu._ ", written_str)
        written_str = re.sub(r"_fri_ _\. ", r"_fri._ ", written_str)
        written_str = re.sub(r"_sat_ _\. ", r"_sat._ ", written_str)
        written_str = re.sub(r"_sun_ _\. ", r"_sun._ ", written_str)

        written_str = re.sub(r"_jan_ _\. ", r"_jan._ ", written_str)
        written_str = re.sub(r"_feb_ _\. ", r"_feb._ ", written_str)
        written_str = re.sub(r"_mar_ _\. ", r"_mar._ ", written_str)
        written_str = re.sub(r"_apr_ _\. ", r"_apr._ ", written_str)
        written_str = re.sub(r"_may_ _\. ", r"_may._ ", written_str)
        written_str = re.sub(r"_jun_ _\. ", r"_jun._ ", written_str)
        written_str = re.sub(r"_jul_ _\. ", r"_jul._ ", written_str)
        written_str = re.sub(r"_aug_ _\. ", r"_aug._ ", written_str)
        written_str = re.sub(r"_sep_ _\. ", r"_sep._ ", written_str)
        written_str = re.sub(r"_oct_ _\. ", r"_oct._ ", written_str)
        written_str = re.sub(r"_nov_ _\. ", r"_nov._ ", written_str)
        written_str = re.sub(r"_dec_ _\. ", r"_dec._ ", written_str)

    if semiotic_class == "date" and lang == "ru":
        # _1 8 . 0 8 . 2 0 0 1_  =>  _1 8_ .08. _2 0 0 1_
        # _1 8 / 0 8 / 2 0 0 1_  =>  _1 8_ /08/ _2 0 0 1_
        # _1 8 - 0 8 - 2 0 0 1_  =>  _1 8_ -08- _2 0 0 1_
        written_str = re.sub(r"([\d]) \. ([01]) ([0123456789]) \. ([\d])", r"\g<1>_ .\g<2>\g<3>. _\g<4>", written_str)
        written_str = re.sub(r"([\d]) / ([01]) ([0123456789]) / ([\d])", r"\g<1>_ /\g<2>\g<3>/ _\g<4>", written_str)
        written_str = re.sub(r"([\d]) - ([01]) ([0123456789]) - ([\d])", r"\g<1>_ -\g<2>\g<3>- _\g<4>", written_str)
        # _1 8 . 8 . 2 0 0 1_  =>  _1 8_ .8. _2 0 0 1_
        # _1 8 / 8 / 2 0 0 1_  =>  _1 8_ /8/ _2 0 0 1_
        # _1 8 - 8 - 2 0 0 1_  =>  _1 8_ -8- _2 0 0 1_
        written_str = re.sub(r"([\d]) \. ([123456789]) \. ([\d])", r"\g<1>_ .\g<2>. _\g<3>", written_str)
        written_str = re.sub(r"([\d]) / ([123456789]) / ([\d])", r"\g<1>_ /\g<2>/ _\g<3>", written_str)
        written_str = re.sub(r"([\d]) - ([123456789]) - ([\d])", r"\g<1>_ -\g<2>- _\g<3>", written_str)

    if semiotic_class == "money":
        # if a span start with currency, move it to the end
        #  "_$ 2 5_"  => "_2 5_ _$<<"    #<< means "at post-processing move to the beginning of th semiotic span"
        written_str = re.sub(
            r"^(_[^0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя]) ([\d].*)$",
            r"_\g<2> \g<1><<",
            written_str,
        )

        #  "_us_ _$ 7 0 0_"  => "_us__$ 7 0 0_"
        written_str = re.sub(r"^_us_ _\$ ([\d].*)$", r"_\g<1> _us__$<<", written_str)

        #  "_2 5 $_"  => "_2 5_ _$_"    #insert space between last digit and dollar sign
        written_str = re.sub(
            r"([\d]) ([^0123456789abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя_]_)",
            r"\g<1>_ _\g<2>",
            written_str,
        )

    if semiotic_class == "time":
        # "_pm_ _1 0_" => "_1 0_ _pm_<<"
        written_str = re.sub(r"^(_[ap]m_) (_[\d].*)$", r"\g<2> \g<1><<", written_str)

        # "_8 : 0 0_ _a._ _m._  => _8:00_ _a._ _m._"
        # "_1 2 : 0 0_ _a._ _m._  => _1 2:00_ _a._ _m._"
        written_str = re.sub(r"(\d) [:.] 0 0_", r"\g<1>:00_", written_str)

        # "_2 : 4 2 : 4 4_" => "_2: 4 2: 4 4_"
        written_str = re.sub(r"(\d) [:.] ", r"\g<1>: ", written_str)

    if semiotic_class == "measure":
        #  "_6 5 8_ _см_ _³ ._" => " _6 5 8_ _³> _см._"
        #  > means "at post-processing swap with the next token to the right"
        written_str = re.sub(
            r"(_[abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя.]+_) (_[³²]_?)",
            r"\g<2>> \g<1>",
            written_str,
        )

    return written_str, spoken, " ".join(same_begin), " ".join(same_end)


def fill_alignment_matrix(
    fline2: str, fline3: str, gline2: str, gline3: str
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Parse Giza++ direct and reverse alignment results and represent them as an alignment matrix

    Args:
        fline2: e.g. "_2 0 1 4_"
        fline3: e.g. "NULL ({ }) twenty ({ 1 }) fourteen ({ 2 3 4 })"
        gline2: e.g. "twenty fourteen"
        gline3: e.g. "NULL ({ }) _2 ({ 1 }) 0 ({ }) 1 ({ }) 4_ ({ 2 })"

    Returns:
        matrix: a numpy array of shape (src_len, dst_len) filled with [0, 1, 2, 3], where 3 means a reliable alignment
         the corresponding words were aligned to one another in direct and reverse alignment runs, 1 and 2 mean that the
         words were aligned only in one direction, 0 - no alignment.
        srctokens: e.g. ["twenty", "fourteen"]
        dsttokens: e.g. ["_2", "0", "1", "4_"]

    For example, the alignment matrix for the above example may look like:
    [[3, 0, 0, 0]
     [0, 2, 2, 3]]
    """
    if fline2 is None or gline2 is None or fline3 is None or gline3 is None:
        raise ValueError(f"empty params")
    srctokens = gline2.split()
    dsttokens = fline2.split()
    pattern = r"([^ ]+) \(\{ ([^\(\{\}\)]*) \}\)"
    src2dst = re.findall(pattern, fline3.replace("({ })", "({  })"))
    dst2src = re.findall(pattern, gline3.replace("({ })", "({  })"))
    if len(src2dst) != len(srctokens) + 1:
        raise ValueError(
            "length mismatch: len(src2dst)="
            + str(len(src2dst))
            + "; len(srctokens)"
            + str(len(srctokens))
            + "\n"
            + gline2
            + "\n"
            + fline3
        )
    if len(dst2src) != len(dsttokens) + 1:
        raise ValueError(
            "length mismatch: len(dst2src)="
            + str(len(dst2src))
            + "; len(dsttokens)"
            + str(len(dsttokens))
            + "\n"
            + fline2
            + "\n"
            + gline3
        )
    matrix = np.zeros((len(srctokens), len(dsttokens)))
    for i in range(1, len(src2dst)):
        token, to_str = src2dst[i]
        if to_str == "":
            continue
        to = list(map(int, to_str.split()))
        for t in to:
            matrix[i - 1][t - 1] = 2

    for i in range(1, len(dst2src)):
        token, to_str = dst2src[i]
        if to_str == "":
            continue
        to = list(map(int, to_str.split()))
        for t in to:
            matrix[t - 1][i - 1] += 1

    return matrix, srctokens, dsttokens


def check_monotonicity(matrix: np.ndarray) -> bool:
    """Check if alignment is monotonous - i.e. the relative order is preserved (no swaps).

    Args:
        matrix: a numpy array of shape (src_len, dst_len) filled with [0, 1, 2, 3], where 3 means a reliable alignment
         the corresponding words were aligned to one another in direct and reverse alignment runs, 1 and 2 mean that the
         words were aligned only in one direction, 0 - no alignment.
    """
    is_sorted = lambda k: np.all(k[:-1] <= k[1:])

    a = np.argwhere(matrix == 3)
    b = np.argwhere(matrix == 2)
    c = np.vstack((a, b))
    d = c[c[:, 1].argsort()]  # sort by second column (less important)
    d = d[d[:, 0].argsort(kind="mergesort")]
    return is_sorted(d[:, 1])


def get_targets(matrix: np.ndarray, dsttokens: List[str], delimiter: str) -> List[str]:
    """Join some of the destination tokens, so that their number becomes the same as the number of input words.
    Unaligned tokens tend to join to the left aligned token.

    Args:
        matrix: a numpy array of shape (src_len, dst_len) filled with [0, 1, 2, 3], where 3 means a reliable alignment
         the corresponding words were aligned to one another in direct and reverse alignment runs, 1 and 2 mean that the
         words were aligned only in one direction, 0 - no alignment.
        dsttokens: e.g. ["_2", "0", "1", "4_"]
    Returns:
        targets: list of string tokens, with one-to-one correspondence to matrix.shape[0]

    Example:
        If we get
            matrix=[[3, 0, 0, 0]
                    [0, 2, 2, 3]]
            dsttokens=["_2", "0", "1", "4_"]
        it gives
            targets = ["_201", "4_"]
        Actually, this is a mistake instead of ["_20", "14_"]. That will be further corrected by regular expressions.
    """
    targets = []
    last_covered_dst_id = -1
    for i in range(len(matrix)):
        dstlist = []
        for j in range(last_covered_dst_id + 1, len(dsttokens)):
            # matrix[i][j] == 3: safe alignment point
            if matrix[i][j] == 3 or (
                j == last_covered_dst_id + 1
                and np.all(matrix[i, :] == 0)  # if the whole line does not have safe points
                and np.all(matrix[:, j] == 0)  # and the whole column does not have safe points, match them
            ):
                if len(targets) == 0:  # if this is first safe point, attach left unaligned columns to it, if any
                    for k in range(0, j):
                        if np.all(matrix[:, k] == 0):  # if column k does not have safe points
                            dstlist.append(dsttokens[k])
                        else:
                            break
                dstlist.append(dsttokens[j])
                last_covered_dst_id = j
                for k in range(j + 1, len(dsttokens)):
                    if np.all(matrix[:, k] == 0):  # if column k does not have safe points
                        dstlist.append(dsttokens[k])
                        last_covered_dst_id = k
                    else:
                        break

        if len(dstlist) > 0:
            targets.append(delimiter.join(dstlist))
        else:
            targets.append("<DELETE>")
    return targets


def get_targets_from_back(matrix: np.ndarray, dsttokens: List[str], delimiter: str) -> List[str]:
    """Join some of the destination tokens, so that their number becomes the same as the number of input words.
    Unaligned tokens tend to join to the right aligned token.

    Args:
        matrix: a numpy array of shape (src_len, dst_len) filled with [0, 1, 2, 3], where 3 means a reliable alignment
         the corresponding words were aligned to one another in direct and reverse alignment runs, 1 and 2 mean that the
         words were aligned only in one direction, 0 - no alignment.
        dsttokens: e.g. ["_2", "0", "1", "4_"]
    Returns:
        targets: list of string tokens, with one-to-one correspondence to matrix.shape[0]

    Example:
        If we get
            matrix=[[3, 0, 0, 0]
                    [0, 2, 2, 3]]
            dsttokens=["_2", "0", "1", "4_"]
        it gives
            targets = ["_2", "014_"]
        Actually, this is a mistake instead of ["_20", "14_"]. That will be further corrected by regular expressions.
    """

    targets = []
    last_covered_dst_id = len(dsttokens)
    for i in range(len(matrix) - 1, -1, -1):
        dstlist = []
        for j in range(last_covered_dst_id - 1, -1, -1):
            if matrix[i][j] == 3 or (
                j == last_covered_dst_id - 1 and np.all(matrix[i, :] == 0) and np.all(matrix[:, j] == 0)
            ):
                if len(targets) == 0:
                    for k in range(len(dsttokens) - 1, j, -1):
                        if np.all(matrix[:, k] == 0):
                            dstlist.append(dsttokens[k])
                        else:
                            break
                dstlist.append(dsttokens[j])
                last_covered_dst_id = j
                for k in range(j - 1, -1, -1):
                    if np.all(matrix[:, k] == 0):
                        dstlist.append(dsttokens[k])
                        last_covered_dst_id = k
                    else:
                        break
        if len(dstlist) > 0:
            targets.append(delimiter.join(list(reversed(dstlist))))
        else:
            targets.append("<DELETE>")
    return list(reversed(targets))
