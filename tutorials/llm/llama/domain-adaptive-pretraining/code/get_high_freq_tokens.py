# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import sys
from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np


# +
def binary_search(arr, low, high, bar=0.98):
    total = arr.sum()
    target = total * bar
    print(arr, target)
    if (high - low) >= 2:
        mid = (high + low) // 2
        s = arr[0:mid].sum()
        if s == target:
            return mid
        elif s < target:
            if arr[0 : mid + 1].sum() >= target:
                return mid + 1
            else:
                return binary_search(arr, mid + 1, high, bar)
        else:
            if arr[0 : mid - 1].sum() <= target:
                return mid
            else:
                return binary_search(arr, low, mid - 1, bar)
    else:
        return low


def binary_search2(arr, low, high, bar=0.98):
    arr_csum = np.cumsum(arr)
    total = arr.sum()
    target = int(total * bar)
    print(arr, arr_csum, target)
    i = bisect_left(arr_csum, target)
    if i != len(arr_csum) and arr_csum[i] == target:
        return arr[i]
    else:
        return low


# -


def get_high_freq_tokens(token_usage_path, high_freq_tokens_path, p_th=0.98):
    """
    Function to identify high frequency tokens from previous frequency analysis based on cutoff threshold. Selects the top-K     tokens in a way that their cumulative frequency accounts for approximately 98%.
    Args:
            token_usage_path (str): Path to saved token usage frequency analysis results
            high_freq_tokens_path (str): path to save selected high-frequency tokens (new tokens to be added)
            p_th (float): Frequency Threshold
    Returns:
           Saves a file with high frequency tokens
    """
    f = open(token_usage_path)
    freq_dict = json.load(f)

    topics = []
    p_ths = []
    for key in freq_dict:
        topics.append(key)
        p_ths.append(p_th)

    tokens = {}
    i = 0
    for topic in topics:
        print(topic)
        freq = freq_dict[topic]
        freq_list = freq["new_freq"]
        freqs = []
        ids = []
        for term in freq_list:
            freqs.append(term[-1])
            ids.append(term[0])
        freqs_np = np.array(freqs)
        th = binary_search(freqs_np, freqs_np.min(), freqs_np.max(), bar=p_ths[i])
        print(th)
        i += 1
        if th > 0:
            tokens[topic] = ids[0:th]
        else:
            raise ValueError("Threshold value is not greater than 0")

    L = []
    for key in tokens:
        L = L + tokens[key]
    L = set(L)

    token_category_dict = {}
    for key in freq_dict:
        temp = freq_dict[key]["new_freq"]
        for tok in temp:
            ids = tok[0]
            name = tok[1]
            cate = tok[2]
            if ids in token_category_dict:
                assert name == token_category_dict[ids][1]
            else:
                token_category_dict[ids] = [cate, name]

    add_tokens = []
    for i in L:
        add_tokens.append(token_category_dict[i][1])

    with open(high_freq_tokens_path, "w") as outfile:
        json.dump(add_tokens, outfile)


if __name__ == "__main__":
    token_usage_path = sys.argv[1]  # token usage frequency
    high_freq_tokens_path = sys.argv[2]
    freq_threshold = float(sys.argv[3])
    get_high_freq_tokens(token_usage_path, high_freq_tokens_path, freq_threshold)
