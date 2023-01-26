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

import csv
import json
import os
import pickle
import random
import re
import string
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm

from nemo.utils import logging
from nemo.utils.env_var_parsing import get_envint

__all__ = [
    "DataProcessor",
    "get_label_stats",
    "get_multi_label_stats",
    "partition_data",
    "write_files",
    "write_data",
    "create_dataset",
    "read_csv",
    "get_dataset",
    "partition",
    "map_entities",
    "get_entities",
    "get_data",
    "reverse_dict",
    "get_intent_labels",
    "get_stats",
    "DATABASE_EXISTS_TMP",
    "MODE_EXISTS_TMP",
    "is_whitespace",
    "write_vocab",
    "if_exist",
    "remove_punctuation_from_sentence",
    "dataset_to_ids",
    "get_freq_weights",
    "get_freq_weights_bce_with_logits_loss",
    "fill_class_weights",
    "normalize_answer",
    "get_labels_to_labels_id_mapping",
    "get_vocab",
    "find_newlines",
    "load_data_indices",
    "chinese_punctuation",
    "check_chinese_char",
    "normalize_chinese_answer",
]

DATABASE_EXISTS_TMP = "{} dataset has already been processed and stored at {}"
MODE_EXISTS_TMP = "{} mode of {} dataset has already been processed and stored at {}"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                # if sys.version_info[0] == 2:
                #     line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


chinese_punctuation = {
    "——",
    "‘",
    "’",
    "“",
    "”",
    "…",
    "、",
    "。",
    "〈",
    "〉",
    "《",
    "》",
    "「",
    "」",
    "『",
    "』",
    "【",
    "】",
    "〔",
    "〕",
    "！",
    "（",
    "）",
    "，",
    "．",
    "：",
    "；",
    "？",
}


def check_chinese_char(ch):
    """Check if a character is in Chinese."""
    if "\u4e00" <= ch <= "\u9fff" or ch in chinese_punctuation:
        return True
    else:
        return False


def normalize_chinese_answer(text):
    """Remove the Chinese punctuation and separate Chinese answers to char-level"""

    def remove_punc(text):
        exclude = chinese_punctuation
        return "".join(ch for ch in text if ch not in exclude)

    def separate_char(text):
        ch_list = []
        for ch in text:
            ch_list.append(ch)
        return ch_list

    return separate_char(remove_punc(text))


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_label_stats(labels, outfile="stats.tsv", verbose=True):
    """
    Args:
        labels: list of all labels
        outfile: path to the file where to save label stats
    Returns:
        total (int): total number of labels
        label_frequencies (list of tuples): each tuple represent (label, label frequency)
        max id of the labels
    """
    labels = Counter(labels)
    total = sum(labels.values())
    out = open(outfile, "w")
    i = 0
    freq_dict = {}
    label_frequencies = labels.most_common()
    for k, v in label_frequencies:
        out.write(f"{k}\t\t{round(v/total,5)}\t\t{v}\n")
        if verbose and i < 3:
            logging.info(f"label: {k}, {v} out of {total} ({(v / total)*100.0:.2f}%).")
        i += 1
        freq_dict[k] = v

    return total, freq_dict, max(labels.keys())


def get_multi_label_stats(labels, outfile="stats.tsv", verbose=True):
    """
    Args:
        labels: list of tuples containing labels for each utterance
            Example: If there are 5 intents in total, then (0,1,1,1,0) represents the labels 
                     for an individual utterance. (0,1,1,1,0) indicates that the utterance has labels
                     at index/line 1,2, and 3 in dict.intents. The list of tuples contain labels for 
                     all utterances.

        outfile: path to the file where to save label stats

    Returns:
        total (int): total number of labels
        freq_dict (list of tuples): each tuple represents class counts in the form of (negative, positive)
    """
    total = len(labels)
    positive_class_total = 0
    class_count_dict = {}

    # Get the count of each label in the label dictionary, both the positive and negative classes
    for label in labels:
        for label_index, val in enumerate(label):
            if label_index not in class_count_dict:
                class_count_dict[label_index] = [0, 0]

            if val == 1:
                positive_class_total += 1
                class_count_dict[label_index][1] += 1
            else:
                class_count_dict[label_index][0] += 1

    if verbose:
        three_most_frequent_classes = sorted(class_count_dict, key=lambda idx: class_count_dict[idx][1], reverse=True)

        for cnt, idx in enumerate(three_most_frequent_classes):
            if cnt > 2:
                break

            positives = class_count_dict[idx][1]
            logging.info(
                f"label: {idx}, {positives} out of {positive_class_total} ({(positives / positive_class_total)*100.0:.2f}%)."
            )

    return total, class_count_dict, len(labels[0]) - 1


def partition_data(intent_queries, slot_tags, split=0.1):
    n = len(intent_queries)
    n_dev = int(n * split)
    dev_idx = set(random.sample(range(n), n_dev))
    dev_intents, dev_slots, train_intents, train_slots = [], [], [], []

    dev_intents.append("sentence\tlabel\n")
    train_intents.append("sentence\tlabel\n")

    for i, item in enumerate(intent_queries):
        if i in dev_idx:
            dev_intents.append(item)
            dev_slots.append(slot_tags[i])
        else:
            train_intents.append(item)
            train_slots.append(slot_tags[i])
    return train_intents, train_slots, dev_intents, dev_slots


def write_files(data, outfile):
    with open(outfile, "w") as f:
        for item in data:
            item = f"{item.strip()}\n"
            f.write(item)


def write_data(data, slot_dict, intent_dict, outfold, mode, uncased):
    intent_file = open(f"{outfold}/{mode}.tsv", "w")
    intent_file.write("sentence\tlabel\n")
    slot_file = open(f"{outfold}/{mode}_slots.tsv", "w")
    for tokens, slots, intent in data:
        text = " ".join(tokens)
        if uncased:
            text = text.lower()
        intent_file.write(f"{text}\t{intent_dict[intent]}\n")
        slots = [str(slot_dict[slot]) for slot in slots]
        slot_file.write(" ".join(slots) + "\n")
    intent_file.close()
    slot_file.close()


def create_dataset(train, dev, slots, intents, uncased, outfold):
    os.makedirs(outfold, exist_ok=True)
    if "O" in slots:
        slots.remove("O")
    slots = sorted(list(slots)) + ["O"]
    intents = sorted(list(intents))
    slots = write_vocab(slots, f"{outfold}/dict.slots.csv")
    intents = write_vocab(intents, f"{outfold}/dict.intents.csv")
    write_data(train, slots, intents, outfold, "train", uncased)
    write_data(dev, slots, intents, outfold, "test", uncased)


def read_csv(file_path):
    rows = []
    with open(file_path, "r") as csvfile:
        read_csv = csv.reader(csvfile, delimiter=",")
        for row in read_csv:
            rows.append(row)
    return rows


def get_dataset(files, dev_split=0.1):
    # entity2value, value2entity = get_entities(files)
    data, slots, intents = get_data(files)
    if len(data) == 1:
        train, dev = partition(data[0], split=dev_split)
    else:
        train, dev = data[0], data[1]
    return train, dev, slots, intents


def partition(data, split=0.1):
    n = len(data)
    n_dev = int(n * split)
    dev_idx = set(random.sample(range(n), n_dev))
    dev, train = [], []

    for i, item in enumerate(data):
        if i in dev_idx:
            dev.append(item)
        else:
            train.append(item)
    return train, dev


def map_entities(entity2value, entities):
    for key in entities:
        if "data" in entities[key]:
            if key not in entity2value:
                entity2value[key] = set([])

            values = []
            for value in entities[key]["data"]:
                values.append(value["value"])
                values.extend(value["synonyms"])
            entity2value[key] = entity2value[key] | set(values)

    return entity2value


def get_entities(files):
    entity2value = {}
    for file in files:
        with open(file, "r") as json_file:
            data = json.load(json_file)
            entity2value = map_entities(entity2value, data["entities"])

    value2entity = reverse_dict(entity2value)
    return entity2value, value2entity


def get_data(files):
    all_data, all_slots, all_intents = [], set(["O"]), set()
    for file in files:
        file_data = []
        with open(file, "r") as json_file:
            data = json.load(json_file)
            for intent in data["intents"]:
                all_intents.add(intent)
                utterances = data["intents"][intent]["utterances"]
                for utterance in utterances:
                    tokens, slots = [], []
                    for frag in utterance["data"]:
                        frag_tokens = frag["text"].strip().split()
                        tokens.extend(frag_tokens)
                        if "slot_name" not in frag:
                            slot = "O"
                        else:
                            slot = frag["slot_name"]
                            all_slots.add(slot)
                        slots.extend([slot] * len(frag_tokens))
                    file_data.append((tokens, slots, intent))
        all_data.append(file_data)
    return all_data, all_slots, all_intents


def reverse_dict(entity2value):
    value2entity = {}
    for entity in entity2value:
        for value in entity2value[entity]:
            value2entity[value] = entity
    return value2entity


def get_intent_labels(intent_file):
    labels = {}
    label = 0
    with open(intent_file, "r") as f:
        for line in f:
            intent = line.strip()
            labels[intent] = label
            label += 1
    return labels


def get_stats(lengths):
    logging.info("Some stats of the lengths of the sequences:")
    lengths = np.asarray(lengths)
    logging.info(
        f"Min: {np.min(lengths)} | \
                 Max: {np.max(lengths)} | \
                 Mean: {np.mean(lengths)} | \
                 Median: {np.median(lengths)}"
    )
    logging.info(f"75 percentile: {np.percentile(lengths, 75):.2f}")
    logging.info(f"99 percentile: {np.percentile(lengths, 99):.2f}")


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def write_vocab(items, outfile):
    vocab = {}
    idx = 0
    with open(outfile, "w") as f:
        for item in items:
            f.write(item + "\n")
            vocab[item] = idx
            idx += 1
    return vocab


def get_labels_to_labels_id_mapping(file):
    """
    Reads labels from the file and returns labels to id mapping dictionary
    Args:
        file: path to file
    Returns:
        labels to id mapping dictionary
    """
    lines = open(file, "r").readlines()
    lines = [line.strip() for line in lines if line.strip()]
    label_ids = {lines[i]: i for i in range(len(lines))}
    return label_ids


def if_exist(outfold, files):
    if not os.path.exists(outfold):
        return False
    for file in files:
        if not os.path.exists(f"{outfold}/{file}"):
            return False
    return True


def remove_punctuation_from_sentence(sentence):
    sentence = re.sub("[" + string.punctuation + "]", "", sentence)
    sentence = sentence.lower()
    return sentence


def dataset_to_ids(
    dataset,
    tokenizer,
    cache_ids=False,
    add_bos_eos=True,
    cache_data_per_node=False,
    use_cache=False,
    remove_trailing_newline=False,
):
    """
    Reads dataset from file line by line, tokenizes each line with tokenizer,
    and returns list of lists which corresponds to ids of tokenized strings.

    Args:
        dataset (str): path to dataset
        tokenizer: tokenizer to convert text into ids
        cache_ids (bool): if True, ids are saved to disk as pickle file
            with similar name (e.g., data.txt --> data.txt.pkl)
        add_bos_eos (bool): whether to add <s> and </s> symbols (e.g., for NMT)
        cache_data_per_node (bool): Cache data on local_rank 0. Use when there is not a shared-filesystem.
        use_cache (bool): Use cached ids if they exist.
        remove_trailing_newline (bool): Remove trailing newline character.
    Returns:
        ids: list of ids which correspond to tokenized strings of the dataset
    """

    cached_ids_dataset = dataset + str(".pkl")
    if use_cache and os.path.isfile(cached_ids_dataset):
        logging.info("Loading cached tokenized dataset ...")
        ids = pickle.load(open(cached_ids_dataset, "rb"))
    else:
        logging.info(f"Tokenizing dataset {dataset}...")
        data = open(dataset, "rb").readlines()
        ids = []
        for sentence in tqdm(data, desc="Tokenizing sentence"):
            text = sentence.decode("utf-8")
            if remove_trailing_newline:
                text = text.rstrip("\n")
            sent_ids = tokenizer.text_to_ids(text)
            if add_bos_eos:
                sent_ids = [tokenizer.bos_id] + sent_ids + [tokenizer.eos_id]
            ids.append(sent_ids)
        if cache_ids and (
            not torch.distributed.is_initialized() or (cache_data_per_node and get_envint("LOCAL_RANK", 0) == 0)
        ):
            logging.info("Caching tokenized dataset ...")
            pickle.dump(ids, open(cached_ids_dataset, "wb"))
    return ids


def get_freq_weights(label_freq):
    """
    Goal is to give more weight to the classes with less samples
    so as to match the ones with the higher frequencies. We achieve this by
    dividing the total frequency by the freq of each label to calculate its weight.
    """
    total_size = 0
    for lf in label_freq.values():
        total_size += lf
    weighted_slots = {label: (total_size / (len(label_freq) * freq)) for label, freq in label_freq.items()}

    return weighted_slots


def get_freq_weights_bce_with_logits_loss(label_freq):
    """
    Calculate positive class weights to be passed to BCEWithLogitsLoss
    https://pytorch.org/docs/1.9.1/generated/torch.nn.BCEWithLogitsLoss.html

    Args: 
        label_freq: dictionary of tuples where keys represents class id, and tuple represents counts of positive and negative classes,
                    positive classes are at index 1 and negative at index 0
    Returns:
        weights: dictionary of labels with their weights
    """
    weights = {}

    for label_id, class_values in label_freq.items():
        positive_class = class_values[1]
        negative_class = class_values[0]

        if positive_class == 0:
            weights[label_id] = 0

        else:
            weights[label_id] = float(negative_class) / float(positive_class)

    return weights


def fill_class_weights(weights, max_id=-1):
    """
    Gets a dictionary of labels with their weights and creates a list with size of the labels filled with those weights.
    Missing labels in the dictionary would get value 1.

    Args:
        weights: dictionary of weights for labels, labels as keys and weights are their values
        max_id: the largest label id in the dataset, default=-1 would consider the largest label in the weights dictionary as max_id
    Returns:
        weights_list: list of weights for labels
    """
    if max_id < 0:
        max_id = 0
        for l in weights.keys():
            max_id = max(max_id, l)

    all_weights = [1.0] * (max_id + 1)
    for i in range(len(all_weights)):
        if i in weights:
            all_weights[i] = weights[i]
    return all_weights


def get_vocab(file):
    lines = open(file, "r").readlines()
    lines = [line.strip() for line in lines if line.strip()]
    labels = {i: lines[i] for i in range(len(lines))}
    return labels


def find_newlines(contents):
    """
    Finds all of the newline positions in a text file.
    """
    start = 0

    while True:
        try:
            # index and split are much faster than Python for loops
            new_start = contents.index(b"\n", start)
            line = (
                contents[start:new_start]
                .replace(b"\xc2\x99", b" ")
                .replace(b"\xc2\xa0", b" ")
                .decode("utf-8", errors="ignore")
            )

            if len(line.split()) > 0:
                yield start

            start = new_start + 1

        except ValueError:
            break


def load_data_indices(idx_file: str, data_file: str, savename: str):
    """
    Loads dataset index file if it exsits
    """
    data_dir = data_file[: data_file.rfind("/")]
    mode = data_file[data_file.rfind("/") + 1 : data_file.rfind(".")]
    idx_file = f"{data_dir}/{mode}_{savename}.pkl"

    if os.path.isfile(idx_file):
        # If the sentence indices file already exists, load from it
        with open(idx_file, "rb") as f:
            indices = pickle.load(f)

            return indices, idx_file, data_dir

    return None, idx_file, data_dir
