# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import re
import string
import time
import unicodedata
from abc import ABC, abstractmethod
from builtins import str as unicode
from contextlib import contextmanager
from typing import List

import nltk
import torch

from nemo.collections.common.parts.preprocessing import parsers
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

_words_re = re.compile("([a-z\-]+'[a-z\-]+|[a-z\-]+)|([^a-z{}]+)")


def _text_preprocessing(text):
    text = unicode(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text) if unicodedata.category(char) != 'Mn')
    text = text.lower()
    text = re.sub("[^ a-z'\".,?!()\[\]:;\-]", "", text)
    return text


def _word_tokenize(text):
    words = _words_re.findall(text)
    words = [re.sub(r'\s(\d)', r'\1', word[1].upper()) if word[0] == '' else word[0] for word in words]
    return words


def download_corpora():
    # Download NLTK datasets if this class is to be instantiated
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger.zip')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('corpora/cmudict.zip')
    except LookupError:
        nltk.download('cmudict', quiet=True)


class G2p:
    def __init__(
        self,
        g2p_library,
        phoneme_dict_path=None,
        use_seq2seq_for_oov=False,
        ignore_ambiguous_words=True,
        text_preprocessing_func=_text_preprocessing,
        word_tokenize_func=_word_tokenize,
    ):
        self._g2p = g2p_library
        self.homograph2features = self._g2p.homograph2features
        self.g2p_dict = self._construct_grapheme2phoneme_dict(phoneme_dict_path)
        self.use_seq2seq_for_oov = use_seq2seq_for_oov
        self.ignore_ambiguous_words = ignore_ambiguous_words

        self.text_preprocessing_func = text_preprocessing_func
        self.word_tokenize_func = word_tokenize_func

    @staticmethod
    def _construct_grapheme2phoneme_dict(phoneme_dict_path=None, encoding='latin-1'):
        if phoneme_dict_path is None:
            from nltk.corpus import cmudict

            return cmudict.dict()

        _alt_re = re.compile(r'\([0-9]+\)')
        g2p_dict = {}
        with open(phoneme_dict_path, encoding=encoding) as file:
            for line in file:
                if len(line) > 0 and ('A' <= line[0] <= 'Z' or line[0] == "'"):
                    parts = line.split('  ')
                    word = re.sub(_alt_re, '', parts[0])
                    word = word.lower()

                    pronunciation = parts[1].strip().split(" ")
                    if word in g2p_dict:
                        g2p_dict[word].append(pronunciation)
                    else:
                        g2p_dict[word] = [pronunciation]
        return g2p_dict

    def handle_ambiguous(self, word):
        if not self.ignore_ambiguous_words or len(self.g2p_dict[word]) == 1:
            return True
        return False

    def __call__(self, text):
        text = self.text_preprocessing_func(text)
        words = self.word_tokenize_func(text)
        words_and_pos_tags = nltk.pos_tag(words)

        prons = []
        for word, pos in words_and_pos_tags:
            word_by_hyphen = word.split("-")

            # punctuation
            if re.search("[a-zA-Z]", word) is None:
                pron = list(word)
            # homograph
            elif word in self.homograph2features:
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            # `'s` suffix
            elif (
                len(word) > 2
                and word.endswith("'s")
                and (word not in self.g2p_dict)
                and (word[:-2] in self.g2p_dict)
                and self.handle_ambiguous(word[:-2])
            ):
                pron = self.g2p_dict[word[:-2]][0] + ["Z"]
            # `s` suffix
            elif (
                len(word) > 1
                and word.endswith("s")
                and (word not in self.g2p_dict)
                and (word[:-1] in self.g2p_dict)
                and self.handle_ambiguous(word[:-1])
            ):
                pron = self.g2p_dict[word[:-1]][0] + ["Z"]
            # g2p dict
            elif word in self.g2p_dict and self.handle_ambiguous(word):
                pron = self.g2p_dict[word][0]
            # word with hyphens
            elif len(word_by_hyphen) > 1 and all(
                [sub_word in self.g2p_dict and self.handle_ambiguous(sub_word) for sub_word in word_by_hyphen]
            ):
                pron = []
                for sub_word in word_by_hyphen:
                    pron.extend(self.g2p_dict[sub_word][0])
                    pron.extend(["-"])
                pron.pop()
            else:
                if self.use_seq2seq_for_oov:
                    # run gru-based seq2seq model from _g2p
                    pron = self._g2p.predict(word)
                else:
                    pron = word

            prons.extend(pron)

        return prons


class Base(ABC):
    """Vocabulary for turning str text to list of int tokens."""

    # fmt: off
    PUNCT = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on
    PAD, BLANK, OOV = '<pad>', '<blank>', '<oov>'

    def __init__(self, labels, *, pad=PAD, blank=BLANK, oov=OOV, sep='', add_blank_at="last_but_one"):
        super().__init__()

        labels = list(labels)
        self.pad, labels = len(labels), labels + [pad]  # Padding

        if add_blank_at is not None:
            self.blank, labels = len(labels), labels + [blank]  # Reserved for blank from QN
        else:
            # use add_blank_at=None only for ASR where blank is added automatically
            self.blank = -1

        self.oov, labels = len(labels), labels + [oov]  # Out Of Vocabulary

        if add_blank_at == "last":
            labels[-1], labels[-2] = labels[-2], labels[-1]
            self.oov, self.blank = self.blank, self.oov

        self.labels = labels
        self.sep = sep

        self._util_ids = {self.pad, self.blank, self.oov}
        self._label2id = {l: i for i, l in enumerate(labels)}
        self._id2label = labels

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.sep.join(self._id2label[t] for t in tokens if t not in self._util_ids)


class Chars(Base):
    """Chars vocabulary."""

    def __init__(
        self, punct=True, spaces=False, apostrophe=True, add_blank_at="last_but_one",
    ):
        labels = []
        self.space, labels = len(labels), labels + [' ']  # Space
        labels.extend(string.ascii_lowercase)
        if apostrophe:
            labels.append("'")  # Apostrophe for saving "don't" and "Joe's"

        if punct:
            labels.extend(self.PUNCT)

        super().__init__(labels, add_blank_at=add_blank_at)

        self.punct = punct
        self.spaces = spaces

        self._parser = parsers.ENCharParser(labels)

    def encode(self, text):
        """See base class."""
        text = self._parser._normalize(text)  # noqa

        if self.spaces:
            for p in set(text) & set(self.PUNCT):
                text = text.replace(p, f' {p} ')
            text = text.strip().replace('  ', ' ')

        return self._parser._tokenize(text)  # noqa


class Phonemes(Base):
    """Phonemes vocabulary."""

    # fmt: off
    VOWELS = (
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AY', 'EH', 'ER', 'EY', 'IH',
        'IY', 'OW', 'OY', 'UH', 'UW',
    )
    CONSONANTS = (
        'B', 'CH', 'D', 'DH', 'F', 'G',
        'HH', 'JH', 'K', 'L', 'M', 'N',
        'NG', 'P', 'R', 'S', 'SH', 'T',
        'TH', 'V', 'W', 'Y', 'Z', 'ZH',
    )
    # fmt: on

    def __init__(
        self,
        punct=True,
        stresses=False,
        spaces=True,
        chars=False,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        oov=Base.OOV,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at="last_but_one",
        pad_with_space=False,
        improved_version_g2p=False,
        phoneme_dict_path=None,
    ):
        labels = []
        self.space, labels = len(labels), labels + [space]  # Space

        if silence is not None:
            self.silence, labels = len(labels), labels + [silence]  # Silence

        labels.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)

        if stresses:
            vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
        labels.extend(vowels)

        if chars:
            labels.extend(string.ascii_lowercase)

        if apostrophe:
            labels.append("'")  # Apostrophe

        if punct:
            labels.extend(self.PUNCT)

        super().__init__(labels, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.punct = punct
        self.stresses = stresses
        self.spaces = spaces
        self.pad_with_space = pad_with_space

        # g2p_en tries to run download_corpora() on import but it is not rank zero guarded
        # Try to check if torch distributed is available, if not get global rank zero to download corpora and make
        # all other ranks sleep for a minute
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            group = torch.distributed.group.WORLD
            if is_global_rank_zero():
                download_corpora()
            torch.distributed.barrier(group=group)
        elif is_global_rank_zero():
            logging.error(
                f"Torch distributed needs to be initialized before you initialized {self}. This class is prone to "
                "data access race conditions. Now downloading corpora from global rank 0. If other ranks pass this "
                "before rank 0, errors might result."
            )
            download_corpora()
        else:
            logging.error(
                f"Torch distributed needs to be initialized before you initialized {self}. This class is prone to "
                "data access race conditions. This process is not rank 0, and now going to sleep for 1 min. If this "
                "rank wakes from sleep prior to rank 0 finishing downloading, errors might result."
            )
            time.sleep(60)

        import g2p_en  # noqa pylint: disable=import-outside-toplevel

        _g2p = g2p_en.G2p()
        _g2p.variables = None

        if improved_version_g2p:
            self.g2p = G2p(_g2p, phoneme_dict_path)
        else:
            self.g2p = _g2p

    def encode(self, text):
        """See base class."""
        ps, space, labels = [], self.labels[self.space], set(self.labels)

        for p in self.g2p(text):  # noqa
            # Remove stress
            if p.isalnum() and len(p) == 3 and not self.stresses:
                p = p[:2]

            # Add space if last one isn't one
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)

            # Add next phoneme
            if (p.isalnum() or p == "'") and p in labels:
                ps.append(p)

            # Add punct and remove space if needed
            if (p in self.PUNCT) and self.punct:
                if not self.spaces and len(ps) > 0 and ps[-1] == space:
                    ps.pop()
                ps.append(p)

        # Remove trailing spaces
        while ps[-1] == space:
            ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._label2id[p] for p in ps]

    @contextmanager
    def set_phone_prob(self, prob=None):
        # Add do nothing since this class doesn't support mixed g2p
        yield
