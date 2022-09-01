# -*- coding: utf-8 -*-
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

import abc
import ctypes.util
import itertools
import string
from contextlib import contextmanager
from logging import ERROR, getLogger
from typing import List

from nemo_text_processing.g2p.data.data_utils import (
    english_text_preprocessing,
    german_text_preprocessing,
    ipa_word_tokenize,
)
from nemo_text_processing.g2p.modules import IPAG2P
from phonemizer.backend import EspeakBackend

from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import (
    get_ipa_character_list,
    get_ipa_punctuation_list,
)
from nemo.utils import logging
from nemo.utils.decorators import experimental


class BaseTokenizer(abc.ABC):
    PAD, BLANK, OOV = '<pad>', '<blank>', '<oov>'

    def __init__(self, tokens, *, pad=PAD, blank=BLANK, oov=OOV, sep='', add_blank_at=None):
        """Abstract class for creating an arbitrary tokenizer to convert string to list of int tokens.
        Args:
            tokens: List of tokens.
            pad: Pad token as string.
            blank: Blank token as string.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
        """
        super().__init__()

        tokens = list(tokens)
        self.pad, tokens = len(tokens), tokens + [pad]  # Padding

        if add_blank_at is not None:
            self.blank, tokens = len(tokens), tokens + [blank]  # Reserved for blank from asr-model
        else:
            # use add_blank_at=None only for ASR where blank is added automatically, disable blank here
            self.blank = None

        self.oov, tokens = len(tokens), tokens + [oov]  # Out Of Vocabulary

        if add_blank_at == "last":
            tokens[-1], tokens[-2] = tokens[-2], tokens[-1]
            self.oov, self.blank = self.blank, self.oov

        self.tokens = tokens
        self.sep = sep

        self._util_ids = {self.pad, self.blank, self.oov}
        self._token2id = {l: i for i, l in enumerate(tokens)}
        self._id2token = tokens

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @abc.abstractmethod
    def encode(self, text: str) -> List[int]:
        """Turns str text into int tokens."""
        pass

    def decode(self, tokens: List[int]) -> str:
        """Turns ints tokens into str text."""
        return self.sep.join(self._id2token[t] for t in tokens if t not in self._util_ids)


class BaseCharsTokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on

    def __init__(
        self,
        chars,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=lambda x: x,
    ):
        """Base class for char-based tokenizer.
        Args:
            chars: string that represents all possible characters.
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
        """

        tokens = []
        self.space, tokens = len(tokens), tokens + [' ']  # Space
        tokens.extend(chars)
        if apostrophe:
            tokens.append("'")  # Apostrophe for saving "don't" and "Joe's"

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, add_blank_at=add_blank_at)

        self.punct = punct
        self.pad_with_space = pad_with_space

        self.text_preprocessing_func = text_preprocessing_func

    def encode(self, text):
        """See base class."""
        cs, space, tokens = [], self.tokens[self.space], set(self.tokens)

        text = self.text_preprocessing_func(text)
        for c in text:
            # Add space if last one isn't one
            if c == space and len(cs) > 0 and cs[-1] != space:
                cs.append(c)
            # Add next char
            elif (c.isalnum() or c == "'") and c in tokens:
                cs.append(c)
            # Add punct
            elif (c in self.PUNCT_LIST) and self.punct:
                cs.append(c)
            # Warn about unknown char
            elif c != space:
                logging.warning(f"Text: [{text}] contains unknown char: [{c}]. Symbol will be skipped.")

        # Remove trailing spaces
        if cs:
            while cs[-1] == space:
                cs.pop()

        if self.pad_with_space:
            cs = [space] + cs + [space]

        return [self._token2id[p] for p in cs]


class EnglishCharsTokenizer(BaseCharsTokenizer):
    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=english_text_preprocessing,
    ):
        """English char-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones and apply lower() function.
        """
        super().__init__(
            chars=string.ascii_lowercase,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )


class GermanCharsTokenizer(BaseCharsTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on

    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=german_text_preprocessing,
        phonemes=True,
    ):
        """Deutsch char-based tokenizer.
        Args:
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Currently, it only applies lower() function.
        """

        de_alphabet = "abcdefghijklmnopqrstuvwxyzäöüß"
        if phonemes:
            de_ipa = "ʊʃŋɜːɛɾəɪçɔøɡœɑÜ„1Q̃ɒʒÄɹÖʌθàó̈ðéɐá"
            de_alphabet += de_ipa
        super().__init__(
            chars=de_alphabet,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )


class EnglishPhonemesTokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
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
        g2p,
        punct=True,
        non_default_punct_list=None,
        stresses=False,
        chars=False,
        *,
        space=' ',
        silence=None,
        apostrophe=True,
        oov=BaseTokenizer.OOV,
        sep='|',  # To be able to distinguish between 2/3 letters codes.
        add_blank_at=None,
        pad_with_space=False,
        text_preprocessing_func=lambda text: english_text_preprocessing(text, lower=False),
    ):
        """English phoneme-based tokenizer.
        Args:
            g2p: Grapheme to phoneme module.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            stresses: Whether to use phonemes codes with stresses (0-2) or not.
            chars: Whether to additionally use chars together with phonemes. It is useful if g2p module can return chars too.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones.
             Note that lower() function shouldn't applied here, in case the text contains phonemes (it will be handled by g2p).
        """

        self.phoneme_probability = None
        if hasattr(g2p, "phoneme_probability"):
            self.phoneme_probability = g2p.phoneme_probability
        tokens = []
        self.space, tokens = len(tokens), tokens + [space]  # Space

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]  # Silence

        tokens.extend(self.CONSONANTS)
        vowels = list(self.VOWELS)

        if stresses:
            vowels = [f'{p}{s}' for p, s in itertools.product(vowels, (0, 1, 2))]
        tokens.extend(vowels)

        if chars or self.phoneme_probability is not None:
            if not chars:
                logging.warning(
                    "phoneme_probability was not None, characters will be enabled even though "
                    "chars was set to False."
                )
            tokens.extend(string.ascii_lowercase)

        if apostrophe:
            tokens.append("'")  # Apostrophe

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.chars = chars if self.phoneme_probability is None else True
        self.punct = punct
        self.stresses = stresses
        self.pad_with_space = pad_with_space

        self.text_preprocessing_func = text_preprocessing_func
        self.g2p = g2p

    def encode(self, text):
        """See base class."""
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)

        text = self.text_preprocessing_func(text)
        g2p_text = self.g2p(text)  # TODO: handle infer

        for p in g2p_text:  # noqa
            # Remove stress
            if p.isalnum() and len(p) == 3 and not self.stresses:
                p = p[:2]

            # Add space if last one isn't one
            if p == space and len(ps) > 0 and ps[-1] != space:
                ps.append(p)
            # Add next phoneme or char (if chars=True)
            elif (p.isalnum() or p == "'") and p in tokens:
                ps.append(p)
            # Add punct
            elif (p in self.PUNCT_LIST) and self.punct:
                ps.append(p)
            # Warn about unknown char/phoneme
            elif p != space:
                logging.warning(
                    f"Text: [{''.join(g2p_text)}] contains unknown char/phoneme: [{p}]. Original text: [{text}]. Symbol will be skipped."
                )

        # Remove trailing spaces
        if ps:
            while ps[-1] == space:
                ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        return [self._token2id[p] for p in ps]

    @contextmanager
    def set_phone_prob(self, prob):
        if hasattr(self.g2p, "phoneme_probability"):
            self.g2p.phoneme_probability = prob
        try:
            yield
        finally:
            if hasattr(self.g2p, "phoneme_probability"):
                self.g2p.phoneme_probability = self.phoneme_probability


@experimental
class IPATokenizer(BaseTokenizer):
    # fmt: off
    PUNCT_LIST = (  # Derived from LJSpeech and "/" additionally
        ',', '.', '!', '?', '-',
        ':', ';', '/', '"', '(',
        ')', '[', ']', '{', '}',
    )
    # fmt: on

    def __init__(
        self,
        g2p,
        punct=True,
        non_default_punct_list=None,
        chars=False,
        *,
        space=' ',
        silence=None,
        apostrophe=False,
        oov=BaseTokenizer.OOV,
        sep='|',  # To be able to distinguish between symbols
        add_blank_at=None,
        pad_with_space=False,
        text_preprocessing_func=lambda text: english_text_preprocessing(text, lower=False),
    ):
        """General-purpose IPA-based tokenizer.
        Args:
            g2p: Grapheme to phoneme module, should be IPAG2P or some subclass thereof.
            punct: Whether to reserve grapheme for basic punctuation or not.
            non_default_punct_list: List of punctuation marks which will be used instead default, if any.
            chars: Whether to additionally use chars together with phonemes. It is useful if g2p module can return chars too.
            space: Space token as string.
            silence: Silence token as string (will be disabled if it is None).
            apostrophe: Whether to use apostrophe or not.
            oov: OOV token as string.
            sep: Separation token as string.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
             Basically, it replaces all non-unicode characters with unicode ones.
             Note that lower() function shouldn't applied here, in case the text contains phonemes (it will be handled by g2p).
             Defaults to English text preprocessing.
        """
        if not hasattr(g2p, "symbols"):
            logging.error(
                f"Please make sure the G2P module passed into the IPATokenizer has a `symbols` attribute. "
                f"This is required in order to build the tokenizer vocabulary.\n"
                f"Expected e.g. IPAG2P, found {type(g2p)}"
            )
            raise ValueError("G2P modules passed into the IPATokenizer must have `symbols` defined.")

        self.phoneme_probability = None
        if hasattr(g2p, "phoneme_probability"):
            self.phoneme_probability = g2p.phoneme_probability

        # Build tokens list
        tokens = sorted(list(g2p.symbols))  # Sort to ensure that vocab is in the same order every time

        if chars or self.phoneme_probability is not None:
            if not chars:
                logging.warning(
                    "phoneme_probability was not None, characters will be enabled even though "
                    "chars was set to False."
                )

            # Add uppercase chars if G2P class uses them, otherwise use lowercase ones
            if g2p.set_graphemes_upper:
                tokens.extend(string.ascii_uppercase)
            else:
                tokens.extend(string.ascii_lowercase)

        if space in g2p.symbols:
            self.space = tokens.index(space)
        else:
            self.space, tokens = len(tokens), tokens + [space]

        if silence is not None:
            self.silence, tokens = len(tokens), tokens + [silence]

        if apostrophe:
            tokens.append("'")

        if punct:
            if non_default_punct_list is not None:
                self.PUNCT_LIST = non_default_punct_list
            tokens.extend(self.PUNCT_LIST)

        super().__init__(tokens, oov=oov, sep=sep, add_blank_at=add_blank_at)

        self.chars = chars if self.phoneme_probability is None else True
        self.punct = punct
        self.pad_with_space = pad_with_space

        self.text_preprocessing_func = text_preprocessing_func
        self.g2p = g2p

    def encode(self, text):
        """See base class for more information."""
        ps, space, tokens = [], self.tokens[self.space], set(self.tokens)

        text = self.text_preprocessing_func(text)
        g2p_text = self.g2p(text)  # Double-check this

        for p in g2p_text:
            if p == space and len(ps) > 0 and ps[-1] != space:
                # Add space if last token isn't one
                ps.append(p)
            elif p in tokens:
                # Add next phoneme or char (if chars=True)
                ps.append(p)
            elif (p in self.PUNCT_LIST) and self.punct:
                # Add punct
                ps.append(p)
            elif p != space:
                logging.warning(
                    f"Text: [{''.join(g2p_text)}] contains unknown char/phoneme: [{p}]."
                    f"Original text: [{text}]. Symbol will be skipped."
                )

        # Remove trailing spaces
        if ps:
            while ps[-1] == space:
                ps.pop()

        if self.pad_with_space:
            ps = [space] + ps + [space]

        # Token index lookups
        return [self._token2id[p] for p in ps]

    @contextmanager
    def set_phone_prob(self, prob):
        if hasattr(self.g2p, "phoneme_probability"):
            self.g2p.phoneme_probability = prob
        try:
            yield
        finally:
            if hasattr(self.g2p, "phoneme_probability"):
                self.g2p.phoneme_probability = self.phoneme_probability


@experimental
class PhonemizerTokenizer(IPATokenizer):
    # fmt: off
    ESPEAK_SYMBOLS = ('ˈ', 'ˌ', "'", '-', 'ː')
    # fmt: on

    def __init__(
        self,
        language,
        phoneme_dict,
        punct=True,
        pad_with_space=False,
        non_default_chars=None,
        non_default_punct_list=None,
        with_stress=True,
        ignore_ambiguous_words=True,
        heteronyms=None,
        espeak_logging_level=ERROR,
    ):
        """IPA tokenizer using the Phonemizer library with an eSpeak backend

        https://github.com/bootphon/phonemizer
        http://espeak.sourceforge.net/

        To use this class you will need to install eSpeak separately from NeMo.

        For Linux: apt-get update && apt-get install espeak-ng
        Other OS: https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#installation

        eSpeak provides out-of-the-box text normalization & g2p for a large number of languages.
        This tokenizer incorporates that functionality with a phoneme dictionary that allows overriding
        pronunciation for specific words that eSpeak may get wrong.

        This class currently supports: English, Spanish, and German. For other languages you will have to provide
        the character set for the language using "non_default_chars" and "non_default_punct_list".

        Args:
            language: Language string http://espeak.sourceforge.net/languages.html
            phoneme_dict (str, Path, Dict): Path to file in CMUdict format or dictionary of CMUdict-like entries.
                (For default behavior, provide local path to scripts/tts_dataset_files/ipa_dict-default.txt)
            punct: Whether to keep punctuation or remove it.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_chars: List of characters which will be used instead of default.
            non_default_punct_list: List of punctuation marks which will be used instead of default.
            with_stress: Whether to include stresses/accents in the IPA output.
            ignore_ambiguous_words: Whether to not handle word via phoneme_dict with ambiguous phoneme sequences.
            heteronyms (str, Path, List): Path to file with heteronyms (every line is new word) or list of words.
            espeak_logging_level: Logging level for eSpeak.
                Defaults to 'logging.ERROR' (to avoid a lot of unhelpful warning logs).
        """

        if not (ctypes.util.find_library('espeak-ng') or ctypes.util.find_library('espeak')):
            raise ImportError(
                "PhonemizerTokenizer requires eSpeak to be installed. "
                "Please follow the instructions at: "
                "https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#installation"
            )

        if non_default_chars:
            char_list = non_default_chars
        else:
            char_list = get_ipa_character_list(language)
            if with_stress:
                char_list.extend(self.ESPEAK_SYMBOLS)
        chars = "".join(char_list)

        if not punct:
            punct_list = []
        elif non_default_punct_list:
            punct_list = non_default_punct_list
        else:
            punct_list = get_ipa_punctuation_list(language)

        espeak_logger = getLogger('espeak')
        espeak_logger.setLevel(espeak_logging_level)
        self.espeak_backend = EspeakBackend(
            language=language, preserve_punctuation=punct, with_stress=with_stress, logger=espeak_logger
        )

        ipa_g2p = IPAG2P(
            phoneme_dict=phoneme_dict,
            apply_to_oov_word=self._text_to_phonemes_espeak,
            word_tokenize_func=ipa_word_tokenize,
            ignore_ambiguous_words=ignore_ambiguous_words,
            heteronyms=heteronyms,
        )
        # Merge chars with all symbols found in the phoneme dictionary.
        ipa_g2p.symbols.update(chars)

        super().__init__(
            g2p=ipa_g2p,
            punct=punct,
            non_default_punct_list=punct_list,
            pad_with_space=pad_with_space,
            text_preprocessing_func=lambda text: text,
        )

    def _text_to_phonemes_espeak(self, text):
        phonemes = self.espeak_backend.phonemize([text])[0]
        phonemes = phonemes.strip()
        return phonemes

    def text_to_phonemes(self, text):
        tokens = self.encode(text)
        phoneme_string = self.decode(tokens)
        phonemes = phoneme_string.replace(self.sep, '')
        return phonemes
