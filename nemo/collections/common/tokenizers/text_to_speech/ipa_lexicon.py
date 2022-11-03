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


# fmt: off

SUPPORTED_LOCALES = ["en-US", "de-DE", "es-ES"]

DEFAULT_PUNCTUATION = (
    ',', '.', '!', '?', '-',
    ':', ';', '/', '"', '(',
    ')', '[', ']', '{', '}',
)

GRAPHEME_CHARACTER_SETS = {
    "en-US": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z'
    ),
    "es-ES": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'Á', 'É', 'Í', 'Ñ',
        'Ó', 'Ú', 'Ü'
    ),
    "de-DE": (
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z', 'Ä', 'É', 'Ê', 'Ñ',
        'Ö', 'Ü'
    ),
}

IPA_CHARACTER_SETS = {
    "en-US": (
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w',
        'x', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ',
        'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ',
        'ʒ', 'ʔ', 'ʲ', '̃', '̩', 'θ', 'ᵻ'
    ),
    "es-ES": (
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'x',
        'ð', 'ŋ', 'ɛ', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɾ', 'ʃ', 'ʊ',
        'ʎ', 'ʒ', 'ʝ', 'β', 'θ'
    ),
    "de-DE": (
        '1', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z', 'ç', 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ',
        'ɒ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ',
        'ʊ', 'ʌ', 'ʒ', '̃', 'θ'
    )
}

# fmt: on


def validate_locale(locale):
    if locale not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale '{locale}'. " f"Supported locales {SUPPORTED_LOCALES}")


def get_grapheme_character_set(locale):
    if locale not in GRAPHEME_CHARACTER_SETS:
        raise ValueError(
            f"Grapheme character set not found for locale '{locale}'. "
            f"Supported locales {GRAPHEME_CHARACTER_SETS.keys()}"
        )
    char_set = set(GRAPHEME_CHARACTER_SETS[locale])
    return char_set


def get_ipa_character_set(locale):
    if locale not in IPA_CHARACTER_SETS:
        raise ValueError(
            f"IPA character set not found for locale '{locale}'. " f"Supported locales {IPA_CHARACTER_SETS.keys()}"
        )
    char_set = set(IPA_CHARACTER_SETS[locale])
    return char_set


def get_ipa_punctuation_list(locale):
    if locale is None:
        return sorted(list(DEFAULT_PUNCTUATION))

    validate_locale(locale)

    punct_set = set(DEFAULT_PUNCTUATION)
    if locale in ["de-DE", "es-ES"]:
        # https://en.wikipedia.org/wiki/Guillemet#Uses
        punct_set.update(['«', '»', '‹', '›'])
    if locale == "de-DE":
        punct_set.update(['„', '“'])
    elif locale == "es-ES":
        punct_set.update(['¿', '¡'])

    punct_list = sorted(list(punct_set))
    return punct_list
