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
DEFAULT_PUNCTUATION = (
    ',', '.', '!', '?', '-',
    ':', ';', '/', '"', '(',
    ')', '[', ']', '{', '}',
)


CHARACTER_SETS = {
    "en-us": (
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w',
        'x', 'z', 'æ', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɚ',
        'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɬ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʌ',
        'ʒ', 'ʔ', 'ʲ', '̃', '̩', 'θ', 'ᵻ'
    ),
    "es": (
        'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
        'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'x',
        'ð', 'ŋ', 'ɛ', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɾ', 'ʃ', 'ʊ',
        'ʎ', 'ʒ', 'ʝ', 'β', 'θ'
    ),
    "de": (
        '1', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z', 'ç', 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ',
        'ɒ', 'ɔ', 'ə', 'ɛ', 'ɜ', 'ɡ', 'ɪ', 'ɹ', 'ɾ', 'ʃ',
        'ʊ', 'ʌ', 'ʒ', '̃', 'θ'
    )
}
# fmt: on


def get_ipa_character_list(language):
    if language not in CHARACTER_SETS:
        raise ValueError(f"Character set not found for language {language}")
    char_list = list(CHARACTER_SETS[language])
    return char_list


def get_ipa_punctuation_list(language):
    punct_list = list(DEFAULT_PUNCTUATION)
    if language in ["de", "es"]:
        # https://en.wikipedia.org/wiki/Guillemet#Uses
        punct_list.extend(['«', '»', '‹', '›'])
    if language == "de":
        punct_list.extend(['„', '“'])
    elif language == "es":
        punct_list.extend(['¿', '¡'])

    return punct_list
