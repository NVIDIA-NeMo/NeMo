#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

"""Prepare mixed text/IPA input for EasyMagpieTTS inference.

Examples:
    python prepare_phoneme_text_input.py --language en --text "The glimble fox arrived." --word glimble
    python prepare_phoneme_text_input.py --language es --text "La casa lupina canta." --prob 0.5 --seed 1234
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

DEFAULT_PHONEMIZER_LANGUAGE_MAP = {
    # phonemizer does not accept "en"; "en-gb" matches `espeak-ng -v en --ipa -q`.
    "en": "en-gb",
    "de": "de",
    "es": "es",
    "fr": "fr-fr",
    "hi": "hi",
    "it": "it",
    "vi": "vi",
    "zh": "cmn",
    "ru": "ru",
    "ja": "ja",
    "ko": "ko",
    "ar": "ar",
    "he": "he",
    "nl": "nl",
    "pl": "pl",
    "pt": "pt",
    "pt-BR": "pt",
    "ar-AE": "ar",
    "ar-MSA": "ar",
    "ar-SA": "ar",
    "ar-SY": "ar",
    "ko-KR": "ko",
}
_WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def _validate_probability(name: str, value: float):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be in range [0.0, 1.0], received {value}")


def _phonemize_with_espeak(text: str, language: str, phonemizer_language_map: Optional[Dict[str, str]] = None) -> str:
    try:
        from phonemizer import phonemize
    except ImportError as e:
        raise ImportError("`phonemizer` is required. Install it and ensure the espeak-ng backend is available.") from e

    language_map = dict(DEFAULT_PHONEMIZER_LANGUAGE_MAP)
    if phonemizer_language_map:
        language_map.update(dict(phonemizer_language_map))
    phonemizer_language = language_map.get(language, language)

    phonemized = phonemize(
        [text],
        language=phonemizer_language,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=True,
        language_switch="remove-flags",
        words_mismatch="ignore",
        njobs=1,
    )
    return phonemized[0] if isinstance(phonemized, list) else str(phonemized)


def _coalesce_adjacent_spans(text: str, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    coalesced_spans = []
    for start, end in spans:
        if coalesced_spans and text[coalesced_spans[-1][1] : start].isspace():
            coalesced_spans[-1] = (coalesced_spans[-1][0], end)
        else:
            coalesced_spans.append((start, end))
    return coalesced_spans


def phonemize_selected_words(
    text: str,
    words: List[str],
    language: str,
    phonemizer_language_map: Optional[Dict[str, str]] = None,
    case_sensitive: bool = False,
    bop_marker: str = "<bop>",
    eop_marker: str = "<eop>",
) -> str:
    """Replace all exact word matches with IPA spans."""
    if not words:
        return text

    if case_sensitive:
        selected_words = set(words)
        selected_spans = [match.span() for match in _WORD_RE.finditer(text) if match.group(0) in selected_words]
    else:
        selected_words = {word.lower() for word in words}
        selected_spans = [
            match.span() for match in _WORD_RE.finditer(text) if match.group(0).lower() in selected_words
        ]

    if not selected_spans:
        raise ValueError(f"No exact word match found for: {', '.join(words)}")

    output_parts = []
    cursor = 0
    for start, end in _coalesce_adjacent_spans(text, selected_spans):
        output_parts.append(text[cursor:start])
        ipa_text = _phonemize_with_espeak(text[start:end], language, phonemizer_language_map)
        output_parts.append(f"{bop_marker}{ipa_text}{eop_marker}")
        cursor = end
    output_parts.append(text[cursor:])

    return ''.join(output_parts)


def partially_phonemize_text(
    text: str,
    language: str,
    partial_phoneme_word_prob: float,
    phonemizer_language_map: Optional[Dict[str, str]] = None,
    bop_marker: str = "<bop>",
    eop_marker: str = "<eop>",
) -> str:
    """Replace sampled word spans with IPA spans."""
    _validate_probability("prob", partial_phoneme_word_prob)
    if partial_phoneme_word_prob == 0.0 or not text:
        return text

    selected_spans = [match.span() for match in _WORD_RE.finditer(text) if random.random() < partial_phoneme_word_prob]
    if not selected_spans:
        return text

    output_parts = []
    cursor = 0
    for start, end in _coalesce_adjacent_spans(text, selected_spans):
        output_parts.append(text[cursor:start])
        ipa_text = _phonemize_with_espeak(text[start:end], language, phonemizer_language_map)
        output_parts.append(f"{bop_marker}{ipa_text}{eop_marker}")
        cursor = end
    output_parts.append(text[cursor:])

    return ''.join(output_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Create a mixed regular text + IPA-span string for EasyMagpieTTS inference."
    )
    parser.add_argument("--text", type=str, required=True, help="Input text to partially phonemize.")
    parser.add_argument("--language", "-l", type=str, default="en", help="Language code, e.g. en or es.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prob", "-p", type=float, help="Per-word probability for random phoneme replacement.")
    mode.add_argument("--word", "-w", action="append", help="Exact word to phonemize. Can be passed multiple times.")

    parser.add_argument("--seed", type=int, default=None, help="Random seed for --prob mode.")
    parser.add_argument("--case-sensitive", action="store_true", help="Use case-sensitive matching for --word.")
    parser.add_argument("--phonemizer-language", type=str, default=None, help="Override phonemizer/espeak language.")
    parser.add_argument("--bop-marker", type=str, default="<bop>", help="Beginning marker for IPA spans.")
    parser.add_argument("--eop-marker", type=str, default="<eop>", help="Ending marker for IPA spans.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    phonemizer_language_map = None
    if args.phonemizer_language:
        phonemizer_language_map = {args.language: args.phonemizer_language}

    if args.word:
        output = phonemize_selected_words(
            text=args.text,
            words=args.word,
            language=args.language,
            phonemizer_language_map=phonemizer_language_map,
            case_sensitive=args.case_sensitive,
            bop_marker=args.bop_marker,
            eop_marker=args.eop_marker,
        )
    else:
        output = partially_phonemize_text(
            text=args.text,
            language=args.language,
            partial_phoneme_word_prob=args.prob,
            phonemizer_language_map=phonemizer_language_map,
            bop_marker=args.bop_marker,
            eop_marker=args.eop_marker,
        )

    print(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
