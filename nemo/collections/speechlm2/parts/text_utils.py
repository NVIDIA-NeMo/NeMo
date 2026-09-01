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
import re

import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.common.tokenizers import AutoTokenizer

_whisper_normalizer = EnglishTextNormalizer()

# ---------------------------------------------------------------------------
# Timestamp constants
#
# The speechlm2 pipeline uses two distinct timestamp conventions:
#
# 1. **Training-data timestamps** — integer frame indices written into
#    transcripts by the force-aligner, e.g. ``<|0|> Hey <|3|>``.
#    Pattern: ``<|INT|>``
#
# 2. **Inference-output timestamps** — floating-point seconds inserted by
#    ``tokens_to_str(eval_text_turn_taking=True)`` to annotate where the
#    model's BOS/EOS boundaries fall:
#      * ``<|0.8|>``  — turn start  (BOS position)
#      * ``<$0.72$>`` — turn end    (EOS position)
#    These are NOT vocabulary tokens; they are annotations added after
#    decoding.
# ---------------------------------------------------------------------------

SECONDS_PER_FRAME = 0.08

# Training-data format: integer frame indices (with capture group for parsing)
TRAINING_TIMESTAMP_RE = re.compile(r"<\|(\d+)\|>")

# Inference-output format: float seconds marking turn boundaries
TIMESTAMP_BOS_RE = re.compile(r"<\|[\d.]+\|>")  # turn start
TIMESTAMP_EOS_RE = re.compile(r"<\$[\d.]+\$>")  # turn end

_MULTI_SPACE_RE = re.compile(r"\s+")


def format_bos_timestamp(frame_pos: int) -> str:
    """Format a BOS (turn-start) timestamp annotation from a frame index."""
    return f"<|{round(frame_pos * SECONDS_PER_FRAME, 3)}|>"


def format_eos_timestamp(frame_pos: int) -> str:
    """Format an EOS (turn-end) timestamp annotation from a frame index."""
    return f"<${round(frame_pos * SECONDS_PER_FRAME, 3)}$>"


def format_eot_timestamp(frame_pos: int) -> str:
    """Format an EOT (end-of-text) timestamp annotation from a frame index."""
    return f"<{round(frame_pos * SECONDS_PER_FRAME, 3)}>"


def strip_timestamps(text: str) -> str:
    """Strip all timestamp tokens (both training-data and inference-output formats).

    Handles:
      - Training-data timestamps: ``<|0|>``, ``<|10|>``  (integer frames)
      - Inference BOS timestamps: ``<|0.8|>``             (float seconds)
      - Inference EOS timestamps: ``<$0.72$>``            (float seconds)
    """
    text = TRAINING_TIMESTAMP_RE.sub("", text)
    text = TIMESTAMP_EOS_RE.sub("", text)
    # TIMESTAMP_BOS_RE is a superset of TRAINING_TIMESTAMP_RE, but we keep
    # both calls so the intent is clear and either order is safe.
    text = TIMESTAMP_BOS_RE.sub("", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def get_special_token_strings(tokenizer, pad_id: int, model_cfg=None) -> set[str]:
    """Collect all special token strings that should be stripped from decoded text.

    Derives tokens from the tokenizer and model config at runtime.

    Args:
        tokenizer: Tokenizer with ``ids_to_tokens``, ``bos_token``, and
            ``eos_token`` attributes.
        pad_id: Pad token ID (typically from ``DuplexSTTModel.text_pad_id``).
        model_cfg: Optional model config (OmegaConf or dict).  When provided,
            ``user_bos_token`` and ``user_eos_token`` are included in the set
            (e.g. ``'^'`` and ``'$'`` for some checkpoints).
    """
    pad_str = tokenizer.ids_to_tokens([pad_id])[0]
    tokens = {pad_str}
    if getattr(tokenizer, 'bos_token', None):
        tokens.add(tokenizer.bos_token)
    if getattr(tokenizer, 'eos_token', None):
        tokens.add(tokenizer.eos_token)
    if model_cfg is not None:
        for key in ('user_bos_token', 'user_eos_token'):
            tok = model_cfg.get(key, None) if hasattr(model_cfg, 'get') else None
            if tok:
                tokens.add(tok)
    return tokens


def get_special_token_ids(tokenizer, pad_id: int, model_cfg=None) -> set[int]:
    """Collect special token IDs that should bypass sampling (greedy-only).

    These tokens (pad, BOS, EOS, and optionally user turn markers) must not
    be subject to top-p / temperature / repetition-penalty sampling, otherwise
    EOS may be randomly sampled and generation may not stop properly.

    Args:
        tokenizer: Tokenizer with ``bos_id``, ``eos_id`` attributes and a
            ``text_to_ids`` method.
        pad_id: Pad token ID (typically from ``DuplexSTTModel.text_pad_id``).
        model_cfg: Optional model config (OmegaConf or dict).  When provided,
            ``user_bos_token`` and ``user_eos_token`` are resolved to IDs and
            included in the set.
    """
    ids = {pad_id}
    if getattr(tokenizer, 'bos_id', None) is not None:
        ids.add(tokenizer.bos_id)
    if getattr(tokenizer, 'eos_id', None) is not None:
        ids.add(tokenizer.eos_id)
    if model_cfg is not None:
        for key in ('user_bos_token', 'user_eos_token'):
            tok = model_cfg.get(key, None) if hasattr(model_cfg, 'get') else None
            if tok and hasattr(tokenizer, 'text_to_ids'):
                tok_ids = tokenizer.text_to_ids(tok)
                if tok_ids:
                    ids.add(tok_ids[0])
    return ids


def clean_pred_text(
    text: str,
    special_token_strings: set[str] | None = None,
) -> str:
    """Clean prediction text for fair WER comparison.

    Strips special tokens (pad, BOS, EOS, user turn markers) and timestamp
    annotations, then applies ``EnglishTextNormalizer`` -- the same normalizer
    used by the offline eval metrics in ``speechlm2.parts.metrics.wer``.

    Args:
        text: Raw decoded text that may contain special tokens and timestamps.
        special_token_strings: Set of vocabulary token strings to remove
            (pad, BOS, EOS, user turn markers such as ``'^'``).  Obtain via
            :func:`get_special_token_strings` at pipeline init time.
            When ``None``, only timestamp annotations are stripped.
    """
    if not text:
        return ""
    if special_token_strings:
        for tok in special_token_strings:
            text = text.replace(tok, '')
    text = strip_timestamps(text)
    return _whisper_normalizer(text)


def _decode_tokens_with_specials(
    token_strings: list[str],
    tokenizer,
    pad_token_str: str,
    keep_pad: bool = False,
) -> str:
    """Decode token strings with proper byte-level BPE handling.

    Groups consecutive non-special tokens and decodes each group via
    ``tokenizer.tokens_to_text()`` (HF ``convert_tokens_to_string``), which
    properly reverses byte-level BPE encoding (e.g. ``âĢĻ`` -> ``'``).
    Special tokens (BOS, EOS, PAD) are never passed to
    ``convert_tokens_to_string``.  BOS/EOS are always kept as literal
    strings so that turn boundaries are visible.  PAD tokens are kept
    only when *keep_pad* is True.

    Args:
        token_strings: Raw token strings from ``tokenizer.ids_to_tokens()``.
        tokenizer: Tokenizer with ``tokens_to_text``, ``bos_token``, and
            ``eos_token`` attributes (NeMo ``AutoTokenizer`` or similar).
        pad_token_str: String representation of the pad token.
        keep_pad: If True, preserve all special tokens as literal strings
            in the output.  If False, strip them.
    """
    bos = getattr(tokenizer, 'bos_token', None)
    eos = getattr(tokenizer, 'eos_token', None)

    # All tokens that must not go through convert_tokens_to_string.
    special_tokens = {pad_token_str}
    if bos:
        special_tokens.add(bos)
    if eos:
        special_tokens.add(eos)

    result_parts: list[str] = []
    segment: list[str] = []

    for tok in token_strings:
        if tok in special_tokens:
            if segment:
                result_parts.append(tokenizer.tokens_to_text(segment))
                segment = []
            if tok == pad_token_str:
                if keep_pad:
                    result_parts.append(tok)
            else:
                result_parts.append(tok)
        else:
            segment.append(tok)

    if segment:
        result_parts.append(tokenizer.tokens_to_text(segment))

    return ''.join(result_parts)


def tokens_to_str(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    tokenizer: AutoTokenizer,
    pad_id: int,
    eval_text_turn_taking: bool = False,
    show_eot_timestamps: bool = False,
    keep_pad: bool = False,
) -> list[str]:
    """
    Convert token IDs to text strings with proper byte-level BPE decoding.

    When ``eval_text_turn_taking`` is True, BOS/EOS/EOT token positions are
    replaced by timestamp annotations (these are **not** vocabulary tokens;
    they are human-readable annotations added here):

    * ``<|t|>``  -- turn start  (BOS position, seconds)
    * ``<$t$>``  -- turn end    (EOS position, seconds)
    * ``<t>``    -- end-of-text (first pad after BOS, seconds)

    Note: training-data timestamps use integer frame indices (``<|10|>``),
    while these inference-output timestamps use float seconds (``<|0.8|>``).
    Both ``<|...|>`` formats are stripped by :func:`strip_timestamps`.

    Args:
        tokens: Token IDs tensor (B, T)
        lengths: Length of each sequence (B,)
        tokenizer: Tokenizer for decoding
        pad_id: Pad token ID to filter out
        eval_text_turn_taking: If True, insert timestamps at bos/eos positions
        show_eot_timestamps: If True, also insert timestamps at end-of-text (first pad after BOS)
        keep_pad: If True, preserve all special tokens (including pad) as literal
            strings in the output.  Useful for "raw" output that shows the full
            token stream.  If False (default), special tokens are stripped.

    Returns:
        List of decoded text strings
    """
    pad_token_str = tokenizer.ids_to_tokens([pad_id])[0]
    ans = []

    # Helper function to filter special tokens from token IDs.
    # This filtering is applied regardless of eval_text_turn_taking mode.
    def filter_special_tokens(token_ids):
        # Filter out pad
        token_ids = token_ids[token_ids != pad_id]
        # Filter out agent bos/eos
        token_ids = token_ids[token_ids != tokenizer.bos]
        token_ids = token_ids[token_ids != tokenizer.eos]
        return token_ids

    for hyp_ids, hyp_len in zip(tokens.cpu(), lengths.cpu()):
        if eval_text_turn_taking:
            # Insert timestamps to the text
            agent_bos_positions = (hyp_ids == tokenizer.bos).nonzero(as_tuple=True)[0].tolist()
            agent_eos_positions = (hyp_ids == tokenizer.eos).nonzero(as_tuple=True)[0].tolist()

            # Detect end-of-text (EOT) positions: find first pad after each BOS
            agent_eot_positions = []
            if show_eot_timestamps:
                for bos_pos in agent_bos_positions:
                    # Find the corresponding EOS position for this BOS
                    matching_eos = [eos for eos in agent_eos_positions if eos > bos_pos]
                    end_search_pos = matching_eos[0] if matching_eos else len(hyp_ids)

                    # Search for first pad token after BOS
                    for pos in range(bos_pos + 1, end_search_pos):
                        if hyp_ids[pos] == pad_id:
                            agent_eot_positions.append(pos)
                            break

            # Combine and sort all positions with their types
            all_positions = []
            for pos in agent_bos_positions:
                all_positions.append((pos, 'bos'))
            for pos in agent_eos_positions:
                all_positions.append((pos, 'eos'))
            for pos in agent_eot_positions:
                all_positions.append((pos, 'eot'))

            # Sort by position
            all_positions.sort(key=lambda x: x[0])

            start_idx = 0
            out_str = []
            for pos, pos_type in all_positions:
                text_ids = hyp_ids[start_idx:pos]
                # Filter out special tokens before converting to text
                text_ids = filter_special_tokens(text_ids)
                start_idx = pos
                out_str.append(tokenizer.ids_to_text(text_ids))
                if pos_type == 'bos':
                    out_str.append(format_bos_timestamp(pos))
                elif pos_type == 'eos':
                    out_str.append(format_eos_timestamp(pos))
                else:  # eot
                    out_str.append(format_eot_timestamp(pos))
            # Filter the remaining tokens after the last position
            remaining_ids = filter_special_tokens(hyp_ids[start_idx:])
            out_str.append(tokenizer.ids_to_text(remaining_ids))
            ans.append(" ".join(out_str))
        else:
            hyp_ids = hyp_ids[:hyp_len]
            toks = tokenizer.ids_to_tokens(hyp_ids.tolist())
            ans.append(_decode_tokens_with_specials(toks, tokenizer, pad_token_str=pad_token_str, keep_pad=keep_pad))
    return ans
