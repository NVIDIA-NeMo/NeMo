# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Inference-only NemotronDuplexH model for vLLM-Omni.

A minimal extension of the upstream :class:`NemotronHForCausalLM` that:

1. Accepts pre-computed acoustic encoder embeddings per step via
   ``acoustic_embedding`` in the per-request payload (one row per
   scheduled token). The prefill step receives the *system prompt as
   raw text* via ``system_prompt`` in the same payload; the
   model's :meth:`preprocess` tokenizes it in-process (using a
   HuggingFace tokenizer loaded once in ``__init__`` from the
   checkpoint dir) and constructs the prefill combined embedding
   itself as

       prompt_embed = embed_tokens([BOS] + text_ids + [EOS])
                    + embed_tokens(pad_id)
                    + embed_asr_tokens(pad_id)

   It then *clears* the buffer entry by returning
   ``{"system_prompt": None}`` as its update dict so that subsequent
   decode steps fall through to the decode branch. The producer
   should still send ``system_prompt=None`` on every decode chunk to
   make the intent explicit, but the actual clearing happens
   consumer-side because the orchestrator's serialization filters
   ``None`` values.
2. Embeds up to two additional per-step token id streams, each fed
   **autoregressively from the model itself** via a per-request buffer
   that ``postprocess`` keeps populated after every step:

   - ``input_asr_ids``      – the ASR channel (``predict_user_text``
     checkpoints), embedded with its own ``embed_asr_tokens`` table.
   - ``input_function_ids`` – the function channel
     (``use_function_head`` checkpoints), embedded with the *text*
     ``embed_tokens`` table and scaled by
     ``duplex_function_channel_weight``, mirroring
     ``DuplexSTTModel.build_input_embedding``.

   Which channels exist is checkpoint-dependent. ASR and function can
   both be enabled. The converter records whichever heads it found.

3. Combines the enabled signals into the input embedding fed to the
   NemotronH backbone:

       hidden_in = embed_tokens(input_ids)
                 [+ embed_asr_tokens(input_asr_ids)]
                 [+ embed_tokens(input_function_ids) * weight]
                 + acoustic_embedding

4. Adds a parallel head per enabled channel (``asr_head`` /
   ``function_head``) that produces one token at every decoding step.
   The head matmul and the ``argmax`` run in :meth:`make_omni_output`
   (which the runner invokes *outside* the CUDA-graph wrapper) on the
   full-batch ``hidden_states`` returned by :meth:`forward`. The tokens
   are exposed under ``OmniOutput.multimodal_outputs["asr_tokens"]`` and
   ``["function_tokens"]``, and :meth:`postprocess` stashes the
   request's last id of each back into the corresponding buffer so the
   next step's :meth:`preprocess` can read it as that channel's
   autoregressive input.

   Returning a dict-with-tensor directly from :meth:`forward` is
   unsafe under FULL CUDA graphs: ``weak_ref_tensors`` cannot weak-ref
   tensors nested inside dicts, and the wrapper coerces ``NamedTuple``
   to a plain ``tuple`` on replay. Routing the multimodal output
   through :meth:`make_omni_output` keeps every cudagraph-replayed
   value a plain ``Tensor``.

Text token sampling uses a custom vLLM logits processor which calls the same
PyTorch sampler as the native backend, then forces vLLM's greedy sampler to
the selected token. The auxiliary channels are always greedy.
"""

from collections.abc import Iterable
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from nemo.collections.speechlm2.parts.logit_boosts import (
    LogitBoosts,
    apply_logit_boosts,
)
from nemo.utils import logging as logger


def _is_system_prompt_prefill(
    system_prompt: Any,
    runner_is_prefill: bool,
    request_id: str,
    prompt_token_cache: dict[str, list[int]],
) -> bool:
    """Distinguish initial prompt slices from streaming prompt extensions."""
    has_system_prompt = isinstance(system_prompt, str) and bool(
        system_prompt.strip()
    )
    return has_system_prompt or (
        runner_is_prefill and request_id in prompt_token_cache
    )


def _is_internal_prefill_token(
    is_prompt_prefill: bool,
    runner_is_prefill: bool,
    has_acoustic_embedding: bool,
) -> bool:
    """True for vLLM's 1-token generate after the system prompt is in KV.

    The runner labels every streaming extension as prefill because the
    prompt grows before that token is computed. After a prompt longer
    than the 64-token long-prefill threshold, the last prompt slice pops
    the token cache; the engine then schedules one more token (internal
    ``t_0``) with ``_omni_is_prefill=True`` and no ``acoustic_embedding``.
    A prefill-only ``generate_step`` (empty ``is_first`` frame) does not
    queue audio before that happens, so the decode path would assert.
    Real client steps always carry an acoustic frame.
    """
    return (
        not is_prompt_prefill
        and runner_is_prefill
        and not has_acoustic_embedding
    )


class NemotronDuplexHForCausalLM(NemotronHForCausalLM):
    """NemotronH + optional per-step ASR and function token channels."""

    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True

    # No ``gpu_resident_buffer_keys``; see the note in ``eartts.py``. The keys
    # this stage passes between ``postprocess`` and the next ``preprocess`` are
    # flat, which that mechanism cannot express.

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # NemotronH backbone weights live under
            # `stt_model.llm.backbone.*` in the duplex checkpoint and need to
            # land under our `model.*`.
            "stt_model.llm.backbone": "model",
            "stt_model.llm": "model",
            "stt_model.embed_tokens": "model.embed_tokens",
            "stt_model.embed_asr_tokens": "embed_asr_tokens",
            "stt_model.lm_head": "lm_head",
            "stt_model.asr_head": "asr_head",
            "stt_model.function_head": "function_head",
            # Bare-NemotronH naming, kept as a fallback.
            "backbone": "model",
        },
        orig_to_new_substr={"A_log": "A", "embeddings": "embed_tokens"},
        # Fusing q/k/v into ``qkv_proj`` is done here. This class replaces
        # ``NemotronHForCausalLM.hf_to_vllm_mapper`` wholesale, so the stacked
        # mapping has to be restated or attention projections fail to load.
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config

        # Missing flags default to ASR on / function off. Fresh conversions
        # always write both flags from the checkpoint weights.
        self.use_asr_head = bool(getattr(config, "use_asr_head", True))
        self.use_function_head = bool(getattr(config, "use_function_head", False))
        self.function_channel_weight = float(getattr(config, "duplex_function_channel_weight", 1.0))

        if self.use_asr_head:
            self.embed_asr_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )

            self.asr_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "asr_head"),
            )

        # The function channel has no embedding table of its own: its feedback
        # token is embedded with the text ``embed_tokens``, exactly as
        # ``DuplexSTTModel`` does.
        if self.use_function_head:
            self.function_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                prefix=maybe_prefix(prefix, "function_head"),
            )

        # Tokenizer is used in ``preprocess`` to convert the
        # ``additional_information["system_prompt"]`` text into token
        # IDs on the prefill chunk. Loaded from the checkpoint dir so
        # the vocabulary aligns with ``embed_tokens``. Cached once on
        # init to keep the per-step preprocess fast.
        model_path = vllm_config.model_config.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # The runner can split a prompt at its 64-token long-prefill threshold.
        # Cache the full tokenization in this model process until every slice
        # for a request has been consumed.
        self._prompt_token_cache: dict[str, list[int]] = {}
        self._seeded_channel_requests: set[str] = set()
        self._channel_state: dict[str, dict[str, torch.Tensor]] = {}
        # This pipeline is configured with max_num_seqs=1. Keep a fallback
        # for vLLM-Omni paths that rewrite the internal request id between
        # streaming segments.
        self._last_channel_state: dict[str, torch.Tensor] = {}
        self._current_is_prompt_prefill = False

        # Special token IDs used to construct the prefill prompt:
        # ``[BOS] + text_ids + [EOS]`` and the pad embedding added to
        # every prefill position (mirrors the reference STT recipe
        # where the BOS / pad embeddings are both ``embed_tokens(pad_id)``).
        self.pad_token_id = int(config.pad_token_id)
        self.bos_token_id = int(config.bos_token_id)
        self.eos_token_id = int(config.eos_token_id)

        # User (ASR) channel boosts, read from the converted config so this
        # model applies them exactly as DuplexSTTModel does. The agent-channel
        # boosts arrive per request through the shared text sampling hook.
        self.user_logit_boosts = LogitBoosts.user_from_cfg(config)
        if self.user_logit_boosts:
            logger.info(
                "NemotronDuplexH user logit boosts: "
                f"{self.user_logit_boosts.as_dict()}"
            )
        self._last_asr_token = torch.full(
            (1,), self.pad_token_id, dtype=torch.long
        )
        self._last_function_token = torch.full(
            (1,), self.pad_token_id, dtype=torch.long
        )

        # Per-position pad embedding added on every prefill step: the pad id
        # embedded once per *enabled* auxiliary channel, shape
        # ``(hidden_size,)``. Materialized at the end of
        # :meth:`load_weights` because the embedding tables are not
        # populated yet in ``__init__``. Registered as a *non-persistent*
        # buffer so it follows ``.to(device)`` / dtype casts with the
        # rest of the module but is **not** saved in the state_dict
        # (it is fully derived from ``embed_tokens`` /
        # ``embed_asr_tokens`` which are already saved — duplicating it
        # in the checkpoint would just be a footgun).
        self.register_buffer("_pad_combined_emb", None, persistent=False)

    # ------------------------------------------------------------------ #
    #  producer-side helper                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_prefix_len(tokenizer: PreTrainedTokenizerBase, system_prompt: str) -> int:
        """Length of the prefill chunk for a given system prompt.

        Mirrors the in-model tokenization done by :meth:`preprocess`:

            [BOS] + tokenizer.encode(system_prompt, add_special_tokens=False) + [EOS]

        The streaming producer needs this number to size the
        placeholder ``prompt_token_ids`` it hands vLLM on the prefill
        chunk — vLLM schedules off that list's length, while the
        actual embedding is constructed inside :meth:`preprocess`
        from the ``system_prompt`` string.

        Exposed as a ``@staticmethod`` so callers can compute the
        length without instantiating the model (which would download
        the full checkpoint). They just need any tokenizer compatible
        with the model's vocabulary — typically
        ``AutoTokenizer.from_pretrained(<converted nemotron dir>)``, the
        same instance used to decode output tokens.
        """
        text_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
        return len(text_ids) + 2  # +2 for BOS / EOS wrapped in preprocess

    # ------------------------------------------------------------------ #
    #  preprocess                                                        #
    # ------------------------------------------------------------------ #

    def _materialize_pad_combined_emb(self) -> None:
        embed_weight = self.model.embed_tokens.weight
        device = embed_weight.device
        dtype = embed_weight.dtype
        pad_tokens = torch.full(
            (1,), self.pad_token_id, device=device, dtype=torch.long
        )
        pad_emb = self.model.embed_tokens(pad_tokens).to(dtype).squeeze(0)
        combined_pad = pad_emb
        if self.use_asr_head:
            combined_pad = combined_pad + self.embed_asr_tokens(
                pad_tokens
            ).to(dtype).squeeze(0)
        if self.use_function_head:
            combined_pad = (
                combined_pad
                + pad_emb * self.function_channel_weight
            )
        self._pad_combined_emb = combined_pad.detach()

    def _embeds_without_acoustic(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Text + pad-channel embeddings, used for prompt slices and internal ``t_0``."""
        if self._pad_combined_emb is None:
            self._materialize_pad_combined_emb()
        target_dtype = self.model.embed_tokens.weight.dtype
        text_emb = self.model.embed_tokens(input_ids).to(target_dtype)
        return input_ids, text_emb + self._pad_combined_emb, {"system_prompt": None}

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Combine text/asr/speech embeddings into a single per-token vector.

        Three paths:

        * **Prefill construction.** When
          ``additional_information["system_prompt"]`` is a non-empty
          string, this is the prefill chunk. We tokenize the prompt
          in-process as ``[BOS] + tokenizer.encode(prompt) + [EOS]``,
          embed it with ``model.embed_tokens``, and add the pad
          embedding of each enabled auxiliary channel (which the
          reference STT recipe folds in uniformly across every prefill
          position):

              prefill_combined = embed_tokens(prompt_token_ids)
                               + _pad_combined_emb

          The producer-supplied ``input_ids`` for this chunk are
          placeholders (their length must match the tokenized prompt
          length so vLLM's scheduling sees the right prefill size);
          they are returned unchanged so vLLM's bookkeeping is
          consistent. We then *clear* the buffer entry by returning
          ``{"system_prompt": None}`` in the update dict. Clearing has
          to happen here because the orchestrator's serialization
          (:func:`vllm_omni.data_entry_keys.serialize_payload`)
          silently drops ``None`` values, so the producer cannot
          overwrite the buffer with ``None`` via the streaming-input
          merge. Decode chunks may carry ``system_prompt=None``, but it
          has no effect; the state transition happens only here.

        * **Internal ``t_0`` after a chunked prompt.** vLLM may split the
          prompt at its 64-token long-prefill threshold and then
          schedule a one-token continuation labeled
          ``_omni_is_prefill`` with no ``acoustic_embedding``. That
          token is discarded by the session (``output_count <= 1``);
          embed it like prefill (text + pad, no user acoustics) so a
          prefill-only ``generate_step`` cannot kill the engine.

        * **Decode (single-token step).** Builds the combined embedding
          per scheduled token from:

          - ``input_ids``            – per-step text token id (one per
                                       scheduled token; standard vLLM
                                       autoregressive feedback).
          - ``input_asr_ids``        – per-step ASR token id, written
                                       back by :meth:`postprocess` on
                                       every step. ASR channel only.
          - ``input_function_ids``   – per-step function token id, same
                                       write-back path. Function
                                       channel only.
          - ``acoustic_embedding``   – per-step acoustic encoder
                                       embedding, sourced from
                                       ``additional_information``.

          ``input_embeds`` is the runner's pre-allocated scratch buffer
          on this path and its contents are ignored.
        """
        device = input_ids.device
        n = int(input_ids.shape[0])

        # Prefill vs decode is detected directly on the value of
        # ``system_prompt``: a non-empty string means prefill, anything
        # else (``None`` / missing / empty) means decode. The buffer
        # flips from str → ``None`` inside this method itself (see the
        # update dict returned below) because serialization drops
        # ``None`` and the producer's "send None on each decode chunk"
        # pattern alone is not enough to clear the slot.
        system_prompt = info_dict.get("system_prompt")
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        request_id = str(
            info_dict.get("global_request_id")
            or info_dict.get("request_id")
            or ""
        )
        has_system_prompt = isinstance(system_prompt, str) and bool(
            system_prompt.strip()
        )
        is_prompt_prefill = _is_system_prompt_prefill(
            system_prompt,
            is_prefill,
            request_id,
            self._prompt_token_cache,
        )
        has_acoustic = isinstance(info_dict.get("acoustic_embedding"), torch.Tensor)
        is_internal_t0 = _is_internal_prefill_token(
            is_prompt_prefill, is_prefill, has_acoustic
        )
        # vLLM labels every one-token streaming extension as prefill because
        # the prompt grows before that token is computed. Prompt slices and
        # the engine-internal t_0 after them seed auxiliary feedback with PAD;
        # client-visible decode steps (which always carry acoustic_embedding)
        # do not.
        self._current_is_prompt_prefill = is_prompt_prefill or is_internal_t0
        if is_prompt_prefill:
            # [BOS] + encode(text, add_special_tokens=False) + [EOS].
            # ``add_special_tokens=False`` keeps full control of which
            # specials get wrapped around the text (the underlying HF
            # tokenizer would otherwise prepend its own BOS, which may
            # or may not equal ``config.bos_token_id``).
            if has_system_prompt:
                text_ids = self.tokenizer.encode(
                    system_prompt, add_special_tokens=False
                )
                prompt_token_ids = [
                    self.bos_token_id,
                    *text_ids,
                    self.eos_token_id,
                ]
                self._prompt_token_cache[request_id] = prompt_token_ids
            else:
                prompt_token_ids = self._prompt_token_cache[request_id]
            prompt_len = len(prompt_token_ids)
            expected_prompt_len = info_dict.get("duplex_prompt_len")
            if expected_prompt_len is not None:
                assert prompt_len == int(expected_prompt_len), (
                    f"system_prompt tokenizes to {prompt_len} ids but vLLM "
                    f"tracks a prompt of length {expected_prompt_len}"
                )
            offset = int(info_dict.get("duplex_token_offset", 0) or 0)
            end = offset + n
            if not (0 <= offset < end <= prompt_len):
                # Cache still populated but the runner scheduled past the
                # prompt (internal t_0). Same pad-embedding path as below.
                self._prompt_token_cache.pop(request_id, None)
                return self._embeds_without_acoustic(input_ids)

            prompt_tokens = torch.tensor(
                prompt_token_ids[offset:end],
                device=device,
                dtype=torch.long,
            )
            _, prefill_combined, updates = self._embeds_without_acoustic(
                prompt_tokens
            )
            if end == prompt_len:
                self._prompt_token_cache.pop(request_id, None)
            return input_ids, prefill_combined, updates

        if is_internal_t0:
            return self._embeds_without_acoustic(input_ids)

        combined = self.model.embed_tokens(input_ids)

        request_id = str(
            info_dict.get("global_request_id")
            or info_dict.get("request_id")
            or ""
        )
        cached_channel_state = self._channel_state.get(
            request_id
        ) or self._last_channel_state
        for key, value in cached_channel_state.items():
            if not isinstance(info_dict.get(key), torch.Tensor):
                info_dict[key] = value
        if self.use_asr_head and not isinstance(
            info_dict.get("input_asr_ids"), torch.Tensor
        ):
            info_dict["input_asr_ids"] = self._last_asr_token
        if self.use_function_head and not isinstance(
            info_dict.get("input_function_ids"), torch.Tensor
        ):
            info_dict[
                "input_function_ids"
            ] = self._last_function_token
        if request_id not in self._seeded_channel_requests:
            # The initial prompt output is not guaranteed to run postprocess
            # before the first direct StreamingInput update on a one-stage
            # engine. Seed optional feedback channels exactly as native does;
            # subsequent missing state remains an error.
            if self.use_asr_head and not isinstance(
                info_dict.get("input_asr_ids"), torch.Tensor
            ):
                info_dict["input_asr_ids"] = torch.full(
                    (n,),
                    self.pad_token_id,
                    device=device,
                    dtype=torch.long,
                )
            if self.use_function_head and not isinstance(
                info_dict.get("input_function_ids"), torch.Tensor
            ):
                info_dict["input_function_ids"] = torch.full(
                    (n,),
                    self.pad_token_id,
                    device=device,
                    dtype=torch.long,
                )
            self._seeded_channel_requests.add(request_id)

        if self.use_asr_head:
            asr_ids = self._channel_ids(info_dict, "input_asr_ids", n, device)
            combined = combined + self.embed_asr_tokens(asr_ids)

        if self.use_function_head:
            function_ids = self._channel_ids(info_dict, "input_function_ids", n, device)
            combined = combined + self.model.embed_tokens(function_ids) * self.function_channel_weight

        # Per-step acoustic encoder embedding, sourced from
        # ``additional_information["acoustic_embedding"]``.
        acoustic = info_dict.get("acoustic_embedding")
        assert isinstance(acoustic, torch.Tensor), (
            "acoustic_embedding is required in the per-step payload on every decode step; "
            f"got {type(acoustic).__name__} with available keys {sorted(info_dict)}"
        )
        acoustic = acoustic.to(device=device, dtype=combined.dtype)
        assert acoustic.dim() == 2, f"acoustic_embedding must be 2D, got shape {tuple(acoustic.shape)}"
        assert acoustic.shape[0] == n, (
            f"acoustic_embedding length {acoustic.shape[0]} does not match scheduled token count {n}"
        )
        combined = combined + acoustic

        return input_ids, combined, {}

    @staticmethod
    def _channel_ids(info_dict: dict[str, Any], key: str, n: int, device: torch.device) -> torch.Tensor:
        """Read one auxiliary channel's per-step feedback ids from the payload."""
        ids = info_dict.get(key)
        assert isinstance(ids, torch.Tensor), (
            f"{key} is required on every decode step but is "
            f"{type(ids).__name__}; available keys {sorted(info_dict)}"
        )
        ids = ids.to(device=device, dtype=torch.long).reshape(-1)
        assert ids.numel() == n, f"{key} length {ids.numel()} does not match scheduled token count {n}"
        return ids

    # ------------------------------------------------------------------ #
    #  postprocess - autoregressive feedback for the auxiliary channels  #
    # ------------------------------------------------------------------ #

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: dict[str, Any] | None = None,
        **info_dict: Any,
    ) -> dict[str, Any]:
        """Stash this request's last auxiliary tokens as the next step's input.

        ``hidden_states`` is a slice of the full-batch hidden_states tensor,
        and each ``multimodal_outputs`` entry is the corresponding full-batch
        token tensor produced by :meth:`make_omni_output`. We pick the token
        aligned with the last position of this request's slice.

        On the prefill chunk the function channel is seeded with the pad id
        instead of the prompt's own prediction, because native starts decoding
        with ``gen_function`` still at its ``text_pad_id`` fill value and only
        feeds back real function tokens from the second frame onwards.
        """
        assert multimodal_outputs
        start = hidden_states.storage_offset() // hidden_states.stride(0)
        last_idx = start + hidden_states.shape[0] - 1
        # The initial prompt can be split into a 64-token slice plus a
        # one-token continuation, so tensor length alone cannot identify it.
        is_prefill = self._current_is_prompt_prefill

        def last_token(key: str) -> torch.Tensor:
            tokens = multimodal_outputs.get(key)
            assert isinstance(tokens, torch.Tensor), f"{key} missing from multimodal_outputs"
            return tokens[last_idx : last_idx + 1].detach().to(torch.long)

        updates: dict[str, Any] = {}
        if self.use_asr_head:
            updates["input_asr_ids"] = last_token("asr_tokens")
        if self.use_function_head:
            if is_prefill:
                updates["input_function_ids"] = torch.full(
                    (1,), self.pad_token_id, device=hidden_states.device, dtype=torch.long
                )
            else:
                updates["input_function_ids"] = last_token("function_tokens")
        request_id = str(
            info_dict.get("global_request_id")
            or info_dict.get("request_id")
            or ""
        )
        if request_id:
            channel_state = {
                key: value.detach()
                for key, value in updates.items()
                if isinstance(value, torch.Tensor)
            }
            self._channel_state[request_id] = channel_state
            self._last_channel_state = channel_state
        return updates

    # ------------------------------------------------------------------ #
    #  forward                                                           #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Run the backbone and return its hidden states.

        ASR tokens are produced by :meth:`make_omni_output`, which the
        runner invokes *outside* the CUDA-graph wrapper. This keeps the
        captured graph's output a plain ``Tensor`` (or
        ``IntermediateTensors``) — both types that ``weak_ref_tensors``
        handles correctly. Returning a ``NamedTuple`` containing a
        ``dict[str, Tensor]`` directly here would corrupt the dict's
        tensors on FULL graph replay (the wrapper coerces
        ``NamedTuple`` -> plain ``tuple`` and cannot weak-ref tensors
        nested in dicts).

        IMPORTANT — cudagraph mode requirement
        --------------------------------------
        This model must be run with ``cudagraph_mode="PIECEWISE"`` (or
        ``enforce_eager=True``). The streaming-input pattern used here
        keeps extending each request's prompt with every audio chunk,
        so ``num_computed_tokens < num_prompt_tokens`` is permanently
        true and Mamba's metadata builder always classifies the request
        as a *prefill* (because
        :func:`split_decodes_and_prefills` is called with
        ``treat_short_extends_as_decodes=False`` in
        ``Mamba2AttentionMetadataBuilder._compute_common_metadata``).

        With FULL cudagraph mode, the persistent
        ``state_indices_tensor_d`` buffer is only updated when
        ``num_prefills == 0``, so for streaming it stays at the
        capture-time dummy value (0) while the FULL decode graph is
        still dispatched (the dispatcher only checks ``query_len``).
        The captured Mamba kernel then reads slot 0 of ``mamba_cache``
        instead of the real slot, producing garbage hidden states.
        PIECEWISE side-steps this because the Mamba layer runs eagerly
        and reads the freshly-computed metadata tensor, and the prefill
        code path correctly *writes* the chunk into Mamba state on
        every step (which is essential — there is no separate "prefill"
        phase in this streaming setup).
        """
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    # ------------------------------------------------------------------ #
    #  make_omni_output - runs eagerly outside the CUDA graph wrapper    #
    # ------------------------------------------------------------------ #

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | IntermediateTensors | OmniOutput,
        **_: Any,
    ) -> OmniOutput:
        """Wrap backbone hidden states with the auxiliary channel tokens.

        Invoked by :class:`OmniGPUModelRunner._model_forward` after the
        CUDA-graph wrapper has returned, so the auxiliary head matmuls +
        ``argmax`` here run eagerly. They operate on the full-batch
        ``hidden_states`` tensor in a single GEMM each, so the cost is
        negligible relative to the backbone forward.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, IntermediateTensors):
            return OmniOutput(
                text_hidden_states=model_outputs,
                intermediate_tensors=model_outputs,
            )

        hidden = model_outputs
        multimodal_outputs: dict[str, torch.Tensor] = {}
        if self.use_asr_head:
            asr_logits = self.logits_processor(self.asr_head, hidden)
            # The ASR head's logits never reach vLLM's sampler, so the
            # user-channel boosts are applied here rather than in the shared
            # text logits processor. Same arithmetic as DuplexSTTModel.
            apply_logit_boosts(
                asr_logits,
                self.user_logit_boosts,
                pad_id=self.pad_token_id,
                bos_id=self.bos_token_id,
                eos_id=self.eos_token_id,
            )
            multimodal_outputs["asr_tokens"] = torch.argmax(asr_logits, dim=-1).to(torch.long)
            self._last_channel_state["input_asr_ids"] = (
                multimodal_outputs["asr_tokens"][-1:]
                .detach()
            )
            self._last_asr_token.copy_(
                multimodal_outputs["asr_tokens"][-1:]
            )
        if self.use_function_head:
            function_logits = self.logits_processor(self.function_head, hidden)
            multimodal_outputs["function_tokens"] = torch.argmax(function_logits, dim=-1).to(torch.long)
            if self._current_is_prompt_prefill:
                function_state = torch.full(
                    (1,),
                    self.pad_token_id,
                    device=hidden.device,
                    dtype=torch.long,
                )
            else:
                function_state = multimodal_outputs[
                    "function_tokens"
                ][-1:].detach()
            self._last_channel_state[
                "input_function_ids"
            ] = function_state
            self._last_function_token.copy_(function_state)

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    # ------------------------------------------------------------------ #
    #  weight loading                                                    #
    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp"])
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Now that the embedding tables are populated, materialize the
        # per-prefill pad embedding into the ``_pad_combined_emb`` buffer
        # declared in ``__init__``. See the buffer's registration site
        # for why this lives here rather than in ``__init__`` (embedding
        # tables are empty there) and why it's non-persistent (derived
        # from weights that are already saved).
        self._materialize_pad_combined_emb()

        return loaded
