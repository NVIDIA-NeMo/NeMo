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

"""Inference-only EarTTS model definition for vLLM-Omni.

The model architecture (RMSNorm, MLP, MLPLayer, GatedProjectedSumRMSNorm,
PrecomputedSubwordEmbedding, EarTTSInputEmbedding, MoGHead,
MaskGITSampler, EarTTSModel) matches the PyTorch EarTTS modules.
Classifier-free guidance (CFG) is driven by per-request metadata and keeps
its role, pair, and scale contract in model-owned stable-address buffers.

The original NeMo model used a character-aware subword encoder (a small
transformer over per-character embeddings) followed by additive
subword-continuation and BOS/EOS flag embeddings to embed text tokens.
Those operations are deterministic per token id, so the checkpoint
converter runs them once over the full vocabulary and stores the result
as a single ``nn.Embedding`` (see :class:`PrecomputedSubwordEmbedding`).

The outer :class:`EarTTSForCausalLM` exposes the minimal vLLM-Omni
preprocess/postprocess hooks. Per-request inputs are passed via
``additional_information``.

Inputs (only one mode — streaming text token ids), named after the
categories of :class:`~vllm_omni.data_entry_keys.OmniPayload` so that the
same two keys work whether they arrive on the request or over an
inter-stage connector, which accepts nothing outside that schema:

* ``embed.voice`` (prefill only): Tensor of shape
  ``(Tref, hidden_size)``. The user-supplied speaker latent that
  replaces ``embed_code(rvq_sum(acoustic_tokens))`` on every pre-BOS
  prefill position. ``Tref`` is also the prefill placeholder length
  (the user passes ``prompt_token_ids = [0] * Tref``).
* ``ids.output`` (every decode step): Python ``list[int]`` of the text
  tokens the producer has sent most recently. :meth:`preprocess` takes
  its **last** entry, so decode step ``k`` consumes ``t_k`` whether the
  producer sends one token per step or a growing history.

There is no whole-utterance text path: callers must always provide token
ids per step via the streaming-text contract above.

Per-step flow:

1. ``preprocess`` writes the per-token tensors consumed by
   :class:`EarTTSInputEmbedding` —  ``acoustic_tokens (BTx31)``,
   ``text_tokens (BT)``, ``text_mask (BT)``, ``bos_mask (BT)``,
   ``speaker_latent (BT x hidden_size)`` — into the model-owned
   static-address buffers at the request's flat-batch offset. Returns
   placeholder ``input_ids`` and the ``inputs_embeds`` slice it
   received from the runner unchanged (the actual embedding is
   computed inside the compiled ``forward``; the buffer's contents
   are ignored).

   Prefill is fully derived from ``speaker_latent``:

     * ``acoustic_tokens`` = ``model.sil_tokens`` broadcast to every
       prefill position (only the BOS frame's audio embedding actually
       contributes to the model output; the rest are replaced by the
       speaker latent inside :class:`EarTTSInputEmbedding`).
     * ``text_tokens`` = ``[PAD] * (Tref - 1) + [EOS]``.
     * ``text_mask``   = ``[0] * (Tref - 2) + [1, 1]``.
     * ``bos_mask``    = ``[0] * (Tref - 1) + [1]``.
     * ``speaker_latent`` = the user-supplied ``embed.voice`` tensor.

   Decode each step (chooses ``acoustic_tokens`` in this order):

     * ``text_token == EOS`` (``2``) → ``model.sil_tokens``.
     * First decode step (``ear_decode_offset == 0``) → the acoustic pad
       id (``codebook_size``) broadcast across all quantizers.
     * Otherwise → previous-step codes stashed by :meth:`postprocess`
       as ``last_acoustic_codes``.

     ``text_tokens = ids.output[-1]``,
     ``text_mask = 1``, ``bos_mask = 0``,
     ``speaker_latent = 0`` (no replacement on decode).

2. ``forward`` slices the buffers up to ``num_tokens`` and calls the
   compiled :class:`EarTTSModel` (embedding + Gemma3 backbone). The
   compiled :class:`EarTTSSamplerModel` (MaskGIT) is invoked
   conditionally on decode positions. Generated codes are copied into
   a stable ``_out_codes`` buffer for :meth:`make_omni_output`.

3. ``compute_logits`` returns trivial logits so vLLM's standard
   sampler always picks index ``0`` — the actual audio output is the
   codes tensor exposed via :meth:`make_omni_output`.

4. ``postprocess`` stashes the last frame's codes as
   ``last_acoustic_codes`` for the next step's :meth:`preprocess`.
"""

import bisect
import hashlib
import math
from collections.abc import Iterable
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers.generation.logits_process import (
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import (
    ignore_torch_compile,
    support_torch_compile,
)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.models.gemma3 import Gemma3Model
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput



def _prepare_cfg_sampling_batch(
    hidden_states: torch.Tensor,
    cfg_enabled: torch.Tensor,
    cfg_is_uncond: torch.Tensor,
    cfg_pair_id: torch.Tensor,
    cfg_scale: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Order a complete CFG batch and describe its logical pairs.

    All operations stay on-device: CUDA graph replay can therefore change the
    role/pair/scale buffers without depending on Python-side state. A batch is
    guided only when every valid row has exactly one enabled opposite-role
    companion with the same pair id. Padded rows are ignored; the conditional
    row's scale is authoritative for each pair.

    Returns ordered hidden states, ordered role/scale tensors, an active-row
    mask, each row's partner and conditional representative indices, and the
    inverse permutation used to restore the runner's original row order.
    """
    batch_size = int(hidden_states.shape[0])
    identity = torch.arange(batch_size, device=hidden_states.device)

    pair_match = (
        valid[:, None]
        & valid[None, :]
        & cfg_enabled[:, None]
        & cfg_enabled[None, :]
        & (cfg_pair_id[:, None] == cfg_pair_id[None, :])
        & (cfg_is_uncond[:, None] != cfg_is_uncond[None, :])
    )
    partner_count = pair_match.sum(dim=1)
    partner = pair_match.to(torch.long).argmax(dim=1)
    complete = valid.any() & ((~valid) | (cfg_enabled & (partner_count == 1))).all()

    # Lexicographic order: valid rows first, then conditional before
    # unconditional, with pair ids sorted identically inside both role blocks.
    order = torch.argsort(cfg_pair_id, stable=True)
    order = order[torch.argsort(cfg_is_uncond[order].to(torch.long), stable=True)]
    order = order[torch.argsort((~valid[order]).to(torch.long), stable=True)]
    order = torch.where(complete, order, identity)
    inverse_order = torch.argsort(order)

    hidden_states = hidden_states[order]
    cfg_is_uncond = cfg_is_uncond[order]
    cfg_pair_id = cfg_pair_id[order]
    cfg_scale = cfg_scale[order]
    valid = valid[order]
    active = complete & valid

    ordered_pair_match = (
        active[:, None]
        & active[None, :]
        & (cfg_pair_id[:, None] == cfg_pair_id[None, :])
        & (cfg_is_uncond[:, None] != cfg_is_uncond[None, :])
    )
    partner = ordered_pair_match.to(torch.long).argmax(dim=1)
    conditional_rep = torch.where(cfg_is_uncond, partner, identity)
    conditional_scale = torch.where(
        cfg_is_uncond,
        cfg_scale[partner],
        cfg_scale,
    )
    return (
        hidden_states,
        cfg_is_uncond,
        conditional_scale,
        active,
        partner,
        conditional_rep,
        inverse_order,
    )


def _apply_cfg_after_mlp(
    x: torch.Tensor,
    cfg_is_uncond: torch.Tensor,
    cfg_scale: torch.Tensor,
    cfg_active: torch.Tensor,
    cfg_partner: torch.Tensor,
) -> torch.Tensor:
    """Apply EarTTS CFG to MoG MLP outputs before all projections."""
    partner_x = x[cfg_partner]
    conditional_x = torch.where(cfg_is_uncond[:, None], partner_x, x)
    unconditional_x = torch.where(cfg_is_uncond[:, None], x, partner_x)
    guided_x = conditional_x + cfg_scale[:, None].to(x.dtype) * (
        conditional_x - unconditional_x
    )
    return torch.where(cfg_active[:, None], guided_x, x)


# ---------------------------------------------------------------------------
# Shared EarTTS building blocks, matching the native DuplexEARTTS modules.
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Normalize in fp32 and cast back at the end, so low-precision
        # activations do not lose the mean-square accumulation.
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MLPLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.post_norm = RMSNorm(hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre_norm(x)
        y = self.mlp(y)
        y = self.post_norm(y)
        x = x + y
        return x


class GatedProjectedSumRMSNorm(nn.Module):
    def __init__(
        self,
        audio_dim,
        text_dim,
        hidden_dim,
        final_norm=True,
        num_codebooks=31,
        init_residual_scale=0.5,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        nn.init.normal_(self.audio_proj.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.audio_proj.bias)
        nn.init.normal_(self.text_proj.weight, mean=0.0, std=0.015)
        nn.init.zeros_(self.text_proj.bias)

        # FP32 gate params
        self.gate = nn.Parameter(
            torch.zeros(hidden_dim, dtype=torch.float32), requires_grad=False
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(init_residual_scale, dtype=torch.float32),
            requires_grad=False,
        )

        self.final_norm = RMSNorm(hidden_dim) if final_norm else nn.Identity()

    def forward(self, audio_emb, text_emb):
        audio_emb = audio_emb / self.num_codebooks

        # projections run in model dtype (BF16)
        audio_h = self.audio_proj(audio_emb)
        text_h = self.text_proj(text_emb)

        dtype = audio_h.dtype

        gate = torch.sigmoid(self.gate)  # FP32
        res = torch.sigmoid(self.residual_scale)  # FP32

        h = gate.to(dtype) * audio_h + (1 - gate).to(dtype) * text_h
        h = res.to(dtype) * h
        h = self.final_norm(h.float()).to(dtype)

        return h


class PrecomputedSubwordEmbedding(nn.Module):
    """Per-token text embedding lookup baked out at checkpoint-conversion time.

    The original NeMo model embeds text with a character-aware subword
    encoder (a small transformer over per-character embeddings) followed
    by additive subword-continuation and BOS/EOS flag embeddings. All of
    those operations are deterministic per token id, so the converter
    runs them once over the full vocabulary and stores the result here
    as a single ``nn.Embedding``.
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_subwords = nn.Embedding(vocab_size, hidden_size)

    def forward(self, subword_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_subwords(subword_ids)


class EarTTSInputEmbedding(nn.Module):
    """Module that takes text tokens, audio tokens and prepares input
    embedding for EarTTS model.
    """

    def __init__(self, config):
        super().__init__()

        hidden_size = config.hidden_size
        vocab_size = config.emb_vocab_size

        # allows to embed acoustic tokens into a single embeddings
        self.rvq_embs = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size + 1, config.latent_size)
                for _ in range(config.num_quantizers)
            ]
        )
        self.embed_code = nn.Linear(config.latent_size, hidden_size, bias=False)
        # Pre-computed per-token text embedding lookup. Replaces the
        # original char-aware subword encoder + subword-flag + BOS/EOS
        # additive embeddings; all of those are deterministic per token
        # id and are baked into this single table by the checkpoint
        # converter.
        self.embed_subword = PrecomputedSubwordEmbedding(vocab_size, hidden_size)
        self.bos_emb = nn.Parameter(torch.empty(hidden_size))
        # Learned classifier-free text-conditioning embedding. The audio and
        # speaker branches remain unchanged for unconditional rows.
        self.null_emb = nn.Parameter(torch.empty(hidden_size))

        self.use_gated_fusion_for_text_audio = config.use_gated_fusion_for_text_audio
        if self.use_gated_fusion_for_text_audio:
            self.gated_fusion_audio_text = GatedProjectedSumRMSNorm(
                hidden_size, hidden_size, hidden_size, config.num_quantizers
            )

        self.use_audio_prompt_frozen_projection = (
            config.use_audio_prompt_frozen_projection
        )
        if self.use_audio_prompt_frozen_projection:
            self.audio_prompt_projection_W = nn.Parameter(
                torch.empty(hidden_size, hidden_size),
                requires_grad=False,
            )

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        bos_mask: torch.Tensor,
        speaker_latent: Optional[torch.Tensor] = None,
        cfg_is_uncond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Works for context and generation phases to prepare total input
        embeddings for EarTTS model.

        Inputs:
            acoustic_tokens: (BT x 31) - audio tokens
            text_tokens: (BT) - text token to embed
            text_mask: (BT) - masks text embeddings for prefill
            bos_mask: (BT) - specifies where BOS is applied (first frame of prefill)
            speaker_latent: (BT x hidden_size) - external speaker latent.
                Non-zero rows replace ``embed_code(rvq_sum(...))`` at
                pre-BOS prefill positions; zero rows (decode steps and
                the BOS frame) leave ``audio_emb`` untouched. Pass an
                all-zero tensor on decode.

        Returns:
            embedding of shape (BT x dim)
        """

        # prepare bos emb that is applied to audio embedding
        bos_emb = bos_mask.unsqueeze(1) * self.bos_emb  # BT x dim

        acoustic_tokens = acoustic_tokens.transpose(0, 1)  # 31 x BT
        audio_emb = sum(
            emb(acoustic_tokens[i]) for i, emb in enumerate(self.rvq_embs)
        )  # BT x latent_size
        audio_emb = self.embed_code(audio_emb)  # BT x hidden_size

        if self.use_audio_prompt_frozen_projection:
            if speaker_latent is None:
                # No external latent -> derive one from the acoustic
                # tokens, matching DuplexEARTTS when no speaker prompt
                # is supplied. vLLM-Omni callers always pass a real or
                # zero latent, so they take the other branch.
                latent_provided = torch.zeros_like(bos_mask).unsqueeze(-1)
                latent = torch.nn.functional.linear(
                    audio_emb, self.audio_prompt_projection_W.T
                )
            else:
                # ``latent_provided`` is non-zero exactly on the rows
                # the user populated with a real speaker latent
                # (prefill pre-BOS positions). Decode rows are filled
                # with zeros by ``preprocess``, so they read as "not
                # provided" here.
                latent_provided = (
                    speaker_latent.abs().sum(-1, keepdim=True) > 0
                )  # (BT, 1)
                latent = speaker_latent

            # Replace only at pre-BOS positions of prefill -- i.e.
            # ``bos_mask == 0 AND latent was actually provided``. This
            # excludes:
            #   * the BOS frame (``bos_mask == 1``), where
            #     ``embed_code(acoustic_tokens)`` of ``sil_tokens``
            #     survives (this is the audio_emb the backbone sees on
            #     the BOS frame).
            #   * AR decode steps (``latent_provided == False`` because
            #     ``speaker_latent`` is all zeros).
            replace_mask = (bos_mask.unsqueeze(-1) == 0) & latent_provided
            audio_emb = torch.where(replace_mask, latent, audio_emb)

        audio_emb = audio_emb + bos_emb

        # Embed text tokens via the pre-computed lookup (subword-flag and
        # BOS/EOS additions are baked into the table at conversion time).
        # Apply the mask that zeroes this embedding on prefill positions.
        text_emb = self.embed_subword(text_tokens) * text_mask.unsqueeze(1)  # BT x dim
        if cfg_is_uncond is not None:
            text_emb = torch.where(
                cfg_is_uncond.unsqueeze(1),
                self.null_emb.to(text_emb.dtype),
                text_emb,
            )

        # prepare total embedding by combining audio and text branches
        if self.use_gated_fusion_for_text_audio:
            # Gated fusion needs ``audio_emb`` and ``text_emb`` as
            # separate inputs (it learns a per-feature gate to mix
            # them), which is why neither branch can be folded into a
            # single precomputed ``inputs_embeds`` tensor outside the
            # compiled forward.
            total_emb = self.gated_fusion_audio_text(audio_emb, text_emb)
        else:
            total_emb = audio_emb + text_emb  # BT x dim
        return total_emb


def gumbel_like(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Generates a tensor of Gumbel noise with the same shape as the input
    tensor. Used for the Gumbel-Max trick.
    """
    u = torch.rand_like(tensor)
    return -torch.log(-torch.log(u + eps) + eps)


def batch_matmul(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Performs a batched matrix multiplication using PyTorch's native functions.
    In NeMo this is implemented as a custom kernel using triton.

    Args:
        x: ``[batch_size, d_in]``
        w: ``[num_weights, d_out, d_in]``
        y: ``[batch_size]``

    Returns:
        Tensor of shape ``[batch_size, d_out]``.
    """
    return torch.bmm(w[y], x.unsqueeze(2)).squeeze(2)


class MoGHead(nn.Module):
    """A Mixture of Gaussians (MoG) prediction head.

    This module takes a hidden state and predicts the parameters for a
    mixture of Gaussian distributions. It's suitable for modeling
    continuous, multi-modal data.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        out_size: int,
        num_layers: int,
        num_predictions: int,
        low_rank: Optional[int] = 64,
        top_p_or_k: Optional[Union[float, int]] = 1.0,
        min_log_std: float = -4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_size = out_size
        self.low_rank = low_rank
        self.num_predictions = num_predictions
        self.min_log_std = min_log_std
        self.top_p_or_k = top_p_or_k

        self.logits_processor = (
            TopPLogitsWarper(self.top_p_or_k)
            if isinstance(self.top_p_or_k, float)
            else (
                TopKLogitsWarper(self.top_p_or_k)
                if isinstance(self.top_p_or_k, int)
                else None
            )
        )

        self.mlp_stack = nn.Sequential(
            *[
                MLPLayer(hidden_size, intermediate_size, eps=eps)
                for _ in range(num_layers)
            ],
            RMSNorm(hidden_size, eps=eps),
        )

        if low_rank is None:
            self.proj_logits = nn.Linear(hidden_size, num_predictions, bias=False)
            self.proj_mus = nn.Linear(
                hidden_size, num_predictions * out_size, bias=False
            )
            self.proj_logs = nn.Linear(hidden_size, 1, bias=False)
        else:
            assert low_rank < out_size
            self.proj_logits = nn.Linear(hidden_size, num_predictions, bias=False)
            self.proj_mus = nn.Linear(
                hidden_size, num_predictions * low_rank, bias=False
            )
            self.proj_logs = nn.Linear(hidden_size, 1, bias=False)
            self.proj_else = nn.Linear(hidden_size, out_size, bias=False)
            self.low_mat = nn.Parameter(
                torch.empty(num_predictions, out_size, low_rank)
            )

    def forward(
        self,
        x: torch.Tensor,
        cfg_is_uncond: Optional[torch.Tensor] = None,
        cfg_scale: Optional[torch.Tensor] = None,
        cfg_active: Optional[torch.Tensor] = None,
        cfg_partner: Optional[torch.Tensor] = None,
        cfg_conditional_rep: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bt = x.size(0)
        n, d = self.num_predictions, self.low_rank or self.out_size

        x = self.mlp_stack(x)
        if (
            cfg_is_uncond is not None
            and cfg_scale is not None
            and cfg_active is not None
            and cfg_partner is not None
        ):
            # Native EarTTS guidance is applied after the MoG MLP stack and
            # before proj_logits/proj_mus/proj_logs/proj_else.
            x = _apply_cfg_after_mlp(
                x,
                cfg_is_uncond=cfg_is_uncond,
                cfg_scale=cfg_scale,
                cfg_active=cfg_active,
                cfg_partner=cfg_partner,
            )

        logits = self.proj_logits(x)

        # Apply top-p or top-k filtering to the mixture logits
        if self.logits_processor is not None:
            logits = self.logits_processor(None, logits.view(-1, n)).view_as(logits)

        # Sample a mixture component using the Gumbel-Max trick
        gumbel = gumbel_like(logits)
        if cfg_active is not None and cfg_conditional_rep is not None:
            gumbel = torch.where(
                cfg_active[:, None],
                gumbel[cfg_conditional_rep],
                gumbel,
            )
        mixture_indices = (nn.functional.log_softmax(logits, dim=-1) + gumbel).argmax(
            -1
        )

        # Select the mean corresponding to the sampled component
        mu = batch_matmul(
            x.view(bt, -1),
            self.proj_mus.weight.detach().view(n, d, -1),
            mixture_indices.view(bt),
        ).view(bt, d)
        if self.proj_mus.bias is not None:
            mu += self.proj_mus.bias.detach().view(n, d)[mixture_indices]

        if self.low_rank:
            mu = batch_matmul(
                mu.view(bt, -1),
                self.low_mat.detach().view(n, self.out_size, -1),
                mixture_indices.view(bt),
            ).view(bt, self.out_size)
            mu_res = self.proj_else(x)
        else:
            mu_res = torch.zeros((bt, d), device=x.device)

        logs = self.proj_logs(x).clamp_min(self.min_log_std)
        return mu * torch.exp(logs) + mu_res, logs


class MaskGITSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_quantizers = self.config.num_quantizers
        self.codebook_size = self.config.codebook_size
        self.noise_scale = self.config.noise_scale
        self.debug_cfg_contract = bool(
            getattr(self.config, "debug_cfg_contract", False)
        )

        # pre-compute how many tokens are unmasked at each iteration
        rates = np.linspace(0.0, 1.0, self.config.num_iter + 1)[:-1].reshape(-1, 1)
        masking_rates = np.power(
            1 - np.power(rates, self.config.exponent), 1 / self.config.exponent
        )
        num_maskings = np.ceil(masking_rates * self.num_quantizers).astype(int)
        num_maskings_shifted = np.pad(
            num_maskings[1:], ((0, 1), (0, 0)), constant_values=0
        )
        sampling_per_step = num_maskings - num_maskings_shifted
        sampling_per_step_flat = sampling_per_step.flatten()
        # Drop any values at the beginning that are 0
        first_nonzero = np.argmax(sampling_per_step_flat != 0)
        self.num_to_sample = sampling_per_step_flat[first_nonzero:].tolist()

        # create layers used for acoustic tokens embedding
        self.rvq_embs = nn.Parameter(
            torch.empty(
                self.config.num_quantizers,
                self.config.codebook_size,
                self.config.latent_size,
            )
        )
        self.embed_code = nn.Linear(
            self.config.latent_size, self.config.hidden_size, bias=False
        )
        # MoG head for generation (uncompiled part)
        self.mog_head = MoGHead(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            out_size=self.config.latent_size,
            num_layers=self.config.mog_num_layers,
            num_predictions=self.config.mog_num_predictions,
            low_rank=self.config.mog_low_rank,
            top_p_or_k=self.config.top_p_or_k,
            min_log_std=self.config.mog_min_log_std,
            eps=self.config.mog_eps,
        )

    def _depthsum_embedding(self, code: torch.Tensor) -> torch.Tensor:
        """Embeds all codes into a single embedding."""
        embs = nn.functional.pad(
            self.rvq_embs, [0, 0, 0, 1]
        )  # num_quantizers x (codebook_size + 1) x latent_size
        res = nn.functional.embedding(code[0], embs[0])
        for i in range(1, len(embs)):
            res = res + nn.functional.embedding(code[i], embs[i])
        return res

    def _depthsum_encoding_step_reshaped(
        self,
        r: torch.Tensor,  # [B*T, hidden_size]
        code: torch.Tensor,  # [num_quantizers, B*T]
        depth_str: int,
        k: int,
    ) -> torch.Tensor:
        """RVQ encoding with reshaped code tensor."""
        for i in range(depth_str, depth_str + k):
            # Compute distances: ||emb||² - 2⟨r, emb⟩
            idx_sel = (
                self.rvq_embs[i].pow(2).sum(-1)  # [vocab_size]
                - 2 * (r @ self.rvq_embs[i].T)  # [B*T, vocab_size]
            ).argmin(-1)  # [B*T]

            # Update residual
            emb_i = nn.functional.embedding(
                idx_sel,
                self.rvq_embs[i],
            )  # [B*T, latent_size]
            r = r - emb_i

            # Store selected indices
            code[i] = idx_sel

        return code

    def forward(
        self,
        hidden_states: torch.Tensor,
        cfg_enabled: Optional[torch.Tensor] = None,
        cfg_is_uncond: Optional[torch.Tensor] = None,
        cfg_pair_id: Optional[torch.Tensor] = None,
        cfg_scale: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs the iterative unmasking process for a single
        generation step.
        """

        device = hidden_states.device
        batch_size = int(hidden_states.shape[0])
        if cfg_enabled is None:
            cfg_enabled = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if cfg_is_uncond is None:
            cfg_is_uncond = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if cfg_pair_id is None:
            cfg_pair_id = torch.full(
                (batch_size,),
                -1,
                dtype=torch.long,
                device=device,
            )
        if cfg_scale is None:
            cfg_scale = torch.zeros(batch_size, dtype=torch.float32, device=device)
        if valid is None:
            valid = torch.ones(batch_size, dtype=torch.bool, device=device)

        (
            hidden_states,
            cfg_is_uncond,
            cfg_scale,
            cfg_active,
            cfg_partner,
            cfg_conditional_rep,
            inverse_order,
        ) = _prepare_cfg_sampling_batch(
            hidden_states,
            cfg_enabled=cfg_enabled,
            cfg_is_uncond=cfg_is_uncond,
            cfg_pair_id=cfg_pair_id,
            cfg_scale=cfg_scale,
            valid=valid,
        )
        complete_contract = ((~cfg_enabled) | cfg_active).all()
        if self.debug_cfg_contract and not bool(
            complete_contract.item()
        ):
            raise RuntimeError(
                "Incomplete EarTTS CFG model batch: "
                f"enabled={cfg_enabled.tolist()} "
                f"is_uncond={cfg_is_uncond.tolist()} "
                f"pair_id={cfg_pair_id.tolist()} "
                f"valid={valid.tolist()} "
                f"active={cfg_active.tolist()}"
            )

        # Initialize the full code tensor
        code = (
            torch.zeros(
                (self.num_quantizers, hidden_states.shape[0]),
                dtype=torch.long,
                device=device,
            )
            + self.codebook_size
        )
        # Iteratively unmask the continuous part of the code
        cnt = 0
        for k in self.num_to_sample:
            # Prepare input for the MoG head
            mog_input_embeds = self.embed_code(
                self._depthsum_embedding(code)
            )  # (BT x hidden_size)
            mog_input_embeds += hidden_states

            mog_mu, mog_logs = self.mog_head(
                mog_input_embeds,
                cfg_is_uncond=cfg_is_uncond,
                cfg_scale=cfg_scale,
                cfg_active=cfg_active,
                cfg_partner=cfg_partner,
                cfg_conditional_rep=cfg_conditional_rep,
            )
            normal = torch.randn_like(mog_mu)
            normal = torch.where(
                cfg_active[:, None],
                normal[cfg_conditional_rep],
                normal,
            )
            z = mog_mu + torch.exp(mog_logs) * normal * self.noise_scale
            code = self._depthsum_encoding_step_reshaped(z, code, cnt, k)
            # Match PyTorch EarTTS CFG: every MaskGIT iteration feeds the
            # conditional code trajectory back into both KV streams.
            code = torch.where(
                cfg_active.unsqueeze(0),
                code[:, cfg_conditional_rep],
                code,
            )

            cnt += k
        return code.transpose(0, 1)[inverse_order]  # BT x num_quantizers


@support_torch_compile
class EarTTSModel(nn.Module):
    """Embedding preparation + Gemma3 backbone (compiled together).

    MaskGIT sampling lives in :class:`EarTTSSamplerModel` so the iterative
    sampler can be skipped on prefill positions while still being CUDA-graph
    captured for decode-only batches. See :meth:`EarTTSForCausalLM.forward`.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.total_emb = EarTTSInputEmbedding(config)
        self.backbone = Gemma3Model(vllm_config=vllm_config, prefix=prefix)

        # Per-codebook silence acoustic tokens. Registered as a
        # persistent int32 buffer (loaded from the checkpoint under
        # ``model.sil_tokens``) rather than nn.Parameter so that vLLM's
        # automatic float dtype casting (e.g. ``model.to(bfloat16)``)
        # leaves it untouched.
        self.register_buffer(
            "sil_tokens",
            # Zero is a safe dummy-loader default; production checkpoints
            # overwrite this persistent buffer in ``load_weights``.
            torch.zeros(int(config.num_quantizers), dtype=torch.int32),
            persistent=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        acoustic_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        bos_mask: torch.Tensor,
        speaker_latent: torch.Tensor,
        cfg_is_uncond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through embeddings and backbone transformer.
        Returns the backbone's ``hidden_states``.
        """
        total_emb = self.total_emb(
            acoustic_tokens=acoustic_tokens,
            text_tokens=text_tokens,
            text_mask=text_mask,
            bos_mask=bos_mask,
            speaker_latent=speaker_latent,
            cfg_is_uncond=cfg_is_uncond,
        )
        hidden_states = self.backbone(
            input_ids, positions, intermediate_tensors, inputs_embeds=total_emb
        )
        return hidden_states


@support_torch_compile
class EarTTSSamplerModel(nn.Module):
    """MaskGIT sampler in its own compile group.

    Hosting the sampler in a separate ``@support_torch_compile`` module
    is what makes it possible for :meth:`EarTTSForCausalLM.forward` to:

    * Capture and replay a CUDA-graph for decode-only batches (where
      every position needs sampling).
    * Skip the sampler entirely on prefill positions, where the audio
      output isn't actually needed.
    * Run the sampler on a sliced subset of positions in mixed
      prefill+decode batches, with a ``BatchDescriptor`` override so the
      sampler's CUDA-graph cache is hit at the padded decode-batch size.

    The :meth:`forward` operates on a stable-address scratch buffer
    (:attr:`_sampler_input`) so callers can pass a transient slice
    (e.g. ``hidden_states[decode_idx]``) without breaking CUDA-graph
    replay. The non-compiled :meth:`sample` wrapper does that copy and
    then invokes the compiled :meth:`forward`.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.sampler = MaskGITSampler(config)

        # Stable-address scratch buffer for the sampler's input. Every
        # CUDA-graph replay must read from the same ``data_ptr()``; the
        # caller may pass either the full backbone output or a fresh
        # ``hidden_states[decode_idx]`` slice, so we copy into this
        # buffer (in :meth:`sample`) before invoking :meth:`forward`.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        hidden_size = config.hidden_size
        dtype = vllm_config.model_config.dtype
        self._sampler_input = torch.zeros(
            max_num_tokens, hidden_size, dtype=dtype
        )
        self._sampler_cfg_enabled = torch.zeros(max_num_tokens, dtype=torch.bool)
        self._sampler_cfg_is_uncond = torch.zeros(max_num_tokens, dtype=torch.bool)
        self._sampler_cfg_pair_id = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._sampler_cfg_scale = torch.zeros(max_num_tokens, dtype=torch.float32)
        self._sampler_valid = torch.zeros(max_num_tokens, dtype=torch.bool)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cfg_enabled: torch.Tensor,
        cfg_is_uncond: torch.Tensor,
        cfg_pair_id: torch.Tensor,
        cfg_scale: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled — runs MaskGIT on a (stable-address) hidden buffer."""
        return self.sampler(
            hidden_states,
            cfg_enabled=cfg_enabled,
            cfg_is_uncond=cfg_is_uncond,
            cfg_pair_id=cfg_pair_id,
            cfg_scale=cfg_scale,
            valid=valid,
        )

    def sample(
        self,
        hidden_states: torch.Tensor,
        *,
        cfg_enabled: torch.Tensor,
        cfg_is_uncond: torch.Tensor,
        cfg_pair_id: torch.Tensor,
        cfg_scale: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Non-compiled wrapper — copies into the stable buffer first.

        Mirrors the qwen3-tts code-predictor pattern: transient inputs
        are first written into a model-owned static-address buffer so
        the captured CUDA-graph for the compiled :meth:`forward` always
        reads from the recorded ``data_ptr()``.
        """
        seq_len = int(hidden_states.shape[0])
        buf = self._sampler_input[:seq_len]
        buf.copy_(hidden_states)
        enabled_buf = self._sampler_cfg_enabled[:seq_len]
        role_buf = self._sampler_cfg_is_uncond[:seq_len]
        pair_buf = self._sampler_cfg_pair_id[:seq_len]
        scale_buf = self._sampler_cfg_scale[:seq_len]
        valid_buf = self._sampler_valid[:seq_len]
        enabled_buf.copy_(cfg_enabled)
        role_buf.copy_(cfg_is_uncond)
        pair_buf.copy_(cfg_pair_id)
        scale_buf.copy_(cfg_scale)
        valid_buf.copy_(valid)
        return self(
            buf,
            enabled_buf,
            role_buf,
            pair_buf,
            scale_buf,
            valid_buf,
        )


# ---------------------------------------------------------------------------
# Outer model — the vLLM-Omni preprocess/postprocess entry point.
# ---------------------------------------------------------------------------


# Placeholder token id used to fill the per-step ``input_ids`` returned
# by :meth:`preprocess`. Must be a valid id in ``[0, config.vocab_size)``
# but is otherwise unused — the actual decode-vs-prefill behaviour is
# driven by the per-token buffers populated in :meth:`preprocess`.
#
# The width of the dummy logits tensor returned by
# :meth:`compute_logits` is taken from ``config.vocab_size`` (see
# :class:`EarTTSConfig`) so vLLM's sampler / ``LogitsProcessor`` and the
# model agree on the logits shape. ``compute_logits`` returns
# ``[0, -inf, ..., -inf]`` so the sampler's argmax always picks index 0
# regardless of how wide ``vocab_size`` is — the real audio output is
# the codes tensor exposed via :meth:`make_omni_output`.
_DUMMY_TOKEN_ID = 0


@ignore_torch_compile
@support_torch_compile
class EarTTSForCausalLM(nn.Module, SupportsPP):
    """EarTTS for vLLM-Omni.

    Inputs (passed via ``additional_information``):

      * ``embed.voice`` (prefill chunk 0 only) — Tensor of shape
        ``(Tref, hidden_size)`` carrying the user-supplied speaker
        latent. The user must also pass ``prompt_token_ids = [0] *
        Tref`` so the prefill placeholder length matches.
      * ``ids.output`` — Python ``list[int]`` of the most recently sent
        text tokens; preprocess consumes the last entry, so decode step
        ``k`` consumes ``t_k``.
      * CFG metadata (on every prefill/decode chunk): ``cfg_enabled``,
        ``cfg_role`` (``"cond"`` or ``"uncond"``), ``cfg_pair_id``, and
        ``cfg_scale``. Unconditional rows replace only text conditioning
        with ``model.total_emb.null_emb``. Complete decode pairs are sampled
        with native EarTTS guidance and receive identical acoustic codes.

    Per-step flow (see module docstring for details):

    ``preprocess`` populates five model-owned buffers
    (:attr:`_acoustic_tokens`, :attr:`_text_tokens`, :attr:`_text_mask`,
    :attr:`_bos_mask`, :attr:`_speaker_latent`) at each request's
    flat-batch offset. ``forward`` slices them up to ``num_tokens`` and
    runs the compiled :class:`EarTTSModel` (embedding + Gemma3
    backbone) for every position, then conditionally invokes the
    compiled :class:`EarTTSSamplerModel` (MaskGIT) to produce codes.
    The sampler is skipped on prefill positions — see :meth:`forward`
    for details. The generated codes (BTx31) are written to
    :attr:`_out_codes` and exposed as a multimodal output by
    :meth:`make_omni_output` (see there for the key it uses).
    ``postprocess`` stashes the final-frame codes under
    ``last_acoustic_codes`` for the next decode step's
    :meth:`preprocess`.

    Sampler skipping mirrors the qwen3-tts code-predictor pattern:

    * **Profile / dummy run** (``attn_metadata is None``) and
      **decode-only batches** (``max_query_len == 1``) run the sampler
      on every token so the captured CUDA graph covers all of
      ``cudagraph_capture_sizes``.
    * **Mixed prefill+decode batches**: only decode-token positions go
      through the sampler. The sampler's ``BatchDescriptor`` is
      overridden to the padded decode-batch size so the right captured
      graph is replayed.
    * **Prefill-only batches**: the sampler is skipped entirely.
    * Prefill rows of :attr:`_out_codes` are intentionally not
      written. ``last_acoustic_codes`` returned by :meth:`postprocess`
      after prefill is therefore undefined — :meth:`_preprocess_decode`
      seeds the first decode step's acoustic input with the acoustic
      pad id (``codebook_size``) so this never matters.
    """

    # ``model.sampler.*`` lands on :attr:`sampler_module` (the MaskGIT compile
    # group). Other prefixes (``model.total_emb.``, ``model.backbone.``) match
    # the module layout 1:1.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.sampler.": "sampler_module.sampler.",
        }
    )

    # Omni preprocess/postprocess hooks (consumed by the gpu model runner).
    has_preprocess = True
    has_postprocess = True
    have_multimodal_outputs = True

    # No ``gpu_resident_buffer_keys``: vLLM-Omni's opt-out from the
    # ``model_intermediate_buffer`` D2H round-trip is keyed by
    # ``(type_key, qualifier)`` pairs and is only consulted for *nested*
    # payload entries, so a model whose payloads are flat -- as both stages
    # here are -- cannot express its keys in that form. Declaring flat names
    # is not merely inert, it breaks: the runner unpacks every declared key
    # as a pair as soon as any nested entry arrives, and 0.24 onwards always
    # sends one (``meta``).

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config

        # Embedding + Gemma3 backbone — runs on every position. Built
        # under the default ``"backbone"`` model tag (vLLM's compile
        # cache key for the main model). We don't wrap this in a
        # ``set_model_tag`` block because :func:`set_model_tag` asserts
        # the new tag differs from the current one and the default is
        # already ``"backbone"``.
        self.model = EarTTSModel(
            vllm_config=vllm_config,
            prefix=prefix,
        )

        # MaskGIT sampler in its own compile group, so it can be invoked
        # conditionally (decode positions only, or skipped entirely on
        # prefill-only batches) while still being CUDA-graph captured
        # for decode-only batches over ``cudagraph_capture_sizes``. The
        # ``"sampler"`` tag keys the sampler's compile cache separately
        # from the backbone's.
        with set_model_tag("sampler"):
            self.sampler_module = EarTTSSamplerModel(
                vllm_config=vllm_config,
                prefix=prefix,
            )

        # Pad ids used by buffers / preprocess. Match the conventions of
        # the original EarTTSInputEmbedding: an acoustic token id of
        # ``codebook_size`` is the trailing "no audio" pad row in
        # ``rvq_embs`` (which has ``codebook_size + 1`` entries).
        # How the sampled codes are surfaced from ``make_omni_output``, driven
        # by the stage's pipeline-config ``engine_output_type``:
        #
        # * "audio" (this stage is the final, client-facing one, which is the
        #   split VoiceChat layout): emit under the ``model_outputs`` key.
        #   vLLM-Omni's output processor remaps ``model_outputs`` to the
        #   drainable ``audio`` modality key, so DELTA streaming drains it
        #   after every step and the client receives one frame per step.
        #   Any other key is retained across steps *and* concatenated along
        #   the last dimension (``get_accumulation_strategy`` maps the audio
        #   modality to ``CONCAT_LAST``), which for a ``T x num_quantizers``
        #   code tensor silently widens the per-step frame instead of
        #   appending to it.
        # * otherwise: emit ``audio_codes`` for a downstream stage to consume.
        engine_output_type = getattr(
            vllm_config.model_config, "engine_output_type", None
        )
        self._single_stage_audio = str(engine_output_type or "").lower() == "audio"

        self._num_quantizers: int = int(self.config.num_quantizers)
        self._hidden_size: int = int(self.config.hidden_size)
        self._acoustic_pad_id: int = int(self.config.codebook_size)
        text_pad_id = getattr(self.config, "pad_token_id", None)
        eos_token_id = getattr(self.config, "eos_token_id", None)
        if text_pad_id is None or eos_token_id is None:
            raise ValueError(
                "EarTTS config.json must set pad_token_id and eos_token_id from the "
                "source VoiceChat tokenizer. Re-run convert_duplex_eartts_checkpoint.py."
            )
        self._text_pad_id: int = int(text_pad_id)
        self._eos_token_id: int = int(eos_token_id)

        # ── Persistent stable-address buffers ────────────────────────
        # Plain tensor attributes (not nn.Parameter / not register_buffer):
        #   * AutoWeightsLoader only walks named_parameters() and persistent
        #     registered buffers, so plain attributes are invisible to it
        #     (no spurious "missing weight" errors during load_weights).
        #   * vLLM constructs models inside
        #     ``with torch.device(device_config.device):`` so a bare
        #     ``torch.zeros(...)`` here is allocated directly on the GPU.
        #   * Addresses stay stable across CUDA graph replays as long as
        #     we never re-assign these names (only do in-place writes via
        #     copy_/fill_/indexed assignment), which is what the rest of
        #     this class does. The piecewise CUDAGraphWrapper records
        #     data_ptr() at capture time and expects the same pointer at
        #     replay time — that holds with plain tensors.
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        model_dtype = vllm_config.model_config.dtype

        self._acoustic_tokens = torch.full(
            (max_num_tokens, self._num_quantizers),
            self._acoustic_pad_id,
            dtype=torch.long,
        )
        self._text_tokens = torch.full(
            (max_num_tokens,), self._text_pad_id, dtype=torch.long
        )
        self._text_mask = torch.zeros(max_num_tokens, dtype=torch.long)
        self._bos_mask = torch.zeros(max_num_tokens, dtype=torch.long)
        # Speaker latent buffer — model dtype, hidden_size wide.
        # Decode rows stay all-zero (which the embedding module reads
        # as "latent not provided" so ``audio_emb`` is preserved).
        # Prefill rows are populated from the user-supplied tensor.
        self._speaker_latent = torch.zeros(
            max_num_tokens, self._hidden_size, dtype=model_dtype
        )
        # Per-token CFG contract. These plain tensors follow the same
        # stable-address rules as the model input buffers above and are copied
        # into the sampler's own CUDA-graph scratch buffers before sampling.
        self._cfg_enabled = torch.zeros(max_num_tokens, dtype=torch.bool)
        self._cfg_is_uncond = torch.zeros(max_num_tokens, dtype=torch.bool)
        self._cfg_pair_id = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._cfg_scale = torch.zeros(max_num_tokens, dtype=torch.float32)
        # vLLM-Omni 0.26 computes per-request flat-batch slices but does not
        # pass ``start``/``end`` into preprocess. The CFG scheduler guarantees
        # cond then uncond order, so this cursor reconstructs those slices.
        self._preprocess_cursor = 0
        self._out_codes = torch.zeros(
            max_num_tokens, self._num_quantizers, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compatibility shim — not actually consumed at runtime since
        every forward goes through ``inputs_embeds`` assembled inside
        :meth:`forward`.
        """
        return self.model.backbone.embed_input_ids(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    @staticmethod
    def _unwrap_singleton(value: Any) -> Any:
        """Unwrap a possibly list-wrapped scalar (e.g. ``[tensor]``)."""
        if isinstance(value, list):
            return value[0] if value else None
        return value

    @staticmethod
    def _payload_get(info_dict: dict[str, Any], category: str, qualifier: str) -> Any:
        """Read one ``OmniPayload`` field, e.g. ``("embed", "voice")``.

        Categories arrive as sub-dicts, which is also what lets the runner
        keep a prefill-only field alive while a per-step field in another
        category is replaced.
        """
        sub = info_dict.get(category)
        return sub.get(qualifier) if isinstance(sub, dict) else None

    @classmethod
    def _cfg_scalar(cls, value: Any) -> Any:
        value = cls._unwrap_singleton(value)
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1, (
                "EarTTS CFG metadata tensors must contain one scalar; "
                f"got shape={tuple(value.shape)}."
            )
            return value.item()
        return value

    @classmethod
    def _stable_cfg_pair_id(cls, value: Any) -> int:
        """Map request-provided pair ids to a deterministic signed int64."""
        value = cls._cfg_scalar(value)
        assert value is not None and not isinstance(
            value, bool
        ), "EarTTS CFG requires a non-empty ``cfg_pair_id``."
        if isinstance(value, int):
            assert (
                -(1 << 63) <= value < (1 << 63)
            ), f"EarTTS cfg_pair_id={value} does not fit in int64."
            return value
        if isinstance(value, float):
            assert (
                value.is_integer()
            ), f"EarTTS cfg_pair_id must be integral or a string; got {value}."
            return cls._stable_cfg_pair_id(int(value))
        encoded = str(value).encode("utf-8")
        assert encoded, "EarTTS CFG requires a non-empty ``cfg_pair_id``."
        return int.from_bytes(
            hashlib.blake2b(encoded, digest_size=8).digest(),
            "little",
        ) & ((1 << 63) - 1)

    def _write_cfg_state(
        self,
        *,
        start: int,
        span_len: int,
        info_dict: dict[str, Any],
    ) -> None:
        """Validate one request/chunk's CFG metadata and fill static rows."""
        enabled_value = self._cfg_scalar(info_dict.get("cfg_enabled", False))
        if isinstance(enabled_value, str):
            normalized = enabled_value.strip().lower()
            assert normalized in {
                "true",
                "false",
                "1",
                "0",
            }, f"EarTTS cfg_enabled must be boolean; got {enabled_value!r}."
            cfg_enabled = normalized in {"true", "1"}
        else:
            cfg_enabled = bool(enabled_value)

        cfg_is_uncond = False
        cfg_pair_id = -1
        scale_value = self._cfg_scalar(info_dict.get("cfg_scale"))
        if scale_value is None:
            scale_value = getattr(self.config, "guidance_scale", 0.5)
        cfg_scale = float(scale_value)
        assert math.isfinite(
            cfg_scale
        ), f"EarTTS cfg_scale must be finite; got {cfg_scale}."

        if cfg_enabled:
            role = str(self._cfg_scalar(info_dict.get("cfg_role")) or "").lower()
            assert role in {"cond", "uncond"}, (
                "EarTTS CFG requires cfg_role='cond' or 'uncond'; " f"got {role!r}."
            )
            cfg_is_uncond = role == "uncond"
            cfg_pair_id = self._stable_cfg_pair_id(info_dict.get("cfg_pair_id"))

        end = start + span_len
        self._cfg_enabled[start:end].fill_(cfg_enabled)
        self._cfg_is_uncond[start:end].fill_(cfg_is_uncond)
        self._cfg_pair_id[start:end].fill_(cfg_pair_id)
        self._cfg_scale[start:end].fill_(cfg_scale)

    def _validate_speaker_latent(self, value: Any) -> torch.Tensor:
        """Assert ``speaker_latent`` has shape ``(Tref, hidden_size)``."""
        x = self._unwrap_singleton(value)
        assert isinstance(x, torch.Tensor), (
            f"speaker_latent must be a torch.Tensor; got {type(x).__name__}."
        )
        assert x.ndim == 2 and x.shape[1] == self._hidden_size, (
            "speaker_latent must have shape (Tref, hidden_size="
            f"{self._hidden_size}); got {tuple(x.shape)}."
        )
        return x.to(dtype=self._speaker_latent.dtype).contiguous()

    def _build_prefill_tensors(
        self,
        speaker_latent: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the cached prefill ``(text_tokens, text_mask, bos_mask,
        speaker_latent)`` of length ``prefill_len = speaker_latent.shape[0]``.

        Layout: ``text_tokens = [PAD] * (n - 1) + [EOS]``,
        ``text_mask = [0] * (n - 2) + [1, 1]``,
        ``bos_mask = [0] * (n - 1) + [1]``. Acoustic tokens are not
        cached — :meth:`preprocess` broadcasts ``model.sil_tokens`` at
        every prefill position.
        """
        prefill_len = int(speaker_latent.shape[0])
        assert prefill_len > 0, (
            "speaker_latent must have at least one frame "
            f"(got shape={tuple(speaker_latent.shape)})."
        )

        text_tokens = torch.full(
            (prefill_len,), self._text_pad_id, dtype=torch.long, device=device
        )
        text_tokens[-1] = self._eos_token_id

        text_mask = torch.zeros(prefill_len, dtype=torch.long, device=device)
        text_mask[max(0, prefill_len - 2):] = 1

        bos_mask = torch.zeros(prefill_len, dtype=torch.long, device=device)
        bos_mask[-1] = 1

        speaker_latent = speaker_latent.to(
            device=device, dtype=self._speaker_latent.dtype, non_blocking=True
        ).contiguous()

        return text_tokens, text_mask, bos_mask, speaker_latent

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        *,
        start: int = 0,
        end: int = 0,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build per-request ``(input_ids, inputs_embeds)`` for this step.

        Prefill (``span_len > 1``):
            On the first prefill chunk, constructs the per-position
            prefill tensors of length
            ``prefill_len = speaker_latent.shape[0]``:

            * ``text_tokens``     = ``[PAD] * (prefill_len - 1) + [EOS]``
            * ``text_mask``       = ``[0] * (prefill_len - 2) + [1, 1]``
            * ``bos_mask``        = ``[0] * (prefill_len - 1) + [1]``
            * ``acoustic_tokens`` = ``model.sil_tokens`` broadcast at
              every position (only the BOS frame's ``audio_emb`` is
              actually consumed; the others are replaced by
              ``speaker_latent`` inside the embedding module).
            * ``speaker_latent``  = the user-supplied tensor.

            ``embed.voice`` is the only required
            ``additional_information`` field. Multi-chunk prefill is
            tracked by ``ear_prefill_offset``; the cached
            ``ear_prefill_speaker_latent`` is sliced into each chunk.

        Decode (``span_len == 1``):
            Takes the newest text token from ``ids.output``, which the
            producer (a user-driven :class:`StreamingInput` or an
            upstream stage in an ``async_chunk`` pipeline such as
            ``nemotron_voicechat``) refreshes on every step.

            Acoustic input rules (in order):
              * ``text_token == EOS`` → ``model.sil_tokens``.
              * First decode (``ear_decode_offset == 0``) →
                broadcast acoustic pad id (``codebook_size``).
              * Otherwise → ``last_acoustic_codes`` (stashed by
                :meth:`postprocess` after the previous step).

            ``text_mask = 1``, ``bos_mask = 0``,
            ``speaker_latent = 0``.
        """
        # Normalize: some runner paths still pass per-request state
        # nested under ``additional_information`` instead of flattened.
        nested = info_dict.get("additional_information")
        if isinstance(nested, dict):
            merged = {
                k: v for k, v in info_dict.items() if k != "additional_information"
            }
            for k, v in nested.items():
                merged.setdefault(k, v)
            info_dict = merged

        device = input_ids.device
        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            base = (
                input_embeds
                if input_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            return input_ids, base, {}

        explicit_start = int(start)
        explicit_end = int(end)
        if explicit_end > explicit_start:
            flat_start = explicit_start
        else:
            role = str(
                self._cfg_scalar(info_dict.get("cfg_role"))
                or ""
            ).lower()
            cfg_enabled = bool(
                self._cfg_scalar(
                    info_dict.get("cfg_enabled", False)
                )
            )
            if not cfg_enabled or role == "cond":
                self._preprocess_cursor = 0
            flat_start = self._preprocess_cursor
            self._preprocess_cursor += span_len

        self._write_cfg_state(
            start=flat_start,
            span_len=span_len,
            info_dict=info_dict,
        )

        if span_len > 1:
            return self._preprocess_prefill(
                input_ids=input_ids,
                input_embeds=input_embeds,
                start=flat_start,
                span_len=span_len,
                device=device,
                info_dict=info_dict,
            )
        return self._preprocess_decode(
            input_ids=input_ids,
            input_embeds=input_embeds,
            start=flat_start,
            device=device,
            info_dict=info_dict,
        )

    def _preprocess_prefill(
        self,
        *,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        start: int,
        span_len: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prefill branch of :meth:`preprocess`. Writes one chunk-slice of
        the cached prefill tensors into the static buffers."""
        cached_speaker_latent = info_dict.get("ear_prefill_speaker_latent")

        info_update: dict[str, Any] = {}
        if not isinstance(cached_speaker_latent, torch.Tensor):
            # First chunk: build & cache prefill tensors from the
            # user-supplied speaker latent.
            speaker_latent = self._validate_speaker_latent(
                self._payload_get(info_dict, "embed", "voice")
            )
            (
                cached_text_tokens,
                cached_text_mask,
                cached_bos_mask,
                cached_speaker_latent,
            ) = self._build_prefill_tensors(speaker_latent, device=device)

            info_update["ear_prefill_text_tokens"] = cached_text_tokens
            info_update["ear_prefill_text_mask"] = cached_text_mask
            info_update["ear_prefill_bos_mask"] = cached_bos_mask
            info_update["ear_prefill_speaker_latent"] = cached_speaker_latent
            info_update["ear_prefill_offset"] = 0
            info_update["ear_decode_offset"] = 0
        else:
            cached_text_tokens = info_dict["ear_prefill_text_tokens"]
            cached_text_mask = info_dict["ear_prefill_text_mask"]
            cached_bos_mask = info_dict["ear_prefill_bos_mask"]

        offset = int(info_dict.get("ear_prefill_offset", 0) or 0)
        full_len = int(cached_speaker_latent.shape[0])
        s, e = offset, offset + span_len
        assert 0 <= s and e <= full_len, (
            "prefill chunk overshoots cached prefill: offset="
            f"{offset}, span_len={span_len}, prefill_len={full_len}. "
            "User must pass prompt_token_ids of length "
            "speaker_latent.shape[0]."
        )

        buf_s = start
        buf_e = buf_s + span_len
        self._text_tokens[buf_s:buf_e].copy_(cached_text_tokens[s:e])
        self._text_mask[buf_s:buf_e].copy_(cached_text_mask[s:e])
        self._bos_mask[buf_s:buf_e].copy_(cached_bos_mask[s:e])
        self._speaker_latent[buf_s:buf_e].copy_(cached_speaker_latent[s:e])
        # Acoustic input is sil_tokens broadcast — only the BOS-frame's
        # audio_emb is consumed (the rest get replaced by speaker_latent
        # inside EarTTSInputEmbedding).
        self._acoustic_tokens[buf_s:buf_e] = self.model.sil_tokens.to(
            self._acoustic_tokens.dtype
        )

        info_update["ear_prefill_offset"] = offset + span_len

        # Placeholder input_ids; compiled forward reads the buffers, not these.
        input_ids_out = torch.full_like(input_ids, _DUMMY_TOKEN_ID)
        return input_ids_out, input_embeds, info_update

    def _preprocess_decode(
        self,
        *,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        start: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Decode branch of :meth:`preprocess`. Writes one row at ``start``."""
        chunk_text_tokens = self._payload_get(info_dict, "ids", "output")
        assert isinstance(chunk_text_tokens, list) and chunk_text_tokens, (
            "EarTTS decode requires a non-empty ``ids.output`` list in "
            f"additional_information; got {type(chunk_text_tokens).__name__} "
            f"with available keys {sorted(info_dict)}."
        )

        decode_offset = int(info_dict.get("ear_decode_offset", 0) or 0)
        # The newest token is the one this step consumes: the producer sends
        # exactly one per step, and a producer that resends a history still
        # has the current token at the end.
        text_token_id = int(chunk_text_tokens[-1])

        buf_s = start

        # Acoustic input selection:
        #   * EOS subword → force sil_tokens (return to silence).
        #   * First decode after prefill → seed with the acoustic pad id
        #     (codebook_size) broadcast across all quantizers.
        #   * Otherwise → previous-step predicted codes.
        if text_token_id == self._eos_token_id:
            self._acoustic_tokens[buf_s].copy_(
                self.model.sil_tokens.to(self._acoustic_tokens.dtype)
            )
        elif decode_offset == 0:
            self._acoustic_tokens[buf_s].fill_(self._acoustic_pad_id)
        else:
            last_codes = info_dict.get("last_acoustic_codes")
            assert isinstance(last_codes, torch.Tensor) and last_codes.numel() > 0, (
                "EarTTS decode (offset > 0) requires "
                "``last_acoustic_codes`` from the previous step's "
                "postprocess."
            )
            ac = (
                last_codes.to(device=device, dtype=torch.long)
                .reshape(-1)[: self._num_quantizers]
            )
            self._acoustic_tokens[buf_s, : ac.shape[0]].copy_(ac)

        self._text_tokens[buf_s] = text_token_id
        self._text_mask[buf_s] = 1
        self._bos_mask[buf_s] = 0
        # Decode never replaces audio_emb with a latent.
        self._speaker_latent[buf_s].zero_()

        info_update: dict[str, Any] = {"ear_decode_offset": decode_offset + 1}
        return input_ids, input_embeds, info_update

    # ------------------------------------------------------------------
    # forward — runs the compiled embedding + backbone, then the sampler
    # only on decode positions (skipping the expensive MaskGIT loop on
    # prefill positions).
    # ------------------------------------------------------------------

    def _get_decode_idxs(self):
        """Return ``(decode_token_indices, num_requests)`` for sampler dispatch.

        Mirrors the qwen3-tts code-predictor pattern:

        * ``(None, 0)`` → run sampler on every token. Used during
          profile / dummy runs (no ``attn_metadata``) and decode-only
          batches (``max_query_len == 1``), so the captured CUDA graph
          covers all of ``cudagraph_capture_sizes``.
        * ``(decode_token_indices, num_requests)`` → run sampler only on
          the listed positions. ``decode_token_indices`` is padded up to
          the next captured CUDA-graph size (so the sampler's graph
          cache is hit) and ``num_requests`` is the unpadded count of
          real decode tokens (used to scatter codes back into the right
          rows of :attr:`_out_codes`).
        """
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        if attn_metadata is None:
            # Profile / dummy run. Apply sampler everywhere so capture
            # covers every cudagraph_capture_sizes value.
            return None, 0

        if isinstance(attn_metadata, dict):
            any_layer_meta = next(iter(attn_metadata.values()))
        else:
            any_layer_meta = attn_metadata

        if any_layer_meta.max_query_len == 1:
            # Decode-only batch: every position is a decode position,
            # so just run the sampler over the whole flat batch.
            return None, 0

        start_loc = any_layer_meta.query_start_loc
        tokens_per_req = start_loc[1:] - start_loc[:-1]
        is_decode = (tokens_per_req == 1)
        decode_token_indices = start_loc[:-1][is_decode]

        num_requests = decode_token_indices.shape[0]
        padded_num_requests = num_requests
        if (
            self.vllm_config.compilation_config.cudagraph_mode
            != CUDAGraphMode.NONE
        ):
            sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            idx = bisect.bisect_left(sizes, num_requests)
            if idx < len(sizes):
                padded_num_requests = sizes[idx]
        if padded_num_requests != num_requests:
            decode_token_indices = torch.nn.functional.pad(
                decode_token_indices,
                (0, padded_num_requests - num_requests),
            )
        return decode_token_indices, num_requests

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Run the compiled embedding + backbone over every position,
        then run the compiled MaskGIT sampler only on decode positions
        (the sampler is skipped on prefill-only batches and on prefill
        rows of mixed batches). ``inputs_embeds`` is ignored — the
        actual embedding is assembled inside the compiled
        :class:`EarTTSInputEmbedding` from the per-token buffers
        populated by :meth:`preprocess`.
        """
        num_tokens = int(input_ids.shape[0])

        acoustic_tokens = self._acoustic_tokens[:num_tokens]
        text_tokens = self._text_tokens[:num_tokens]
        text_mask = self._text_mask[:num_tokens]
        bos_mask = self._bos_mask[:num_tokens]
        speaker_latent = self._speaker_latent[:num_tokens]
        cfg_is_uncond = self._cfg_is_uncond[:num_tokens]

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            acoustic_tokens=acoustic_tokens,
            text_tokens=text_tokens,
            text_mask=text_mask,
            bos_mask=bos_mask,
            speaker_latent=speaker_latent,
            cfg_is_uncond=cfg_is_uncond,
        )

        decode_idx, num_req = self._get_decode_idxs()
        if decode_idx is None:
            # Dummy/profile run or decode-only batch: sample everywhere.
            codes = self.sampler_module.sample(
                hidden_states,
                cfg_enabled=self._cfg_enabled[:num_tokens],
                cfg_is_uncond=self._cfg_is_uncond[:num_tokens],
                cfg_pair_id=self._cfg_pair_id[:num_tokens],
                cfg_scale=self._cfg_scale[:num_tokens],
                valid=torch.ones(
                    num_tokens,
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
            )
            self._out_codes[:num_tokens].copy_(codes.to(dtype=torch.long))
        elif num_req > 0:
            # Mixed batch: gather decode positions, override the
            # BatchDescriptor so the sampler's CUDA-graph cache is hit
            # at the padded decode-batch size.
            ctx = get_forward_context()
            orig_batch_descriptor = ctx.batch_descriptor
            ctx.batch_descriptor = BatchDescriptor(
                num_tokens=decode_idx.shape[0],
            )
            decode_hidden = hidden_states[decode_idx]
            sampler_valid = (
                torch.arange(
                    decode_idx.shape[0],
                    device=decode_idx.device,
                )
                < num_req
            )
            codes = self.sampler_module.sample(
                decode_hidden,
                cfg_enabled=self._cfg_enabled[decode_idx],
                cfg_is_uncond=self._cfg_is_uncond[decode_idx],
                cfg_pair_id=self._cfg_pair_id[decode_idx],
                cfg_scale=self._cfg_scale[decode_idx],
                valid=sampler_valid,
            )
            ctx.batch_descriptor = orig_batch_descriptor

            valid_dec_idx = decode_idx[:num_req]
            self._out_codes[valid_dec_idx] = codes[:num_req].to(
                dtype=torch.long
            )
        # Prefill-only batch: sampler skipped. ``_out_codes`` rows for
        # those positions are not written here on purpose — callers
        # must not rely on them; the public per-decode-step contract
        # is driven by ``last_acoustic_codes`` from postprocess and
        # the seed rules in :meth:`_preprocess_decode`.

        return hidden_states

    # ------------------------------------------------------------------
    # compute_logits — sampler bypass (the real output is ``codes``)
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: Union[torch.Tensor, OmniOutput],
        sampling_metadata: Any = None,
    ) -> Optional[torch.Tensor]:
        """Return zero logits of width ``config.vocab_size``.

        ``config.vocab_size`` is what vLLM's sampler / ``LogitsProcessor``
        use to size their working buffers, so deriving the width from
        the same field guarantees the two agree. The sampled token id
        is irrelevant: ``input_ids`` are never consumed by the model
        (the per-step decode behaviour is driven by the buffers
        populated in :meth:`preprocess`), and the real audio output is
        the codes tensor exposed via :meth:`make_omni_output`.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        batch_size = hidden_states.shape[0]
        return hidden_states.new_zeros(batch_size, int(self.config.vocab_size))

    # ------------------------------------------------------------------
    # multimodal output plumbing
    # ------------------------------------------------------------------

    def make_omni_output(
        self,
        model_outputs: Union[torch.Tensor, OmniOutput],
        **_: Any,
    ) -> OmniOutput:
        """Wrap backbone hidden states with the codes generated by the
        sampler (BTx31).

        The key depends on whether this stage is the client-facing one; see
        ``self._single_stage_audio`` in :meth:`__init__`. ``postprocess``
        accepts either.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        num_tokens = int(hidden.shape[0])
        audio_codes = self._out_codes[:num_tokens].clone()
        key = "model_outputs" if self._single_stage_audio else "audio_codes"
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={key: audio_codes},
        )

    # ------------------------------------------------------------------
    # postprocess — stash last-frame codes for the next decode step
    # ------------------------------------------------------------------

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Pull the last-frame codes out of the multimodal output (or,
        as a fallback, out of :attr:`_out_codes` using the slice's
        storage offset) and stash them under ``last_acoustic_codes`` so
        the next step's :meth:`preprocess` can use them as the decode
        input.
        """
        if hidden_states.numel() == 0:
            return {}

        mm = multimodal_outputs or {}
        audio_codes = mm.get("audio_codes")
        if audio_codes is None:
            audio_codes = mm.get("model_outputs")
        if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() > 0:
            # ``hidden_states`` is a slice of the flat batch. Recover
            # the request's last position via storage_offset and pick
            # the corresponding row from ``audio_codes``.
            stride0 = hidden_states.stride(0) or 1
            req_start = hidden_states.storage_offset() // stride0
            last = req_start + hidden_states.shape[0] - 1
            last_codes = audio_codes[last : last + 1].detach()
            return {"last_acoustic_codes": last_codes}

        return {}

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        skip_prefixes: list[str] = []
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")

        # The Gemma3 backbone keeps a vestigial ``embed_tokens`` layer,
        # but this model never consumes ``input_ids`` (every forward
        # goes through ``inputs_embeds`` assembled from the audio / text
        # buffers in :meth:`preprocess`). We expose a 2-class placeholder
        # ``vocab_size`` purely so the vLLM sampler's working buffers
        # match the dummy logits returned by :meth:`compute_logits`.
        # The checkpoint, however, ships ``embed_tokens.weight`` at the
        # original tokenizer vocab size — which trips
        # ``VocabParallelEmbedding``'s
        # ``loaded_weight.shape[output_dim] == self.org_vocab_size``
        # assertion. Truncate (or pad) the loaded weight to
        # ``(config.vocab_size, hidden_size)`` so the assertion passes;
        # the surviving rows are never consumed at runtime.
        target_vocab = int(self.config.vocab_size)
        embed_weight_name = "model.backbone.embed_tokens.weight"

        def _adjusted_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, w in weights:
                if name == embed_weight_name and w.dim() >= 1 and w.shape[0] != target_vocab:
                    if w.shape[0] >= target_vocab:
                        yield name, w[:target_vocab].contiguous()
                    else:
                        pad = torch.zeros(
                            target_vocab - w.shape[0],
                            *w.shape[1:],
                            dtype=w.dtype,
                            device=w.device,
                        )
                        yield name, torch.cat([w, pad], dim=0).contiguous()
                else:
                    yield name, w

        # ``AutoWeightsLoader`` only dispatches into child modules and
        # ``nn.Parameter``s, so any registered buffer (e.g.
        # ``model.sil_tokens``) needs to be routed manually. Resolve the
        # buffer names *after* applying ``hf_to_vllm_mapper`` so the
        # checkpoint key matches the in-model attribute path.
        buffers_dict = dict(self.named_buffers())
        loaded_buffer_names: set[str] = set()

        def _route_buffers(
            stream: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in stream:
                if name in buffers_dict:
                    buf = buffers_dict[name]
                    with torch.no_grad():
                        buf.copy_(weight.to(buf.dtype))
                    loaded_buffer_names.add(name)
                    continue
                yield name, weight

        # ``hf_to_vllm_mapper`` rewrites ``model.sampler.*`` to
        # ``sampler_module.sampler.*`` so the upstream EarTTS checkpoint
        # (which still places the MaskGIT sampler under ``model.``) lands
        # on the dedicated :attr:`sampler_module` compile group.
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        loaded = loader.load_weights(
            _route_buffers(_adjusted_weights()), mapper=self.hf_to_vllm_mapper
        )
        loaded.update(loaded_buffer_names)
        return loaded
