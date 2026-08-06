# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Parallel Expert Encoder (PEE) for ASR / audio.

A PEE hosts several independently-trained *whole* encoders ("experts") inside a
single container module so they share one front-end, batching, and inference
entry point. Unlike a Mixture-of-Experts layer, **PEE does not route** tokens to
a sparse subset of experts: every expert preserves its own, unmodified inference
path and produces its own output. The only thing borrowed from MoE is the
*compute* trick -- the position-wise feed-forward (FFN) matmuls of
shape-compatible experts are batched into a single grouped-GEMM instead of being
issued as many small per-expert kernels.

Concretely:

- ``GGEMMTransformerEncoder`` -- a container that owns N expert encoders
  (e.g. a multilingual-ASR :class:`MoETransformerEncoder`, a dense diarization
  :class:`TransformerEncoder`, a dense sound-event :class:`TransformerEncoder`)
  and exposes a per-expert forward that is bit-for-bit identical to running the
  original encoder standalone.
- ``GroupedFeedForward`` -- a numerically-exact, batched replacement for a list
  of same-shape :class:`FeedForward` modules. It stacks the expert weights and
  evaluates them with two batched matmuls (the portable grouped-GEMM stand-in
  for a fused CUTLASS/Triton grouped-GEMM kernel).
- Helpers to (a) **bucket** expert FFNs by shape so each grouped-GEMM call sees
  a uniform ``(d_model, d_hidden)`` and (b) **zero-pad** a narrow expert FFN up
  to a wider ``d_model`` when a single uniform grouped tensor is required.

The bucketing / padding helpers exist specifically to reconcile the heterogeneous
experts that motivate this module. On the PEE-v2 checkpoints the diarization
(speaker) expert is deliberately half-width at ``d_model = 1024`` while the
speech-MoE and sound experts run at ``d_model = 2048``, yet all three share the
same FFN hidden size of ``1024`` and the same ``head_dim = 128``. See the
module-level "Reconciling the half-width speaker FFN" note below and the project
README for the design rationale.

References:
- "Parallel Expert Encoders" (motivation / theory).
- "Bucketed Group-GEMM Multi-Expert Inference" (kernel / binding technique).
"""

import contextlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import and_masks, create_block_mask

# SDPA backend preference for the packed / serial-SDPA attention paths. We ask
# the dispatcher for FlashAttention-2 first (fastest; eligible only when the
# attn_mask is None, i.e. no materialized padding/bias mask -- see
# ``_packed_attention_step``), then the memory-efficient kernel (accepts an
# additive float bias, used by rel_pos experts and the padded case), and finally
# the math fallback so the call can never silently fail to dispatch.
_SDPA_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

from nemo.collections.asr.modules.moe_transformer_encoder import MoEFeedForward
from nemo.collections.asr.modules.transformer_encoder import (
    FeedForward,
    TransformerEncoder,
    TransformerEncoderConfig,
)
from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking

__all__ = [
    'GroupedFeedForward',
    'GGEMMTransformerEncoder',
    'pad_feedforward',
    'bucket_ffns_by_shape',
    'grouped_ffn_compute',
    'GROUPED_GEMM_BACKENDS',
    'PEE_EXPERT_TASKS',
]

# Available grouped-GEMM backends for the position-wise FFN of a shape bucket.
#   'baddbmm' : one batched GEMM over the stacked experts (default; the portable
#               stand-in for a fused CUTLASS/Triton grouped-GEMM).
#   'loop'    : per-expert ``addmm`` reference; same math, slower, used to validate
#               the batched path and as a fallback where bmm is unavailable.
# A future 'triton'/'cutlass' backend (ragged per-group offsets) plugs in here
# without changing the parameter layout or any call site.
GROUPED_GEMM_BACKENDS = ('baddbmm', 'loop')


def _autocast_compute_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the dtype GEMMs will use for ``x`` under the active autocast policy."""
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return x.dtype


def grouped_ffn_compute(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    drop_rate: float = 0.0,
    training: bool = False,
    backend: str = 'baddbmm',
) -> torch.Tensor:
    """Batched position-wise FFN over ``E`` stacked experts.

    Computes, for every expert ``e``::

        h_e   = gelu(x_e @ w1_e + b1_e)
        out_e = dropout(h_e) @ w2_e + b2_e

    matching the per-expert ``FeedForward`` (``Linear -> GELU -> Dropout ->
    Linear -> Dropout``) exactly, but for all experts at once.

    Args:
        x: ``(E, T, d_model)`` per-expert token batches (same ``T`` per expert).
        w1: ``(E, d_model, d_hidden)``; b1: ``(E, 1, d_hidden)``.
        w2: ``(E, d_hidden, d_model)``; b2: ``(E, 1, d_model)``.
        drop_rate: Dropout probability (applied after GELU and after ``w2``).
        training: Whether dropout is active (pass ``module.training``).
        backend: One of :data:`GROUPED_GEMM_BACKENDS`.

    Returns:
        ``(E, T, d_model)`` stacked expert outputs.
    """
    # LayerNorm commonly leaves ``x`` in fp32 under autocast even though the GEMMs
    # execute in bf16/fp16. Cast all operands explicitly to that compute dtype.
    # This is especially important for eval-time packed weights: they are ordinary
    # cached tensors (not Parameters), so autocast would otherwise re-cast the full
    # multi-GB packed bank on every forward.
    compute_dtype = _autocast_compute_dtype(x)
    x = x.to(compute_dtype)
    w1, b1 = w1.to(compute_dtype), b1.to(compute_dtype)
    w2, b2 = w2.to(compute_dtype), b2.to(compute_dtype)

    if backend == 'baddbmm':
        hidden = torch.baddbmm(b1, x, w1)  # (E, T, d_hidden)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=drop_rate, training=training)
        out = torch.baddbmm(b2, hidden, w2)  # (E, T, d_model)
    elif backend == 'loop':
        outs = []
        for e in range(x.shape[0]):
            h = torch.addmm(b1[e], x[e], w1[e])  # (T, d_hidden)
            h = F.gelu(h)
            h = F.dropout(h, p=drop_rate, training=training)
            outs.append(torch.addmm(b2[e], h, w2[e]))  # (T, d_model)
        out = torch.stack(outs, dim=0)
    else:
        raise ValueError(f"Unknown grouped-GEMM backend '{backend}'; expected one of {GROUPED_GEMM_BACKENDS}.")

    return F.dropout(out, p=drop_rate, training=training)


# Known per-expert task / decoder families. PEE binds only the *encoders*; the
# matching decoder (and the metric it produces) lives at the model level and is
# restored from each expert's own ``.nemo`` checkpoint. ``expert_tasks`` on the
# container records which family each expert belongs to so an eval harness can
# pair the encoder output with the correct decoder and score:
#
#   - ``asr_tdt``    -> TDT decoder + joint -> WER
#       (speech multilingual-ASR MoE expert; RNNT is also possible -- read the
#        decoder type from the restored checkpoint config rather than assuming).
#   - ``sound_rnnt`` -> RNNT decoder + joint -> WER
#       (sound / emotion-recognition expert).
#   - ``diarization`` -> Sortformer head (``sortformer_modules``) -> DER
#       (speaker expert; note the Sortformer model also has an upstream
#        FastConformer encoder and computes sigmoid speaker activities).
PEE_EXPERT_TASKS = ('asr_tdt', 'sound_rnnt', 'diarization')


# ---------------------------------------------------------------------------
# Reconciling the half-width speaker FFN with the wide experts
# ---------------------------------------------------------------------------
#
# The grouped-GEMM batches FFN "units" that share the same ``(d_model, d_hidden)``
# shape. On PEE-v2 the speech-MoE and sound experts contribute units of shape
# ``(2048 -> 1024 -> 2048)``; the speaker's unit is ``(1024 -> 1024 -> 1024)``.
# Two ways to make the speaker participate, both of which MUST preserve its exact
# output (PEE's first principle):
#
#   (A) PAD -- widen to (2048 -> 1024 -> 2048). Pad W1's input side with 1024 zero
#       rows and W2's output side with 1024 zero columns; feed the 1024-d speaker
#       activation in the top half (rest zero) and slice the top 1024 outputs
#       back. Exact (the zeros are structural), one uniform grouped tensor, but
#       2x the FFN FLOPs for that unit. ``pad_feedforward`` builds such a unit
#       standalone; ``_unified_weights`` does the same padding inline.
#
#   (B) BUCKET -- keep the native (1024 -> 1024 -> 1024) FFN and place it in a
#       SEPARATE shape bucket (its own grouped-GEMM group). Exact, zero wasted
#       compute, at the cost of one extra GEMM group. ``bucket_ffns_by_shape``
#       implements the grouping.
#
# (A) is what the fused paths use, and the choice is measured rather than assumed:
# one padded speaker unit against 17 native-width ones wastes 2.10M MAC/token, or
# 2.78% of the dense-mode FFN (3.57% under moe_mode='topk', where the bucket holds
# only 2 dense units). Switching to (B) reclaims that but adds a bucket, i.e.
# +2 baddbmm x n_layers launches; at 30 s / batch 4 the ~151 GFLOP saved is worth
# 0.25-0.50 ms against 0.24-0.48 ms of added launch overhead -- a wash. So padding
# stays while this workload is launch-bound. (B) becomes the better choice once a
# ragged/variable-K grouped-GEMM kernel removes the padding without adding a
# launch, which is the natural replacement at the ``grouped_ffn_compute`` seam.


class GroupedFeedForward(nn.Module):
    """Batched, numerically-exact replacement for ``E`` same-shape ``FeedForward``s.

    Each of the ``E`` experts is the standard NeMo Transformer FFN
    ``Linear(d_model, d_hidden) -> GELU -> Dropout -> Linear(d_hidden, d_model)
    -> Dropout``. Instead of holding ``E`` separate :class:`FeedForward` modules
    and looping over them (``E`` kernel launches per projection), this module
    stacks the expert weights and evaluates all experts with two batched matmuls
    (``torch.baddbmm``), which lowers to a single batched/grouped GEMM on GPU.

    This is the portable stand-in for a fused grouped-GEMM kernel: same math,
    one launch. Swapping in a CUTLASS/Triton grouped-GEMM later only changes the
    two ``baddbmm`` calls, not the parameter layout or the public API.

    Parameter layout (registered as ``nn.Parameter``):

    - ``w1``: ``(E, d_model, d_hidden)`` -- input projections (transposed from the
      ``nn.Linear`` ``(out, in)`` convention so we can do ``x @ w1``).
    - ``b1``: ``(E, 1, d_hidden)``
    - ``w2``: ``(E, d_hidden, d_model)``
    - ``b2``: ``(E, 1, d_model)``

    Args:
        num_experts: Number of expert FFNs ``E`` batched together.
        d_model: Input/output width shared by every expert in this group.
        d_hidden: FFN inner width shared by every expert in this group.
        drop_rate: Dropout probability (matches ``FeedForward``). Defaults to 0.0.
        backend: Grouped-GEMM backend, one of :data:`GROUPED_GEMM_BACKENDS`.
            Defaults to ``'baddbmm'``.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_hidden: int,
        drop_rate: float = 0.0,
        backend: str = 'baddbmm',
    ):
        super().__init__()
        if backend not in GROUPED_GEMM_BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'; expected one of {GROUPED_GEMM_BACKENDS}.")
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.drop_rate = drop_rate
        self.backend = backend

        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, d_hidden))
        self.b1 = nn.Parameter(torch.zeros(num_experts, 1, d_hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_hidden, d_model))
        self.b2 = nn.Parameter(torch.zeros(num_experts, 1, d_model))

        # Match nn.Linear's default Kaiming-uniform init per expert so a freshly
        # constructed GroupedFeedForward behaves like a stack of fresh FeedForwards.
        for e in range(num_experts):
            nn.init.kaiming_uniform_(self.w1[e].t(), a=5**0.5)
            nn.init.kaiming_uniform_(self.w2[e].t(), a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all experts on a per-expert batch.

        Args:
            x: ``(E, T, d_model)`` -- the ``e``-th slice is the (dense) token
               batch for expert ``e``. Every expert sees the same number of
               tokens ``T`` (the PEE setting: all experts process the same
               frames). For ragged/routed token counts, pad to a common ``T`` or
               use a true grouped-GEMM kernel with per-group offsets.

        Returns:
            ``(E, T, d_model)`` -- the stacked expert outputs.
        """
        if x.dim() != 3 or x.shape[0] != self.num_experts or x.shape[2] != self.d_model:
            raise ValueError(
                f"GroupedFeedForward expects input of shape (E={self.num_experts}, T, "
                f"d_model={self.d_model}), got {tuple(x.shape)}."
            )
        return grouped_ffn_compute(
            x,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            drop_rate=self.drop_rate,
            training=self.training,
            backend=self.backend,
        )

    @classmethod
    def from_feedforwards(
        cls, ffns: Sequence[FeedForward], drop_rate: float = 0.0, backend: str = 'baddbmm'
    ) -> "GroupedFeedForward":
        """Build a grouped FFN from a list of identically-shaped ``FeedForward``s.

        The resulting module is numerically equivalent (up to float reduction
        order) to running each input ``FeedForward`` on its own token batch.
        """
        if len(ffns) == 0:
            raise ValueError("from_feedforwards requires at least one FeedForward.")
        w1_0 = ffns[0].net[0]
        w2_0 = ffns[0].net[3]
        d_model = w1_0.in_features
        d_hidden = w1_0.out_features
        for i, ff in enumerate(ffns):
            l1, l2 = ff.net[0], ff.net[3]
            if (l1.in_features, l1.out_features, l2.in_features, l2.out_features) != (
                d_model,
                d_hidden,
                d_hidden,
                d_model,
            ):
                raise ValueError(
                    f"FeedForward {i} has shape mismatch; all experts in a group must share "
                    f"(d_model={d_model}, d_hidden={d_hidden})."
                )
        grouped = cls(num_experts=len(ffns), d_model=d_model, d_hidden=d_hidden, drop_rate=drop_rate, backend=backend)
        with torch.no_grad():
            for e, ff in enumerate(ffns):
                grouped.w1[e].copy_(ff.net[0].weight.t())
                grouped.b1[e, 0].copy_(ff.net[0].bias)
                grouped.w2[e].copy_(ff.net[3].weight.t())
                grouped.b2[e, 0].copy_(ff.net[3].bias)
        return grouped


def pad_feedforward(ffn: FeedForward, target_d_model: int) -> FeedForward:
    """Zero-pad a narrow ``FeedForward`` up to ``target_d_model`` (option A).

    Returns a NEW ``FeedForward`` whose input/output width is ``target_d_model``
    while the FFN hidden width is unchanged. The original weights occupy the top
    ``d_model`` rows/columns; the rest are zeros. Running this padded FFN on
    ``[x; 0]`` (the original activation padded with zeros) and slicing the top
    ``d_model`` outputs reproduces the original FFN exactly, so it can join a
    uniform ``target_d_model`` grouped-GEMM bucket without changing the result.

    Args:
        ffn: Source feed-forward with ``in_features = out_features = d_model``.
        target_d_model: Wider model width to pad up to (>= the source ``d_model``).

    Returns:
        A ``FeedForward`` of width ``target_d_model`` and the same hidden width.
    """
    l1, l2 = ffn.net[0], ffn.net[3]
    d_model = l1.in_features
    d_hidden = l1.out_features
    if target_d_model < d_model:
        raise ValueError(f"target_d_model ({target_d_model}) must be >= source d_model ({d_model}).")

    cfg = TransformerEncoderConfig(
        d_model=target_d_model,
        ff_expansion=d_hidden / target_d_model,
        drop_rate=ffn.net[2].p,
    )
    padded = FeedForward(cfg)
    # FeedForward derives d_hidden via int(ff_expansion * d_model); guard against
    # rounding so the padded hidden width matches the source exactly.
    if padded.net[0].out_features != d_hidden:
        raise ValueError(
            f"Rounding produced hidden={padded.net[0].out_features}, expected {d_hidden}; "
            "pass a target_d_model that divides evenly."
        )
    with torch.no_grad():
        padded.net[0].weight.zero_()
        padded.net[0].weight[:, :d_model].copy_(l1.weight)
        padded.net[0].bias.copy_(l1.bias)  # hidden width unchanged
        padded.net[3].weight.zero_()
        padded.net[3].weight[:d_model, :].copy_(l2.weight)
        padded.net[3].bias.zero_()
        padded.net[3].bias[:d_model].copy_(l2.bias)
    return padded


def bucket_ffns_by_shape(
    ffns: Sequence[FeedForward],
) -> Dict[Tuple[int, int], List[int]]:
    """Group FFN indices by their ``(d_model, d_hidden)`` shape (option B).

    Each returned bucket can be fused into one :class:`GroupedFeedForward` /
    one grouped-GEMM call. Heterogeneous experts (e.g. the 1024-d speaker vs the
    2048-d speech/sound experts) land in separate buckets with no wasted compute.

    Returns:
        Mapping ``(d_model, d_hidden) -> [indices into ``ffns``]``.
    """
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, ff in enumerate(ffns):
        key = (ff.net[0].in_features, ff.net[0].out_features)
        buckets.setdefault(key, []).append(i)
    return buckets


# FlexAttention mask closures. These intentionally mirror the (private) helpers in
# ``transformer_encoder.py`` so the lockstep path builds an identical block mask to
# each expert's own ``forward_internal``. They are trivial and stable; the
# ``verify_grouped_equivalence`` self-check guards against any drift.
def _padding_mask_mod(lengths):
    def pad_mask(b, h, q_idx, kv_idx):
        return kv_idx < lengths[b]

    return pad_mask


def _causal_mask_mod():
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return causal


class GGEMMTransformerEncoder(nn.Module):
    """Container hosting several whole expert encoders behind one entry point.

    Every expert is a standalone NeMo encoder (``TransformerEncoder`` or
    ``MoETransformerEncoder``) and keeps its own, unmodified inference path: a
    ``forward(expert_name, ...)`` call is bit-for-bit identical to running that
    encoder on its own. PEE adds no top-level router and never drops experts --
    the grouped-GEMM batching (see :class:`GroupedFeedForward`) is purely a
    compute optimization layered on top, not a change in semantics.

    Experts may be heterogeneous in ``d_model`` and positional-encoding scheme;
    they only need to share the front-end / frame rate so the same audio can be
    fed to all of them. See the module docstring for how the speaker expert's
    half-width 1024-d FFN is reconciled with the 2048-d experts.

    Decoders are **not** held here. PEE binds encoders only; each expert's decoder
    (RNNT/TDT joint for the speech / sound ASR experts, the Sortformer head for
    the speaker expert) is a model-level component restored from that expert's
    own ``.nemo``. The optional ``expert_tasks`` map records which decoder family
    each expert belongs to (see :data:`PEE_EXPERT_TASKS`) so an eval harness can
    route encoder outputs to the right decoder and compute WER / DER. The encoder
    container itself never runs a decoder.

    Args:
        experts: Mapping ``role -> nn.Module`` of already-constructed expert
            encoders (e.g. ``{"speech": MoETransformerEncoder(...), "speaker":
            TransformerEncoder(...), "sound": TransformerEncoder(...)}``).
        expert_tasks: Optional mapping ``role -> task`` (one of
            :data:`PEE_EXPERT_TASKS`) recording each expert's decoder family for
            downstream WER / DER evaluation. Pure metadata; does not affect the
            encoder forward.
    """

    def __init__(self, experts: Dict[str, nn.Module], expert_tasks: Optional[Dict[str, str]] = None):
        super().__init__()
        if not experts:
            raise ValueError("GGEMMTransformerEncoder requires at least one expert encoder.")
        self.experts = nn.ModuleDict(experts)
        self.expert_names: List[str] = list(experts.keys())

        expert_tasks = dict(expert_tasks or {})
        for role, task in expert_tasks.items():
            if role not in self.experts:
                raise KeyError(f"expert_tasks references unknown expert '{role}'.")
            if task not in PEE_EXPERT_TASKS:
                raise ValueError(
                    f"expert_tasks['{role}'] = '{task}' is not a known task; " f"expected one of {PEE_EXPERT_TASKS}."
                )
        self.expert_tasks: Dict[str, str] = expert_tasks

        # Cache of pre-packed grouped-FFN weights for forward_grouped, keyed by
        # (layer_idx, bucket_names, dtype) -> (w1, b1, w2, b2) in bmm layout. Built
        # lazily in eval (never while training, where params change every step) so
        # forward_grouped stops re-stacking + re-casting weights on every call.
        # Stale after a weight change: call clear_packed_weights() (done automatically
        # on train()/load_state_dict()) to rebuild.
        self._use_packed_grouped: bool = True
        self._packed_cache: Dict[Tuple[object, ...], Tuple[torch.Tensor, ...]] = {}
        # Runtime-dtype RoPE buffers keyed by the source buffer identity. The
        # Transformer encoders store cos/sin in model dtype (usually fp32), while
        # q/k run in bf16 under autocast; caching avoids four casts per expert/layer.
        self._rope_cache: Dict[Tuple[object, ...], Tuple[torch.Tensor, torch.Tensor]] = {}

        # SDPA fast-path toggle (README §4.2). When True (default) the packed /
        # serial-SDPA attention paths (1) drop the additive mask for unpadded,
        # non-causal groups so SDPA can dispatch FlashAttention-2, and (3) wrap
        # every SDPA call in ``sdpa_kernel([FLASH, EFFICIENT, MATH])`` to force the
        # fused backends. Set False to restore the legacy behaviour (always
        # materialize the padding mask; no backend hint) for A/B comparison.
        self.sdpa_fastpath: bool = True

    def train(self, mode: bool = True):
        # Packed weights are an eval-time optimization; invalidate when (re-)entering
        # training so we never read stale, pre-update weights.
        self._packed_cache.clear()
        self._rope_cache.clear()
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        # These caches are plain tensors rather than registered buffers, so Module.to()
        # cannot migrate them safely. Rebuild lazily after any device/dtype transform.
        self._packed_cache.clear()
        self._rope_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def clear_packed_weights(self) -> None:
        """Drop eval-time grouped-FFN and RoPE caches.

        Call after mutating expert parameters outside ``load_state_dict`` so the
        next grouped/packed forward rebuilds its runtime-dtype tensors.
        """
        self._packed_cache.clear()
        self._rope_cache.clear()

    @property
    def expert_d_models(self) -> Dict[str, int]:
        """``role -> d_model`` for every expert that exposes a ``d_model`` attr."""
        return {name: m.d_model for name, m in self.experts.items() if hasattr(m, 'd_model')}

    def get_expert(self, expert_name: str) -> nn.Module:
        """Return the expert encoder module for ``expert_name`` (for pairing with
        its model-level decoder during eval)."""
        if expert_name not in self.experts:
            raise KeyError(f"Unknown expert '{expert_name}'. Available: {self.expert_names}.")
        return self.experts[expert_name]

    def get_expert_task(self, expert_name: str) -> Optional[str]:
        """Return the recorded decoder family for ``expert_name`` (or ``None``)."""
        return self.expert_tasks.get(expert_name)

    def forward(self, expert_name: str, audio_signal, length, bypass_pre_encode: bool = False):
        """Run a single expert along its own (unmodified) inference path.

        Args:
            expert_name: Which expert to run; must be one of ``self.expert_names``.
            audio_signal: ``(B, C, T)`` features (or ``(B, T, D)`` if
                ``bypass_pre_encode``), forwarded as-is to the expert.
            length: ``(B,)`` valid frame counts per sample.
            bypass_pre_encode: Passed through to the expert encoder.

        Returns:
            Whatever the chosen expert returns -- ``(B, D, T')`` encoded output
            and output lengths for the NeMo Transformer encoders.
        """
        if expert_name not in self.experts:
            raise KeyError(f"Unknown expert '{expert_name}'. Available experts: {self.expert_names}.")
        return self.experts[expert_name](audio_signal, length, bypass_pre_encode=bypass_pre_encode)

    def forward_all(self, audio_signal, length, bypass_pre_encode: bool = False) -> Dict[str, object]:
        """Run every expert on the same input and return a ``role -> output`` map.

        This is the reference (non-fused) multi-expert path. The grouped-GEMM
        path will replace the per-expert FFN evaluation here with a single
        bucketed grouped GEMM while leaving these outputs unchanged.
        """
        return {
            name: self.experts[name](audio_signal, length, bypass_pre_encode=bypass_pre_encode)
            for name in self.expert_names
        }

    # -----------------------------------------------------------------------
    # Lockstep fused forward (grouped-GEMM FFN across experts)
    # -----------------------------------------------------------------------
    #
    # Heterogeneous-expert reconciliation (README §3.1 / TODO #2):
    #   * Attention, norms, and positional encoding stay **per-expert** -- each
    #     expert runs its own attention sub-block with its own d_model and
    #     positional scheme (``rel_pos`` vs ``rope``), so nothing is shared or
    #     approximated there.
    #   * Only the position-wise FFN is fused, and only across experts whose
    #     layer-``i`` FFN is a dense :class:`FeedForward` of identical
    #     ``(d_model, d_hidden)`` (shape-bucketing, option B). Experts whose FFN
    #     is a sparse ``MoEFeedForward`` (e.g. the speech expert) run their own
    #     FFN sub-block -- their internal token-routed grouped-GEMM is separate.
    #
    # The pre-/post-layer and per-sub-block logic below mirrors
    # ``TransformerEncoder.forward`` / ``forward_internal`` / ``TransformerBlock``
    # by calling the experts' own public submodules (``pre_encode``, ``pos_enc``,
    # ``embed_norm``, ``layers[i].{norm1,attn,drop,norm2,ffn}``, ``final_norm``,
    # ``out_proj``). It does not modify the base encoder. ``allclose`` (not
    # bitwise) equivalence vs :meth:`forward_all` is asserted by
    # :meth:`verify_grouped_equivalence` (see README §4.1 for why it is ULP-level,
    # not bit-exact) -- run it in eval mode (dropout off).

    @staticmethod
    def _prepend_prefix(
        x: torch.Tensor,
        length: torch.Tensor,
        prefix: Optional[torch.Tensor],
        mode: str = 'extend',
    ):
        """Splice a streaming cache ``prefix`` onto ``x``; returns ``(x, length, chunk)``.

        ``prefix`` is ``(B, P, d_model)`` of already-projected embeddings (the same
        representation ``x`` carries at this point: post-projection, pre-norm), so the
        concatenation is normalized as one sequence -- matching how Sortformer's
        ``forward_streaming_step`` feeds ``[spkcache | fifo | chunk]`` through the
        encoder body in a single pass.

        Two ways to splice, differing in what happens to the *other* experts:

        ``'extend'``
            ``[prefix | x]``, so the prefixed expert grows to ``P + T`` while every
            other expert stays at ``T`` and gets right-padded with zeros to match.
            Those pad frames are masked out, so their attention and FFN work is pure
            waste -- 675/475 = 1.42x on PEE-v2.

        ``'replace'``
            ``[prefix | x[:, P:]]``, i.e. the cache *substitutes* for the expert's own
            leading ``P`` frames rather than extending past them. ``T`` is unchanged,
            so every expert in the group already agrees on ``T`` and nothing is padded.
            The caller feeds a window extended ``P`` frames further back, which turns
            what would have been zero padding for the unprefixed experts into ``P``
            frames of real left context -- same FLOPs, real signal instead of zeros,
            and the group can stay FlashAttention-2 eligible.

        Returns the spliced ``x``, the updated ``length``, and ``chunk`` -- the slice of
        ``x`` the prefixed expert treats as its current chunk (``x[:, P:]`` under
        ``'replace'``, all of ``x`` under ``'extend'``). Streaming callers must push
        ``chunk``, not the raw projection, into their cache.
        """
        if prefix is None:
            return x, length, x
        if mode not in ('extend', 'replace'):
            raise ValueError(f"prefix mode must be 'extend' or 'replace', got {mode!r}.")
        if prefix.dim() != 3 or prefix.shape[0] != x.shape[0] or prefix.shape[-1] != x.shape[-1]:
            raise ValueError(f"prefix must be (B={x.shape[0]}, P, d_model={x.shape[-1]}), got {tuple(prefix.shape)}.")
        p = prefix.shape[1]
        prefix = prefix.to(dtype=x.dtype, device=x.device)
        if mode == 'extend':
            return torch.cat([prefix, x], dim=1), length + p, x
        if p > x.shape[1]:
            raise ValueError(
                f"prefix of {p} frames cannot replace the leading frames of a {x.shape[1]}-frame "
                "input; feed a window extended at least P frames further back, or use "
                "mode='extend'."
            )
        chunk = x[:, p:]
        # The prefix frames are cache and always valid, so a sample whose audio ended
        # inside them still has P valid frames.
        return torch.cat([prefix, chunk], dim=1), torch.clamp(length, min=p), chunk

    @staticmethod
    def _right_pad_to(x: torch.Tensor, t_max: int) -> torch.Tensor:
        """Right-pad ``x`` ``(B, T, D)`` with zeros to ``T = t_max``.

        Padding on the RIGHT is load-bearing: every mask builder here assumes the
        valid frames of a sample are the prefix ``[0, length)``, so appending keeps
        ``_padding_additive_mask`` / ``_no_padding`` correct with ``length`` left at
        the true valid count.
        """
        pad = t_max - x.shape[1]
        if pad <= 0:
            return x
        return F.pad(x, (0, 0, 0, pad))

    def _expert_pre(
        self,
        expert: nn.Module,
        audio_signal,
        length,
        bypass_pre_encode: bool,
        build_block_mask: bool = True,
        prefix: Optional[torch.Tensor] = None,
        t_max: Optional[int] = None,
        return_pre_encode: bool = False,
        prefix_mode: str = 'extend',
    ):
        """Run an expert's pre-layer stack (mirrors ``forward`` + pre-loop of
        ``forward_internal``); returns ``(x, layer_pos_emb, block_mask, length)``.

        With ``prefix`` / ``t_max`` set, the streaming cache is prepended to the
        projected embeddings before the norm and the result is right-padded to a
        common ``t_max`` (see :meth:`_prepend_prefix` / :meth:`_right_pad_to`).
        ``return_pre_encode`` additionally yields the projected chunk embeddings
        (pre-prefix, pre-norm) so a caller can push them into its cache without
        recomputing the projection.
        """
        if not bypass_pre_encode and audio_signal.shape[-2] != expert._feat_in:
            raise ValueError(f"Expert expects feat_in={expert._feat_in} on dim -2, got {audio_signal.shape[-2]}.")
        if bypass_pre_encode and audio_signal.shape[-1] != expert.d_model:
            raise ValueError(
                f"Expert expects d_model={expert.d_model} on dim -1 when bypassing pre-encode, "
                f"got {audio_signal.shape[-1]}."
            )
        if bypass_pre_encode:
            expert.update_max_seq_length(seq_length=audio_signal.size(1), device=audio_signal.device)
        else:
            expert.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)

        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(1) if bypass_pre_encode else audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )

        if not bypass_pre_encode:
            if isinstance(expert.pre_encode, FeatureStacking):
                x, length = expert.pre_encode(audio_signal, length)
            else:
                x = torch.transpose(audio_signal, 1, 2)
            if isinstance(expert.pre_encode, nn.Linear):
                x = expert.pre_encode(x)
            elif not isinstance(expert.pre_encode, FeatureStacking):
                x, length = expert.pre_encode(x=x, lengths=length)
            length = length.to(torch.int64)
        else:
            x = audio_signal
            length = length.to(torch.int64)

        # Projected chunk embeddings, before the norm -- what a streaming caller
        # pushes into its cache. Under prefix_mode='replace' this is the post-splice
        # chunk, i.e. the leading P frames consumed by the cache are excluded, so its
        # length has to drop by the same amount to stay in step.
        pre_len = length
        n_before = x.shape[1]
        x, length, x_proj = self._prepend_prefix(x, length, prefix, mode=prefix_mode)
        proj_length = (pre_len - (n_before - x_proj.shape[1])).clamp(min=0)

        if expert.self_attention_model == "rope":
            if expert.xscale:
                x = x * expert.xscale
            x = expert.dropout_pre_encoder(x)
            pos_emb = None
        elif expert.pos_enc is not None:
            x, pos_emb = expert.pos_enc(x=x)
        else:
            pos_emb = None
        x = expert.embed_norm(x)
        if t_max is not None:
            x = self._right_pad_to(x, t_max)  # `length` stays at the true valid count

        block_mask = None
        if build_block_mask:
            B, T, _ = x.shape
            if expert.attn_mode == "causal":
                mask_mod = and_masks(_causal_mask_mod(), _padding_mask_mod(length))
            else:
                mask_mod = _padding_mask_mod(length)
            block_mask = create_block_mask(mask_mod, B=B, H=1, Q_LEN=T, KV_LEN=T, device=x.device)
        layer_pos_emb = pos_emb if expert.self_attention_model == "rel_pos" else None
        if return_pre_encode:
            return x, layer_pos_emb, block_mask, length, (x_proj, proj_length)
        return x, layer_pos_emb, block_mask, length

    def _experts_pre(
        self,
        encs: Dict[str, nn.Module],
        audio_signal: torch.Tensor,
        length: Optional[torch.Tensor],
        bypass_pre_encode: bool,
        build_block_mask: bool,
        prefix: Optional[Dict[str, torch.Tensor]] = None,
        return_pre_encode: bool = False,
        prefix_mode: str = 'extend',
    ) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], object, torch.Tensor]]:
        """Prepare every expert, batching compatible ``FeatureStacking`` projections.

        All PEE-v2 experts consume the same 128-bin features with the same stacking
        factor. Their learned projections have a common input width but heterogeneous
        output widths (2048/1024/2048). In eval, this method stacks/pads their weights
        once and evaluates one ``bmm`` over an expanded view of the common stacked
        features. The wider padded output intentionally uses more workspace to turn
        three small projection launches into one.

        Unsupported/training/bypass cases retain the original per-expert path.
        FlexAttention masks are also shared by experts with the same attention mode;
        SDPA callers request no block mask.

        Streaming (``prefix``): a ``{name: (B, P, d_model)}`` map of cache embeddings
        prepended to that expert's projected chunk before the norm. Because a prefixed
        expert then runs longer than the others, every expert is right-padded with
        zeros to the common ``T_max`` while ``length`` keeps its true valid count --
        so the group can share one packed-attention call. ``return_pre_encode`` also
        returns ``{name: (x_proj, out_length)}``, the projected chunk embeddings the
        prefix path consumes, so a caller can update its cache without reprojecting.
        """
        names = list(encs)
        prefix = prefix or {}
        for name in prefix:
            if name not in encs:
                raise KeyError(f"prefix references unknown expert '{name}'. Available: {names}.")
        if prefix and build_block_mask:
            # A prefix makes experts ragged, so they get right-padded to a common T
            # -- a FlexAttention BlockMask built for the unpadded T would no longer
            # describe the tensor. Streaming runs through the SDPA paths, which build
            # their mask from `length` per layer, so this combination never arises.
            raise ValueError(
                "prefix is only supported on the SDPA paths (build_block_mask=False); "
                "forward_grouped's FlexAttention masks cannot describe the padded T."
            )
        feature_stackers = [encs[name].pre_encode for name in names]
        can_batch = (
            not self.training
            and not bypass_pre_encode
            and feature_stackers
            and all(isinstance(pre, FeatureStacking) for pre in feature_stackers)
            and len({pre.subsampling_factor for pre in feature_stackers}) == 1
            and len({pre.proj.in_features for pre in feature_stackers}) == 1
            and len({encs[name]._feat_in for name in names}) == 1
        )
        if not can_batch:
            # Pad to the observed max rather than a precomputed one: the padding in
            # `_expert_pre` is applied after the norm, so appending it here instead
            # is the same tensor and works for any pre_encode kind.
            out = {
                name: self._expert_pre(
                    encs[name],
                    audio_signal,
                    length,
                    bypass_pre_encode,
                    build_block_mask=build_block_mask,
                    prefix=prefix.get(name),
                    return_pre_encode=return_pre_encode,
                    prefix_mode=prefix_mode,
                )
                for name in names
            }
            if prefix:
                t_max = max(v[0].shape[1] for v in out.values())
                out = {n: (self._right_pad_to(v[0], t_max),) + tuple(v[1:]) for n, v in out.items()}
            if return_pre_encode:
                return ({n: v[:4] for n, v in out.items()}, {n: v[4] for n, v in out.items()})
            return out

        feat_in = encs[names[0]]._feat_in
        if audio_signal.shape[-2] != feat_in:
            raise ValueError(f"Experts expect feat_in={feat_in} on dim -2, got {audio_signal.shape[-2]}.")
        for expert in encs.values():
            expert.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)

        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),),
                audio_signal.size(-1),
                dtype=torch.int64,
                device=audio_signal.device,
            )
        else:
            length = length.to(torch.int64)

        factor = feature_stackers[0].subsampling_factor
        stacked = audio_signal.transpose(1, 2)
        B, T, C = stacked.shape
        pad_size = (factor - (T % factor)) % factor
        if pad_size:
            stacked = F.pad(stacked, (0, 0, 0, pad_size))
        T_out = (T + pad_size) // factor
        stacked = stacked.reshape(B * T_out, C * factor)
        out_length = feature_stackers[0].compute_num_out_frames(length)

        target_d = max(pre.proj.out_features for pre in feature_stackers)
        compute_dtype = _autocast_compute_dtype(stacked)
        key = ('pre_encode', tuple(names), target_d, compute_dtype)
        packed = self._packed_cache.get(key)
        if packed is None:
            with torch.no_grad():
                weights = []
                for pre in feature_stackers:
                    weight = pre.proj.weight.t()
                    weights.append(F.pad(weight, (0, target_d - weight.shape[1])))
                packed = (torch.stack(weights).to(compute_dtype).contiguous(),)
            self._packed_cache[key] = packed
        weights = packed[0]
        shared_input = stacked.to(compute_dtype).unsqueeze(0).expand(len(names), -1, -1)
        projected = torch.bmm(shared_input, weights)

        # Under 'extend' a prefixed expert grows past the others and they get padded up
        # to it. Under 'replace' the prefix consumes that expert's own leading frames,
        # so every expert already shares T_out and nothing is padded.
        t_max = T_out
        if prefix_mode == 'extend':
            t_max += max((p.shape[1] for p in prefix.values()), default=0)

        mask_cache = {}
        result = {}
        pre_encode_out = {}
        for slot, name in enumerate(names):
            expert = encs[name]
            x = projected[slot, :, : expert.d_model].reshape(B, T_out, expert.d_model)
            n_before = x.shape[1]
            x, length_n, chunk = self._prepend_prefix(x, out_length, prefix.get(name), mode=prefix_mode)
            # Pre-norm chunk embeddings a streaming caller pushes into its cache. Under
            # 'replace' this excludes the leading frames the cache stood in for, so it
            # lines up with the lc/rc the caller passes to streaming_update. Measure the
            # drop against the PRE-splice frame count -- under 'extend' the chunk keeps
            # all of them and its length must not move.
            if return_pre_encode:
                n_dropped = n_before - chunk.shape[1]
                pre_encode_out[name] = (chunk, (out_length - n_dropped).clamp(min=0))

            if expert.self_attention_model == "rope":
                if expert.xscale:
                    x = x * expert.xscale
                x = expert.dropout_pre_encoder(x)
                pos_emb = None
            elif expert.pos_enc is not None:
                x, pos_emb = expert.pos_enc(x=x)
            else:
                pos_emb = None
            x = expert.embed_norm(x)
            if prefix:
                x = self._right_pad_to(x, t_max)  # `length_n` stays the true count

            block_mask = None
            if build_block_mask:
                mask_key = expert.attn_mode
                block_mask = mask_cache.get(mask_key)
                if block_mask is None:
                    if expert.attn_mode == "causal":
                        mask_mod = and_masks(_causal_mask_mod(), _padding_mask_mod(out_length))
                    else:
                        mask_mod = _padding_mask_mod(out_length)
                    block_mask = create_block_mask(mask_mod, B=B, H=1, Q_LEN=T_out, KV_LEN=T_out, device=x.device)
                    mask_cache[mask_key] = block_mask
            layer_pos_emb = pos_emb if expert.self_attention_model == "rel_pos" else None
            result[name] = (x, layer_pos_emb, block_mask, length_n)
        if return_pre_encode:
            return result, pre_encode_out
        return result

    def _expert_post(self, expert: nn.Module, x, length):
        """Run an expert's post-layer stack (mirrors post-loop of ``forward_internal``)."""
        x = expert.final_norm(x)
        if expert.out_proj is not None:
            x = expert.out_proj(x)
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        return x, length.to(dtype=torch.int64)

    # -----------------------------------------------------------------------
    # Unified FFN step: fuse EVERY expert's FFN -- the dense experts, the speech
    # MoE's per-token experts (run dense, then recombined with the router top-k),
    # AND the narrower speaker FFN (zero-padded up to the widest d_model) -- into
    # one grouped GEMM per (d_hidden) bucket. On the PEE-v2 checkpoints this
    # collapses the per-layer FFN to a single grouped GEMM over 18 units (16 MoE
    # + sound + speaker), instead of 1 small GEMM + a 16-iteration MoE index_add
    # loop. Under moe_mode='topk' the MoE leaves for its own capacity-padded call
    # and the shared bucket holds the 2 dense units.
    # -----------------------------------------------------------------------
    def _unified_weights(self, group, target_d: int, d_hidden: int, layer_idx: int, dtype: torch.dtype):
        """Stack the FFN weights of every unit in ``group`` into the grouped-bmm
        layout, zero-padding any unit whose ``d_model < target_d`` (option A; the
        pad rows/cols are structurally zero so the unit's output is unchanged).

        Cached per ``(layer_idx, d_hidden, target_d, names, dtype)`` in eval so
        repeated calls skip the ``stack`` + ``.to(dtype)`` that would otherwise
        dominate the FFN cost; rebuilt live (grad-preserving) under training.
        """
        units, srcs = [], []
        for p in group:
            for ff in p['units']:
                units.append(ff)
                srcs.append(ff.net[0].in_features)

        def _stack():
            w1s, b1s, w2s, b2s = [], [], [], []
            for ff, src_d in zip(units, srcs):
                w1 = ff.net[0].weight.t()  # (src_d, d_hidden)
                w2 = ff.net[3].weight.t()  # (d_hidden, src_d)
                b1 = ff.net[0].bias.unsqueeze(0)  # (1, d_hidden) -- hidden never padded
                b2 = ff.net[3].bias.unsqueeze(0)  # (1, src_d)
                if src_d != target_d:
                    pad = target_d - src_d
                    w1 = F.pad(w1, (0, 0, 0, pad))  # pad input rows -> (target_d, d_hidden)
                    w2 = F.pad(w2, (0, pad))  # pad output cols -> (d_hidden, target_d)
                    b2 = F.pad(b2, (0, pad))  # -> (1, target_d)
                w1s.append(w1)
                b1s.append(b1)
                w2s.append(w2)
                b2s.append(b2)
            return (torch.stack(w1s, 0), torch.stack(b1s, 0), torch.stack(w2s, 0), torch.stack(b2s, 0))

        if self.training or not self._use_packed_grouped:
            return _stack()
        key = (layer_idx, 'unified', d_hidden, target_d, tuple(p['name'] for p in group), dtype)
        cached = self._packed_cache.get(key)
        if cached is None:
            with torch.no_grad():
                w1, b1, w2, b2 = _stack()
                cached = (
                    w1.to(dtype).contiguous(),
                    b1.to(dtype).contiguous(),
                    w2.to(dtype).contiguous(),
                    b2.to(dtype).contiguous(),
                )
            self._packed_cache[key] = cached
        return cached

    def _unified_ffn_step(self, encs, state, layer_idx: int, backend: str, moe_mode: str = 'dense') -> None:
        """FFN sub-block for layer ``layer_idx`` fusing *all* experts' FFNs.

        Builds one grouped GEMM per inner-width (``d_hidden``) bucket over every
        FFN unit: dense experts contribute one unit, narrower experts are
        zero-padded up to the bucket's widest ``d_model``, and the speech MoE is
        handled per ``moe_mode``:

        - ``'dense'`` (default): the MoE's ``num_experts`` experts join the shared
          bucket and run on *all* tokens, then are recombined with the router's
          renormalized top-k weights. Exact, one big batched GEMM, but ~``num_experts
          / top_k`` redundant FFN FLOPs.
        - ``'topk'``: only the routed top-k expert/token pairs are computed, in a
          separate capacity-padded batched GEMM (:meth:`_moe_topk_ffn_step`).
          Exact (capacity = max expert load, no drops); far less compute/memory
          when compute-bound, at the cost of gather/scatter + a second launch.

        Residual updates are applied in place.
        """
        if moe_mode not in ('dense', 'topk'):
            raise ValueError(f"moe_mode must be 'dense' or 'topk', got {moe_mode!r}.")
        plans = []
        for n in self.expert_names:
            ffn = encs[n].layers[layer_idx].ffn
            if isinstance(ffn, MoEFeedForward):
                plans.append({'name': n, 'kind': 'moe', 'units': list(ffn.experts), 'moe': ffn})
            elif isinstance(ffn, FeedForward):
                plans.append({'name': n, 'kind': 'dense', 'units': [ffn], 'moe': None})
            else:
                # Anything we cannot express as FeedForward units runs its own path.
                layer = encs[n].layers[layer_idx]
                state[n]['x'] = state[n]['x'] + layer.drop(ffn(layer.norm2(state[n]['x'])))

        if moe_mode == 'topk':
            # Sparse MoE: compute only the routed top-k pairs in their own grouped
            # call; dense experts (+ zero-padded speaker) stay in the shared bucket.
            for p in plans:
                if p['kind'] == 'moe':
                    self._moe_topk_ffn_step(encs, state, layer_idx, p, backend)
            grouped = [p for p in plans if p['kind'] == 'dense']
        else:
            grouped = [p for p in plans if p['kind'] in ('dense', 'moe')]
        if not grouped:
            return

        by_hidden: Dict[int, List[dict]] = {}
        for p in grouped:
            by_hidden.setdefault(p['units'][0].net[0].out_features, []).append(p)

        for d_hidden, group in by_hidden.items():
            target_d = max(p['units'][0].net[0].in_features for p in group)
            rows, layout, slot = [], [], 0
            for p in group:
                n = p['name']
                layer = encs[n].layers[layer_idx]
                h = layer.norm2(state[n]['x'])  # (B, T, src_d)
                B, T, src_d = h.shape
                hf = h.reshape(B * T, src_d)
                hf_p = hf if src_d == target_d else F.pad(hf, (0, target_d - src_d))
                n_units = len(p['units'])
                entry = {
                    'name': n,
                    'kind': p['kind'],
                    'src_d': src_d,
                    'slot': slot,
                    'n_units': n_units,
                    'B': B,
                    'T': T,
                    'W': None,
                }
                if p['kind'] == 'moe':
                    moe = p['moe']
                    gate = moe.router(hf)  # (N, num_experts) softmax probs
                    topv, topi = torch.topk(gate, moe.top_k, dim=-1)
                    if moe.top_k > 1:
                        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
                    Wmat = torch.zeros(B * T, moe.num_experts, dtype=gate.dtype, device=gate.device)
                    entry['W'] = Wmat.scatter_(1, topi, topv)
                    rows.extend([hf_p] * n_units)  # dense MoE: every expert sees all tokens
                else:
                    rows.append(hf_p)
                layout.append(entry)
                slot += n_units

            H = torch.stack(rows, dim=0)  # (E_total, N, target_d)
            compute_dtype = _autocast_compute_dtype(H)
            w1, b1, w2, b2 = self._unified_weights(group, target_d, d_hidden, layer_idx, compute_dtype)
            drop_rate = group[0]['units'][0].net[2].p
            out = grouped_ffn_compute(H, w1, b1, w2, b2, drop_rate=drop_rate, training=self.training, backend=backend)

            for entry in layout:
                n, slot, src_d = entry['name'], entry['slot'], entry['src_d']
                B, T = entry['B'], entry['T']
                layer = encs[n].layers[layer_idx]
                if entry['kind'] == 'dense':
                    o = out[slot][:, :src_d].reshape(B, T, src_d)
                else:
                    ne = entry['n_units']
                    o_slots = out[slot : slot + ne][:, :, :src_d]  # (ne, N, src_d)
                    # fp32 recombine to mirror the MoE's fp32 index_add accumulation.
                    Wt = entry['W'].t().unsqueeze(-1).float()  # (ne, N, 1)
                    o = (o_slots.float() * Wt).sum(0).to(state[n]['x'].dtype).reshape(B, T, src_d)
                state[n]['x'] = state[n]['x'] + layer.drop(o)

    def _moe_weights(self, name: str, moe, layer_idx: int, dtype: torch.dtype):
        """Stack the MoE's ``num_experts`` expert FFNs into grouped-bmm layout
        ``(ne, d_model, d_hidden)`` / ``(ne, d_hidden, d_model)``; cached in eval."""

        def _stack():
            ffns = list(moe.experts)
            return (
                torch.stack([f.net[0].weight.t() for f in ffns], 0),
                torch.stack([f.net[0].bias.unsqueeze(0) for f in ffns], 0),
                torch.stack([f.net[3].weight.t() for f in ffns], 0),
                torch.stack([f.net[3].bias.unsqueeze(0) for f in ffns], 0),
            )

        if self.training or not self._use_packed_grouped:
            return _stack()
        key = (layer_idx, 'moe', name, dtype)
        cached = self._packed_cache.get(key)
        if cached is None:
            with torch.no_grad():
                w1, b1, w2, b2 = _stack()
                cached = (
                    w1.to(dtype).contiguous(),
                    b1.to(dtype).contiguous(),
                    w2.to(dtype).contiguous(),
                    b2.to(dtype).contiguous(),
                )
            self._packed_cache[key] = cached
        return cached

    def _moe_topk_ffn_step(self, encs, state, layer_idx: int, p: dict, backend: str) -> None:
        """Sparse MoE FFN: compute ONLY the routed top-k expert/token pairs.

        Mirrors :meth:`MoEFeedForward.forward` but batches the per-expert matmuls
        into a single grouped GEMM. Tokens routed to each expert are gathered into
        a capacity-padded buffer ``(num_experts, C, d_model)`` with ``C`` = the max
        per-expert load (so nothing is dropped -> exact), run through one batched
        GEMM, then scattered back with the router weights (fp32 accumulation, to
        mirror the reference ``index_add_``). This computes ``~N*top_k`` token-FFNs
        instead of the dense path's ``N*num_experts``.
        """
        n = p['name']
        moe = p['moe']
        layer = encs[n].layers[layer_idx]
        x = state[n]['x']
        h = layer.norm2(x)
        B, T, d = h.shape
        N = B * T
        x_flat = h.reshape(N, d)
        ne, top_k = moe.num_experts, moe.top_k

        gate = moe.router(x_flat)  # (N, ne) softmax probs
        topv, topi = torch.topk(gate, top_k, dim=-1)
        if top_k > 1:
            topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)

        M = N * top_k
        flat_expert = topi.reshape(-1)  # (M,)
        flat_weight = topv.reshape(-1)  # (M,)
        flat_token = torch.arange(N, device=x.device).unsqueeze(1).expand(N, top_k).reshape(-1)  # (M,)
        counts = torch.bincount(flat_expert, minlength=ne)  # (ne,)
        capacity = int(counts.max().item()) if M > 0 else 0
        if capacity == 0:
            return  # no tokens routed (degenerate); nothing to add

        # Sort dispatch rows by expert so each expert's tokens are contiguous, then
        # compute each row's slot within its expert segment (0..count_e-1).
        order = torch.sort(flat_expert, stable=True).indices
        s_expert = flat_expert[order]
        s_token = flat_token[order]
        s_weight = flat_weight[order]
        seg_start = torch.zeros(ne, dtype=torch.long, device=x.device)
        if ne > 1:
            seg_start[1:] = torch.cumsum(counts, 0)[:-1]
        within = torch.arange(M, device=x.device) - seg_start[s_expert]  # (M,)

        # Capacity-padded per-expert token buffer (unused slots stay zero).
        buf = x_flat.new_zeros(ne, capacity, d)
        buf[s_expert, within] = x_flat.index_select(0, s_token)

        compute_dtype = _autocast_compute_dtype(buf)
        w1, b1, w2, b2 = self._moe_weights(n, moe, layer_idx, compute_dtype)
        out = grouped_ffn_compute(
            buf,
            w1,
            b1,
            w2,
            b2,
            drop_rate=moe.experts[0].net[2].p,
            training=self.training,
            backend=backend,
        )  # (ne, capacity, d)

        disp_out = out[s_expert, within]  # (M, d) -- one row per (token, expert) pair
        acc = torch.zeros(N, d, dtype=torch.float32, device=x.device)
        acc.index_add_(0, s_token, disp_out.float() * s_weight.float().unsqueeze(-1))
        o = acc.to(x.dtype).reshape(B, T, d)
        state[n]['x'] = x + layer.drop(o)

    def forward_grouped(
        self,
        audio_signal,
        length,
        bypass_pre_encode: bool = False,
        backend: str = 'baddbmm',
        moe_mode: str = 'dense',
    ) -> Dict[str, object]:
        """Lockstep multi-expert forward that fuses same-shape dense FFNs.

        Drives all experts layer-by-layer: at each layer it runs every expert's
        attention sub-block (per-expert, unchanged), then evaluates the
        position-wise FFN of *every* expert in one grouped GEMM per ``d_hidden``
        bucket (see :meth:`_unified_ffn_step`): the speech MoE's per-token experts
        are run dense and recombined with the router top-k, and the narrower
        speaker FFN is zero-padded up to the bucket's widest ``d_model``.

        Output matches :meth:`forward_all` (``role -> (x, length)``) to within
        ULP tolerance (not bitwise; README §4.1). All experts must share
        ``n_layers`` and consume the same ``audio_signal`` / ``length``.

        Args:
            audio_signal, length, bypass_pre_encode: as in :meth:`forward`.
            backend: grouped-GEMM backend (:data:`GROUPED_GEMM_BACKENDS`).
        """
        if backend not in GROUPED_GEMM_BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'; expected one of {GROUPED_GEMM_BACKENDS}.")
        encs = {name: self.experts[name] for name in self.expert_names}
        for name, e in encs.items():
            if not hasattr(e, 'layers') or not hasattr(e, 'n_layers'):
                raise TypeError(
                    f"Expert '{name}' is not a flex TransformerEncoder-family module; "
                    f"forward_grouped requires per-layer access."
                )
        n_layers_set = {e.n_layers for e in encs.values()}
        if len(n_layers_set) != 1:
            raise ValueError(f"forward_grouped requires equal n_layers across experts, got {n_layers_set}.")
        n_layers = n_layers_set.pop()

        state: Dict[str, dict] = {}
        prepared = self._experts_pre(encs, audio_signal, length, bypass_pre_encode, build_block_mask=True)
        for name, e in encs.items():
            x, pos_emb, block_mask, ln = prepared[name]
            state[name] = {'x': x, 'pos_emb': pos_emb, 'block_mask': block_mask, 'length': ln}

        for i in range(n_layers):
            # 1) per-expert attention sub-block (mirrors TransformerBlock line 1)
            for name, e in encs.items():
                layer = e.layers[i]
                s = state[name]
                s['x'] = s['x'] + layer.drop(
                    layer.attn(layer.norm1(s['x']), block_mask=s['block_mask'], pos_emb=s['pos_emb'])
                )
            # 2) FFN sub-block: fuse ALL experts' FFN units (dense + speech-MoE
            #    experts + zero-padded speaker) into grouped GEMMs by d_hidden.
            self._unified_ffn_step(encs, state, i, backend, moe_mode=moe_mode)

        return {
            name: self._expert_post(encs[name], state[name]['x'], state[name]['length']) for name in self.expert_names
        }

    # -----------------------------------------------------------------------
    # Fully-packed forward (BGG-MEI): batched-SDPA attention over packed heads
    # + grouped FFN. This is the path that actually cuts kernel launches and
    # avoids per-expert flex-attention (which does not compile for head_dim that
    # the Triton flex template cannot fit in shared memory). See README §"Packed".
    # -----------------------------------------------------------------------

    @staticmethod
    def _padding_additive_mask(length: torch.Tensor, T: int, dtype: torch.dtype) -> torch.Tensor:
        """Additive key-padding mask ``(B, 1, 1, T)``: 0 for valid keys, -inf for pads."""
        device = length.device
        valid = torch.arange(T, device=device)[None, :] < length[:, None]  # (B, T)
        mask = torch.zeros(length.shape[0], 1, 1, T, dtype=dtype, device=device)
        return mask.masked_fill(~valid[:, None, None, :], torch.finfo(dtype).min)

    @staticmethod
    def _no_padding(length: torch.Tensor, T: int) -> bool:
        """True iff every sequence fills all ``T`` frames (no key padding).

        Synchronizes once (host read) -- callers cache the result for the whole
        forward (length is layer-invariant) so the 32-layer attention loop stays
        sync-free. When True the SDPA padding mask can be dropped entirely, which
        lets the dispatcher pick the FlashAttention-2 kernel (it requires
        ``attn_mask=None``).
        """
        return bool(torch.all(length >= T))

    def _opt_additive_mask(self, length: torch.Tensor, T: int, dtype: torch.dtype, device, no_pad: bool, causal: bool):
        """Build the additive attention mask, or ``None`` when none is needed.

        Returns ``None`` when there is neither padding nor causal masking (so SDPA
        gets ``attn_mask=None`` and can dispatch FlashAttention-2). Otherwise
        returns a float bias broadcastable to ``(B, H, T, T)`` combining the
        key-padding mask (skipped when ``no_pad``) and the causal mask.
        """
        base = None
        if not no_pad:
            base = self._padding_additive_mask(length, T, dtype)  # (B, 1, 1, T)
        if causal:
            causal_add = torch.zeros(T, T, dtype=dtype, device=device).masked_fill(
                torch.ones(T, T, dtype=torch.bool, device=device).triu(1), torch.finfo(dtype).min
            )  # (T, T) -> broadcasts to (B, 1, T, T)
            base = causal_add if base is None else base + causal_add
        return base

    def _group_additive_mask(
        self,
        gkey,
        gnames: List[str],
        head_counts: List[int],
        state: dict,
        T: int,
        dtype: torch.dtype,
        device,
        causal: bool,
    ):
        """Additive attention mask for one packed-head group, honouring per-expert lengths.

        When every expert in the group shares a length this is the usual
        ``(B, 1, 1, T)`` key-padding mask (or ``None`` when nothing needs masking, so
        SDPA can dispatch FlashAttention-2). When lengths differ -- the streaming case,
        where the speaker carries a spkcache prefix and the others are right-padded to
        match -- each expert's ``(B, 1, 1, T)`` mask is expanded over its own head count
        and concatenated on the head axis into ``(B, Hg, 1, T)``, which SDPA broadcasts
        over queries. That is ``B*Hg*T`` elements (~108K at B=4/Hg=40/T=675) rather than
        the ``B*Hg*T*T`` (~73M) a full query-major mask would need.

        Cached on ``state`` under the group key: lengths do not change across layers,
        so the 24-layer loop builds this once.
        """
        cache = state.setdefault('_mask_cache', {})
        ckey = (gkey, causal, dtype)
        if ckey in cache:
            return cache[ckey]

        # The group can skip its padding mask only if EVERY member is unpadded.
        no_pad = all(state[n]['no_pad'] for n in gnames) and self.sdpa_fastpath
        lengths = [state[n]['length'] for n in gnames]
        uniform = no_pad or all(torch.equal(lengths[0], ln) for ln in lengths[1:])

        if uniform:
            base = self._opt_additive_mask(lengths[0], T, dtype, device, no_pad, causal)
        else:
            # (B, Hg, 1, T): each expert's key-padding mask over its own heads.
            parts = [
                self._padding_additive_mask(ln, T, dtype).expand(-1, hn, -1, -1)
                for ln, hn in zip(lengths, head_counts)
            ]
            base = torch.cat(parts, dim=1)
            if causal:
                causal_add = torch.zeros(T, T, dtype=dtype, device=device).masked_fill(
                    torch.ones(T, T, dtype=torch.bool, device=device).triu(1), torch.finfo(dtype).min
                )
                base = base + causal_add  # (B, Hg, 1, T) + (T, T) -> (B, Hg, T, T)
        cache[ckey] = base
        return base

    def _apply_rope_cached(self, rope, q: torch.Tensor, k: torch.Tensor):
        """Apply RoPE using cos/sin buffers cached in the q/k runtime dtype."""
        cos = rope.cos
        sin = rope.sin
        key = (
            id(rope),
            cos.data_ptr(),
            sin.data_ptr(),
            cos.device,
            cos.dtype,
            q.dtype,
        )
        runtime = self._rope_cache.get(key)
        if runtime is None:
            runtime = (cos.to(q.dtype), sin.to(q.dtype))
            self._rope_cache[key] = runtime
        cos, sin = runtime

        t_q = q.size(2)
        t_k = k.size(2)
        cache_len = t_k - t_q
        cos_k = cos[:t_k].view(1, 1, t_k, rope.d_k_rot)
        sin_k = sin[:t_k].view(1, 1, t_k, rope.d_k_rot)
        cos_q = cos[cache_len:t_k].view(1, 1, t_q, rope.d_k_rot)
        sin_q = sin[cache_len:t_k].view(1, 1, t_q, rope.d_k_rot)
        return rope._apply_rotary(q, cos_q, sin_q), rope._apply_rotary(k, cos_k, sin_k)

    def _expert_qkv(self, attn, x: torch.Tensor, pos_emb):
        """Project ``x`` to per-head q/k/v for one expert and return an optional
        additive score bias, reproducing the flex-attention math in
        :class:`MultiHeadAttention` (rope rotation / Transformer-XL rel-pos) so the
        batched-SDPA result matches the per-expert flex path to ULP tolerance.

        Returns ``(q, k, v, bias)`` with q/k/v shaped ``(B, H, T, D)`` and ``bias``
        either ``None`` (rope / no_pos) or ``(B, H, T, T)`` (rel_pos, pre-scaled).
        """
        B, T, _ = x.shape
        H, D = attn.n_heads, attn.head_dim
        qkv = attn.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, H, T, D)
        if attn.qk_norm:
            q = attn.q_norm(q).to(v.dtype)
            k = attn.k_norm(k).to(v.dtype)
        if attn._uses_rope:
            q, k = self._apply_rope_cached(attn.rope, q, k)
            return q, k, v, None
        if attn._uses_rel_pos:
            # Mirror MultiHeadAttention._build_rel_pos_score_mod, but return the bias
            # tensor (SDPA additive mask) instead of a flex score_mod closure.
            p = attn.linear_pos(pos_emb).view(pos_emb.size(0), -1, H, D).transpose(1, 2)
            bias_u = attn.pos_bias_u.view(1, H, 1, D).to(q.dtype)
            bias_v = attn.pos_bias_v.view(1, H, 1, D).to(q.dtype)
            matrix_bd = torch.matmul(q + bias_v, p.transpose(-2, -1))  # (B, H, T, 2T-1)
            rel_pos_bias = attn._rel_shift(matrix_bd)[..., :T] * (D**-0.5)  # (B, H, T, T)
            return q + bias_u, k, v, rel_pos_bias
        return q, k, v, None  # no_pos

    def _packed_attention_step(self, encs, state, layer_idx: int) -> None:
        """Run every expert's attention for layer ``layer_idx`` as batched SDPA over
        packed heads (one call per (head_dim, pos-scheme, attn_mode) group), then
        apply each expert's out-projection + residual in place on ``state``."""
        names = self.expert_names
        # Compute per-expert q/k/v (+bias) from the pre-attn LayerNorm of each expert.
        qkv = {}
        groups: Dict[Tuple[int, str, str], List[str]] = {}
        for n in names:
            e = encs[n]
            layer = e.layers[layer_idx]
            attn = layer.attn
            h = layer.norm1(state[n]['x'])
            q, k, v, bias = self._expert_qkv(attn, h, state[n]['pos_emb'])
            qkv[n] = (q, k, v, bias)
            key = (attn.head_dim, attn.self_attention_model if attn._uses_rel_pos else 'nobias', e.attn_mode)
            groups.setdefault(key, []).append(n)

        for gkey, gnames in groups.items():
            B, _, T, D = qkv[gnames[0]][0].shape
            dtype = qkv[gnames[0]][0].dtype
            device = qkv[gnames[0]][0].device
            causal = encs[gnames[0]].attn_mode == 'causal'
            Q = torch.cat([qkv[n][0] for n in gnames], dim=1)  # (B, Hg, T, D)
            K = torch.cat([qkv[n][1] for n in gnames], dim=1)
            V = torch.cat([qkv[n][2] for n in gnames], dim=1)
            has_bias = any(qkv[n][3] is not None for n in gnames)
            # Additive padding(+causal) mask; None when fully packed (no padding,
            # non-causal) so the no-bias path can hit FlashAttention-2. Experts in a
            # group can have DIFFERENT lengths (a streaming prefix lengthens one of
            # them), so the mask is per expert, expanded over that expert's own heads
            # -- using the first expert's length for the whole group would let the
            # shorter experts attend to their own right padding. Lengths are
            # layer-invariant, so this is built once per forward and cached.
            base = self._group_additive_mask(
                gkey, gnames, [qkv[n][0].shape[1] for n in gnames], state, T, dtype, device, causal
            )
            with sdpa_kernel(_SDPA_BACKENDS) if self.sdpa_fastpath else contextlib.nullcontext():
                if has_bias:
                    # (B, Hg, T, T) additive mask = per-expert rel-pos bias (or 0) [+ pad/causal].
                    # When `base` is per-expert (B, Hg, 1, T) it must be sliced to each
                    # expert's own head span rather than added whole.
                    per_head_base = base is not None and base.shape[1] > 1
                    parts, hoff = [], 0
                    for n in gnames:
                        Hn = qkv[n][0].shape[1]
                        b = (
                            qkv[n][3]
                            if qkv[n][3] is not None
                            else torch.zeros(B, Hn, T, T, dtype=dtype, device=device)
                        )
                        if base is not None:
                            b = b + (base[:, hoff : hoff + Hn] if per_head_base else base)
                        hoff += Hn
                        parts.append(b)
                    attn_mask = torch.cat(parts, dim=1)
                    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
                elif base is None:
                    # Fully packed, no mask -> FlashAttention-2 eligible (is_causal
                    # handled here only because base is None implies non-causal).
                    out = F.scaled_dot_product_attention(Q, K, V)
                else:
                    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=base)
            # Split heads back per expert, out-project, residual.
            off = 0
            for n in gnames:
                layer = encs[n].layers[layer_idx]
                Hn = qkv[n][0].shape[1]
                o = out[:, off : off + Hn]  # (B, Hn, T, D)
                off += Hn
                o = o.transpose(1, 2).contiguous().view(B, T, Hn * D)
                o = layer.attn.out_proj(o)
                state[n]['x'] = state[n]['x'] + layer.drop(o)

    def forward_packed(
        self,
        audio_signal,
        length,
        bypass_pre_encode: bool = False,
        backend: str = 'baddbmm',
        moe_mode: str = 'dense',
        prefix: Optional[Dict[str, torch.Tensor]] = None,
        return_pre_encode: bool = False,
        prefix_mode: str = 'extend',
    ) -> Dict[str, object]:
        """Lockstep multi-expert forward using batched-SDPA packed-head attention
        and grouped FFN -- the launch-count-reducing path (BGG-MEI).

        Per layer: (1) every expert's attention is computed as one batched
        ``scaled_dot_product_attention`` per (head_dim, pos-scheme, attn_mode)
        group over concatenated heads -- on PEE-v2 all three experts share
        ``head_dim = 128``, so that is a single call over 40 heads (16 + 8 + 16)
        with no padded head slots, since heads (not ``d_model``) are the packing
        unit. Then (2) every expert's FFN is fused into one grouped GEMM per
        ``d_hidden`` bucket (speech-MoE experts run dense + recombined, speaker
        zero-padded; see :meth:`_unified_ffn_step`). Output matches
        :meth:`forward_all` to ULP tolerance (README §4.1); attention switches from
        flex to SDPA, so it is allclose, not bit-exact, vs the native flex path.

        Args:
            prefix: Optional ``{name: (B, P, d_model)}`` of streaming-cache embeddings
                spliced into that expert's projected chunk before the norm, so the
                expert attends over ``[cache | chunk]``.
            prefix_mode: How the prefix is spliced (see :meth:`_prepend_prefix`).
                ``'extend'`` grows the prefixed expert to ``P + T`` and right-pads the
                others with zeros to match -- correct, but the pad frames are masked
                out, so that work is wasted and the group loses FlashAttention-2
                eligibility. ``'replace'`` has the cache stand in for the expert's own
                leading ``P`` frames, leaving ``T`` unchanged and nothing padded; feed a
                window extended ``P`` frames further back and the unprefixed experts
                spend those slots on real left context instead of zeros.
            return_pre_encode: Also return ``{name: (chunk, chunk_length)}`` -- the
                pre-norm chunk embeddings -- so a streaming caller can push them into
                its cache without reprojecting. Under ``'replace'`` this excludes the
                leading frames the cache stood in for.

        Returns:
            ``{name: (out, length)}``, or ``({name: (out, length)}, {name: (x_proj,
            out_length)})`` when ``return_pre_encode``.
        """
        encs = {name: self.experts[name] for name in self.expert_names}
        for name, e in encs.items():
            if not hasattr(e, 'layers') or not hasattr(e, 'n_layers'):
                raise TypeError(f"Expert '{name}' is not a flex TransformerEncoder-family module.")
        n_layers_set = {e.n_layers for e in encs.values()}
        if len(n_layers_set) != 1:
            raise ValueError(f"forward_packed requires equal n_layers, got {n_layers_set}.")
        n_layers = n_layers_set.pop()

        state: Dict[str, dict] = {}
        prepared = self._experts_pre(
            encs,
            audio_signal,
            length,
            bypass_pre_encode,
            build_block_mask=False,
            prefix=prefix,
            return_pre_encode=return_pre_encode,
            prefix_mode=prefix_mode,
        )
        if return_pre_encode:
            prepared, pre_encode_out = prepared
        for name, e in encs.items():
            x, pos_emb, block_mask, ln = prepared[name]
            # Cache the no-padding flag once (length is layer-invariant) so the
            # per-layer attention loop never re-syncs to decide whether the SDPA
            # mask can be dropped (FlashAttention-2 path).
            state[name] = {
                'x': x,
                'pos_emb': pos_emb,
                'block_mask': block_mask,
                'length': ln,
                'no_pad': self._no_padding(ln, x.shape[1]),
            }

        for i in range(n_layers):
            self._packed_attention_step(encs, state, i)
            # FFN: fuse ALL experts' FFN units (dense + speech-MoE experts +
            # zero-padded speaker) into grouped GEMMs by d_hidden.
            self._unified_ffn_step(encs, state, i, backend, moe_mode=moe_mode)

        out = {
            name: self._expert_post(encs[name], state[name]['x'], state[name]['length']) for name in self.expert_names
        }
        if return_pre_encode:
            return out, pre_encode_out
        return out

    def _sdpa_attention_single(self, e, layer_idx: int, s: dict) -> None:
        """Run a single expert's attention via SDPA (same FlashAttention kernel as
        :meth:`forward_packed`) but **without** head packing, applying out-proj +
        residual in place on ``s``.

        This is the per-expert analogue of :meth:`_packed_attention_step`; the two
        produce ULP-identical attention. Its purpose is to provide a fair "serial
        but FlashAttention" baseline so benchmarks can separate the flex->SDPA
        kernel-swap win from the head-packing win (see :meth:`forward_serial_sdpa`).
        """
        layer = e.layers[layer_idx]
        attn = layer.attn
        h = layer.norm1(s['x'])
        q, k, v, bias = self._expert_qkv(attn, h, s['pos_emb'])
        B, Hn, T, D = q.shape
        dtype = q.dtype
        causal = e.attn_mode == 'causal'
        no_pad = s['no_pad'] and self.sdpa_fastpath
        base = self._opt_additive_mask(s['length'], T, dtype, q.device, no_pad, causal)
        with sdpa_kernel(_SDPA_BACKENDS) if self.sdpa_fastpath else contextlib.nullcontext():
            if bias is not None:
                attn_mask = bias if base is None else bias + base
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            elif base is None:
                # Fully packed, no mask -> FlashAttention-2 eligible.
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=base)  # (B, Hn, T, D)
        o = out.transpose(1, 2).contiguous().view(B, T, Hn * D)
        o = attn.out_proj(o)
        s['x'] = s['x'] + layer.drop(o)

    def forward_serial_sdpa(self, audio_signal, length, bypass_pre_encode: bool = False) -> Dict[str, object]:
        """Reference "serial, but FlashAttention" path: **expert-major** -- each
        expert's full stack is run to completion before the next one starts
        (``enc1.forward(); enc2.forward(); enc3.forward()`` order), exactly like
        loading the three checkpoints separately. Per-expert SDPA attention (NOT
        head-packed), each expert's own FFN -- no grouped GEMM, no head concat,
        no cross-expert sharing, single stream.

        This isolates the two effects conflated in ``forward_all`` (native flex)
        vs :meth:`forward_packed` (packed SDPA + grouped FFN):

        - ``forward_all`` -> ``forward_serial_sdpa`` = the flex->SDPA **kernel
          swap** (large only where flex falls back to eager, e.g. head_dim the
          Triton flex template cannot fit in shared memory);
        - ``forward_serial_sdpa`` -> ``forward_packed`` = the genuine BGG-MEI
          **head-packing / launch-reduction** win.

        Unlike the lockstep paths this does not require equal ``n_layers`` across
        experts. Output matches :meth:`forward_all` to ULP tolerance (README
        §4.1); since attention is SDPA rather than flex it is allclose, not
        bit-exact.
        """
        encs = {name: self.experts[name] for name in self.expert_names}
        for name, e in encs.items():
            if not hasattr(e, 'layers') or not hasattr(e, 'n_layers'):
                raise TypeError(f"Expert '{name}' is not a flex TransformerEncoder-family module.")

        # Expert-major: run each encoder's FULL stack (pre-encode -> all layers ->
        # post) to completion before starting the next one. This mirrors the real
        # deployment where the three checkpoints are loaded separately and called
        # as enc1.forward(); enc2.forward(); enc3.forward() -- no interleaving and
        # nothing shared across experts. The only difference vs each expert's
        # native .forward() is that attention runs through SDPA instead of flex
        # (we cannot edit the base encoder to switch its kernel).
        out: Dict[str, object] = {}
        prepared = self._experts_pre(encs, audio_signal, length, bypass_pre_encode, build_block_mask=False)
        for name, e in encs.items():
            x, pos_emb, block_mask, ln = prepared[name]
            s = {
                'x': x,
                'pos_emb': pos_emb,
                'block_mask': block_mask,
                'length': ln,
                'no_pad': self._no_padding(ln, x.shape[1]),
            }
            for i in range(e.n_layers):
                self._sdpa_attention_single(e, i, s)
                layer = e.layers[i]
                s['x'] = s['x'] + layer.drop(layer.ffn(layer.norm2(s['x'])))
            out[name] = self._expert_post(e, s['x'], s['length'])

        return out

    @torch.no_grad()
    def verify_grouped_equivalence(
        self, audio_signal, length, bypass_pre_encode: bool = False, backend: str = 'baddbmm'
    ) -> Dict[str, float]:
        """Return ``role -> max|forward_all - forward_grouped|`` (run in eval mode).

        Use to confirm the lockstep fused path matches the per-expert reference to
        within ULP tolerance (README §4.1). Forces eval mode so dropout is off and
        the comparison is deterministic, restoring the prior mode afterwards.
        """
        was_training = self.training
        self.eval()
        try:
            ref = self.forward_all(audio_signal, length, bypass_pre_encode=bypass_pre_encode)
            grp = self.forward_grouped(audio_signal, length, bypass_pre_encode=bypass_pre_encode, backend=backend)
            return {name: (ref[name][0] - grp[name][0]).abs().max().item() for name in self.expert_names}
        finally:
            if was_training:
                self.train()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Namespace per-expert checkpoint keys under ``experts.<role>.``.

        When merging standalone ``.nemo`` experts into one PEE container, each
        source checkpoint's encoder keys (e.g. ``layers.0.ffn.net.0.weight``) are
        relative to that encoder. If such un-namespaced keys are present for a
        role that exists in this container, they are rewritten to
        ``experts.<role>.layers.0.ffn.net.0.weight`` so the standard loader can
        place them. Keys already namespaced under ``experts.`` are left untouched.
        """
        already = re.compile(r'^' + re.escape(prefix) + r'experts\.')
        bare_role = re.compile(
            r'^' + re.escape(prefix) + r'(' + '|'.join(re.escape(n) for n in self.expert_names) + r')\.'
        )

        keys_to_add = {}
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if already.match(key):
                continue
            m = bare_role.match(key)
            if m:
                role = m.group(1)
                suffix = key[m.end() :]
                keys_to_add[f"{prefix}experts.{role}.{suffix}"] = state_dict[key]
                keys_to_remove.append(key)

        # Loading new weights invalidates any pre-packed grouped-FFN cache.
        self._packed_cache.clear()
        self._rope_cache.clear()

        for key in keys_to_remove:
            del state_dict[key]
        state_dict.update(keys_to_add)
        if keys_to_remove:
            print(
                f"GGEMMTransformerEncoder: Namespaced {len(keys_to_remove)} bare expert keys "
                f"under 'experts.<role>.' for roles {self.expert_names}."
            )

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    @staticmethod
    def extract_encoder_state_dict(
        full_state_dict: Dict[str, torch.Tensor], encoder_attr: str = 'encoder'
    ) -> Dict[str, torch.Tensor]:
        """Pull one encoder's parameters out of a full model ``.nemo`` state dict.

        Each expert ships as a *full* model (preprocessor + encoder(s) + decoder),
        so its checkpoint keys are prefixed by the submodule attribute name. This
        strips the ``<encoder_attr>.`` prefix and returns only that encoder's
        keys, ready to load into the corresponding PEE expert.

        Args:
            full_state_dict: ``model.state_dict()`` of the source expert.
            encoder_attr: Attribute under which the encoder lives in the source
                model. ``'encoder'`` for the ASR experts; ``'transformer_encoder'``
                for the Sortformer speaker expert (whose ``encoder`` attribute is
                the upstream FastConformer, not the PEE transformer encoder).

        Returns:
            ``{stripped_key: tensor}`` for the requested encoder only.
        """
        prefix = f"{encoder_attr}."
        return {key[len(prefix) :]: value for key, value in full_state_dict.items() if key.startswith(prefix)}
