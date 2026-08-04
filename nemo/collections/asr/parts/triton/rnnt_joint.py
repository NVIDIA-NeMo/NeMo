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

"""Fused broadcast-add, activation and dropout for the RNN-T joint network.

The joint hidden state is ``act(encoder[b, t, :] + predictor[b, u, :])`` over a
``[B, T, U + 1, H]`` grid. Writing it as a Triton kernel rather than relying on
``torch.compile`` keeps a single code path: Dynamo forks a graph per shape, dtype and
grad mode, and once its recompile limit is reached it silently falls back to eager,
which changes reduced-precision rounding part-way through a run.

Backward saves only the two projected operands and rebuilds the pre-activation inside the
reduction kernels, so no ``[B, T, U + 1, H]`` tensor stays live across the step.

Dropout is folded in rather than applied afterwards, and its mask is never stored: each kernel
redraws the mask from a per-call seed and the element's position in the grid, which all three agree
on even though they traverse different axes. Applying dropout separately would instead read and
rewrite the whole hidden state and hold a mask beside it, both of which the peak has to cover.
"""

from __future__ import annotations

import torch

from nemo.core.utils.optional_libs import TRITON_AVAILABLE

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

ACTIVATIONS = ("relu", "sigmoid", "tanh")


if TRITON_AVAILABLE:

    @triton.jit
    def _activate_fwd(value, activation: tl.constexpr):
        # No fallback branch: an unlisted activation matches nothing and fails to compile.
        if activation == "relu":
            return tl.maximum(value, 0.0)
        if activation == "sigmoid":
            return tl.sigmoid(value)
        if activation == "tanh":
            return (2.0 * tl.sigmoid(2.0 * value)) - 1.0

    @triton.jit
    def _activate_bwd(pre_activation, activation: tl.constexpr):
        """d act / d pre-activation, recomputed from the pre-activation."""
        if activation == "relu":
            return tl.where(pre_activation > 0.0, 1.0, 0.0)
        output = _activate_fwd(pre_activation, activation)
        if activation == "sigmoid":
            return output * (1.0 - output)
        if activation == "tanh":
            return 1.0 - (output * output)

    @triton.jit
    def _rng_row_stride(hidden_size):
        """Row spacing that keeps every block's first element on a Philox boundary.

        Four consecutive elements share one draw, so rounding each row up to a multiple of four
        makes a block's key exactly ``base_index // 4``. This spacing places draws only; memory is
        addressed with ``hidden_size``.
        """
        return ((hidden_size + 3) // 4) * 4

    @triton.jit
    def _apply_dropout(value, seed, base_index, dropout_p: tl.constexpr, block_hidden: tl.constexpr):
        """Zero the dropped elements of a hidden block and rescale the survivors by 1 / (1 - p).

        An element's draw depends only on its position in the grid ``_rng_row_stride`` spaces, which
        ``base_index`` locates, so the forward and both backward kernels agree without storing a mask.

        Philox produces four values per call, so the block draws one key per four elements and
        unpacks all four.
        """
        if dropout_p > 0.0:
            keys = base_index // 4 + tl.arange(0, block_hidden // 4)
            first, second, third, fourth = tl.rand4x(seed, keys)
            # Interleave back into element order, so the draw does not depend on block_hidden
            uniform = tl.reshape(tl.join(tl.join(first, third), tl.join(second, fourth)), [block_hidden])
            return tl.where(uniform > dropout_p, value / (1.0 - dropout_p), 0.0)
        return value

    @triton.jit
    def _joint_fwd_kernel(
        encoder_ptr,
        predictor_ptr,
        hidden_ptr,
        seed_ptr,
        source_steps,
        target_states,
        hidden_size,
        activation: tl.constexpr,
        dropout_p: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        """Write ``act(encoder + predictor)`` with dropout applied, one (batch, source, target) row each."""
        row = tl.program_id(0).to(tl.int64)
        # Row indexes the flattened (batch, source, target) grid
        target_idx = row % target_states
        source_row = row // target_states
        batch_idx = source_row // source_steps

        offsets = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
        mask = offsets < hidden_size
        encoder = tl.load(encoder_ptr + source_row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        predictor = tl.load(
            predictor_ptr + (batch_idx * target_states + target_idx) * hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        hidden = _activate_fwd(encoder + predictor, activation)
        rng_base = row * _rng_row_stride(hidden_size) + tl.program_id(1) * block_hidden
        hidden = _apply_dropout(hidden, tl.load(seed_ptr), rng_base, dropout_p, block_hidden)
        tl.store(hidden_ptr + row * hidden_size + offsets, hidden, mask=mask)

    @triton.jit
    def _joint_bwd_encoder_kernel(
        encoder_ptr,
        predictor_ptr,
        grad_hidden_ptr,
        grad_encoder_ptr,
        seed_ptr,
        source_steps,
        target_states,
        hidden_size,
        activation: tl.constexpr,
        dropout_p: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        """Sum ``grad_hidden * act'(pre)`` over the target axis, one (batch, source) row each.

        The pre-activation is rebuilt from the two projected inputs, so backward needs neither the
        [B, T, U + 1, H] hidden state nor a pre-activation gradient buffer -- only the small
        [B, T, H] and [B, U + 1, H] operands stay live across the step.
        """
        row = tl.program_id(0).to(tl.int64)  # batch * source_steps + source
        batch_idx = row // source_steps
        offsets = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
        mask = offsets < hidden_size
        encoder = tl.load(encoder_ptr + row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_base = row * target_states * hidden_size + offsets
        predictor_base = batch_idx * target_states * hidden_size + offsets
        seed = tl.load(seed_ptr)
        rng_row_stride = _rng_row_stride(hidden_size)
        rng_base = row * target_states * rng_row_stride + tl.program_id(1) * block_hidden
        total = tl.zeros((block_hidden,), tl.float32)
        for target_idx in tl.range(0, target_states):
            predictor = tl.load(predictor_ptr + predictor_base + target_idx * hidden_size, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_hidden = tl.load(grad_hidden_ptr + grad_base + target_idx * hidden_size, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_hidden = _apply_dropout(
                grad_hidden, seed, rng_base + target_idx * rng_row_stride, dropout_p, block_hidden
            )
            total += grad_hidden * _activate_bwd(encoder + predictor, activation)
        tl.store(grad_encoder_ptr + row * hidden_size + offsets, total, mask=mask)

    @triton.jit
    def _joint_bwd_predictor_kernel(
        encoder_ptr,
        predictor_ptr,
        grad_hidden_ptr,
        grad_predictor_ptr,
        seed_ptr,
        source_steps,
        target_states,
        hidden_size,
        activation: tl.constexpr,
        dropout_p: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        """Sum ``grad_hidden * act'(pre)`` over the source axis, one (batch, target) row each."""
        row = tl.program_id(0).to(tl.int64)  # batch * target_states + target
        batch_idx = row // target_states
        target_idx = row % target_states
        offsets = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
        mask = offsets < hidden_size
        predictor = tl.load(predictor_ptr + row * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        source_stride = target_states * hidden_size
        grad_base = batch_idx * source_steps * source_stride + target_idx * hidden_size + offsets
        encoder_base = batch_idx * source_steps * hidden_size + offsets
        seed = tl.load(seed_ptr)
        rng_row_stride = _rng_row_stride(hidden_size)
        rng_source_stride = target_states * rng_row_stride
        rng_base = (
            batch_idx * source_steps * rng_source_stride
            + target_idx * rng_row_stride
            + tl.program_id(1) * block_hidden
        )
        total = tl.zeros((block_hidden,), tl.float32)
        for source_idx in tl.range(0, source_steps):
            encoder = tl.load(encoder_ptr + encoder_base + source_idx * hidden_size, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_hidden = tl.load(grad_hidden_ptr + grad_base + source_idx * source_stride, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_hidden = _apply_dropout(
                grad_hidden, seed, rng_base + source_idx * rng_source_stride, dropout_p, block_hidden
            )
            total += grad_hidden * _activate_bwd(encoder + predictor, activation)
        tl.store(grad_predictor_ptr + row * hidden_size + offsets, total, mask=mask)


def _block_hidden(hidden_size: int) -> int:
    """Lanes per program, at least four so a block covers whole Philox draws."""
    return max(4, min(1024, triton.next_power_of_2(hidden_size)))


class _JointHiddenState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, encoder: torch.Tensor, predictor: torch.Tensor, activation: str, dropout_p: float):
        batch, source_steps, hidden_size = encoder.shape
        target_states = predictor.shape[1]
        hidden = torch.empty(
            (batch, source_steps, target_states, hidden_size), device=encoder.device, dtype=encoder.dtype
        )
        # The kernels rebuild the whole mask from this one number, so it is all backward needs to
        # keep. It stays on the device because reading it on the host would synchronize per tile.
        seed = torch.empty(1, device=encoder.device, dtype=torch.int32)
        if dropout_p > 0.0:
            seed.random_(0, 2**31 - 1)
        block_hidden = _block_hidden(hidden_size)
        grid = (batch * source_steps * target_states, triton.cdiv(hidden_size, block_hidden))
        _joint_fwd_kernel[grid](
            encoder,
            predictor,
            hidden,
            seed,
            source_steps=source_steps,
            target_states=target_states,
            hidden_size=hidden_size,
            activation=activation,
            dropout_p=dropout_p,
            block_hidden=block_hidden,
            # One warp per row, so each lane carries several elements. The kernel only streams a
            # write, and splitting that stream across more warps adds no bandwidth.
            num_warps=1,
        )
        ctx.save_for_backward(encoder, predictor, seed)
        ctx.activation = activation
        ctx.dropout_p = dropout_p
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden):
        encoder, predictor, seed = ctx.saved_tensors
        grad_hidden = grad_hidden.contiguous()
        batch, source_steps, hidden_size = encoder.shape
        target_states = predictor.shape[1]
        activation = ctx.activation
        dropout_p = ctx.dropout_p
        block_hidden = _block_hidden(hidden_size)
        hidden_blocks = triton.cdiv(hidden_size, block_hidden)

        # Broadcast-add backward: sum the target axis for the encoder, the source axis for the predictor
        grad_encoder = torch.empty_like(encoder)
        _joint_bwd_encoder_kernel[(batch * source_steps, hidden_blocks)](
            encoder,
            predictor,
            grad_hidden,
            grad_encoder,
            seed,
            source_steps=source_steps,
            target_states=target_states,
            hidden_size=hidden_size,
            activation=activation,
            dropout_p=dropout_p,
            block_hidden=block_hidden,
        )
        grad_predictor = torch.empty_like(predictor)
        _joint_bwd_predictor_kernel[(batch * target_states, hidden_blocks)](
            encoder,
            predictor,
            grad_hidden,
            grad_predictor,
            seed,
            source_steps=source_steps,
            target_states=target_states,
            hidden_size=hidden_size,
            activation=activation,
            dropout_p=dropout_p,
            block_hidden=block_hidden,
        )
        return grad_encoder, grad_predictor, None, None


def joint_hidden_state(
    encoder: torch.Tensor, predictor: torch.Tensor, activation: str, dropout_p: float = 0.0
) -> torch.Tensor:
    """Return ``dropout(act(encoder.unsqueeze(2) + predictor.unsqueeze(1)))``.

    Args:
        encoder: projected encoder states, ``[B, T, H]``.
        predictor: projected prediction-network states, ``[B, U + 1, H]``.
        activation: one of ``ACTIVATIONS``.
        dropout_p: probability of zeroing an element of the activated hidden state; survivors are
            scaled by ``1 / (1 - dropout_p)``. Applied whenever it is positive, so the caller
            decides whether the module is in training mode. The mask is drawn per call and is not
            the one ``torch.nn.functional.dropout`` would have produced for the same torch seed.

    Returns:
        Joint hidden states of shape ``[B, T, U + 1, H]``.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for the fused RNN-T joint activation")
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported RNN-T joint activation: {activation}")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
    if encoder.ndim != 3 or predictor.ndim != 3:
        raise ValueError(f"expected [B, T, H] and [B, U + 1, H], got {tuple(encoder.shape)} {tuple(predictor.shape)}")
    if encoder.shape[0] != predictor.shape[0] or encoder.shape[2] != predictor.shape[2]:
        raise ValueError(f"incompatible joint shapes: {tuple(encoder.shape)} and {tuple(predictor.shape)}")
    return _JointHiddenState.apply(encoder.contiguous(), predictor.contiguous(), activation, dropout_p)
