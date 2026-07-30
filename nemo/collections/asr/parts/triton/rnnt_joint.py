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

"""Fused broadcast-add plus activation for the RNN-T joint network.

The joint hidden state is ``act(encoder[b, t, :] + predictor[b, u, :])`` over a
``[B, T, U + 1, H]`` grid. Writing it as a Triton kernel rather than relying on
``torch.compile`` keeps a single code path: Dynamo forks a graph per shape, dtype and
grad mode, and once its recompile limit is reached it silently falls back to eager,
which changes reduced-precision rounding part-way through a run.

Backward saves only the two projected operands and rebuilds the pre-activation inside the
reduction kernels, so no ``[B, T, U + 1, H]`` tensor stays live across the step.
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
    def _activate(value, activation: tl.constexpr):
        if activation == "relu":
            return tl.maximum(value, 0.0)
        if activation == "sigmoid":
            return tl.sigmoid(value)
        # Triton exposes tanh only through the CUDA-specific libdevice namespace
        return (2.0 * tl.sigmoid(2.0 * value)) - 1.0

    @triton.jit
    def _activation_grad_from_pre(pre_activation, activation: tl.constexpr):
        """d act / d pre-activation, recomputed from the pre-activation."""
        if activation == "relu":
            return tl.where(pre_activation > 0.0, 1.0, 0.0)
        output = _activate(pre_activation, activation)
        if activation == "sigmoid":
            return output * (1.0 - output)
        return 1.0 - (output * output)

    @triton.jit
    def _join_activate_kernel(
        encoder_ptr,
        predictor_ptr,
        hidden_ptr,
        source_steps,
        target_states,
        hidden_size,
        activation: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        # row indexes the flattened (batch, source, target) grid
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
        tl.store(hidden_ptr + row * hidden_size + offsets, _activate(encoder + predictor, activation), mask=mask)

    @triton.jit
    def _join_grad_encoder_kernel(
        encoder_ptr,
        predictor_ptr,
        grad_hidden_ptr,
        grad_encoder_ptr,
        source_steps,
        target_states,
        hidden_size,
        activation: tl.constexpr,
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
        total = tl.zeros((block_hidden,), tl.float32)
        for target_idx in tl.range(0, target_states):
            predictor = tl.load(predictor_ptr + predictor_base + target_idx * hidden_size, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_hidden = tl.load(grad_hidden_ptr + grad_base + target_idx * hidden_size, mask=mask, other=0.0).to(
                tl.float32
            )
            total += grad_hidden * _activation_grad_from_pre(encoder + predictor, activation)
        tl.store(grad_encoder_ptr + row * hidden_size + offsets, total, mask=mask)

    @triton.jit
    def _join_grad_predictor_kernel(
        encoder_ptr,
        predictor_ptr,
        grad_hidden_ptr,
        grad_predictor_ptr,
        source_steps,
        target_states,
        hidden_size,
        activation: tl.constexpr,
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
        total = tl.zeros((block_hidden,), tl.float32)
        for source_idx in tl.range(0, source_steps):
            encoder = tl.load(encoder_ptr + encoder_base + source_idx * hidden_size, mask=mask, other=0.0).to(
                tl.float32
            )
            grad_hidden = tl.load(grad_hidden_ptr + grad_base + source_idx * source_stride, mask=mask, other=0.0).to(
                tl.float32
            )
            total += grad_hidden * _activation_grad_from_pre(encoder + predictor, activation)
        tl.store(grad_predictor_ptr + row * hidden_size + offsets, total, mask=mask)


class _JoinActivate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, encoder: torch.Tensor, predictor: torch.Tensor, activation: str):
        batch, source_steps, hidden_size = encoder.shape
        target_states = predictor.shape[1]
        hidden = torch.empty(
            (batch, source_steps, target_states, hidden_size), device=encoder.device, dtype=encoder.dtype
        )
        block_hidden = min(1024, triton.next_power_of_2(hidden_size))
        grid = (batch * source_steps * target_states, triton.cdiv(hidden_size, block_hidden))
        _join_activate_kernel[grid](
            encoder,
            predictor,
            hidden,
            source_steps=source_steps,
            target_states=target_states,
            hidden_size=hidden_size,
            activation=activation,
            block_hidden=block_hidden,
        )
        ctx.save_for_backward(encoder, predictor)
        ctx.activation = activation
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden):
        encoder, predictor = ctx.saved_tensors
        grad_hidden = grad_hidden.contiguous()
        batch, source_steps, hidden_size = encoder.shape
        target_states = predictor.shape[1]
        activation = ctx.activation
        block_hidden = min(1024, triton.next_power_of_2(hidden_size))
        hidden_blocks = triton.cdiv(hidden_size, block_hidden)

        # broadcast-add backward: sum the target axis for the encoder, the source axis for the predictor
        grad_encoder = torch.empty_like(encoder)
        _join_grad_encoder_kernel[(batch * source_steps, hidden_blocks)](
            encoder,
            predictor,
            grad_hidden,
            grad_encoder,
            source_steps=source_steps,
            target_states=target_states,
            hidden_size=hidden_size,
            activation=activation,
            block_hidden=block_hidden,
        )
        grad_predictor = torch.empty_like(predictor)
        _join_grad_predictor_kernel[(batch * target_states, hidden_blocks)](
            encoder,
            predictor,
            grad_hidden,
            grad_predictor,
            source_steps=source_steps,
            target_states=target_states,
            hidden_size=hidden_size,
            activation=activation,
            block_hidden=block_hidden,
        )
        return grad_encoder, grad_predictor, None


def join_activate(encoder: torch.Tensor, predictor: torch.Tensor, activation: str) -> torch.Tensor:
    """Return ``act(encoder.unsqueeze(2) + predictor.unsqueeze(1))`` without a Dynamo cache.

    Args:
        encoder: projected encoder states, ``[B, T, H]``.
        predictor: projected prediction-network states, ``[B, U + 1, H]``.
        activation: one of ``ACTIVATIONS``.

    Returns:
        Joint hidden states of shape ``[B, T, U + 1, H]``.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for the fused RNN-T joint activation")
    if activation not in ACTIVATIONS:
        raise ValueError(f"Unsupported RNN-T joint activation: {activation}")
    if encoder.ndim != 3 or predictor.ndim != 3:
        raise ValueError(f"expected [B, T, H] and [B, U + 1, H], got {tuple(encoder.shape)} {tuple(predictor.shape)}")
    if encoder.shape[0] != predictor.shape[0] or encoder.shape[2] != predictor.shape[2]:
        raise ValueError(f"incompatible join shapes: {tuple(encoder.shape)} and {tuple(predictor.shape)}")
    return _JoinActivate.apply(encoder.contiguous(), predictor.contiguous(), activation)
