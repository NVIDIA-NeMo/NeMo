# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import torch
import triton
import triton.language as tl


@triton.jit
def _rnnt_logprobs_fwd_kernel(
    logits_ptr,
    targets_ptr,
    source_lengths_ptr,
    target_lengths_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,  # vocab size (with blank)
    blank_id: int,
    target_scores_ptr,
    blank_scores_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Forward kernel for RNN-T log probs. Stores result in `target_scores_ptr` and `blank_scores_ptr`.
    Calculations are performed in float32 (but original tensors can use any precision).
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    # load lengths for source/target
    source_len = tl.load(source_lengths_ptr + batch_i)
    target_len = tl.load(target_lengths_ptr + batch_i)

    if source_i >= source_len or target_i > target_len:
        # no calculations required
        return

    # calculate offset in [B, T, U+1, V] tensor for the current vector with target logits
    flat_index = ((batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i) * num_labels
    logits_ptr += flat_index
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    # stable log softmax calculation
    logits_max = tl.max(logits, axis=0)
    logits_minus_max = logits - logits_max
    denominator = tl.log(tl.sum(tl.exp(logits_minus_max), axis=0))
    blank_logit = tl.load(logits_ptr + blank_id).to(tl.float32)
    flat_index_output = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    tl.store(blank_scores_ptr + flat_index_output, blank_logit - logits_max - denominator)

    # calculate log prob for target if needed
    if target_i < target_len:
        target_id = tl.load(targets_ptr + batch_i * (max_target_len_plus_1 - 1) + target_i)
        target_logit = tl.load(logits_ptr + target_id).to(tl.float32)
        tl.store(target_scores_ptr + flat_index_output, target_logit - logits_max - denominator)


@triton.jit
def _rnnt_logprobs_bwd_kernel(
    logits_ptr,
    grad_logits_ptr,
    targets_ptr,
    source_lengths_ptr,
    target_lengths_ptr,
    max_source_len: int,
    max_target_len_plus_1: int,
    num_labels: int,
    blank_id: int,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    upstream_scale_ptr,
    clamp: float,
    CLAMP_GRAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for RNN-T log probs. Stores result in `grad_target_scores_ptr` and `grad_blank_scores_ptr`.
    We recalculate part of the forward here to avoid using extra memory in forward.
    Calculations are performed in float32 (but original tensors can use any precision).
    """
    batch_i = tl.program_id(axis=0).to(tl.int64)
    source_i = tl.program_id(axis=1).to(tl.int64)
    target_i = tl.program_id(axis=2).to(tl.int64)

    # load lengths for source/target
    source_len = tl.load(source_lengths_ptr + batch_i)
    target_len = tl.load(target_lengths_ptr + batch_i)
    valid_state = (source_i < source_len) & (target_i <= target_len)

    # calculate offset in [B, T, U+1, V] tensor for the current vector with target logits/grad_logits
    flat_index = ((batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i) * num_labels
    logits_ptr += flat_index
    grad_logits_ptr += flat_index

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_labels
    logits = tl.load(logits_ptr + col_offsets, mask=mask & valid_state, other=-float("inf")).to(tl.float32)
    # stable log softmax calculation
    logits_max = tl.max(logits, axis=0)
    logits_minus_max = logits - logits_max
    unnormalized = tl.exp(logits_minus_max)
    denominator_sum = tl.sum(unnormalized, axis=0)
    # softmax for gradient
    softmax = unnormalized / denominator_sum

    flat_index_grad = (batch_i * max_source_len + source_i) * max_target_len_plus_1 + target_i
    blank_grad = tl.load(grad_blank_scores_ptr + flat_index_grad, mask=valid_state, other=0.0).to(tl.float32)
    target_i_valid = valid_state & (target_i < target_len)
    target_grad = tl.load(grad_target_scores_ptr + flat_index_grad, mask=target_i_valid, other=0.0).to(tl.float32)
    target_id = tl.load(targets_ptr + batch_i * (max_target_len_plus_1 - 1) + target_i, mask=target_i_valid, other=-1)

    if CLAMP_GRAD:
        # Numba clamps the unit-scale per-sample gradient before multiplying by the loss
        # reduction or AMP scale, so divide that scale out, clamp, and reapply it below.
        upstream_scale = tl.load(upstream_scale_ptr + batch_i).to(tl.float32)
        inverse_scale = tl.where(upstream_scale != 0.0, 1.0 / upstream_scale, 0.0)
        blank_grad *= inverse_scale
        target_grad *= inverse_scale

    grad_not_in_targets = (-softmax) * (blank_grad + target_grad)
    # Add both deltas instead of overwriting one with the other. This also keeps
    # malformed target==blank inputs mathematically correct.
    grad = grad_not_in_targets
    grad += tl.where(col_offsets == blank_id, blank_grad, 0.0)
    grad += tl.where(col_offsets == target_id, target_grad, 0.0)
    if CLAMP_GRAD:
        grad = tl.maximum(tl.minimum(grad, clamp), -clamp)
        grad *= upstream_scale
    grad = tl.where(valid_state, grad, 0.0)
    tl.store(grad_logits_ptr + col_offsets, grad, mask=mask)


def _validate_rnnt_logprobs_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    source_lengths: torch.Tensor | None,
    target_lengths: torch.Tensor | None,
) -> None:
    """Validate the tensor layout assumed by the pointer arithmetic below."""
    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [B, T, U + 1, V], got {tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape [B, U], got {tuple(targets.shape)}")
    expected_targets = (logits.shape[0], logits.shape[2] - 1)
    if targets.shape != expected_targets:
        raise ValueError(f"targets must have shape {expected_targets}, got {tuple(targets.shape)}")
    if not 0 <= blank_id < logits.shape[-1]:
        raise ValueError(f"blank_id={blank_id} must be in [0, {logits.shape[-1]})")
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")
    if targets.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    if targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must use int32 or int64 indices")

    for name, lengths in (("source_lengths", source_lengths), ("target_lengths", target_lengths)):
        if lengths is None:
            continue
        if lengths.shape != (logits.shape[0],):
            raise ValueError(f"{name} must have shape ({logits.shape[0]},), got {tuple(lengths.shape)}")
        if lengths.device != logits.device:
            raise ValueError(f"{name} and logits must be on the same device")
        if lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must use int32 or int64 values")


class RnntLogProbs(torch.autograd.Function):
    """
    Function to calculate log probabilities for target and blank labels for RNN-T, supporting torch.autograd.
    """

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        targets: torch.Tensor,
        blank_id: int,
        source_lengths: torch.Tensor | None,
        target_lengths: torch.Tensor | None,
        clamp: float,
        reuse_logits_for_grad: bool,
        upstream_scale: torch.Tensor | None,
    ):
        """

        Args:
            ctx: ctx object for storing the context
            logits: Joint tensor of size [B, T, U+1, D]
            targets: Targets of size [B, U]
            blank_id: id of the blank output
            source_lengths: optional tensor with lengths for source utterances
            target_lengths: optional tensor with lengths for targets

        Returns:

        """
        targets = targets.contiguous()
        device = logits.device
        float_dtype = torch.float32

        target_scores = torch.zeros(logits.shape[:-1], dtype=float_dtype, device=device)
        blank_scores = torch.zeros_like(target_scores)
        if source_lengths is None:
            source_lengths = torch.full([logits.shape[0]], fill_value=logits.shape[1], dtype=torch.int, device=device)
        else:
            source_lengths = source_lengths.contiguous()
        if target_lengths is None:
            target_lengths = torch.full(
                [logits.shape[0]], fill_value=logits.shape[2] - 1, dtype=torch.int, device=device
            )
        else:
            target_lengths = target_lengths.contiguous()

        # run Triton kernel
        _rnnt_logprobs_fwd_kernel[(logits.shape[0], logits.shape[1], logits.shape[2])](
            logits_ptr=logits,
            targets_ptr=targets,
            source_lengths_ptr=source_lengths,
            target_lengths_ptr=target_lengths,
            max_source_len=logits.shape[1],
            max_target_len_plus_1=logits.shape[2],
            num_labels=logits.shape[3],
            blank_id=blank_id,
            target_scores_ptr=target_scores,
            blank_scores_ptr=blank_scores,
            BLOCK_SIZE=triton.next_power_of_2(logits.shape[-1]),
        )

        # saving for backward
        ctx.save_for_backward(logits, targets, source_lengths, target_lengths)
        ctx.blank_id = blank_id
        ctx.clamp = float(clamp) if clamp > 0.0 else 0.0
        ctx.reuse_logits_for_grad = reuse_logits_for_grad
        ctx.reused_logits_consumed = False
        # Held outside save_for_backward: the loss fills it during its own backward, which
        # runs before ours because these scores are what it consumes.
        ctx.upstream_scale = upstream_scale
        return target_scores, blank_scores

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        """
        Backward calculation for RNN-T log-probs.

        Args:
            ctx: ctx object for storing the context
            grad_target_scores: upstream gradient for targets
            grad_blank_scores:  upstream gradient for blank scores

        Returns:
            gradient for logits, None for all other arguments for `forward`
        """
        if ctx.reuse_logits_for_grad:
            if ctx.reused_logits_consumed:
                raise RuntimeError("reuse_logits_for_grad=True only supports one backward pass")
            ctx.reused_logits_consumed = True
        (logits, targets, source_lengths, target_lengths) = ctx.saved_tensors
        blank_id = ctx.blank_id
        clamp = ctx.clamp
        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()
        grad_logits = logits if ctx.reuse_logits_for_grad else torch.zeros_like(logits)
        # Any valid pointer will do when clamping is off: CLAMP_GRAD is a constexpr, so the
        # only branch that reads this argument is compiled out.
        upstream_scale = ctx.upstream_scale if ctx.upstream_scale is not None else grad_blank_scores
        _rnnt_logprobs_bwd_kernel[(logits.shape[0], logits.shape[1], logits.shape[2])](
            logits_ptr=logits,
            grad_logits_ptr=grad_logits,
            source_lengths_ptr=source_lengths,
            target_lengths_ptr=target_lengths,
            targets_ptr=targets,
            max_source_len=logits.shape[1],
            max_target_len_plus_1=logits.shape[2],
            num_labels=logits.shape[3],
            blank_id=blank_id,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            upstream_scale_ptr=upstream_scale,
            clamp=clamp,
            CLAMP_GRAD=clamp > 0.0,
            BLOCK_SIZE=triton.next_power_of_2(logits.shape[-1]),
        )
        return grad_logits, None, None, None, None, None, None, None


def rnnt_logprobs_triton(
    logits: torch.Tensor,
    targets: torch.Tensor,
    blank_id: int,
    source_lengths: torch.Tensor | None = None,
    target_lengths: torch.Tensor | None = None,
    clamp: float = -1.0,
    reuse_logits_for_grad: bool = False,
    upstream_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given logits, calculate log probabilities for blank and target labels needed for transducer loss calculation.
    Optimized implementation in Triton.

    Args:
        logits: Joint tensor of size [B, T, U+1, D]
        targets: Targets of size [B, U]
        blank_id: id of the blank output
        source_lengths: optional tensor with lengths for source utterances
        target_lengths: optional tensor with lengths for targets
        clamp: clamp the unit-scale standard RNN-T gradient before applying its
            per-sample upstream scale
        reuse_logits_for_grad: overwrite logits with their gradient during backward; only safe for private,
            disposable logits; a second backward through the same graph raises
        upstream_scale: ``[B]`` float32 buffer carrying each sample's upstream gradient, filled by
            ``rnnt_loss_triton`` during its backward. Required when clamping, because the clamp
            applies to the unit-scale gradient and autograd has already folded that scale in.

    Returns:
        Tuple of tensors with log probabilities for targets and blank labels, both of size [B, T, U+1].
        For the non-existent targets (U+1 or beyond target_lengths) output is zero.
    """
    _validate_rnnt_logprobs_inputs(logits, targets, blank_id, source_lengths, target_lengths)
    if clamp > 0.0:
        if upstream_scale is None:
            raise ValueError("Clamping the RNN-T gradient requires upstream_scale")
        if upstream_scale.shape != (logits.shape[0],) or upstream_scale.dtype != torch.float32:
            raise ValueError(
                f"upstream_scale must be a float32 tensor of shape ({logits.shape[0]},), "
                f"got {tuple(upstream_scale.shape)} of {upstream_scale.dtype}"
            )
    return RnntLogProbs.apply(
        logits, targets, blank_id, source_lengths, target_lengths, clamp, reuse_logits_for_grad, upstream_scale
    )
