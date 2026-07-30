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

"""Exact RNN-T with a bounded joint workspace and activation recomputation."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo.collections.asr.parts.triton.rnnt_joint import join_activate
from nemo.collections.asr.parts.triton.rnnt_loss import MAX_TARGET_TOKENS
from nemo.core.utils.optional_libs import TRITON_AVAILABLE

_DEFAULT_MAX_JOINT_ROWS = 200_000


def _validate_joint(joint, blank: int) -> torch.nn.Dropout | None:
    """Validate the narrow RNNTJoint structure consumed by the flash path."""
    if joint.is_adapter_available() or joint.masking_prob > 0.0:
        raise ValueError("Flash RNN-T does not support adapters or HAINAN masking")
    if joint.num_extra_outputs != 0 or blank != joint.num_classes_with_blank - 1:
        raise ValueError("Flash RNN-T requires standard RNN-T with a final blank output")
    if len(joint.joint_net) not in (2, 3) or not isinstance(joint.joint_net[-1], torch.nn.Linear):
        raise ValueError("Flash RNN-T requires activation, optional dropout, and one output linear layer")

    output = joint.joint_net[-1]
    if output.out_features != joint.num_classes_with_blank:
        raise ValueError("Flash RNN-T requires the joint output to include every label and the blank")
    if not 0 <= blank < output.out_features:
        raise ValueError(f"blank={blank} must be in [0, {output.out_features})")

    dropout = joint.joint_net[1] if len(joint.joint_net) == 3 else None
    if dropout is not None and not isinstance(dropout, torch.nn.Dropout):
        raise ValueError("Flash RNN-T only supports torch.nn.Dropout in the joint network")
    if joint.log_softmax is True or joint.temperature != 1.0:
        raise ValueError("Flash RNN-T requires unnormalized joint logits with temperature 1")
    return dropout


class FlashRNNTLoss(torch.nn.Module):
    """Exact Flash RNN-T loss configured for the fused joint path.

    ``RNNTJoint.fused_batch_size`` supplies ``max_samples_per_chunk``. It limits
    samples in each chunk, not the number of chunks. A local batch of size ``B``
    produces ``ceil(B / fused_batch_size)`` chunks.
    A chunk costs what its longest member costs, so smaller chunks trim padding
    more aggressively: with a 4x spread of utterance lengths, chunks of 4 run
    roughly twice as fast as chunks of 32 at the same peak memory. Prefer the
    smallest value that still saturates the GPU. ``max_joint_rows`` independently
    sets the target row budget for the disposable workspace within each chunk.

    ``max_target_tokens`` is a safety guard, not a compile-time reservation.
    The actual padded target length selects the Triton scan width, so leaving
    the guard at its maximum does not slow ordinary batches. Very large actual
    target dimensions require new, increasingly expensive kernel compilations
    and accumulate more float32 scan error.

    ``max_joint_rows`` applies to the flattened B * T * (U + 1) rows in each
    disposable joint workspace. Source time is divided into balanced tiles so
    that the final tile is not substantially smaller than the others. A tile
    contains at least one source step, so B * (U + 1) can exceed the requested
    budget. Gradient clamping requires the global final blank transition and
    therefore disables time tiling.
    """

    def __init__(
        self,
        blank: int,
        fastemit_lambda: float = 0.0,
        clamp: float = -1.0,
        max_target_tokens: int = MAX_TARGET_TOKENS,
        max_joint_rows: int = _DEFAULT_MAX_JOINT_ROWS,
    ):
        super().__init__()
        if fastemit_lambda < 0.0:
            raise ValueError("fastemit_lambda must be nonnegative")
        if not 0 <= max_target_tokens <= MAX_TARGET_TOKENS:
            raise ValueError(f"max_target_tokens must be in [0, {MAX_TARGET_TOKENS}]")
        if max_joint_rows < 1:
            raise ValueError("max_joint_rows must be positive")
        self.blank = blank
        self.fastemit_lambda = float(fastemit_lambda)
        self.clamp = float(clamp) if clamp > 0.0 else 0.0
        self.max_target_tokens = max_target_tokens
        self.max_joint_rows = max_joint_rows

    def forward(
        self,
        joint,
        encoder: torch.Tensor,
        predictor: torch.Tensor,
        targets: torch.Tensor,
        source_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        max_samples_per_chunk: int,
    ) -> torch.Tensor:
        """Return per-sample losses from encoder [B, T, D] and prediction network [B, U + 1, D] states."""
        return _compute_flash_rnnt(
            joint=joint,
            encoder=encoder,
            predictor=predictor,
            targets=targets,
            source_lengths=source_lengths,
            target_lengths=target_lengths,
            blank=self.blank,
            fastemit_lambda=self.fastemit_lambda,
            clamp=self.clamp,
            max_samples_per_chunk=max_samples_per_chunk,
            max_target_tokens=self.max_target_tokens,
            max_joint_rows=self.max_joint_rows,
        )


def _chunk_scores(
    projected_encoder: torch.Tensor,
    projected_predictor: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    activation: str,
    dropout_p: float,
    blank: int,
    clamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate transition scores for one disposable joint chunk."""
    from nemo.collections.asr.parts.triton.rnnt_logprobs import rnnt_logprobs_triton

    hidden = join_activate(projected_encoder, projected_predictor, activation)
    hidden = F.dropout(hidden, p=dropout_p, training=True)
    logits = F.linear(hidden, output_weight, output_bias)
    return rnnt_logprobs_triton(
        logits,
        targets,
        blank,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        clamp=clamp,
        reuse_logits_for_grad=True,
    )


def _balanced_time_tile_size(
    source_steps: int,
    chunk_batch: int,
    target_states: int,
    max_joint_rows: int,
) -> int:
    """Choose balanced source tiles subject to a one-source-step minimum."""
    max_time_tile = max(1, max_joint_rows // (chunk_batch * target_states))
    num_tiles = (source_steps + max_time_tile - 1) // max_time_tile
    return (source_steps + num_tiles - 1) // num_tiles


def _time_tiled_chunk_scores(
    projected_encoder: torch.Tensor,
    projected_predictor: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor,
    activation: str,
    dropout_p: float,
    blank: int,
    clamp: float,
    max_joint_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Checkpoint one balanced source tile at a time."""
    source_steps = projected_encoder.shape[1]
    if clamp > 0.0:
        # Clamping rescales by a per-sample upstream factor recovered from the global final
        # blank transition, which is only present in a tile spanning all of source time.
        time_tile = source_steps
    else:
        time_tile = _balanced_time_tile_size(
            source_steps,
            projected_encoder.shape[0],
            projected_predictor.shape[1],
            max_joint_rows,
        )

    target_score_tiles = []
    blank_score_tiles = []
    for source_begin in range(0, source_steps, time_tile):
        source_end = min(source_begin + time_tile, source_steps)
        tile_source_lengths = (source_lengths - source_begin).clamp(min=0, max=source_end - source_begin)
        target_score_tile, blank_score_tile = checkpoint(
            _chunk_scores,
            projected_encoder[:, source_begin:source_end],
            projected_predictor,
            targets,
            tile_source_lengths,
            target_lengths,
            output_weight,
            output_bias,
            activation,
            dropout_p,
            blank,
            clamp,
            use_reentrant=False,
            # Backward recomputation must use the same mask as the forward tile.
            preserve_rng_state=dropout_p > 0.0,
        )
        target_score_tiles.append(target_score_tile)
        blank_score_tiles.append(blank_score_tile)

    if clamp > 0.0 and len(target_score_tiles) != 1:
        # Splitting source time under clamping corrupts gradients without changing the loss,
        # so refuse rather than train on silently wrong gradients.
        raise ValueError(
            "Gradient clamping recovers each sample's upstream scale from the global final blank "
            f"transition, so a chunk must be scored in one source tile, got {len(target_score_tiles)}"
        )

    return torch.cat(target_score_tiles, dim=1), torch.cat(blank_score_tiles, dim=1)


def _compute_flash_rnnt(
    joint,
    encoder: torch.Tensor,
    predictor: torch.Tensor,
    targets: torch.Tensor,
    source_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    fastemit_lambda: float,
    clamp: float,
    max_samples_per_chunk: int,
    max_target_tokens: int,
    max_joint_rows: int,
) -> torch.Tensor:
    """Run exact RNN-T with a bounded vocabulary workspace recomputed in backward."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for flash RNN-T training")
    if not encoder.is_cuda:
        raise RuntimeError("Flash RNN-T training requires CUDA tensors")
    if max_samples_per_chunk < 1:
        raise ValueError("max_samples_per_chunk must be positive")
    dropout = _validate_joint(joint, blank)
    batch = encoder.shape[0]
    if predictor.shape[0] != batch or targets.shape[0] != batch:
        raise ValueError("encoder, predictor, and targets must have the same batch size")
    if source_lengths.shape != (batch,) or target_lengths.shape != (batch,):
        raise ValueError("source_lengths and target_lengths must contain one value per batch item")

    order = torch.argsort(target_lengths, stable=True)
    source_lengths = source_lengths.index_select(0, order)
    target_lengths = target_lengths.index_select(0, order)

    num_chunks = (batch + max_samples_per_chunk - 1) // max_samples_per_chunk
    padded_batch = num_chunks * max_samples_per_chunk
    length_pairs = torch.stack((source_lengths, target_lengths), dim=1)
    if padded_batch != batch:
        length_pairs = F.pad(length_pairs, (0, 0, 0, padded_batch - batch))
    # Python slicing needs concrete shapes; synchronize once for the reduced
    # per-chunk maxima rather than materializing every sample length.
    chunk_maxima = length_pairs.view(num_chunks, max_samples_per_chunk, 2).amax(dim=1).tolist()
    if chunk_maxima[-1][1] > max_target_tokens:
        raise ValueError(
            f"Batch target length {chunk_maxima[-1][1]} exceeds configured max_target_tokens={max_target_tokens}"
        )

    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    encoder = encoder.index_select(0, order)
    predictor = predictor.index_select(0, order)
    targets = targets.index_select(0, order)

    projected_encoder = joint.project_encoder(encoder)
    projected_predictor = joint.project_prednet(predictor)

    from nemo.collections.asr.parts.triton.rnnt_loss import rnnt_loss_triton

    target_score_chunks = []
    blank_score_chunks = []
    output = joint.joint_net[-1]
    dropout_p = dropout.p if dropout is not None and dropout.training else 0.0
    for chunk_index, (max_source, max_target) in enumerate(chunk_maxima):
        begin = chunk_index * max_samples_per_chunk
        end = min(begin + max_samples_per_chunk, batch)
        chunk_target_scores, chunk_blank_scores = _time_tiled_chunk_scores(
            projected_encoder[begin:end, :max_source],
            projected_predictor[begin:end, : max_target + 1],
            targets[begin:end, :max_target],
            source_lengths[begin:end],
            target_lengths[begin:end],
            output.weight,
            output.bias,
            joint.activation,
            dropout_p,
            blank,
            clamp,
            max_joint_rows,
        )
        padding = (
            0,
            predictor.shape[1] - max_target - 1,
            0,
            encoder.shape[1] - max_source,
        )
        if any(padding):
            chunk_target_scores = F.pad(chunk_target_scores, padding)
            chunk_blank_scores = F.pad(chunk_blank_scores, padding)
        target_score_chunks.append(chunk_target_scores)
        blank_score_chunks.append(chunk_blank_scores)

    target_scores = torch.cat(target_score_chunks)
    blank_scores = torch.cat(blank_score_chunks)

    losses = rnnt_loss_triton(
        target_scores[..., :-1],
        blank_scores,
        source_lengths,
        target_lengths,
        fastemit_lambda,
    )
    return losses.index_select(0, inverse_order)
