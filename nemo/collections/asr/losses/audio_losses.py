# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


from typing import List, Optional

import numpy as np
import torch

from nemo.core.classes import Loss, Typing, typecheck
from nemo.core.neural_types import AudioSignal, LengthsType, LossType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['SDRLoss']


def temporal_mean(
    input: torch.Tensor,
    input_length: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    keepdim: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Calculate mean along temporal dimension with optionally
    averaging only over valid samples (based on the input length).

    Args:
        input: Batch of signals, shape (B, T, C)
        input_length: Optional, length of each example in the batch, shape (B,)
        mask: Optional, temporal mask for each example in the batch, shape (B, T)
        keepdim: Whether to keep the temporal dimension
        eps: Regularization to avoid division by zero

    Returns:
        (B, C, 1) if keepdim=True, otherwise (B, C)
    """
    if (input_length is not None) and (mask is not None):
        raise RuntimeError(
            'Argument `input_length` is mutually exclusive with `mask`. Both cannot be used at the same time.'
        )

    if input_length is None and mask is None:
        # No length information, assume all samples are valid
        mean = torch.mean(input, dim=-1, keepdim=keepdim)
    elif input_length is not None:
        assert (input_length <= input.shape[-1]).all(), f'Check input length {input_length}, input shape {input.shape}'
        # Average only over valid elements
        mean = []
        for b, b_len in enumerate(input_length):
            mean_b = torch.sum(input[b, :, :b_len], axis=-1, keepdim=keepdim) / b_len
            mean.append(mean_b)
        mean = torch.stack(mean, axis=0)
    elif mask is not None:
        # Average using temporal mask
        mean = mask.unsqueeze(1) * input
        mean = torch.sum(mean, axis=-1, keepdim=keepdim)
        normalization = torch.sum(mask, axis=-1, keepdim=keepdim)
        mean = mean / (normalization.unsqueeze(1) + eps)
    else:
        raise RuntimeError(f'Unexpected input with both input_length and mask provided.')

    return mean


def calculate_sdr_batch(
    estimate: torch.Tensor,
    target: torch.Tensor,
    input_length: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    scale_invariant: bool = False,
    remove_mean: bool = True,
    sdr_max: Optional[float] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Calculate signal-to-distortion ratio per channel.

        SDR = 10 * log10( ||t||_2^2 / (||e-t||_2^2 + alpha * ||t||^2)

    where
        alpha = 10^(-sdr_max/10)

    Optionally, apply scale-invariant scaling on the target signal.

    Args:
        estimate: estimated signal, shape (B, T, C)
        target: target signal, shape (B, T, C)
        input_length: Optional, length of valid samples, shape (B,)
        mask: Optional, temporal mask, shape (B, T)
        scale_invariant: Use scale invariant SDR
        remove_mean: If True, mean will be removed before calculating SDR
        eps: Small regularization constant

    Returns:
        SDR in dB for each channel, shape (B, C)
    """
    assert (
        estimate.shape == target.shape
    ), f'Estimate shape ({estimate.shape}) not matching target shape ({target.shape})'

    if remove_mean:
        estimate = estimate - temporal_mean(estimate, input_length=input_length, mask=mask, keepdim=True, eps=eps)
        target = target - temporal_mean(target, input_length=input_length, mask=mask, keepdim=True, eps=eps)

    if scale_invariant:
        estimate_dot_target = temporal_mean(
            estimate * target, input_length=input_length, mask=mask, keepdim=True, eps=eps
        )
        target_pow = temporal_mean(torch.abs(target) ** 2, input_length=input_length, mask=mask, keepdim=True, eps=eps)
        target_scale = estimate_dot_target / (target_pow + eps)
        target = target_scale * target

    distortion = estimate - target

    target_pow = temporal_mean(torch.abs(target) ** 2, input_length=input_length, mask=mask, eps=eps)
    distortion_pow = temporal_mean(torch.abs(distortion) ** 2, input_length=input_length, mask=mask, eps=eps)

    if sdr_max is not None:
        distortion_pow = distortion_pow + 10 ** (-sdr_max / 10) * target_pow

    sdr = target_pow / (distortion_pow + eps)
    sdr = 10 * torch.log10(sdr + eps)

    return sdr


class SDRLoss(Loss, Typing):
    """
    Computes signal-to-distortion ratio (SDR) loss with weighted average across channels.

    Args:
        weight: weight for SDR of each output channel, used for averaging the loss across channels. Defaults to `None` (averaging).
        reduction: batch reduction. Defaults to `mean` over the batch.
        scale_invariant: If `True`, use scale-invariant SDR. Defaults to `False`.
        remove_mean: Remove mean before calculating the loss. Defaults to `True`.
        sdr_max: Soft thresholding of the loss to SDR_max.
        eps: Small value for regularization.
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        reduction: str = 'mean',
        scale_invariant: bool = False,
        remove_mean: bool = True,
        sdr_max: Optional[float] = None,
        eps: float = 1e-10,
    ):
        super().__init__()

        # SDR weight buffer
        if weight is not None:
            if any([w <= 0 for w in weight]):
                raise ValueError(f'Weight must be positive! Current value: {weight}')
            elif not np.isclose(sum(weight), 1, atol=1e-6):
                raise ValueError(f'Weight should add to one, current weight: {weight}')
            weight = torch.tensor(weight).reshape(1, -1)
            logging.info(f'Channel weight set to %s', weight)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]

        # Batch reduction
        self.reduction = reduction
        if reduction == 'mean':
            self.reduce = torch.mean
        else:
            raise ValueError(f'Unexpected reduction mode {reduction}.')

        # SDR calculation setup
        self.scale_invariant = scale_invariant
        self.remove_mean = remove_mean
        self.sdr_max = sdr_max
        self.eps = eps

    @property
    def input_types(self):
        """Input types definitions for SDRLoss.
        """
        signal_shape = ('B', 'C', 'T')
        return {
            "estimate": NeuralType(signal_shape, AudioSignal()),
            "target": NeuralType(signal_shape, AudioSignal()),
            "input_length": NeuralType(tuple('B'), LengthsType(), optional=True),
            "mask": NeuralType(('B', 'T'), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        """Output types definitions for SDRLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(
        self,
        estimate: torch.Tensor,
        target: torch.Tensor,
        input_length: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """For input batch of multi-channel signals, calculate SDR between estimate and target for each channel,
        perform averaging across channels (weighting optional), and apply reduction across the batch.

        Args:
            estimate: Batch of signals, shape (B, T, C)
            target: Batch of signals, shape (B, T, C)
            input_length: Batch of lengths, shape (B,)
            mask: Batch of temporal masks, shape (B, T)

        Returns:
            Scalar loss.
        """

        sdr = calculate_sdr_batch(
            estimate=estimate,
            target=target,
            input_length=input_length,
            mask=mask,
            scale_invariant=self.scale_invariant,
            remove_mean=self.remove_mean,
            sdr_max=self.sdr_max,
            eps=self.eps,
        )

        # channel averaging
        if self.weight is None:
            sdr = torch.mean(sdr, dim=1)
        else:
            # weighting across channels
            sdr = sdr * self.weight
            sdr = torch.sum(sdr, dim=1)

        # reduction
        sdr = self.reduce(sdr)

        return -sdr
