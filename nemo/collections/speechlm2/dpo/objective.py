# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Small, explicit, reference-subtracted pairwise DPO objective."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DPOPairObjective:
    """The un-reduced standard DPO values for one ordered pair."""

    margin: torch.Tensor
    loss: torch.Tensor


def dpo_pair_objective(
    *,
    chosen_policy_logp: torch.Tensor,
    rejected_policy_logp: torch.Tensor,
    chosen_reference_logp: torch.Tensor,
    rejected_reference_logp: torch.Tensor,
    beta: float,
) -> DPOPairObjective:
    """Return ``-logsigmoid(beta * ((pi_c-ref_c) - (pi_r-ref_r)))``.

    The caller controls reduction. Keeping this function pair-local makes the
    distributed ownership scaling visible in the training module.
    """

    if not 0.0 < beta <= 10.0:
        raise ValueError(f"beta must be in (0, 10], got {beta}")
    values = (chosen_policy_logp, rejected_policy_logp, chosen_reference_logp, rejected_reference_logp)
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("DPO log-probability tensors must have equal shapes")
    margin = (chosen_policy_logp - chosen_reference_logp) - (rejected_policy_logp - rejected_reference_logp)
    return DPOPairObjective(margin=margin, loss=-F.logsigmoid(beta * margin))
