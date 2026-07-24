# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import math

import torch

from nemo.collections.speechlm2.dpo.objective import dpo_pair_objective
from nemo.collections.speechlm2.dpo.surface import (
    ATTENTION_LAYERS,
    HERO2_ACOUSTIC_LAYERS,
    MAMBA_LAYERS,
    SELECTED_SCALAR_COUNT,
    selected_parameter_names,
)


def test_dpo_initial_identity_has_log2_but_nonzero_preference_gradient():
    chosen = torch.tensor(3.0, requires_grad=True)
    rejected = torch.tensor(1.0, requires_grad=True)
    objective = dpo_pair_objective(
        chosen_policy_logp=chosen,
        rejected_policy_logp=rejected,
        chosen_reference_logp=torch.tensor(3.0),
        rejected_reference_logp=torch.tensor(1.0),
        beta=0.2,
    )
    assert objective.margin.item() == 0.0
    assert math.isclose(objective.loss.item(), math.log(2.0), rel_tol=0, abs_tol=1e-7)
    objective.loss.backward()
    assert chosen.grad.item() < 0.0
    assert rejected.grad.item() > 0.0


def test_dpo_positive_margin_reduces_loss():
    identity = dpo_pair_objective(
        chosen_policy_logp=torch.tensor(0.0), rejected_policy_logp=torch.tensor(0.0),
        chosen_reference_logp=torch.tensor(0.0), rejected_reference_logp=torch.tensor(0.0), beta=0.2,
    )
    improved = dpo_pair_objective(
        chosen_policy_logp=torch.tensor(2.0), rejected_policy_logp=torch.tensor(0.0),
        chosen_reference_logp=torch.tensor(0.0), rejected_reference_logp=torch.tensor(0.0), beta=0.2,
    )
    assert improved.margin.item() > 0.0
    assert improved.loss.item() < identity.loss.item()


def test_hero2_partial_acoustic_surface_contract_is_fixed():
    names = selected_parameter_names()
    assert len(names) == 269
    assert len(set(names)) == len(names)
    assert len(ATTENTION_LAYERS) == 6
    assert len(MAMBA_LAYERS) == 23
    assert HERO2_ACOUSTIC_LAYERS == (30, 31)
    assert SELECTED_SCALAR_COUNT == 1_074_327_616
    assert names[-2:] == ("perception.proj.weight", "perception.proj.bias")
