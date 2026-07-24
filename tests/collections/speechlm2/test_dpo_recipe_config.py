# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path
from types import SimpleNamespace

import pytest

import torch
from lightning import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.plugins.precision.half import HalfPrecision
from omegaconf import OmegaConf

from nemo.collections.speechlm2.dpo.data import PreferenceBatch, PreferencePair, rank_active_slots
from nemo.collections.speechlm2.dpo.model import DPOSALMAutomodel, _gradient_layout
from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel


class _ManualClipModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))


def test_hero2_r5_dpo_config_has_explicit_finite_two_pass_accounting():
    root = Path(__file__).parents[3]
    cfg = OmegaConf.load(root / "examples/speechlm2/conf/salm_dpo_hero2_ami_historical_r5.yaml")
    assert cfg.trainer.devices == 8
    assert cfg.trainer.num_nodes == 1
    assert cfg.trainer.max_steps == 20
    assert cfg.trainer.gradient_clip_val is None
    assert cfg.dpo.explicit_passes == 2
    assert cfg.dpo.expected_updates == 20
    assert cfg.dpo.pairs_per_update == 435
    assert cfg.dpo.source_shards == 10
    assert cfg.data.expected_rows == cfg.dpo.pairs_per_update * cfg.dpo.source_shards
    assert cfg.data.shuffle is False and cfg.data.cycle is False
    assert cfg.dpo.beta == 0.2
    assert cfg.dpo.learning_rate == 2.5e-6
    assert cfg.dpo.optimizer.betas == [0.9, 0.95]
    assert cfg.dpo.lora is False and cfg.dpo.peft is False and cfg.dpo.adapters is False


def test_435_pair_schedule_has_fixed_multirank_ownership_and_five_padding_slots():
    active = rank_active_slots(pairs_per_update=435, world_size=8)
    assert active == (55, 55, 55, 54, 54, 54, 54, 54)
    assert sum(active) == 435
    assert 8 * max(active) - sum(active) == 5
    # Ownership restarts per source shard.  Therefore all ten shards have the
    # same 55-slot local accumulation shape, instead of a rotating global
    # modulo partition caused by the 435 % 8 offset.
    assert [active for _ in range(10)] == [(55, 55, 55, 54, 54, 54, 54, 54)] * 10


def test_preference_batch_accepts_stock_lightning_bf16_input_conversion():
    """The normal Lightning precision plugin may reconstruct the batch."""

    pair = PreferencePair(
        pair_id="p", source_id="s", prompt="<audio>", chosen="yes", rejected="no",
        audio=torch.ones(4, dtype=torch.float32), active=True,
    )
    batch = PreferenceBatch(global_step=1, dpo_pass=1, source_shard=1, pairs=(pair,))
    converted = HalfPrecision("bf16-true").convert_input(batch)
    assert converted.global_step == 1
    assert converted.dpo_pass == 1 and converted.source_shard == 1
    assert converted.pairs[0].pair_id == "p"
    assert converted.pairs[0].prompt == "<audio>"
    assert converted.pairs[0].audio.dtype is torch.bfloat16
    # Conversion returns a framework-owned transport copy; loader-provided
    # samples retain their original data for the model's reference cache.
    assert batch.pairs[0].audio.dtype is torch.float32


def test_stock_lightning_accepts_historical_manual_clip_when_trainer_clip_is_null():
    trainer = Trainer(accelerator="cpu", devices=1, gradient_clip_val=None, logger=False, enable_checkpointing=False)
    module = _ManualClipModule()
    module._trainer = trainer
    optimizer = torch.optim.AdamW(module.parameters())
    module.weight.grad = torch.tensor([2.0])
    module.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
    assert module.weight.grad.item() == pytest.approx(1.0, abs=1e-5)


def test_dpo_uses_inherited_mesh_aware_clip_once_with_historical_norm():
    """DPO must not bypass SALMAutomodel's existing mixed-DTensor handler."""

    assert DPOSALMAutomodel.configure_gradient_clipping is SALMAutomodel.configure_gradient_clipping
    calls = []

    class Harness:
        cfg = SimpleNamespace(dpo=SimpleNamespace(gradient_clip_norm=1.0))

        def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
            calls.append((optimizer, gradient_clip_val, gradient_clip_algorithm))

    optimizer = object()
    DPOSALMAutomodel._clip_selected_gradients(Harness(), optimizer)
    assert calls == [(optimizer, 1.0, "norm")]


def test_selected_gradient_layout_receipt_is_data_free_for_local_tensors():
    entry = _gradient_layout("perception.proj.weight", torch.ones((2, 3), dtype=torch.float32))
    assert entry == {
        "name": "perception.proj.weight",
        "tensor_type": "torch.Tensor",
        "global_shape": [2, 3],
        "local_shape": [2, 3],
        "dtype": "torch.float32",
        "layout": {"kind": "local"},
    }
