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


def test_generic_dpo_config_is_path_free_and_requires_experiment_values():
    root = Path(__file__).parents[3]
    cfg = OmegaConf.load(root / "examples/speechlm2/conf/salm_dpo.yaml")
    required = (
        "trainer.devices",
        "trainer.num_nodes",
        "trainer.precision",
        "trainer.max_steps",
        "model.base_experiment_config",
        "dpo.source_checkpoint",
        "dpo.output_root",
        "dpo.beta",
        "dpo.learning_rate",
        "dpo.pairs_per_update",
        "dpo.checkpoint_steps",
        "data.cuts_path",
        "data.expected_rows",
    )
    assert all(
        OmegaConf.is_missing(OmegaConf.select(cfg, key.rsplit(".", 1)[0]), key.rsplit(".", 1)[1]) for key in required
    )
    assert cfg.trainer.gradient_clip_val is None
    assert cfg.dpo.explicit_passes == 2
    assert cfg.data.shuffle is False and cfg.data.cycle is False
    assert cfg.dpo.lora is False and cfg.dpo.peft is False and cfg.dpo.adapters is False
    rendered = OmegaConf.to_yaml(cfg)
    assert "/lustre" not in rendered
    assert "AMI" not in rendered
    assert "Hero" not in rendered


def test_nondivisible_schedule_has_fixed_multirank_ownership_and_padding():
    active = rank_active_slots(pairs_per_update=11, world_size=3)
    assert active == (4, 4, 3)
    assert sum(active) == 11
    assert 3 * max(active) - sum(active) == 1
    # Ownership restarts per source shard, so every shard has the same local
    # accumulation shape instead of a rotating global modulo partition.
    assert [active for _ in range(4)] == [(4, 4, 3)] * 4


def test_preference_batch_accepts_stock_lightning_bf16_input_conversion():
    """The normal Lightning precision plugin may reconstruct the batch."""

    pair = PreferencePair(
        pair_id="p",
        source_id="s",
        prompt="<audio>",
        chosen="yes",
        rejected="no",
        audio=torch.ones(4, dtype=torch.float32),
        active=True,
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


def test_stock_lightning_accepts_manual_clip_when_trainer_clip_is_null():
    trainer = Trainer(accelerator="cpu", devices=1, gradient_clip_val=None, logger=False, enable_checkpointing=False)
    module = _ManualClipModule()
    module._trainer = trainer
    optimizer = torch.optim.AdamW(module.parameters())
    module.weight.grad = torch.tensor([2.0])
    module.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
    assert module.weight.grad.item() == pytest.approx(1.0, abs=1e-5)


def test_dpo_uses_inherited_mesh_aware_clip_once_with_configured_norm():
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


def test_dpo_policy_pair_owns_precision_forward_context_and_requires_fp32_audio():
    events = []

    class Context:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")

    class Plugin:
        def forward_context(self):
            return Context()

    class Harness:
        _trainer = SimpleNamespace(precision_plugin=Plugin())

        def _encoded_pair(self, pair):
            return "chosen", "rejected"

        def _completion_logprob(self, encoded, audio):
            assert events == ["enter"] or events == ["enter", "completion"]
            events.append("completion")
            return encoded, audio

    audio = torch.tensor([0.123456789, -0.987654321], dtype=torch.float32)
    pair = PreferencePair("p", "s", "<audio>", "yes", "no", audio, True)

    chosen, rejected = DPOSALMAutomodel._policy_pair(Harness(), pair)

    assert events == ["enter", "completion", "completion", "exit"]
    assert chosen == ("chosen", audio) and rejected == ("rejected", audio)
    assert torch.equal(pair.audio, audio)

    pair.audio = pair.audio.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="FP32 audio"):
        DPOSALMAutomodel._policy_pair(Harness(), pair)


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
