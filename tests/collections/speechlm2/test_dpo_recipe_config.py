# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from pathlib import Path

from omegaconf import OmegaConf


def test_hero2_r5_dpo_config_has_explicit_finite_two_pass_accounting():
    root = Path(__file__).parents[3]
    cfg = OmegaConf.load(root / "examples/speechlm2/conf/salm_dpo_hero2_ami_historical_r5.yaml")
    assert cfg.trainer.devices == 8
    assert cfg.trainer.num_nodes == 1
    assert cfg.trainer.max_steps == 20
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
