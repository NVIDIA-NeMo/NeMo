# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Run the native finite-DPO cached-reference identity audit with zero updates.

This entrypoint uses the same model construction, strict source-DCP load,
finite Lhotse data module, reference capture, prompt formatter, policy logprob,
and DPO objective as ``salm_dpo_train.py``.  A normal Lightning callback runs
after reference capture and stops the fit loop before batch 1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf, open_dict

from nemo.collections.speechlm2.dpo import FiniteLhotsePreferenceDataModule
from nemo.collections.speechlm2.dpo.zero_margin import (
    DPOZeroMarginPreflightModel,
    ZeroMarginPreflightCallback,
    validate_preflight_artifacts,
)
from nemo.core.config import hydra_runner
from nemo.utils.trainer_utils import resolve_trainer_cfg


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prepare_model_config(cfg):
    base = OmegaConf.load(cfg.model.base_experiment_config)
    OmegaConf.resolve(base)
    model = OmegaConf.create(OmegaConf.to_container(base.model, resolve=True))
    with open_dict(model):
        model.init_from_checkpoint = None
        model.pretrained_llm_weights = True
        model.pretrained_asr_weights = True
        model.init_configure_model = False
        model.torch_dtype = "bfloat16"
        model.dpo = OmegaConf.to_container(cfg.dpo, resolve=True)
    trainer = OmegaConf.create(OmegaConf.to_container(base.trainer, resolve=True))
    with open_dict(trainer):
        trainer.devices = cfg.trainer.devices
        trainer.num_nodes = cfg.trainer.num_nodes
        trainer.precision = cfg.trainer.precision
        trainer.max_steps = cfg.trainer.max_steps
        # Lightning only honors ``trainer.should_stop`` before batch 1 when no
        # minimum epoch/step floor is active.  Keep these explicit and persist
        # the resolved trainer config below as part of the preflight receipt.
        trainer.min_steps = 0
        trainer.min_epochs = 0
        trainer.enable_checkpointing = False
        trainer.logger = False
        trainer.log_every_n_steps = 1
        trainer.use_distributed_sampler = False
        trainer.gradient_clip_val = cfg.trainer.gradient_clip_val
    return model, trainer


def _validate_preflight_config(cfg) -> None:
    checks = {
        "trainer.max_steps": int(cfg.trainer.max_steps) == 1,
        "dpo.expected_updates": int(cfg.dpo.expected_updates) == 1,
        "dpo.explicit_passes": int(cfg.dpo.explicit_passes) == 2,
        "data.expected_rows_product": int(cfg.data.expected_rows)
        == int(cfg.data.pairs_per_update) * int(cfg.data.source_shards),
        "dpo_data_pairs_per_update": int(cfg.dpo.pairs_per_update) == int(cfg.data.pairs_per_update),
        "dpo_data_source_shards": int(cfg.dpo.source_shards) == int(cfg.data.source_shards),
        "dpo_data_world_size": int(cfg.dpo.world_size) == int(cfg.data.world_size),
        "trainer_world_size": int(cfg.trainer.devices) * int(cfg.trainer.num_nodes) == int(cfg.dpo.world_size),
        "finite_ordered_data": cfg.data.shuffle is False and cfg.data.cycle is False,
        "no_checkpoint_steps": list(cfg.dpo.checkpoint_steps) == [],
    }
    if not all(checks.values()):
        raise ValueError(f"invalid no-update DPO preflight config: {checks}")


@hydra_runner(config_path="conf", config_name="salm_dpo")
def main(cfg) -> None:
    OmegaConf.resolve(cfg)
    _validate_preflight_config(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("SpeechLM2 DPO zero-margin preflight requires CUDA")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    try:
        seed_everything(int(cfg.dpo.seed), workers=True)
        root = Path(str(cfg.dpo.output_root))
        rank = torch.distributed.get_rank()
        if rank == 0:
            if root.exists():
                raise FileExistsError(f"DPO zero-margin output root must be fresh: {root}")
            root.mkdir(parents=True)
            OmegaConf.save(cfg, root / "effective_config.yaml")
        torch.distributed.barrier()

        callback = ZeroMarginPreflightCallback(
            root,
            expected_rows=int(cfg.data.expected_rows),
            pairs_per_shard=int(cfg.data.pairs_per_update),
            source_shards=int(cfg.data.source_shards),
            world_size=int(cfg.data.world_size),
            beta=float(cfg.dpo.beta),
            learning_rate=float(cfg.dpo.learning_rate),
        )
        model_cfg, trainer_cfg = _prepare_model_config(cfg)
        if rank == 0:
            OmegaConf.save(model_cfg, root / "resolved_model_config.yaml")
            OmegaConf.save(trainer_cfg, root / "resolved_trainer_config.yaml")
        trainer_kwargs = resolve_trainer_cfg(trainer_cfg)
        trainer = Trainer(**trainer_kwargs, callbacks=[callback])
        with trainer.init_module():
            model = DPOZeroMarginPreflightModel(OmegaConf.to_container(model_cfg, resolve=True))
        datamodule = FiniteLhotsePreferenceDataModule(cfg.data)
        trainer.fit(model, datamodule=datamodule)
        if int(trainer.global_step) != 0:
            raise RuntimeError("zero-margin preflight returned after an optimizer update")
        torch.distributed.barrier()
        if rank == 0:
            receipt = validate_preflight_artifacts(
                root,
                expected_shards=int(cfg.data.source_shards),
                pairs_per_shard=int(cfg.data.pairs_per_update),
                expected_world_size=int(cfg.data.world_size),
                expected_beta=float(cfg.dpo.beta),
                expected_learning_rate=float(cfg.dpo.learning_rate),
            )
            _write_json(
                root / "ZERO_MARGIN_PREFLIGHT_DONE.json",
                {
                    "schema": "speechlm2.dpo.zero-margin-preflight-done.v1",
                    "status": "passed",
                    "optimizer_step_count": 0,
                    "checked_active_pairs": receipt["pointwise_summary"]["checked_active_pairs"],
                    "receipt": str(root / "ZERO_MARGIN_PREFLIGHT.json"),
                },
            )
        torch.distributed.barrier()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
