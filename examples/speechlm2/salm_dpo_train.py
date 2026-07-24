# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Train a finite direct-Lhotse SpeechLM2 DPO trajectory.

Example:
    torchrun --standalone --nproc-per-node=8 examples/speechlm2/salm_dpo_train.py \
      dpo.output_root=/path/to/fresh/output

All algorithmic settings are in the YAML.  This entrypoint performs no runtime
source rewriting, package copying, import-hook installation, or compatibility
overlay.  The NeMo-Speech source used to execute it must be the tracked source
snapshot described by the run provenance.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from omegaconf import OmegaConf, open_dict

from nemo.collections.speechlm2.dpo import DPOSALMAutomodel, FiniteLhotsePreferenceDataModule
from nemo.core.config import hydra_runner
from nemo.utils.trainer_utils import resolve_trainer_cfg


def _prepare_model_config(cfg):
    base = OmegaConf.load(cfg.model.base_experiment_config)
    OmegaConf.resolve(base)
    model = OmegaConf.create(OmegaConf.to_container(base.model, resolve=True))
    with open_dict(model):
        model.init_from_checkpoint = None
        # The strict DCP below is the single final model-weight authority.
        # Avoid the stale 5600 training checkpoint, but retain the configured
        # ASR archive during construction: it supplies the perception
        # preprocessor/encoder schema absent from the experiment YAML.  The
        # strict Hero2 step-14400 DCP immediately overwrites those temporary
        # construction weights before references or updates are permitted.
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
        trainer.enable_checkpointing = False
        trainer.logger = False
        trainer.log_every_n_steps = 1
        trainer.use_distributed_sampler = False
        # Lightning disallows trainer-managed clipping for manual optimization.
        # DPOSALMAutomodel applies the historical global-norm clip explicitly
        # immediately before AdamW.step(), so preserve that mechanism while
        # disabling the incompatible Trainer-level duplicate.
        trainer.gradient_clip_val = 0.0
    return model, trainer


@hydra_runner(config_path="conf", config_name="salm_dpo_hero2_ami_historical_r5")
def main(cfg) -> None:
    OmegaConf.resolve(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("SpeechLM2 DPO requires CUDA")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    seed_everything(int(cfg.dpo.seed), workers=True)
    root = Path(str(cfg.dpo.output_root))
    if torch.distributed.get_rank() == 0:
        if root.exists():
            raise FileExistsError(f"DPO output root must be fresh: {root}")
        root.mkdir(parents=True)
        OmegaConf.save(cfg, root / "effective_config.yaml")
    torch.distributed.barrier()
    model_cfg, trainer_cfg = _prepare_model_config(cfg)
    trainer = Trainer(**resolve_trainer_cfg(trainer_cfg))
    with trainer.init_module():
        model = DPOSALMAutomodel(OmegaConf.to_container(model_cfg, resolve=True))
    datamodule = FiniteLhotsePreferenceDataModule(cfg.data)
    trainer.fit(model, datamodule=datamodule)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
