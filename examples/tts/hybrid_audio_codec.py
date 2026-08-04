# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Train an additive semantic-code plus variational-residual codec from a reference .nemo config."""

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.tts.models import AudioCodecModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def reference_model_config(cfg: DictConfig) -> DictConfig:
    """Load the exact reference topology and apply only hybrid/training overrides."""
    model_cfg = AudioCodecModel.restore_from(cfg.reference_model, return_config=True)

    with open_dict(model_cfg):
        model_cfg.hybrid_codec = OmegaConf.create(OmegaConf.to_container(cfg.hybrid_codec, resolve=True))
        model_cfg.train_ds = OmegaConf.create(OmegaConf.to_container(cfg.train_ds, resolve=True))
        model_cfg.validation_ds = OmegaConf.create(OmegaConf.to_container(cfg.validation_ds, resolve=True))
        model_cfg.log_config = None
        model_cfg.max_steps = cfg.max_steps
        model_cfg.steps_per_epoch = cfg.max_steps
        model_cfg.max_epochs = cfg.trainer.max_epochs
        model_cfg.disc_start_epoch = cfg.disc_start_epoch
        model_cfg.mmd_loss_start_epoch = cfg.mmd_loss_start_epoch

        # The embedded semantic model is a submodule here; it must not construct
        # the original standalone model's historical train/validation loaders.
        with open_dict(model_cfg.semantic_codec):
            model_cfg.semantic_codec.train_ds = None
            model_cfg.semantic_codec.validation_ds = None
            model_cfg.semantic_codec.log_config = None

        if cfg.reconstruction_only:
            # A fast local smoke test does not need adversarial, SLM, or legacy MMD losses.
            model_cfg.discriminator = None
            model_cfg.use_slm_loss = False
            model_cfg.feature_loss_scale = 0.0
            model_cfg.mmd_loss_scale = 0.0
            model_cfg.mmd_time_loss_scale = 0.0
            model_cfg.disc_start_epoch = 2

    return model_cfg


@hydra_runner(config_path="conf/audio_codec", config_name="hybrid_audio_codec_32000")
def main(cfg: DictConfig) -> None:
    logging.info('\nConfig Params:\n%s', OmegaConf.to_yaml(cfg, resolve=True))
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager"))

    model_cfg = reference_model_config(cfg)
    model = AudioCodecModel(cfg=model_cfg, trainer=trainer)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
