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

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_finetune_model import MegatronT5FinetuneModel
from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5GLUEModel
from nemo.collections.nlp.models.language_modeling.megatron_t0_model import MegatronT0Model
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPPlugin,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager


@hydra_runner(config_path="conf", config_name="megatron_t5_config_finetune_glue_mnli")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    plugins = [
        NLPDDPPlugin(
            no_ddp_communication_hook=True,
            gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
            find_unused_parameters=False,
        )
    ]
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path
    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # Get the T5 Base configuration.
    t5_cfg = MegatronT5FinetuneModel.restore_from(
        restore_path=cfg.model.restore_from_path, trainer=trainer, return_config=True
    )

    # Override the T5 configuration with the one from the config file.
    OmegaConf.set_struct(t5_cfg, True)
    with open_dict(t5_cfg):
        t5_cfg.masked_softmax_fusion = False
        t5_cfg.megatron_amp_O2 = cfg.model.get('megatron_amp_O2', False)
        t5_cfg.hidden_dropout = cfg.model.get('hidden_dropout', 0.1)
        t5_cfg.attention_dropout = cfg.model.get('attention_dropout', 0.1)
        t5_cfg.data = cfg.model.data
        t5_cfg.precision = cfg.trainer.precision
        t5_cfg.optim = cfg.model.optim
        t5_cfg.micro_batch_size = cfg.model.data.train_ds.micro_batch_size
        t5_cfg.global_batch_size = cfg.model.data.train_ds.global_batch_size
        # XNLI has eval languages in the yaml config.
        if hasattr(cfg.model, 'eval_languages'):
            t5_cfg.eval_languages = cfg.model.eval_languages

    if hasattr(cfg.model.data.train_ds, 'task_name'):
        model = MegatronT5GLUEModel.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
    elif hasattr(cfg.model.data.train_ds, 'file_names'):
        model = MegatronT0Model.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
    else:
        model = MegatronT5FinetuneModel.restore_from(
            restore_path=cfg.model.restore_from_path,
            trainer=trainer,
            override_config_path=t5_cfg,
            save_restore_connector=NLPSaveRestoreConnector(),
        )

    trainer.fit(model)
    trainer.validate(model)
    if hasattr(cfg.model.data, 'test_ds'):
        trainer.test(model)


if __name__ == '__main__':
    main()
