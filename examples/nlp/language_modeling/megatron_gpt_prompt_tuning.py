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

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPPlugin,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.exp_manager import StatelessTimer, exp_manager


"""
Can currently only prompt tune on one task at a time, but can
run inference with multiple soft-prompts/tasks within a batch.

Datasets should be formatted with in a json file like:
{"prompt_tag": <tag1>, "text": <text1>, "answer": <answer1>}
{"prompt_tag": <tag1>, "text": <text2>, "answer": <answer2>}
{"prompt_tag": <tag1>, "text": <text3>, "answer": <answer3>}

Example Usage for first prompt tuning task:

EXPR_NAME='winogrande_prompt_tuning'
RESTORE_PATH='megatron_gpt.nemo'
GPUS=1
MAX_STEPS=4800
PROMPT_LENGTH=20

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.devices=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        restore_from_path=$RESTORE_PATH \
        exp_manager.name=$EXPR_NAME \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=[] \
        +model.new_prompt_tags=['Winogrande'] \
        +model.new_prompt_init_text=['disambiguate pronoun noun names pick correct name fill blank'] \
        +model.new_prompt_init_methods=['text'] \
        model.data.data_prefix=None \
        +model.data.train_ds='winogrande_prompt_tuning_train.jsonl' \
        +model.data.valid_ds='winogrande_prompt_tuning_val.jsonl' \
        +model.data.batch_size=32 \
        model.optim.lr=2e-3 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=320 \
        model.optim.sched.constant_steps=2240 \
        model.encoder_seq_length=2048


Example Usage for second prompt tuning task:
Be sure to update model.exsiting_prompt_tags with tags from previous prompt tuning session
and to use the .nemo file saved at the end of the last prompt tuning session

EXPR_NAME='rte_prompt_tuning'
RESTORE_PATH='winograde_megatron_gpt.nemo'
GPUS=1
MAX_STEPS=780
PROMPT_LENGTH=20
VAL_CHECK_INTERVAL=50

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.devices=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        trainer.val_check_interval=$VAL_CHECK_INTERVAL \
        restore_from_path=$RESTORE_PATH \
        exp_manager.name=$EXPR_NAME \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=['Winogrande'] \
        +model.new_prompt_tags=['RTE'] \
        +model.new_prompt_init_text=['entailment cause relationship imply label text'] \
        +model.new_prompt_init_methods=['text'] \
        model.data.data_prefix=None \
        +model.data.train_ds='RTE_prompt_tuning_train.jsonl' \
        +model.data.valid_ds='RTE_prompt_tuning_val.jsonl' \
        +model.data.batch_size=32 \
        model.optim.lr=2e-4 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=78 \
        model.optim.sched.constant_steps=545 \
        model.encoder_seq_length=2048


Example Usage for third prompt tuning task:
Be sure to update model.exsiting_prompt_tags with tags from previous prompt tuning sessions
and to use the .nemo file saved at the end of the last prompt tuning sessions

EXPR_NAME='boolq_prompt_tune'
GPUS=1
MAX_STEPS=2950
PROMPT_LENGTH=20
RESTORE_PATH='rte_winogrande_megatron_gpt.nemo'

echo "Prompt tuning starting"
python megatron_gpt_prompt_tuning.py \
        --config-name=megatron_gpt_config \
        trainer.devices=$GPUS \
        trainer.max_steps=$MAX_STEPS \
        exp_manager.name=$EXPR_NAME \
        exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
        restore_from_path=$RESTORE_PATH \
        +model.use_soft_prompts=True \
        +model.num_prompt_tokens=$PROMPT_LENGTH \
        +model.existing_prompt_tags=['Winogrande, RTE'] \
        +model.new_prompt_tags=['BoolQ'] \
        +model.new_prompt_init_text=['true false question answer reading comprehension'] \
        +model.new_prompt_init_methods=['text'] \
        +model.calc_loss_on_answer_only=False \
        model.data.data_prefix=None \
        +model.data.train_ds='boolq_prompt_tuning_train.jsonl' \
        +model.data.valid_ds='boolq_prompt_tuning_val.jsonl' \
        +model.data.batch_size=32 \
        model.optim.lr=2e-4 \
        model.optim.sched.min_lr=2e-6 \
        model.optim.sched.warmup_steps=295 \
        model.optim.sched.constant_steps=2063 \
        model.encoder_seq_length=2048

"""


@hydra_runner(config_path="conf", config_name="megatron_prompt_tuning_gpt")
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

    app_state = AppState()
    if cfg.model.tensor_model_parallel_size > 1 or cfg.model.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            _,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=cfg.model.tensor_model_parallel_size,
            pipeline_model_parallel_size_=cfg.model.pipeline_model_parallel_size,
        )

    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    model = MegatronGPTModel.restore_from(cfg.restore_from_path, cfg.model, trainer=trainer)
    trainer.fit(model)


if __name__ == '__main__':
    main()
