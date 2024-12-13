# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import nemo_run as run

from nemo.collections import vlm


def configure_recipe(nodes: int = 1, gpus_per_node: int = 8, pretrain=False, language_model_from_pretrained=None):
    """Configure the recipe"""
    if pretrain:
        recipe = vlm.llava_next_7b.pretrain_recipe(
            dir="./outputs/checkpoints/llava",  # Path to store checkpoints
            name="llava_pretrain",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
            language_model_from_pretrained=language_model_from_pretrained,
        )
    else:
        recipe = vlm.llava_next_7b.finetune_recipe(
            dir="./outputs/checkpoints/llava",  # Path to store checkpoints
            name="llava_finetune",
            num_nodes=nodes,
            num_gpus_per_node=gpus_per_node,
        )
    recipe.trainer.max_steps = 20
    recipe.trainer.val_check_interval = 20
    recipe.model.config.freeze_vision_model = True
    return recipe


def local_executor_torchrun(nodes: int = 1, devices: int = 8) -> run.LocalExecutor:
    # pylint: disable=C0115,C0116
    # Env vars for jobs are configured here
    env_vars = {}

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def run_pretraining(language_model_from_pretrained=None):
    # pylint: disable=C0115,C0116
    recipe = configure_recipe(pretrain=True, language_model_from_pretrained=language_model_from_pretrained)
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor)


def run_finetuning():
    # pylint: disable=C0115,C0116
    recipe = configure_recipe(pretrain=False)
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)

    run.run(recipe, executor=executor)


# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining(language_model_from_pretrained='/root/.cache/nemo/models/lmsys/vicuna-7b-v1.5/')
    # run_finetuning()
