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
from packaging.version import Version as PkgVersion
from utils import get_torch_version_str

import nemo.lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data.hf_dataset import SquadHFDataModule

DATA_PATH = '/lustre/fsw/coreai_dlalgo_llm/boxiangw/squad'


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


if __name__ == '__main__':
    if PkgVersion(get_torch_version_str()) >= PkgVersion("2.4"):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B-Instruct')
        parser.add_argument('--devices', default=2)
        parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
        parser.add_argument('--max-steps', type=int, default=100)
        args = parser.parse_args()

        recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
            model_name=args.model,
            name="sft",
            num_nodes=1,
            num_gpus_per_node=args.devices,
            peft_scheme='none',
            max_steps=args.max_steps,
        )
        recipe.trainer.val_check_interval = 50

        tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)
        recipe.data = run.Config(
            SquadHFDataModule,
            path_or_dataset=DATA_PATH,
            split="train[:100]",
            pad_token_id=tokenizer.tokenizer.eos_token_id,
            tokenizer=run.Config(AutoTokenizer, pretrained_model_name=args.model),
        )

        recipe.trainer.strategy = run.Config(nl.FSDP2Strategy, data_parallel_size=2, tensor_parallel_size=1)
        recipe.trainer.plugins = None
        executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
        run.run(recipe, executor=executor)
