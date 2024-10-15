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

# NOTE: This script is just an example of using NeMo checkpoints for generating outputs and is subject to change without notice.

import os

import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams

import nemo.lightning as nl
from nemo.collections.llm import api

if __name__ == "__main__":
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=2,
        num_nodes=1,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    )
    prompts = [
        "Hello, how are you?",
        "How many r's are in the word 'strawberry'?",
        "Which number is bigger? 10.119 or 10.19?",
    ]
    results = api.generate(
        path=os.path.join(os.environ["NEMO_HOME"], "models", "meta-llama/Meta-Llama-3-8B"),
        prompts=prompts,
        trainer=trainer,
        inference_params=CommonInferenceParams(temperature=0.1, top_k=10, num_tokens_to_generate=512),
        text_only=True,
    )
    if torch.distributed.get_rank() == 0:
        for i, r in enumerate(results):
            print(prompts[i])
            print("*" * 50)
            print(r)
            print("\n\n")
