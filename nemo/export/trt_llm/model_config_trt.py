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

import os
import shutil
from pathlib import Path
from typing import List, Union

from nemo.export.trt_llm.model_config import ModelConfig
from nemo.export.trt_llm.tensorrt_llm_model import LMHeadModelBuilder


def model_config_to_tensorrt_llm(
    model_configs: List[ModelConfig],
    engine_dir: Union[str, Path],
    world_size: int = 1,
    max_input_len: int = 200,
    max_output_len: int = 200,
    max_batch_size: int = 1,
    max_beam_width: int = 1,
    max_prompt_embedding_table_size: int = 0,
    use_inflight_batching: bool = False,
    paged_kv_cache: bool = False,
    enable_context_fmha: bool = True,
    enable_multi_block_mode: bool = False,
    use_refit: bool = False,
    use_lora_plugin: str = None,
    lora_target_modules: List[str] = None,
    max_lora_rank: int = 64,
):
    """The API to convert a torch or huggingface model represented as ModelConfig to tensorrt_llm.

    Args:
        model_configs: The list of ModelConfig converted, 1 for each GPU.
        engine_dir: The target output directory to save the built tensorrt_llm engines.
        gpus: the number of inference gpus for multi gpu inferencing.
        max_input_len: The max input sequence length.
        max_output_len: The max output sequence length.
        max_batch_size: The max batch size.
        max_beam_width: The max beam search width.
        max_prompt_embedding_table_size: max size of the prompt embedding table.
        use_inflight_batching (bool): if True, enables inflight batching for TensorRT-LLM Triton backend.
        paged_kv_cache (bool): if True, uses kv cache feature of the TensorRT-LLM.
        enable_context_fmha (bool): if True, use fused Context MultiHeadedAttention.
        enable_multi_block_mode (bool): enable faster decoding in multihead attention. Required for long context.
    """
    engine_dir = Path(engine_dir)
    if os.path.exists(engine_dir):
        shutil.rmtree(engine_dir)

    for rank in range(world_size):
        model_configs[rank].use_prompt_tuning = max_prompt_embedding_table_size > 0
        model_configs[rank].max_lora_rank = max_lora_rank
        builder = LMHeadModelBuilder(model_configs[rank])
        builder.build(
            output_dir=engine_dir,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            parallel_build=False,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
            use_inflight_batching=use_inflight_batching,
            paged_kv_cache=paged_kv_cache,
            enable_context_fmha=enable_context_fmha,
            enable_multi_block_mode=enable_multi_block_mode,
            use_refit=use_refit,
            use_lora_plugin=use_lora_plugin,
            lora_target_modules=lora_target_modules,
            max_lora_rank=max_lora_rank,
        )
