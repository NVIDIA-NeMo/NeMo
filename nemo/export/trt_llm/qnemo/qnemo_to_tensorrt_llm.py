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


import glob
import os
import warnings
from typing import List, Optional

from modelopt.deploy.llm import build_tensorrt_llm

from nemo.export.trt_llm.qnemo.utils import CONFIG_NAME, WEIGHTS_NAME


def qnemo_to_tensorrt_llm(
    nemo_checkpoint_path: str,
    engine_dir: str,
    max_input_len: int,
    max_output_len: int,
    max_batch_size: int,
    max_prompt_embedding_table_size: int,
    tensor_parallel_size: int = None,
    pipeline_parallel_size: int = None,
    use_parallel_embedding: bool = False,
    paged_kv_cache: bool = True,
    remove_input_padding: bool = True,
    enable_multi_block_mode: bool = False,
    use_lora_plugin: str = None,
    lora_target_modules: Optional[List[str]] = None,
    max_lora_rank: int = 64,
    max_num_tokens: int = None,
    opt_num_tokens: int = None,
):
    """Build TensorRT-LLM engine with ModelOpt build_tensorrt_llm function."""
    assert not lora_target_modules, f"LoRA is not supported for quantized checkpoints, got {lora_target_modules}"

    warnings.warn(
        "Note that setting tensor_parallel_size and pipeline_parallel_size parameters"
        " for quantized models should be done on calibration step with nemo.export.quantize module."
        " These parameters are ignored when building and running TensorRT-LLM engine below.",
        UserWarning,
        stacklevel=3,
    )

    warnings.warn(
        "Also use_parallel_embedding, paged_kv_cache, remove_input_padding, enable_multi_block_mode, max_num_tokens"
        " and opt_num_tokens parameters are set by ModelOpt build_tensorrt_llm function in the optimal way and are"
        " ignored on engine build step.",
        UserWarning,
        stacklevel=3,
    )

    num_build_workers = len(glob.glob(os.path.join(nemo_checkpoint_path, WEIGHTS_NAME.format("*"))))
    assert num_build_workers, f"No TensorRT-LLM weight files found in {nemo_checkpoint_path}"

    build_tensorrt_llm(
        pretrained_config=os.path.join(nemo_checkpoint_path, CONFIG_NAME),
        engine_dir=engine_dir,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        max_beam_width=1,
        num_build_workers=num_build_workers,
        enable_sparsity=False,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
    )
