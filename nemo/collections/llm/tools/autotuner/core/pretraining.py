# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import Any, Dict, List, Optional

import nemo_run as run

from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
from nemo.collections.llm.tools.autotuner.core.utils import _load_args_from_config_dir, validate_all_configs
from nemo.lightning.run.plugins import PerfEnvPlugin

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def lepton_executor(
    nodes: int,
    devices: int,
    resource_shape: str = "gpu.8xh200",
    container_image: str = "nvcr.io/nvidia/nemo:25.04",
    nemo_run_dir: str = "/nemo-workspace/nemo-run",
    mount_path: str = "/nemo-workspace",
    mount_from: str = "node-nfs:shared",
    node_group: str = "nebius-h200-01",
    hf_token: Optional[str] = None,
    wandb_api_key: Optional[str] = None,
    torch_home: str = "/nemo-workspace/.cache",
    pythonpath: str = "/nemo-workspace/nemo-run:$PYTHONPATH",
) -> run.LeptonExecutor:
    """Create a Lepton executor for training with dynamic configuration.

    Includes performance optimization environment variables for:
    - NCCL communication buffer optimization
    - Flash Attention and cuDNN fused attention
    - Memory usage logging
    - Tokenizer parallelism control
    """
    mounts = [{"path": "/", "mount_path": mount_path, "from": mount_from}]
    env_vars = {
        "PYTHONPATH": pythonpath,
        "TORCH_HOME": torch_home,
        # Performance optimization environment variables (from llama31_utils.py)
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FLASH_ATTN": "1",  # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NVTE_FUSED_ATTN": "1",  # Enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1",  # Print memory allocation
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
        env_vars["TRANSFORMERS_OFFLINE"] = "0"  # Enable online downloads when HF token is provided
    if wandb_api_key:
        env_vars["WANDB_API_KEY"] = wandb_api_key

    return run.LeptonExecutor(
        resource_shape=resource_shape,
        container_image=container_image,
        nemo_run_dir=nemo_run_dir,
        mounts=mounts,
        node_group=node_group,
        nodes=nodes,
        nprocs_per_node=devices,
        env_vars=env_vars,
        launcher="torchrun",
    )


def run_pretraining(
    base_config,
    configs: Dict,
    base_config_matches: List[str] = None,
    sequential: bool = False,
    executor_config: Dict[str, Any] = None,
    memory_analysis: Dict[str, Dict[str, Any]] = None,
    run_all: bool = False,
):
    """Run pretraining only without results collection."""
    logger.info("Starting AutoTune pretraining...")

    if base_config_matches is None:
        base_config_matches = []
    if executor_config is None:
        executor_config = {}
    if memory_analysis is None:
        memory_analysis = {}

    configs_to_run = {}
    skipped_configs = {}
    base_config_will_run = True

    base_analysis = memory_analysis.get("base_config", {})
    base_will_oom = base_analysis.get("will_oom", False)
    if base_will_oom and not run_all:
        base_config_will_run = False
        skipped_configs["base_config"] = "Potential CUDA OOM"
        logger.warning("Skipping base_config due to potential CUDA OOM (use --run-all to force)")

    for config_name, config_obj in configs.items():
        analysis = memory_analysis.get(config_name, {})
        will_oom = analysis.get("will_oom", False)
        if will_oom and not run_all:
            skipped_configs[config_name] = "Potential CUDA OOM"
            logger.warning(f"Skipping {config_name} due to potential CUDA OOM (use --run-all to force)")
        else:
            configs_to_run[config_name] = config_obj

    total_configs = len(configs) + (1 if not base_config_matches else 0)
    configs_to_run_count = len(configs_to_run) + (1 if base_config_will_run and not base_config_matches else 0)
    skipped_count = len(skipped_configs)

    logger.info(f"Configuration filtering summary:")
    logger.info(f"  Total configurations: {total_configs}")
    logger.info(f"  Configurations to run: {configs_to_run_count}")
    logger.info(f"  Skipped configurations: {skipped_count}")

    if configs_to_run_count == 0:
        logger.error("No configurations to run! All were filtered out due to potential OOM.")
        logger.error("Use --run-all flag to run anyway, or adjust your configuration parameters.")
        return {
            'total_configs': total_configs,
            'configs_run': 0,
            'configs_skipped': skipped_count,
            'skipped_configs': skipped_configs,
            'status': 'no_configs_to_run',
        }

    logger.info("Executor Settings...")
    logger.info(executor_config)

    executor = lepton_executor(
        nodes=base_config.trainer.num_nodes, devices=base_config.trainer.devices, **executor_config
    )

    logger.info("Running filtered configurations...")

    with run.Experiment("pretrain-magic") as exp:
        if not base_config_matches and base_config_will_run:
            plugins = [
                PerfEnvPlugin(
                    enable_vboost=True,
                    nccl_pp_comm_chunksize=(
                        2097152 if base_config.trainer.strategy.pipeline_model_parallel_size > 1 else None
                    ),
                )
            ]
            exp.add(base_config, executor=executor, name="base-config", plugins=plugins)
            logger.info("Added base_config to experiment")
        elif not base_config_matches and not base_config_will_run:
            logger.info("Skipped base_config due to potential CUDA OOM")
        else:
            logger.info(f"Skipping base_config as it matches: {', '.join(base_config_matches)}")

        idx = 2

        def extract_config_number(config_name):
            """Extract numerical part from config name for sorting."""
            try:
                if config_name.startswith('config-'):
                    return int(config_name.split('-')[-1])
                elif config_name.startswith('base_config'):
                    return 0  # base_config comes first
                else:
                    return float('inf')  # unknown configs go last
            except (ValueError, IndexError):
                return float('inf')  # invalid configs go last

        sorted_configs = sorted(configs_to_run.items(), key=lambda x: extract_config_number(x[0]))
        for config_name, recipe in sorted_configs:
            plugins = [
                PerfEnvPlugin(
                    enable_vboost=True,
                    nccl_pp_comm_chunksize=(
                        2097152 if recipe.trainer.strategy.pipeline_model_parallel_size > 1 else None
                    ),
                )
            ]
            if config_name in base_config_matches:
                exp.add(recipe, executor=executor, name=f'base-config', plugins=plugins)
                logger.info(f"Added {config_name} as base_config_equivalent (matches base config)")
            else:
                exp.add(recipe, executor=executor, name=f'config-{idx}', plugins=plugins)
                logger.info(f"Added {config_name} as config-{idx}")
                idx = idx + 1

        exp.run(sequential=sequential)

    logger.info("AutoTune pretraining completed successfully!")
    if base_config_matches:
        logger.info(
            f"Note: Base config was not run separately as it matches {len(base_config_matches)} generated config(s)"
        )
    if skipped_count > 0:
        logger.info(f"Note: {skipped_count} configuration(s) were skipped due to potential CUDA OOM")

    return {
        'total_configs': total_configs,
        'configs_run': configs_to_run_count,
        'configs_skipped': skipped_count,
        'skipped_configs': skipped_configs,
        'status': 'completed',
    }
