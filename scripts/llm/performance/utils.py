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

import argparse
import os
import sys
from typing import Dict, List

import nemo_run as run
from lightning.pytorch.callbacks.callback import Callback
from nemo_run.config import NEMORUN_HOME

from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model import GPTModel
from nemo.collections.llm.recipes.llama3_8b import MegatronCommOverlapCallback
from nemo.lightning.base import DEFAULT_NEMO_CACHE_HOME
from nemo.utils import logging

DEFAULT_NEMO_HOME = os.getenv('NEMO_HOME', DEFAULT_NEMO_CACHE_HOME)


def slurm_executor(
    account: str,
    partition: str,
    log_dir: str,
    nodes: int,
    num_gpus_per_node: int,
    time_limit: str = "00:30:00",
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    custom_mounts: List[str] = [],
    custom_env_vars: Dict[str, str] = {},
    custom_srun_args: List[str] = [],
    hf_token: str = None,
    nemo_home: str = DEFAULT_NEMO_HOME,
) -> run.SlurmExecutor:
    """
    Slurm cluster definition with appropriate cluster params and NeMo container params needed for pre-training
    and fine-tuning experiments
    """
    err_msgs = []
    if log_dir != NEMORUN_HOME:
        err_msgs.append(f"\nRun `export NEMORUN_HOME={log_dir}` in your shell environment and rerun this script.")
    if len(err_msgs) > 0:
        logging.error("\n".join(err_msgs))
        sys.exit(1)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "False",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "1",
        "NVTE_FLASH_ATTN": "1",
        "NEMO_LOG_MEMORY_USAGE": "1",
        "NEMORUN_HOME": log_dir,
    }
    mounts = []
    srun_args = ["--mpi=pmix"]

    if nemo_home != DEFAULT_NEMO_CACHE_HOME:  # DO NOT change this 'DEFAULT_NEMO_HOME'/'NEMO_HOME'
        env_vars.update({"NEMO_HOME": nemo_home})
        mounts.extend([f"{nemo_home}:{nemo_home}"])
    if hf_token is not None:
        env_vars.update({"HF_TOKEN": hf_token, "TRANSFORMERS_OFFLINE": "0"})

    env_vars |= custom_env_vars
    mounts.extend(custom_mounts)
    srun_args.extend(custom_srun_args)

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.LocalTunnel(
            job_dir=os.path.join(log_dir, "experiments"),
        ),
        nodes=nodes,
        ntasks_per_node=num_gpus_per_node,
        container_image=container_image,
        container_mounts=mounts,
        env_vars=env_vars,
        srun_args=srun_args,
        time=time_limit,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(),
    )

    return executor


def hf_tokenizer(model_name: str) -> run.Config[AutoTokenizer]:
    """
    HuggingFace tokenizer.

    Args:
        model_name (str): corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument.
                For more details please refer to-
                huggingface.co/docs/transformers/v4.47.1/en/model_doc/auto#transformers.AutoTokenizer
    """
    log_msg = [
        f"`AutoTokenizer` first searches for tokenizer files locally stored in {DEFAULT_NEMO_HOME}.",
        "(from env var `NEMO_HOME`- can be changed using '-nh/--nemo_home' CLI arg).",
        "If files are missing locally, `AutoTokenizer` will try downloading from HuggingFace. In this case-",
        "make sure env vars 'TRANSFORMERS_OFFLINE':'0' and 'HF_TOKEN':'<token_value>' are set in your sbatch script.",
        "Both of these will be set automatically if you provide '-hf/--hf_token' CLI arg.",
    ]
    logging.warning(" ".join(log_msg))

    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


def import_ckpt_experiment(executor: run.SlurmExecutor, model: run.Config[GPTModel], source: str):
    """
    Downloads/Acceses checkpoint to be used for fine-tuning. `import_ckpt` first tries find the nemo checkpoint in
    <NEMO_HOME>/models/. For eg: for llama3 8b, the path will look like- <NEMO_HOME>/models/meta-llama/Meta-Llama-3-8B
    If missing, tries to downloads at the same location from HuggingFace and converts it nemo format.

    Args:
        source (str): HuggingFace URL. For eg- hf://meta-llama/Meta-Llama-3-70B
    """
    from copy import deepcopy

    from nemo.collections.llm import import_ckpt

    import_executor = deepcopy(executor)
    import_executor.ntasks_per_node = 1
    import_executor.nodes = 1

    return run.Partial(import_ckpt, model=model, source=source, overwrite=False), import_executor, "import_ckpt_exp"


def isfile_train_pack_metadata(hf_model_uri: str, data_config: run.Config[SquadDataModule]) -> bool:
    """
    This method is used for fine-tuning. It checks if packed train data for a partiular
    sequence length exists locally. This is needed to set data flag (force_redownload=True)
    which avoids experiment crash in case files are missing.
    """
    datasets_dir = os.getenv("NEMO_DATASETS_CACHE", os.path.join(DEFAULT_NEMO_HOME, "datasets"))
    model_dir = hf_model_uri.replace("/", "--")
    metadata_filename = f"{data_config.seq_length}_metadata.jsonl"

    train_pack_metadata_filepath = os.path.join(datasets_dir, "squad", "packed", model_dir, metadata_filename)

    return os.path.exists(train_pack_metadata_filepath) and os.path.isfile(train_pack_metadata_filepath)


def get_comm_overlap_callback_idx(callbacks: List[Callback]) -> int | None:
    """
    nemo.lightning.Trainer has a list of callbacks defined. This method identifies index of MegatronCommOverlapCallback
    from the list defined in recipes in nemo.collections.llm.recipes. The index is needed to override ddp communication
    params
    """
    if callbacks:  # default is None in lightning
        for idx, callback in enumerate(callbacks):
            if callback.__fn_or_cls__ == MegatronCommOverlapCallback:
                return idx
    return None


def parse_cli_args():
    """
    Command line arguments correspong to Slurm cluster and NeMo2.0 for running pre-training and
    fine-tuning experiments.
    """
    parser = argparse.ArgumentParser(description="NeMo2.0 Performance Pretraining and Fine-Tuning")

    parser.add_argument(
        "-a",
        "--account",
        type=str,
        help="Slurm account to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Slurm partition to use for experiment",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        help=f"Directory for logging experiment results. Defaults to {NEMORUN_HOME}",
        required=False,
        default=NEMORUN_HOME,
    )
    parser.add_argument(
        "-t",
        "--time_limit",
        type=str,
        help="Maximum time limit to run experiment for. Defaults to 30 minutes (format- 'HH:MM:SS')",
        required=False,
        default="00:30:00",
    )
    container_img_msg = [
        "NeMo container to use for experiment. Defaults to latest dev container- 'nvcr.io/nvidia/nemo:dev'",
        "Make sure your NGC credentials are accessible in your environment.",
    ]
    parser.add_argument(
        "-i",
        "--container_image",
        type=str,
        help=" ".join(container_img_msg),
        required=False,
        default="nvcr.io/nvidia/nemo:dev",
    )
    parser.add_argument(
        "-c",
        "--compute_dtype",
        type=str,
        help="Compute precision. Options- bf16 or fp8. Defaults to bf16",
        required=False,
        default="bf16",
    )
    parser.add_argument(
        "-ep",
        "--enable_profiling",
        help="Enable Nsys profiling. Diabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        help="Enable tensorboard logging. Disabled by default",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--finetuning",
        help="Finetuning scheme to use. Options- 'sft', 'lora'. Defaults is 'lora'",
        default='lora',
    )
    parser.add_argument(
        "-hf",
        "--hf_token",
        type=str,
        help="HuggingFace token. Defaults to None. Required for accessing tokenizers and checkpoints.",
        default=None,
    )
    nemo_home_msg = [
        "Sets env var `NEMO_HOME` (on compute node using sbatch script)- directory where NeMo searches",
        "for models and checkpoints. This saves a lot of time (especially for bigger models) if checkpoints already",
        f"exist here. Missing files will be downloaded here from HuggingFace. Defaults to {DEFAULT_NEMO_HOME}",
    ]
    parser.add_argument(
        "-nh",
        "--nemo_home",
        type=str,
        help=" ".join(nemo_home_msg),
        default=DEFAULT_NEMO_HOME,
    )
    parser.add_argument(
        "-d",
        "--dryrun",
        help="If true, prints sbatch script to terminal without launching experiment.",
        required=False,
        action="store_true",
    )

    return parser
