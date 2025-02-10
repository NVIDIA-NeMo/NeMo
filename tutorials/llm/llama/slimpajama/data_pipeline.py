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

from typing import Optional

import nemo_run as run
from data.download import download_slimpajama
from data.extract import run_extraction
from data.preprocess import preprocess_data


def slurm_executor(  # pylint: disable=C0116
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    tasks_per_node: int,
    time: str = "01:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
    ssh_key_file_path: Optional[str] = None,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and tasks_per_node):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."  # pylint: disable=line-too-long
        )

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    # Required to run on CPU nodes
    env_vars = {"NVIDIA_VISIBLE_DEVICES": "void"}
    if custom_env_vars:
        env_vars |= custom_env_vars

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir,
            identity=ssh_key_file_path,
        ),
        nodes=nodes,
        ntasks_per_node=tasks_per_node,
        mem="0",
        exclusive=True,
        packager=run.GitArchivePackager(subpath="examples/llm/slimpajama"),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def docker_executor():  # pylint: disable=C0116
    packager = run.GitArchivePackager(subpath="examples/llm/slimpajama")
    executor = run.DockerExecutor(
        packager=packager,
        ipc_mode="host",
        shm_size="30g",
        env_vars={"PYTHONUNBUFFERED": "1"},
        volumess=["/path/to/save/data:/data"],
        container_image="python:3.11",
        ulimits=["memlock:-1", "stack:67108864"],
    )
    return executor


def run_data_pipeline():  # pylint: disable=C0116
    executor = docker_executor()
    with run.Experiment("slimpajama-data-pipeline") as exp:
        exp.add(
            download_slimpajama(
                include_pattern='--include "train/chunk1/*_1*zst"',
            ),
            name="download_slimpajama",
            executor=executor,
        )

        # Use NeMo image for the remaining tasks
        executor.container_image = "nvcr.io/nvidia/nemo:nightly"
        exp.add(run.Partial(run_extraction, data_dir="/data/slimpajama"), executor=executor)

        # examples/llm/slimpajama is automatically mounted to /nemo_run/code
        exp.add(run.Script("/nemo_run/code/data/concat.sh", args=["/data/slimpajama/train", "1"]), executor=executor)
        exp.add(
            run.Partial(
                preprocess_data,
                data_dir="/data/slimpajama",
                output_dir="/data/slimpajama_megatron",
                tokenizer_model="/data/tokenizer/tokenizer.model",
                tokenizer_library="sentencepiece",
                vocab_file_path="/data/tokenizer/tokenizer.vocab",
            ),
            executor=executor,
        )

        exp.run(sequential=True, tail_logs=True, detach=True)


if __name__ == "__main__":
    run_data_pipeline()
