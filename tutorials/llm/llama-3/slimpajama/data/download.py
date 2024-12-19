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


def download_slimpajama(include_pattern: str = "", exclude_pattern: str = ""):
    """
    Configure run.Script to download SlimPajama dataset from HuggingFace.

    Args:
        include_pattern: Include pattern for HuggingFace CLI.
        exclude_pattern: Exclude pattern for HuggingFace CLI.
    """
    hf_cli_cmd = "huggingface-cli download cerebras/SlimPajama-627B {include_pattern} {exclude_pattern} --quiet --repo-type dataset --local-dir /data/slimpajama --cache-dir /data/slimpajama"  # pylint: disable=line-too-long
    hf_cli_cmd = hf_cli_cmd.format(include_pattern=include_pattern, exclude_pattern=exclude_pattern)

    download_script = """
pip install "huggingface_hub[cli,hf_transfer]"

retry_command() {
    local max_retries=$1
    local sleep_time=$2
    local retry_count=0
    local command=${@:3}

    echo "Running $command"
    while [ $retry_count -lt $max_retries ]; do
        eval $command
        if [ $? -eq 0 ]; then
            echo "Command succeeded"
            return 0
        else
            echo "Command failed. Attempt: $((retry_count + 1))"
            retry_count=$((retry_count + 1))
            sleep $sleep_time
        fi
    done

    echo "Command failed after $max_retries retries"
    return 1
}

export HF_HUB_DOWNLOAD_TIMEOUT=30
export HF_ENABLE_HF_TRANSFER=True
"""

    download_script += f"retry_command 5 5 {hf_cli_cmd}\n"
    download_task = run.Script(inline=download_script)
    return download_task
