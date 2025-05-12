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
from typing import Optional

import nemo_run as run
import torch


@run.cli.factory
def torchrun(devices: Optional[int] = None) -> run.Config[run.LocalExecutor]:
    """
    Local executor using torchrun.

    Args:
        devices (Optional[int]): Number of devices to use. If None, it will use all available CUDA devices.

    Returns:
        run.Config[run.LocalExecutor]: Configuration for the local executor using torchrun.
    """
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    }

    if devices is None:
        if torch.cuda.is_available():
            devices = torch.cuda.device_count()
        else:
            raise RuntimeError(
                "Cannot infer the 'ntasks_per_node' parameter as CUDA is not available: please specify explicitely."
            )

    executor = run.Config(
        run.LocalExecutor,
        ntasks_per_node=devices,
        launcher="torchrun",
        env_vars=env_vars,
    )

    return executor
