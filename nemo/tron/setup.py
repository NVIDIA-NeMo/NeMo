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

import time

import torch

from nemo.tron.config import ConfigContainer
from nemo.tron.data.dataset import setup_data_iterators
from nemo.tron.init import initialize_megatron, set_jit_fusion_options
from nemo.tron.model import get_model_from_config
from nemo.tron.optim import setup_optimizer
from nemo.tron.state import GlobalState
from nemo.tron.utils import append_to_progress_log, barrier_and_log, print_rank_0


def setup(
    cfg: ConfigContainer,
    train_valid_test_dataset_provider,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(
        cfg=cfg, get_embedding_ranks=get_embedding_ranks, get_position_embedding_ranks=get_position_embedding_ranks
    )

    timers = state.timers

    if cfg.megatron_lm_config.log_progress:
        append_to_progress_log(cfg.megatron_lm_config.save, "Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(state)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor([state.start_time], dtype=torch.double, device="cuda")
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print_rank_0("time to initialize megatron (seconds): {:.3f}".format(time.time() - state.start_time))
    barrier_and_log("after megatron is initialized")

    checkpointing_context = {}

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model = get_model_from_config(
        cfg.model_config,
        cfg.ddp_config,
        use_torch_fsdp2=cfg.megatron_lm_config.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer_config.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.megatron_lm_config.data_parallel_random_init,
    )
    optimizer, scheduler = setup_optimizer(cfg, model)

    timers("model-and-optimizer-setup").stop()
    barrier_and_log("after model, optimizer, and learning rate scheduler are built")

    # Data stuff.
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
    train_data_iterator, valid_data_iterator, test_data_iterator = setup_data_iterators(
        cfg=cfg,
        train_state=state.train_state,
        model_length=len(model),
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
    )
    timers("train/valid/test-data-iterators-setup").stop()
    barrier_and_log("after dataloaders are built")

    # if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
    #     ft_integration.get_rank_monitor_client().init_workload_monitoring()
    #     ft_timeouts = ft_integration.get_rank_monitor_client().timeouts
    #     print_rank_0(f"Fault tolerance client initialized. Timeouts: {ft_timeouts}")

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"], barrier=True)

    return model, optimizer, scheduler, train_data_iterator, valid_data_iterator, test_data_iterator
