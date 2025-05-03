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

import math
import os
import time
from datetime import datetime
from typing import Optional

import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches

from nemo.automodel.checkpointing import save_checkpoint
from nemo.automodel.config import ConfigContainer
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import (
    append_to_progress_log,
    barrier_and_log,
    get_world_size_safe,
    is_last_rank,
    print_rank_last,
)


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    if torch.distributed.get_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    loss_scale,
    report_memory_flag,
    grad_norm,
    params_norm,
    config: ConfigContainer,
    global_state: GlobalState,
):
    """Log training metrics to console and tensorboard.

    Args:
        loss_dict: Dictionary containing losses for the current step.
        total_loss_dict: Dictionary containing accumulated losses.
        learning_rate: Current learning rate.
        loss_scale: Current loss scale for mixed precision training.
        report_memory_flag: Whether to report memory usage.
        grad_norm: Gradient norm after clipping.
        params_norm: Model parameters norm.
        config: Configuration container.
        global_state: Global state container.

    Returns:
        Updated report_memory_flag.
    """
    timers = global_state.timers
    train_state = global_state.train_state
    tb_logger = global_state.tensorboard_logger
    wandb_logger = global_state.wandb_logger
    logger_config = config.logger_config
    train_config = config.train_config

    # Advanced, skipped, and Nan iterations tracking
    advanced_iters_key = "advanced iterations"
    nan_iters_key = "nan iterations"

    # Track advanced iterations
    total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1

    # Track NaN losses
    got_nan = False
    for key in loss_dict:
        value = (
            loss_dict[key][0].float().item() / loss_dict[key][1].float().item()
            if isinstance(loss_dict[key], tuple)
            else loss_dict[key].float().item()
        )
        total_loss_dict[key] = total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float, device="cuda")) + value

    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

    # Timers to log
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "all-grads-sync",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
    ]

    # Calculate batch size
    batch_size = train_config.micro_batch_size * config.data_parallel_size * get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if logger_config.log_timers_to_tensorboard and (train_state.step % logger_config.tensorboard_log_interval == 0):
        reset_in_tb = False if hasattr(timers, "write_to_wandb") else True
        timers.write(timers_to_log, tb_logger, train_state.step, normalizer=total_iterations, reset=reset_in_tb)
        if hasattr(timers, "write_to_wandb"):
            timers.write_to_wandb(
                timers_to_log, wandb_logger, train_state.step, normalizer=total_iterations, reset=True
            )

    if tb_logger and (train_state.step % logger_config.tensorboard_log_interval == 0):
        if config.profiling_config:
            if config.profiling_config.record_memory_history and is_last_rank():
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump

                with open(config.profiling_config.memory_snapshot_path, "wb") as f:
                    dump(snapshot, f)

        if wandb_logger:
            wandb_logger.log({"samples vs steps": global_state.train_state.consumed_train_samples}, train_state.step)
        tb_logger.add_scalar("learning-rate", learning_rate, train_state.step)
        tb_logger.add_scalar(
            "learning-rate vs samples", learning_rate, global_state.train_state.consumed_train_samples
        )
        if wandb_logger:
            wandb_logger.log({"learning-rate": learning_rate}, train_state.step)

        if global_state.train_state.skipped_train_samples > 0:
            tb_logger.add_scalar(
                "skipped-train-samples", global_state.train_state.skipped_train_samples, train_state.step
            )
            if wandb_logger:
                wandb_logger.log(
                    {"skipped-train-samples": global_state.train_state.skipped_train_samples}, train_state.step
                )
        tb_logger.add_scalar("batch-size", batch_size, train_state.step)
        tb_logger.add_scalar("batch-size vs samples", batch_size, global_state.train_state.consumed_train_samples)
        if wandb_logger:
            wandb_logger.log({"batch-size": batch_size}, train_state.step)

        for key in loss_dict:
            if key == "lm loss":
                loss_sum = loss_dict[key][0].float().item()
                total_num_tokens = loss_dict[key][1].float().item()
                value = loss_sum / total_num_tokens
                tb_logger.add_scalar(key, value, train_state.step)
                tb_logger.add_scalar(key + " vs samples", value, global_state.train_state.consumed_train_samples)
                if wandb_logger:
                    wandb_logger.log({key: value}, train_state.step)
                    wandb_logger.log({"total_num_tokens": total_num_tokens}, train_state.step)
                    wandb_logger.log({"loss_sum": loss_sum}, train_state.step)

        if logger_config.log_loss_scale_to_tensorboard:
            tb_logger.add_scalar("loss-scale", loss_scale, train_state.step)
            tb_logger.add_scalar("loss-scale vs samples", loss_scale, global_state.train_state.consumed_train_samples)
            if wandb_logger:
                wandb_logger.log({"loss-scale": loss_scale}, train_state.step)
        if logger_config.log_world_size_to_tensorboard:
            tb_logger.add_scalar("world-size", get_world_size_safe(), train_state.step)
            tb_logger.add_scalar(
                "world-size vs samples", get_world_size_safe(), global_state.train_state.consumed_train_samples
            )
            if wandb_logger:
                wandb_logger.log({"world-size": get_world_size_safe()}, train_state.step)
        if grad_norm is not None:
            tb_logger.add_scalar("grad-norm", grad_norm, train_state.step)
            tb_logger.add_scalar("grad-norm vs samples", grad_norm, global_state.train_state.consumed_train_samples)
            if wandb_logger:
                wandb_logger.log({"grad-norm": grad_norm}, train_state.step)
        if params_norm is not None:
            tb_logger.add_scalar("params-norm", params_norm, train_state.step)
            tb_logger.add_scalar(
                "params-norm vs samples", params_norm, global_state.train_state.consumed_train_samples
            )
            if wandb_logger:
                wandb_logger.log({"params-norm": params_norm}, train_state.step)
        if logger_config.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            tb_logger.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                train_state.step,
            )
            tb_logger.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                train_state.step,
            )
            tb_logger.add_scalar(
                "mem-max-allocated-bytes",
                mem_stats["allocated_bytes.all.peak"],
                train_state.step,
            )
            tb_logger.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                train_state.step,
            )

    if train_state.step % logger_config.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        if logger_config.log_timers_to_tensorboard:
            if tb_logger:
                tb_logger.add_scalar("iteration-time", elapsed_time_per_iteration, train_state.step)
            if wandb_logger:
                wandb_logger.log({"iteration-time": elapsed_time_per_iteration}, train_state.step)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += " iteration {:8d}/{:8d} |".format(train_state.step, train_config.train_iters)
        log_string += " consumed samples: {:12d} |".format(global_state.train_state.consumed_train_samples)
        if global_state.train_state.skipped_train_samples > 0:
            log_string += " skipped samples: {:12d} |".format(global_state.train_state.skipped_train_samples)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(elapsed_time_per_iteration * 1000.0)

        log_string += f" learning rate: {learning_rate:.6E} |"
        log_string += f" global batch size: {batch_size:5d} |"
        for key in total_loss_dict:
            if key not in [advanced_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device="cuda")
        log_string += f" loss scale: {loss_scale:.1f} |"
        if grad_norm is not None:
            log_string += f" grad norm: {grad_norm:.3f} |"
        if params_norm is not None:
            log_string += f" params norm: {params_norm:.3f} |"
        log_string += " number of nan iterations: {:3d} |".format(total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag:
            report_memory(f"(after {train_state.step} iterations)")
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=logger_config.log_interval)

    return report_memory_flag


def eval_log(prefix: str, total_loss_dict: dict[str, torch.Tensor], global_state: GlobalState):
    tb_logger = global_state.tensorboard_logger
    wandb_logger = global_state.wandb_logger
    string = f" validation loss at {prefix} | "
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if tb_logger:
            tb_logger.add_scalar(
                "{} validation".format(key), total_loss_dict[key].item(), global_state.train_state.step
            )
            tb_logger.add_scalar(
                "{} validation vs samples".format(key),
                total_loss_dict[key].item(),
                global_state.train_state.consumed_train_samples,
            )
            if global_state.cfg.logger_config.log_validation_ppl_to_tensorboard:
                tb_logger.add_scalar("{} validation ppl".format(key), ppl, global_state.train_state.step)
                tb_logger.add_scalar(
                    "{} validation ppl vs samples".format(key), ppl, global_state.train_state.consumed_train_samples
                )

        if wandb_logger and is_last_rank():
            wandb_logger.log({"{} validation".format(key): total_loss_dict[key].item()}, global_state.train_state.step)
            if global_state.cfg.logger_config.log_validation_ppl_to_tensorboard:
                wandb_logger.log({"{} validation ppl".format(key): ppl}, global_state.train_state.step)

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)


def reduce_loss(
    loss_store: list[torch.Tensor],
    total_num_tokens: torch.Tensor,
    per_token_loss: bool = True,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Reduce loss across all ranks."""
    loss = torch.sum(torch.stack(loss_store).float()).view(1).clone().detach()

    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=dp_group)

    if per_token_loss:
        denominator = total_num_tokens.clone().detach().to(torch.int)
    else:
        denominator = torch.tensor([len(loss_store)], dtype=torch.int, device="cuda")
    torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    return loss, denominator


def get_start_time_from_progress_log(cfg: ConfigContainer):
    """
    Gets start time of earliest job with same world size. Also returns the number
    of floating-point operations completed in last saved checkpoint.
    """
    assert cfg.checkpoint_config.save is not None
    progress_log_filename = os.path.join(cfg.checkpoint_config.save, "progress.txt")

    # start_time is time when job with same world size started.
    # start_num_floating_point_operations is the number of floating-point operations
    # completed when this job started.
    # latest_num_floating_point_operations is the number of floating-point operations
    # completed in most recent saved checkpoint.
    start_time = None
    start_num_floating_point_operations = None
    latest_num_floating_point_operations = 0

    def _get_field(string, type):
        return type(string.split(": ")[1])

    with open(progress_log_filename, "r") as f:
        for line in f:
            line = line.strip()
            line_tokens = line.split("\t")
            world_size_in_line = _get_field(line_tokens[2], int)
            if line_tokens[3] == "Saved checkpoint":
                latest_num_floating_point_operations = _get_field(line_tokens[7], float)
            if world_size_in_line != get_world_size_safe():
                # Re-start search if we see a different world size.
                start_time = None
                start_num_floating_point_operations = None
                continue
            if line_tokens[3] == "Starting job":
                if start_time is None:
                    start_time = line_tokens[0]
                    start_num_floating_point_operations = latest_num_floating_point_operations
    assert (
        start_time is not None and start_num_floating_point_operations is not None
    ), "Should have seen at least one 'Starting job' entry with same world_size"
    return datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"), start_num_floating_point_operations


def compute_throughputs_and_append_to_progress_log(state: GlobalState):
    if state.cfg.checkpoint_config.save is None:
        return

    num_floating_point_operations_so_far = state.train_state.floating_point_operations_so_far

    # Compute job throughput.
    # completed at the start of job.
    job_throughput = (num_floating_point_operations_so_far - state.train_state.floating_point_operations_so_far) / (
        (time.time() - state.start_time) * 10**12 * get_world_size_safe()
    )

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log(state.cfg)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = (num_floating_point_operations_so_far - start_num_floating_point_operations) / (
        elapsed_time * 10**12 * get_world_size_safe()
    )

    tokens_so_far = state.train_state.consumed_train_samples * state.cfg.dataset_config.seq_length
    saved_ckpt_prefix = "Saving async checkpoint" if state.cfg.checkpoint_config.async_save else "Saved checkpoint"
    append_to_progress_log(
        state.cfg.checkpoint_config.save,
        f"{saved_ckpt_prefix}\tIteration: {state.train_state.step}\t"
        f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
        f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
        f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
        f"Tokens (in billions): {tokens_so_far / 10**9:.2f}",
    )


def save_checkpoint_and_time(
    state: GlobalState,
    model,
    optimizer,
    opt_param_scheduler,
):
    timers = state.timers

    # Stop timer to get accurate train interval time and exclude checkpointing duration
    timers("interval-time").stop()
    # Extra barrier is added to make sure all ranks report the max time.
    timer_key = "save-checkpoint"
    timers(timer_key, log_level=0).start(barrier=True)

    save_checkpoint(
        state.cfg.checkpoint_config.save,
        state,
        model,
        optimizer,
        opt_param_scheduler,
        state.cfg.dataset_config.tokenizer,
        save_rng=state.cfg.checkpoint_config.save_rng,
        save_optim=state.cfg.checkpoint_config.save_optim,
    )
    timers(timer_key).stop(barrier=True)
    timers.log([timer_key])

    if state.cfg.logger_config.log_progress:
        compute_throughputs_and_append_to_progress_log(state)

    # Recover timing
    timers("interval-time", log_level=0).start(barrier=True)


def checkpoint_and_decide_exit(
    state: GlobalState,
    model,
    optimizer,
    opt_param_scheduler,
):
    """Save checkpoint and decide whether to exit based on arguments (e.g., if
    --exit-duration-in-mins is set). Actual exit happens in main training loop
    based on the return value of this function."""
    saved_checkpoint = False

    # Exit based on signal handler.
    if state.cfg.train_config.exit_signal_handler:
        signal_handler = state.signal_handler
        if any(signal_handler.signals_received()):
            if state.cfg.checkpoint_config.save:
                save_checkpoint_and_time(
                    state,
                    model,
                    optimizer,
                    opt_param_scheduler,
                )
            barrier_and_log("exiting program after receiving SIGTERM.")

            return True

    # Regular save (persistent).
    if (
        state.cfg.checkpoint_config.save
        and state.cfg.checkpoint_config.save_interval
        and state.train_state.step % state.cfg.checkpoint_config.save_interval == 0
    ):
        save_checkpoint_and_time(
            state,
            model,
            optimizer,
            opt_param_scheduler,
        )
        saved_checkpoint = True

    # Exit based on duration.
    if state.cfg.train_config.exit_duration_in_mins:
        train_time = (time.time() - state.train_state.start_time) / 60.0
        done_cuda = torch.tensor(
            [train_time > state.cfg.checkpoint_config.exit_duration_in_mins], dtype=torch.int, device="cuda"
        )
        torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
        done = done_cuda.item()
        if done:
            if state.cfg.checkpoint_config.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    state,
                    model,
                    optimizer,
                    opt_param_scheduler,
                )
            barrier_and_log(f"exiting program after {train_time} minutes")

            return True

    # Exit based on iterations.
    if state.cfg.train_config.exit_interval and state.train_state.step % state.cfg.train_config.exit_interval == 0:
        if state.cfg.checkpoint_config.save and not saved_checkpoint:
            save_checkpoint_and_time(
                state,
                model,
                optimizer,
                opt_param_scheduler,
            )
        barrier_and_log(f"exiting program at iteration {state.train_state.step}")

        return True

    return False
