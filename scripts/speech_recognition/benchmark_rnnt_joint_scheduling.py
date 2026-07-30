#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Compare loss-agnostic RNN-T joint scheduling before and after full-batch projection."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


METHODS = ("warprnnt_numba", "graph_rnnt", "flash_rnnt")
MODES = ("before", "full_projection")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("target", "tawseem"),
        default=["target", "tawseem"],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--source-length", type=int, default=400)
    parser.add_argument("--target-length", type=int, default=128)
    parser.add_argument("--min-length-fraction", type=float, default=1.0)
    parser.add_argument(
        "--fused-batch-size",
        type=int,
        default=4,
        help="Maximum samples per joint chunk.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0, help="Joint dropout probability.")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Activation dtype. Compare backends at matched precision before quoting a speedup.",
    )
    parser.add_argument("--length-profile", type=Path)
    parser.add_argument("--profile-index", type=int, default=0)
    parser.add_argument(
        "--replay-all-tawseem",
        action="store_true",
        help="Prewarm and measure every stored TAWSEEM length batch once.",
    )
    parser.add_argument("--output", type=Path, default=Path("rnnt_joint_scheduling_results.json"))
    parser.add_argument("--job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=("target", "tawseem"), help=argparse.SUPPRESS)
    return parser


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(int(len(values) * fraction), len(values) - 1)]


def _profile_length_batches(args) -> list[tuple[list[int], list[int]]]:
    batch = args.batch_size or (32 if args.profile == "target" else 48)
    if args.profile == "target":
        if not 0.0 < args.min_length_fraction <= 1.0:
            raise ValueError("--min-length-fraction must be in (0, 1]")
        denominator = max(batch - 1, 1)
        fractions = [
            args.min_length_fraction + (1.0 - args.min_length_fraction) * i / denominator for i in range(batch)
        ]
        source_lengths = [max(1, round(args.source_length * fraction)) for fraction in fractions]
        target_lengths = [max(0, round(args.target_length * fraction)) for fraction in fractions]
        return [(source_lengths, target_lengths)]

    if args.length_profile is None:
        raise ValueError("--length-profile is required for the tawseem profile")
    payload = json.loads(args.length_profile.read_text())
    items = payload["batches"]
    if not args.replay_all_tawseem:
        if not 0 <= args.profile_index < len(items):
            raise ValueError(f"--profile-index must be in [0, {len(items) - 1}], got {args.profile_index}")
        items = [items[args.profile_index]]
    result = []
    for item in items:
        if batch > len(item["input_lengths"]):
            raise ValueError(f"TAWSEEM profile contains only {len(item['input_lengths'])} samples")
        result.append((item["input_lengths"][:batch], item["target_lengths"][:batch]))
    return result


def _fixed_chunks(source_lengths, target_lengths, chunk_size):
    chunks = []
    for begin in range(0, len(source_lengths), chunk_size):
        end = min(begin + chunk_size, len(source_lengths))
        chunks.append(
            argparse.Namespace(
                begin=begin,
                end=end,
                max_source=max(source_lengths[begin:end]),
                max_target=max(target_lengths[begin:end]),
            )
        )
    return chunks


def _build_loss(method, blank):
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    kwargs = None
    if method == "warprnnt_numba":
        kwargs = {"fastemit_lambda": 0.0, "clamp": -1.0}
    elif method == "graph_rnnt":
        kwargs = {
            "use_grid_implementation": True,
            "use_triton": True,
            "cast_to_float32": False,
        }
    elif method == "flash_rnnt":
        kwargs = {
            "fastemit_lambda": 0.0,
            "clamp": -1.0,
        }
    loss = RNNTLoss(num_classes=blank, reduction=None, loss_name=method, loss_kwargs=kwargs)
    if method == "warprnnt_numba":
        # The Numba kernels cannot unify BF16 with their FP32 dynamic-programming state, so they
        # run FP32 whatever the joint uses. That makes a BF16 comparison against them a precision
        # comparison as much as a scheduling one -- use `--dtype float32` for a matched baseline.
        loss._force_float32 = True
    return loss


def _run_job(args) -> dict:
    import torch

    from nemo.collections.asr.modules.rnnt import RNNTJoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    vocab = 1024
    length_batches = _profile_length_batches(args)
    batch = len(length_batches[0][0])
    if any(len(source) != batch for source, _ in length_batches):
        raise ValueError("All replay batches must have the same batch size")
    max_source = max(max(source) for source, _ in length_batches)
    max_target = max(max(target) for _, target in length_batches)
    blank = vocab - 1

    max_chunk_batch = args.fused_batch_size
    joint = RNNTJoint(
        jointnet={
            "encoder_hidden": 512,
            "pred_hidden": 640,
            "joint_hidden": 640,
            "activation": "relu",
            "dropout": args.dropout,
        },
        num_classes=blank,
        log_softmax=False,
        fuse_loss_wer=args.method == "flash_rnnt",
        fused_batch_size=max_chunk_batch if args.method == "flash_rnnt" else None,
    ).to(device=device, dtype=dtype)
    loss = _build_loss(args.method, blank).to(device)
    if args.method == "flash_rnnt":
        joint.set_loss(loss)
        joint.set_wer(object())
    encoder = torch.randn(batch, 512, max_source, device=device, dtype=dtype, requires_grad=True)
    predictor = torch.randn(batch, 640, max_target + 1, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, blank, (batch, max_target), device=device, dtype=torch.int64)
    device_length_batches = [
        (
            torch.tensor(source, device=device, dtype=torch.int64),
            torch.tensor(target, device=device, dtype=torch.int64),
        )
        for source, target in length_batches
    ]
    reported_chunks = [_fixed_chunks(source, target, max_chunk_batch) for source, target in length_batches]

    def iteration_chunks(source_lengths, target_lengths):
        return _fixed_chunks(
            source_lengths.tolist(),
            target_lengths.tolist(),
            max_chunk_batch,
        )

    def clear_gradients():
        encoder.grad = None
        predictor.grad = None
        for parameter in joint.parameters():
            parameter.grad = None

    def iteration(profile_index):
        source_lengths, target_lengths = device_length_batches[profile_index]
        source_lengths_list, target_lengths_list = length_batches[profile_index]
        profile_source = max(source_lengths_list)
        profile_target = max(target_lengths_list)
        profile_encoder = encoder[:, :, :profile_source]
        profile_predictor = predictor[:, :, : profile_target + 1]
        profile_targets = targets[:, :profile_target]
        encoder_t = profile_encoder.transpose(1, 2)
        predictor_t = profile_predictor.transpose(1, 2)
        if args.method == "flash_rnnt":
            value = joint(
                encoder_outputs=profile_encoder,
                decoder_outputs=profile_predictor,
                encoder_lengths=source_lengths,
                transcripts=profile_targets,
                transcript_lengths=target_lengths,
            )[0].mean()
            value.backward()
            return value
        if args.mode == "before":
            projected_encoder = projected_predictor = None
        else:
            projected_encoder = joint.project_encoder(encoder_t)
            projected_predictor = joint.project_prednet(predictor_t)
        losses = []
        for chunk in iteration_chunks(source_lengths, target_lengths):
            begin, end = chunk.begin, chunk.end
            chunk_source, chunk_target = chunk.max_source, chunk.max_target
            if args.mode == "before":
                logits = joint.joint(
                    encoder_t[begin:end, :chunk_source],
                    predictor_t[begin:end, : chunk_target + 1],
                )
            else:
                logits = joint.joint_after_projection(
                    projected_encoder[begin:end, :chunk_source],
                    projected_predictor[begin:end, : chunk_target + 1],
                )
            chunk_loss = loss(
                log_probs=logits,
                targets=profile_targets[begin:end, :chunk_target],
                input_lengths=source_lengths[begin:end],
                target_lengths=target_lengths[begin:end],
            )
            losses.append(chunk_loss)
        value = torch.cat(losses).mean()
        value.backward()
        return value

    if args.replay_all_tawseem:
        # Shape-dependent Triton specializations must not compile in the timed replay.
        for profile_index in range(len(length_batches)):
            clear_gradients()
            iteration(profile_index)
    for warmup_index in range(args.warmup):
        clear_gradients()
        iteration(warmup_index % len(length_batches))
    torch.cuda.synchronize()
    clear_gradients()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_memory = torch.cuda.memory_allocated()

    host_times = []
    gpu_times = []
    last_loss = None
    if args.replay_all_tawseem:
        measured_profiles = range(len(length_batches))
    else:
        measured_profiles = [0] * args.iterations
    measured_samples = 0
    for profile_index in measured_profiles:
        clear_gradients()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        host_start = time.perf_counter()
        start_event.record()
        last_loss = iteration(profile_index)
        end_event.record()
        end_event.synchronize()
        host_times.append((time.perf_counter() - host_start) * 1000.0)
        gpu_times.append(start_event.elapsed_time(end_event))
        measured_samples += len(length_batches[profile_index][0])

    peak = torch.cuda.max_memory_allocated()
    host_p50 = statistics.median(host_times)
    result = {
        "method": args.method,
        "mode": args.mode,
        "profile": args.profile,
        "batch_size": batch,
        "fused_batch_size": max_chunk_batch,
        "dropout": args.dropout,
        "dtype": args.dtype,
        "profile_batches": len(length_batches),
        "chunks_p50": statistics.median([len(profile_chunks) for profile_chunks in reported_chunks]),
        "chunk_sizes": [chunk.end - chunk.begin for chunk in reported_chunks[0]],
        "loss": float(last_loss.detach()),
        "host_p50_ms": host_p50,
        "host_p95_ms": _percentile(host_times, 0.95),
        "gpu_p50_ms": statistics.median(gpu_times),
        "samples_per_second": measured_samples * 1000.0 / sum(host_times),
        "peak_allocated_gib": peak / 2**30,
        "incremental_allocated_gib": (peak - base_memory) / 2**30,
    }
    print(json.dumps(result), flush=True)
    return result


def _run_all(args) -> None:
    script = Path(__file__).resolve()
    results = []
    for profile in args.profiles:
        for method in args.methods:
            for mode in args.modes:
                command = [
                    sys.executable,
                    str(script),
                    "--job",
                    "--method",
                    method,
                    "--mode",
                    mode,
                    "--profile",
                    profile,
                    "--fused-batch-size",
                    str(args.fused_batch_size),
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--dropout",
                    str(args.dropout),
                    "--dtype",
                    args.dtype,
                    "--source-length",
                    str(args.source_length),
                    "--target-length",
                    str(args.target_length),
                    "--min-length-fraction",
                    str(args.min_length_fraction),
                ]
                if args.batch_size is not None:
                    command.extend(("--batch-size", str(args.batch_size)))
                if args.length_profile is not None:
                    command.extend(("--length-profile", str(args.length_profile)))
                if args.replay_all_tawseem:
                    command.append("--replay-all-tawseem")
                else:
                    command.extend(("--profile-index", str(args.profile_index)))
                process = subprocess.run(command, text=True, capture_output=True, env=os.environ.copy())
                if process.returncode:
                    results.append(
                        {
                            "profile": profile,
                            "method": method,
                            "mode": mode,
                            "status": "error",
                            "error": process.stderr[-4000:],
                        }
                    )
                    continue
                results.append(json.loads(process.stdout.strip().splitlines()[-1]))
                print(
                    f"{profile:7} {method:17} {mode:15} "
                    f"{results[-1]['host_p50_ms']:8.2f} ms {results[-1]['samples_per_second']:8.2f} samples/s",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    args = _parser().parse_args()
    if args.job:
        _run_job(args)
    else:
        _run_all(args)


if __name__ == "__main__":
    main()
