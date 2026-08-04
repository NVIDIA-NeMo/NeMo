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

"""Compare RNN-T loss backends through the fused joint, the path training runs.

Every backend is driven by ``RNNTJoint`` with ``fuse_loss_wer=True``, so the chunk loop, the
per-chunk length narrowing and the memory bookkeeping are the production ones rather than a
reimplementation. ``--fused-batch-size`` accepts several values to sweep the chunk size.
"""

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
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
        nargs="+",
        default=[4],
        help="Maximum samples per joint chunk. Several values sweep them.",
    )
    parser.add_argument(
        "--max-joint-rows",
        type=int,
        default=None,
        help="Flash workspace row budget. Defaults to the loss's own value.",
    )
    parser.add_argument(
        "--clamp",
        type=float,
        default=-1.0,
        help="Gradient clamp. Above zero this activates the per-sample rescaling both backends use.",
    )
    parser.add_argument(
        "--encoder-hidden",
        type=int,
        default=512,
        help="Encoder projection width. Sweeping it separates terms that scale with it from joint_hidden.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.0, help="Joint dropout probability.")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
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


def _build_loss(method, blank, dtype, max_joint_rows=None, clamp=-1.0):
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    kwargs = None
    if method == "warprnnt_numba":
        kwargs = {"fastemit_lambda": 0.0, "clamp": clamp}
    elif method == "graph_rnnt":
        kwargs = {
            "use_grid_implementation": True,
            "use_triton": True,
            "cast_to_float32": False,
        }
    elif method == "flash_rnnt":
        kwargs = {
            "fastemit_lambda": 0.0,
            "clamp": clamp,
        }
        if max_joint_rows is not None:
            kwargs["max_joint_rows"] = max_joint_rows
    # Each backend decides for itself whether it can consume the joint's dtype. Forcing FP32 here
    # would silently turn a narrow-precision arm into an FP32 one and make the comparison a
    # precision comparison rather than a backend comparison.
    return RNNTLoss(num_classes=blank, reduction="mean_batch", loss_name=method, loss_kwargs=kwargs)


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

    (max_chunk_batch,) = args.fused_batch_size
    joint = RNNTJoint(
        jointnet={
            "encoder_hidden": args.encoder_hidden,
            "pred_hidden": 640,
            "joint_hidden": 640,
            "activation": "relu",
            "dropout": args.dropout,
        },
        num_classes=blank,
        log_softmax=False,
        fuse_loss_wer=True,
        fused_batch_size=max_chunk_batch,
    ).to(device=device, dtype=dtype)
    loss = _build_loss(args.method, blank, args.dtype, args.max_joint_rows, args.clamp).to(device)
    joint.set_loss(loss)
    joint.set_wer(object())
    encoder = torch.randn(batch, args.encoder_hidden, max_source, device=device, dtype=dtype, requires_grad=True)
    predictor = torch.randn(batch, 640, max_target + 1, device=device, dtype=dtype, requires_grad=True)
    targets = torch.randint(0, blank, (batch, max_target), device=device, dtype=torch.int64)
    device_length_batches = [
        (
            torch.tensor(source, device=device, dtype=torch.int64),
            torch.tensor(target, device=device, dtype=torch.int64),
        )
        for source, target in length_batches
    ]

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
        value = joint(
            encoder_outputs=encoder[:, :, :profile_source],
            decoder_outputs=predictor[:, :, : profile_target + 1],
            encoder_lengths=source_lengths,
            transcripts=targets[:, :profile_target],
            transcript_lengths=target_lengths,
        )[0]
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
        "profile": args.profile,
        "batch_size": batch,
        "fused_batch_size": max_chunk_batch,
        "dropout": args.dropout,
        "dtype": args.dtype,
        "encoder_hidden": args.encoder_hidden,
        "max_joint_rows": args.max_joint_rows,
        "clamp": args.clamp,
        "profile_batches": len(length_batches),
        "chunks": -(-batch // max_chunk_batch),
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
            for fused_batch_size in args.fused_batch_size:
                _run_one(args, script, results, profile, method, fused_batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


def _run_one(args, script, results, profile, method, fused_batch_size) -> None:
    command = [
        sys.executable,
        str(script),
        "--job",
        "--method",
        method,
        "--profile",
        profile,
        "--fused-batch-size",
        str(fused_batch_size),
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
    if args.max_joint_rows is not None:
        command.extend(("--max-joint-rows", str(args.max_joint_rows)))
    command.extend(("--clamp", str(args.clamp)))
    command.extend(("--encoder-hidden", str(args.encoder_hidden)))
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
                "fused_batch_size": fused_batch_size,
                "status": "error",
                "error": process.stderr[-4000:],
            }
        )
        print(f"{profile:7} {method:17} fbs={fused_batch_size:<4} FAILED", flush=True)
        return
    results.append(json.loads(process.stdout.strip().splitlines()[-1]))
    print(
        f"{profile:7} {method:17} fbs={fused_batch_size:<4} "
        f"{results[-1]['host_p50_ms']:8.2f} ms {results[-1]['samples_per_second']:8.2f} samples/s "
        f"{results[-1]['incremental_allocated_gib']:6.2f} GiB",
        flush=True,
    )


def main() -> None:
    args = _parser().parse_args()
    if args.job:
        _run_job(args)
    else:
        _run_all(args)


if __name__ == "__main__":
    main()
