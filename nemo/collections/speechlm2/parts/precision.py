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
import os
from contextlib import contextmanager

import torch


def inference_precision_in_effect(
    *,
    allow_tf32: bool,
    matmul_precision: str,
    deterministic: bool,
) -> bool:
    """Whether torch is *currently* configured the way these arguments ask.

    Read straight back off torch rather than tracked in a module-level flag:
    the question that matters to a caller is "are my settings applied?", not
    "did someone call my context manager?". Reading the real state answers it
    without adding mutable global state of our own, stays correct if the
    switches were set some other way, and also catches a scope entered with
    the *wrong* config.

    The flash/mem-efficient SDP kernels that :func:`inference_precision`
    toggles are deliberately not checked: they follow from ``deterministic``,
    and can be legitimately disabled for unrelated reasons.
    """
    return (
        bool(torch.backends.cudnn.allow_tf32) == allow_tf32
        and bool(torch.backends.cuda.matmul.allow_tf32) == allow_tf32
        and torch.get_float32_matmul_precision() == matmul_precision
        and torch.are_deterministic_algorithms_enabled() == deterministic
    )


@contextmanager
def inference_precision(
    *,
    allow_tf32: bool = True,
    matmul_precision: str = "medium",
    deterministic: bool = False,
):
    """Apply the process-wide precision and determinism switches, then restore.

    These are torch-level globals rather than per-model state, so they have to
    be applied before any weights load and stay in effect for the whole run.
    Scoped rather than set-and-forget because leaving them on would silently
    change every later computation in the process: a deterministic run inside
    a test session would otherwise seed the RNGs and disable the fast
    attention kernels for everything that follows it.

    ``deterministic`` guarantees identical text outputs across runs for the
    same input even when sampling is enabled, by seeding the global RNGs and
    forcing deterministic CUDA kernels. It costs speed, and vLLM engines
    cannot honour it -- callers that offer a choice of engine must reject the
    combination themselves.
    """
    saved = (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    saved_cpu_rng = torch.get_rng_state() if deterministic else None
    saved_cuda_rng = torch.cuda.get_rng_state_all() if deterministic and torch.cuda.is_available() else None

    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision(matmul_precision)

    if deterministic:
        # CuBLAS reads this once, at the first CUDA matmul in the process, so
        # restoring it on exit would not undo anything. Left set: the only
        # cost is a 32 KB workspace reservation.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    torch.backends.cuda.enable_flash_sdp(not deterministic)
    torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)

    try:
        yield
    finally:
        (
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
            matmul,
            flash,
            mem_efficient,
            was_deterministic,
            warn_only,
        ) = saved
        torch.set_float32_matmul_precision(matmul)
        torch.backends.cuda.enable_flash_sdp(flash)
        torch.backends.cuda.enable_mem_efficient_sdp(mem_efficient)
        torch.use_deterministic_algorithms(was_deterministic, warn_only=warn_only)
        if saved_cpu_rng is not None:
            torch.set_rng_state(saved_cpu_rng)
        if saved_cuda_rng is not None:
            torch.cuda.set_rng_state_all(saved_cuda_rng)


@contextmanager
def fp32_precision():
    """
    Workaround for precision related issues when training with bf16-true PyTorch Lightning precision setting.
    In bf16-true, PTL changes PyTorch's default dtype, which may break implicit assumptions for some models.
    This context manager restores default float32 precision and runs the computation in float32 autocast context.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32):
            yield
    finally:
        torch.set_default_dtype(default_dtype)
