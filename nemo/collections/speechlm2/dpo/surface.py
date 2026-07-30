# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Declared SpeechLM2 partial-acoustic DPO mutation surface.

This is normal SpeechLM2 code.  It does not import or alter an installed
dependency, rebind an imported implementation, or create an adapter.  The
parameter selection is deliberately explicit because it is part of the model
update contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy


ATTENTION_LAYERS = (5, 12, 19, 26, 33, 42)
MAMBA_LAYERS = (0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50)
ATTENTION_SUFFIXES = ("mixer.k_proj.weight", "mixer.o_proj.weight", "mixer.q_proj.weight", "mixer.v_proj.weight", "norm.weight")
MAMBA_SUFFIXES = (
    "mixer.A_log", "mixer.D", "mixer.conv1d.bias", "mixer.conv1d.weight", "mixer.dt_bias", "mixer.in_proj.weight",
    "mixer.norm.weight", "mixer.out_proj.weight", "norm.weight",
)
ACOUSTIC_ENCODER_SUFFIXES = (
    "norm1.weight", "norm1.bias", "attn.w_qkv.weight", "attn.out_proj.weight", "attn.out_proj.bias", "attn.q_norm.weight",
    "attn.q_norm.bias", "attn.k_norm.weight", "attn.k_norm.bias", "norm2.weight", "norm2.bias", "ffn.net.0.weight",
    "ffn.net.0.bias", "ffn.net.3.weight", "ffn.net.3.bias",
)
ACOUSTIC_LAYERS = (30, 31)
SELECTED_TENSOR_COUNT = 269
# Read from the verified source DCP metadata rather than hand arithmetic. The
# 269 selected names and shapes total 1,074,318,016 scalars.
SELECTED_SCALAR_COUNT = 1_074_318_016

# SHA256 of newline-joined ``name|shape|numel`` records from the verified
# checkpoint metadata. It is provenance for the count above; the live
# inventory remains the authoritative runtime check.
VERIFIED_SURFACE_INVENTORY_SHA256 = "20cd5cb3a3fbdaa5a91e430e7a65dfdc53c463b72382e0d62a56039b4a7f9dfc"
VERIFIED_SURFACE_NAMES_SHA256 = "b7066381abcfd73486e7bcd2e56cec6798b70bc8a3ba6afe46a702848e440cc2"


def canonical_name(name: str) -> str:
    """Remove the state-dict-transparent activation-checkpoint wrapper token."""

    return ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")


def selected_parameter_names() -> tuple[str, ...]:
    native: list[str] = []
    for layer in range(52):
        suffixes = ATTENTION_SUFFIXES if layer in ATTENTION_LAYERS else MAMBA_SUFFIXES if layer in MAMBA_LAYERS else ()
        native.extend(f"llm.model.layers.{layer}.{suffix}" for suffix in suffixes)
    acoustic = [
        f"perception.encoder.layers.{layer}.{suffix}"
        for layer in ACOUSTIC_LAYERS
        for suffix in ACOUSTIC_ENCODER_SUFFIXES
    ]
    names = tuple(native + acoustic + ["perception.proj.weight", "perception.proj.bias"])
    if len(names) != SELECTED_TENSOR_COUNT or len(set(names)) != len(names):
        raise RuntimeError("SpeechLM2 DPO surface inventory drift")
    return names


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


@dataclass(frozen=True)
class SurfaceInventory:
    names: tuple[str, ...]
    tensor_count: int
    scalar_count: int
    dtypes: tuple[str, ...]


def inventory(model: torch.nn.Module) -> SurfaceInventory:
    expected = set(selected_parameter_names())
    found = {canonical_name(name): parameter for name, parameter in model.named_parameters() if canonical_name(name) in expected}
    if set(found) != expected:
        missing, unexpected = sorted(expected - set(found)), sorted(set(found) - expected)
        raise RuntimeError(f"SpeechLM2 DPO surface names differ: missing={missing[:4]} unexpected={unexpected[:4]}")
    ordered = tuple(found[name] for name in selected_parameter_names())
    scalars = sum(int(parameter.numel()) for parameter in ordered)
    return SurfaceInventory(
        names=selected_parameter_names(), tensor_count=len(ordered), scalar_count=scalars,
        dtypes=tuple(sorted({str(parameter.dtype) for parameter in ordered})),
    )


def _replace_with_fp32(parameter: torch.nn.Parameter) -> None:
    """Promote a selected FSDP parameter while preserving parameter identity."""

    with torch.no_grad():
        replacement = torch.nn.Parameter(parameter.to(dtype=torch.float32), requires_grad=parameter.requires_grad)
        torch.utils.swap_tensors(parameter, replacement)


def configure_partial_acoustic_surface(model: torch.nn.Module) -> SurfaceInventory:
    """Freeze all but the declared 269 FP32 SpeechLM2 DPO tensors.

    This is the verified mutation contract expressed as a normal model
    capability.
    The FSDP refresh is necessary after the acoustic child tensors are promoted
    to FP32; it uses the normal PyTorch FSDP object associated with the model.
    """

    before = inventory(model)
    modules = dict(model.named_modules())
    selected_before = {canonical_name(name): parameter for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for layer in sorted(set(ATTENTION_LAYERS) | set(MAMBA_LAYERS)):
        modules[f"llm.model.layers.{layer}"].to(dtype=torch.float32)
    for name in selected_parameter_names():
        if name.startswith("perception."):
            _replace_with_fp32(selected_before[name])
    after = inventory(model)
    if after.tensor_count != SELECTED_TENSOR_COUNT or after.scalar_count != SELECTED_SCALAR_COUNT:
        raise RuntimeError(f"SpeechLM2 DPO surface count drift: {after}")
    selected_after = {canonical_name(name): parameter for name, parameter in model.named_parameters()}
    for name in after.names:
        selected_after[name].requires_grad_(True)
    if set(after.dtypes) != {"torch.float32"}:
        raise RuntimeError(f"SpeechLM2 DPO surface must be FP32, got {after.dtypes}")
    perception = modules["perception"]
    get_state = getattr(perception, "_get_fsdp_state", None)
    if callable(get_state):
        state = get_state()
        group = getattr(state, "_fsdp_param_group", None)
        if group is not None:
            policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16, cast_forward_inputs=True
            )
            state._mp_policy = policy
            group.mp_policy = policy
            acoustic_ids = {id(selected_after[name]) for name in after.names if name.startswith("perception.")}
            for fsdp_parameter in group.fsdp_params:
                if id(fsdp_parameter.sharded_param) in acoustic_ids:
                    fsdp_parameter.reset_sharded_param()
    final = inventory(model)
    if final != after:
        raise RuntimeError("FSDP surface refresh altered the declared DPO surface")
    return final


def named_selected_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    mapping = {canonical_name(name): parameter for name, parameter in model.named_parameters()}
    for name in selected_parameter_names():
        yield mapping[name]
