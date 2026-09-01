# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Stable capability contract for optional VoiceChat output channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AuxiliaryOutputCapabilities:
    """Which optional VoiceChat output channels this run can produce.

    Both backends surface both auxiliary channels, so a head being present in
    the checkpoint is the same thing as its output being available.
    """

    has_asr_head: bool
    has_function_head: bool

    def to_dict(self) -> dict[str, bool]:
        """Return a JSON-serializable representation with stable field names."""
        return asdict(self)


def _head_flag(stt_model: Any, *names: str) -> bool:
    """Whether any of *names* marks a head as present on this checkpoint.

    Native checkpoints expose the ASR head as ``predict_user_text``;
    converted Nemotron configs use ``use_asr_head``. The attribute is
    authoritative when the model defines one; the config is the fallback.
    """
    for name in names:
        value = getattr(stt_model, name, None)
        if value is not None:
            return bool(value)
    cfg = getattr(stt_model, "cfg", None)
    if cfg is None:
        return False
    return any(bool(cfg.get(name, False)) for name in names)


def derive_auxiliary_output_capabilities(stt_model: Any) -> AuxiliaryOutputCapabilities:
    """Derive optional-channel capabilities from the checkpoint's heads."""
    return AuxiliaryOutputCapabilities(
        has_asr_head=_head_flag(stt_model, "predict_user_text", "use_asr_head"),
        has_function_head=_head_flag(stt_model, "use_function_head"),
    )
