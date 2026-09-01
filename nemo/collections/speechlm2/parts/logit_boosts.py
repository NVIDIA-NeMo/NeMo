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

"""Inference-time special-token logit boosts for the duplex text channels.

Both the PyTorch and vLLM runtimes bias the pad/BOS/EOS logits of the agent
text channel and the user (ASR) channel before picking a token. Keeping the
values and the arithmetic here means the two cannot disagree about what
``inference_pad_boost`` and friends do.

The boosts are read from config rather than passed as arguments because the
models that consume them are also used offline, where there is no inference
wrapper to thread them through. See
``nemo/collections/speechlm2/inference/model_wrappers/config_overrides.py``
for how the streaming pipeline sets them.
"""

from dataclasses import dataclass
from typing import Any

import torch

AGENT_BOOST_KEYS = ("inference_pad_boost", "inference_bos_boost", "inference_eos_boost")
USER_BOOST_KEYS = (
    "inference_user_pad_boost",
    "inference_user_bos_boost",
    "inference_user_eos_boost",
)


@dataclass(frozen=True)
class LogitBoosts:
    """Additive logit offsets for one channel's pad/BOS/EOS tokens.

    ``None`` and ``0.0`` both mean "leave this token alone"; config treats a
    falsy value as unset.
    """

    pad: float | None = None
    bos: float | None = None
    eos: float | None = None

    def __bool__(self) -> bool:
        return bool(self.pad or self.bos or self.eos)

    def as_dict(self) -> dict[str, float | None]:
        return {"pad": self.pad, "bos": self.bos, "eos": self.eos}

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "LogitBoosts":
        if not values:
            return cls()
        return cls(
            pad=values.get("pad"),
            bos=values.get("bos"),
            eos=values.get("eos"),
        )

    @staticmethod
    def _cfg_get(cfg: Any, key: str) -> Any:
        """Read *key* from an OmegaConf/dict config or an HF ``PretrainedConfig``."""
        if hasattr(cfg, "get"):
            try:
                return cfg.get(key, None)
            except TypeError:
                pass
        return getattr(cfg, key, None)

    @classmethod
    def _from_cfg(cls, cfg: Any, keys: tuple[str, str, str]) -> "LogitBoosts":
        if cfg is None:
            return cls()
        pad, bos, eos = (cls._cfg_get(cfg, key) for key in keys)
        return cls(
            pad=float(pad) if pad else None,
            bos=float(bos) if bos else None,
            eos=float(eos) if eos else None,
        )

    @classmethod
    def agent_from_cfg(cls, cfg: Any) -> "LogitBoosts":
        """Boosts for the agent text channel (``inference_*_boost``)."""
        return cls._from_cfg(cfg, AGENT_BOOST_KEYS)

    @classmethod
    def user_from_cfg(cls, cfg: Any) -> "LogitBoosts":
        """Boosts for the user/ASR channel (``inference_user_*_boost``)."""
        return cls._from_cfg(cfg, USER_BOOST_KEYS)


def apply_logit_boosts(
    logits: torch.Tensor,
    boosts: LogitBoosts,
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> torch.Tensor:
    """Add *boosts* to the matching vocabulary entries of *logits*, in place.

    Indexing on the last dimension, so this accepts both the ``(B, T, V)``
    tensors the PyTorch heads produce and the ``(V,)`` slice a vLLM
    logits processor receives.
    """
    if not boosts:
        return logits
    for token_id, value in ((pad_id, boosts.pad), (bos_id, boosts.bos), (eos_id, boosts.eos)):
        if value:
            logits[..., token_id] += value
    return logits
