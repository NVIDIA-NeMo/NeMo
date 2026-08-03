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

import logging
from functools import wraps

from torch import nn

logger = logging.getLogger(__name__)

_COMPATIBILITY_MARKER = "_nemo_speech_nemotron_h_layer_compatibility"
_HF_CONFIG_KWARGS = ("cache_dir", "revision", "token", "local_files_only", "subfolder")


def remove_automodel_backend_for_hf_fallback(
    model_path_or_name: str,
    kwargs: dict,
    *,
    trust_remote_code: bool = False,
) -> bool:
    """Remove Automodel-only backend configuration before its Hugging Face fallback.

    Automodel consumes ``backend`` for native model implementations but
    forwards it to Hugging Face model constructors on the fallback path. Those
    constructors do not accept this Automodel-specific keyword. Resolve the same
    native-vs-HF choice up front and remove only the incompatible fallback kwarg.

    Returns:
        ``True`` when ``backend`` was removed; otherwise ``False``.
    """
    if "backend" not in kwargs:
        return False

    try:
        from nemo_automodel._transformers.model_init import get_is_hf_model
        from transformers import AutoConfig

        config = kwargs.get("config")
        if config is None:
            config_kwargs = {key: kwargs[key] for key in _HF_CONFIG_KWARGS if key in kwargs}
            config = AutoConfig.from_pretrained(
                model_path_or_name,
                trust_remote_code=trust_remote_code,
                **config_kwargs,
            )

        uses_hf_model = kwargs.get("quantization_config") is not None or get_is_hf_model(
            config,
            force_hf=bool(kwargs.get("force_hf", False)),
        )
    except Exception as error:
        logger.warning("Could not determine Automodel implementation; leaving backend unchanged: %s", error)
        return False

    if not uses_hf_model:
        return False

    # TODO(Dongji): Remove after Automodel consumes backend before entering its HF fallback.
    kwargs.pop("backend")
    logger.warning("Ignoring Automodel backend configuration for Hugging Face fallback model %s", model_path_or_name)
    return True


class _ModuleDictLayersAdapter:
    """Expose a ModuleDict through the list operations used by Automodel 0.4."""

    def __init__(self, layers: nn.ModuleDict):
        self._layers = layers
        # ModuleDict preserves insertion order. Native Nemotron-V3 registers
        # its numeric string keys in decoder order; sorting would put "10"
        # before "2" and produce the wrong layer sequence.
        self._keys = tuple(layers)

    def __iter__(self):
        return iter(self._layers.values())

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, index: int):
        return self._layers[self._key(index)]

    def __setitem__(self, index: int, layer: nn.Module):
        self._layers[self._key(index)] = layer

    def _key(self, index: int) -> str:
        if not isinstance(index, int):
            raise TypeError(f"ModuleDict layer indices must be integers, got {type(index).__name__}")
        return self._keys[index]


class _BackboneLayersAdapter:
    """Minimal adapter for Automodel 0.4's exact ``model.backbone.layers`` access.

    This deliberately is not an ``nn.Module`` and does not emulate a complete
    Hugging Face backbone. The pinned parallelizer only reads ``.layers``; the
    real Automodel 0.4 smoke test guards that narrow contract.
    """

    def __init__(self, layers):
        self.layers = layers


def install_nemotron_h_layer_compatibility() -> bool:
    """Make Automodel 0.4's Nemotron-H parallelizer accept the native Nemotron-V3 layout.

    Automodel 0.4 assumes every ``NemotronHForCausalLM`` stores decoder blocks
    in ``model.backbone.layers``. The native Nemotron-V3 implementation stores
    them in ``model.model.layers`` as a ``ModuleDict``. Temporarily expose that
    container through the legacy interface while the old parallelizer runs.

    Returns:
        ``True`` when the compatibility wrapper was installed; ``False`` when
        Automodel already contains the upstream fix or the wrapper was installed
        previously.
    """
    try:
        from nemo_automodel.components.distributed import parallelizer
    except ImportError as error:
        logger.warning("Nemotron-V3 layer compatibility not installed: %s", error)
        return False

    if hasattr(parallelizer, "_nemotronh_decoder_blocks"):
        return False

    try:
        strategy_cls = parallelizer.NemotronHParallelizationStrategy
    except AttributeError as error:
        logger.warning("Nemotron-V3 layer compatibility not installed: %s", error)
        return False

    original_parallelize = strategy_cls.parallelize
    if getattr(original_parallelize, _COMPATIBILITY_MARKER, False):
        return False

    @wraps(original_parallelize)
    def parallelize_with_native_layout(self, model, *args, **kwargs):
        if hasattr(model, "backbone"):
            return original_parallelize(self, model, *args, **kwargs)

        inner_model = getattr(model, "model", None)
        layers = getattr(inner_model, "layers", None)
        if isinstance(layers, nn.ModuleDict):
            legacy_layers = _ModuleDictLayersAdapter(layers)
        elif isinstance(layers, nn.ModuleList):
            legacy_layers = layers
        else:
            return original_parallelize(self, model, *args, **kwargs)

        model.backbone = _BackboneLayersAdapter(legacy_layers)
        try:
            return original_parallelize(self, model, *args, **kwargs)
        finally:
            del model.backbone

    setattr(parallelize_with_native_layout, _COMPATIBILITY_MARKER, True)
    strategy_cls.parallelize = parallelize_with_native_layout
    # TODO(Dongji): Remove this shim after Speech pins an Automodel build containing PR #2638.
    logger.warning("Installed temporary native Nemotron-V3 compatibility for the Automodel 0.4 parallelizer")
    return True
