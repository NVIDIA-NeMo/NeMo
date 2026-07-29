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

import builtins
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import torch.nn as nn

_COMPAT_PATH = Path(__file__).resolve().parents[3] / "nemo/collections/speechlm2/parts/automodel_compat.py"
_COMPAT_SPEC = importlib.util.spec_from_file_location("_speechlm2_automodel_compat", _COMPAT_PATH)
_COMPAT_MODULE = importlib.util.module_from_spec(_COMPAT_SPEC)
_COMPAT_SPEC.loader.exec_module(_COMPAT_MODULE)
install_nemotron_h_layer_compatibility = _COMPAT_MODULE.install_nemotron_h_layer_compatibility


class _NativeNemotronV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({"0": nn.Linear(2, 2), "1": nn.ReLU()})


class _HuggingFaceNemotronH(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([nn.Linear(2, 2), nn.ReLU()])


def _install_fake_legacy_automodel(monkeypatch, *, has_upstream_fix=False, has_strategy=True):
    class LegacyNemotronHParallelizationStrategy:
        """Minimal reproduction of Automodel 0.4's hard-coded layer access."""

        def parallelize(self, model):
            layers = model.backbone.layers
            visited = list(layers)
            layers[0] = nn.Identity()
            return visited

    parallelizer = ModuleType("nemo_automodel.components.distributed.parallelizer")
    if has_strategy:
        parallelizer.NemotronHParallelizationStrategy = LegacyNemotronHParallelizationStrategy
    if has_upstream_fix:
        parallelizer._nemotronh_decoder_blocks = lambda model: model

    distributed = ModuleType("nemo_automodel.components.distributed")
    distributed.parallelizer = parallelizer
    components = ModuleType("nemo_automodel.components")
    components.distributed = distributed
    automodel = ModuleType("nemo_automodel")
    automodel.components = components

    monkeypatch.setitem(sys.modules, "nemo_automodel", automodel)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components", components)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed", distributed)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.parallelizer", parallelizer)
    return parallelizer


def test_legacy_parallelizer_supports_native_nemotron_v3(monkeypatch):
    parallelizer = _install_fake_legacy_automodel(monkeypatch)

    assert install_nemotron_h_layer_compatibility() is True

    model = _NativeNemotronV3()
    visited = parallelizer.NemotronHParallelizationStrategy().parallelize(model)

    assert len(visited) == 2
    assert all(isinstance(layer, nn.Module) for layer in visited)
    assert isinstance(model.model.layers["0"], nn.Identity)
    assert not hasattr(model, "backbone")


def test_legacy_parallelizer_preserves_huggingface_layout(monkeypatch):
    parallelizer = _install_fake_legacy_automodel(monkeypatch)

    install_nemotron_h_layer_compatibility()

    model = _HuggingFaceNemotronH()
    original_backbone = model.backbone
    visited = parallelizer.NemotronHParallelizationStrategy().parallelize(model)

    assert len(visited) == 2
    assert model.backbone is original_backbone
    assert isinstance(model.backbone.layers[0], nn.Identity)


def test_compatibility_is_not_installed_when_upstream_fix_exists(monkeypatch):
    parallelizer = _install_fake_legacy_automodel(monkeypatch, has_upstream_fix=True)
    original_parallelize = parallelizer.NemotronHParallelizationStrategy.parallelize

    assert install_nemotron_h_layer_compatibility() is False
    assert parallelizer.NemotronHParallelizationStrategy.parallelize is original_parallelize


def test_missing_parallelizer_is_best_effort(monkeypatch, caplog):
    original_import = builtins.__import__

    def import_without_parallelizer(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "nemo_automodel.components.distributed" and "parallelizer" in fromlist:
            raise ImportError("parallelizer moved")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_parallelizer)
    with caplog.at_level(logging.WARNING):
        assert install_nemotron_h_layer_compatibility() is False
    assert "Nemotron-V3 layer compatibility not installed: parallelizer moved" in caplog.text


def test_missing_strategy_is_best_effort(monkeypatch, caplog):
    _install_fake_legacy_automodel(monkeypatch, has_strategy=False)

    with caplog.at_level(logging.WARNING):
        assert install_nemotron_h_layer_compatibility() is False
    assert "NemotronHParallelizationStrategy" in caplog.text
