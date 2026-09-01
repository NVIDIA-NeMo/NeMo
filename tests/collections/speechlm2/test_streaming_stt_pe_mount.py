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

"""The ParallelExpertEncoder mount must exist on BOTH streaming model paths.

`StreamingSTTModelAutomodel` subclasses `StreamingSTTModel` but builds its own perception module in
`configure_model` instead of reusing the base class's `__init__`, so the mount is NOT inherited --
it has to be repeated. Without it, `model.pe_encoder_path` / `model.parallel_expert_encoder` are
silently ignored under Automodel and training runs with a plain ASR encoder.

These are source-level checks so they stay cheap and do not need checkpoints or a GPU.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "nemo/collections/speechlm2/models/streaming_stt_model.py"
AUTOMODEL = REPO_ROOT / "nemo/collections/speechlm2/models/streaming_stt_model_automodel.py"

MOUNTS = ("setup_parallel_expert_encoder", "setup_parallel_expert_encoder_from_checkpoints")


def _function_node(path, name):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _called_names(node):
    """{callee_name: first_lineno} for both plain and method calls inside `node`."""
    out = {}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name and name not in out:
            out[name] = sub.lineno
    return out


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, fn",
    [(BASE, "__init__"), (AUTOMODEL, "configure_model")],
    ids=["StreamingSTTModel", "StreamingSTTModelAutomodel"],
)
def test_both_paths_mount_the_parallel_expert_encoder(path, fn):
    calls = _called_names(_function_node(path, fn))
    assert "setup_perception" in calls, f"{path.name}:{fn} no longer builds perception here"
    for mount in MOUNTS:
        assert mount in calls, f"{path.name}:{fn} does not mount the PE encoder via {mount}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, fn",
    [(BASE, "__init__"), (AUTOMODEL, "configure_model")],
    ids=["StreamingSTTModel", "StreamingSTTModelAutomodel"],
)
def test_pe_mount_runs_after_perception_and_before_freeze(path, fn):
    """Order matters: the mount replaces `perception.encoder`, and `_apply_freeze_config` must run
    afterwards so a composite encoder's `apply_internal_freeze` is applied to the MOUNTED encoder
    rather than the one the mount discards."""
    calls = _called_names(_function_node(path, fn))
    assert calls["setup_perception"] < calls[MOUNTS[0]], "PE mount runs before perception is built"
    assert calls[MOUNTS[0]] < calls["_apply_freeze_config"], "PE mount runs after the freeze config"
