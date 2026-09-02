# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from pathlib import Path
from types import SimpleNamespace

from nemo.collections.asr.parts.utils import wfst_utils


def test_kaldi_word_lattice_draw_moves_rendered_file(monkeypatch, tmp_path):
    class Source:
        def __init__(self, source):
            self.source = source

        def __str__(self):
            return self.source

    class Digraph:
        def __init__(self, graph_attr):
            self.graph_attr = graph_attr
            self.body = []

        def render(self, filename, directory, format, cleanup):
            rendered_path = Path(directory) / f"{filename}.{format}"
            rendered_path.write_text("rendered lattice", encoding="utf-8")
            return str(rendered_path)

    monkeypatch.setattr(wfst_utils, "_KALDIFST_AVAILABLE", True)
    monkeypatch.setattr(wfst_utils, "_GRAPHVIZ_AVAILABLE", True)
    monkeypatch.setattr(
        wfst_utils,
        "kaldifst",
        SimpleNamespace(draw=lambda *args, **kwargs: "digraph tree {\n0 -> 1\n}"),
        raising=False,
    )
    monkeypatch.setattr(wfst_utils, "graphviz", SimpleNamespace(Source=Source, Digraph=Digraph), raising=False)
    monkeypatch.setattr(wfst_utils, "_is_notebook", lambda: False)

    lattice = wfst_utils.KaldiWordLattice.__new__(wfst_utils.KaldiWordLattice)
    lattice._lattice = object()
    lattice._symbol_table = None
    lattice._auxiliary_tables = None
    output_path = tmp_path / "lattice.svg"

    lattice.draw(filename=output_path)

    assert output_path.read_text(encoding="utf-8") == "rendered lattice"
