# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_NOT_SPACE, GraphFst
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class WordFst(GraphFst):
    """
    Finite state transducer for classifying word. Considers sentence boundary exceptions.
        e.g. sleep -> tokens { name: "sleep" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)

        punct = PunctuationFst().graph
        self.graph = pynini.closure(pynini.difference(NEMO_NOT_SPACE, punct.project("input")), 1)

        if not deterministic:
            self.graph = pynini.closure(
                pynini.difference(
                    self.graph, pynini.union("$", "€", "₩", "£", "¥", "#", "$", "%") + pynini.closure(NEMO_DIGIT, 1)
                ),
                1,
            )

        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
