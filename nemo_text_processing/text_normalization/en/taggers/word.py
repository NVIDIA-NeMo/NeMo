# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)

try:
    import pynini
    from pynini.lib import pynutil
    from pynini.examples import plurals

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

        symbols_to_exclude = (pynini.union("$", "€", "₩", "£", "¥", "#", "%") | NEMO_DIGIT).optimize()
        graph = pynini.closure(pynini.difference(NEMO_NOT_SPACE, symbols_to_exclude), 1)

        # leave phones of format [HH AH0 L OW1] untouched
        phoneme_unit = pynini.closure(NEMO_ALPHA, 1) + pynini.closure(NEMO_DIGIT)
        phoneme = (
            pynini.accep(pynini.escape("["))
            + pynini.closure(phoneme_unit + pynini.accep(" "))
            + phoneme_unit
            + pynini.accep(pynini.escape("]"))
        )

        if not deterministic:
            phoneme = (
                pynini.accep(pynini.escape("["))
                + pynini.closure(pynini.accep(" "), 0, 1)
                + pynini.closure(phoneme_unit + pynini.accep(" "))
                + phoneme_unit
                + pynini.closure(pynini.accep(" "), 0, 1)
                + pynini.accep(pynini.escape("]"))
            )
        self.graph = plurals._priority_union(convert_space(phoneme), graph, NEMO_SIGMA)
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
