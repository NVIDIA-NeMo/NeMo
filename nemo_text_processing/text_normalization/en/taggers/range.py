# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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


from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst, convert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class RangeFst(GraphFst):
    """
    This class is a composite class of two other class instances
    
    Args:
        time: composed tagger and verbalizer
        date: composed tagger and verbalizer
        cardinal: tagger
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
        lm: whether to use for hybrid LM
    """

    def __init__(
        self, time: GraphFst, date: GraphFst, cardinal: GraphFst, deterministic: bool = True, lm: bool = False
    ):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)
        self.graph = time + delete_space + pynini.cross("-", " to ") + delete_space + time
        # only 4 digit year for date
        date_year = (NEMO_DIGIT ** 4 + pynini.closure(pynini.accep("s"), 0, 1)) @ date

        year_to_year_graph = date_year + delete_space + pynini.cross("-", " to ") + delete_space + date_year
        self.graph |= year_to_year_graph

        cardinal = cardinal.graph_with_and
        # cardinal ----
        # excluding 4-digit cardinal
        up_to_three_or_five_digits = (NEMO_DIGIT ** (1, 3)) | (NEMO_DIGIT ** (5, ...))
        up_to_three_or_five_digits = pynini.compose(up_to_three_or_five_digits, cardinal)
        cardinal_to_cardinal_graph = (
            up_to_three_or_five_digits
            + delete_space
            + pynini.cross("-", " to ")
            + delete_space
            + up_to_three_or_five_digits
        )

        if not deterministic and not lm:
            cardinal_to_cardinal_graph |= cardinal + delete_space + pynini.cross("-", " to ") + delete_space + cardinal

        range_graph = cardinal_to_cardinal_graph | (
            cardinal + delete_space + pynini.cross(":", " to ") + delete_space + cardinal
        )
        for x in [" x ", "x"]:
            range_graph |= cardinal + pynini.closure(pynini.cross(x, pynini.union(" by ", " times ")) + cardinal, 1)
        for x in ["*", " * "]:
            range_graph |= cardinal + pynini.closure(pynini.cross(x, " times ") + cardinal, 1)

        if not deterministic or lm:
            range_graph |= cardinal + delete_space + pynini.cross("-", " minus ") + delete_space + cardinal

            # supports "No. 12" -> "Number 12"
            range_graph |= (
                (pynini.cross(pynini.union("NO", "No"), "Number") | pynini.cross("no", "number"))
                + pynini.closure(pynini.union(". ", " "), 0, 1)
                + cardinal
            )

            for x in ["+", " + "]:
                range_graph |= cardinal + pynini.closure(pynini.cross(x, " plus ") + cardinal, 1)
            for x in ["/", " / "]:
                range_graph |= cardinal + pynini.closure(pynini.cross(x, " divided by ") + cardinal, 1)

        self.graph |= range_graph
        # ---- cardinal

        self.graph = self.graph.optimize()
        graph = pynutil.insert("name: \"") + convert_space(self.graph).optimize() + pynutil.insert("\"")
        self.fst = graph.optimize()
