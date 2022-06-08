# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.


import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        "2" -> ordinal { integer: "второе" } }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        alternative_formats: alternative format for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: dict, alternative_formats: dict, deterministic=False):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        one_thousand_alternative = alternative_formats['one_thousand_alternative']
        separators = alternative_formats['separators']

        ordinal = number_names['ordinal_number_names']

        ordinal |= ordinal @ one_thousand_alternative
        ordinal_numbers = separators @ ordinal

        # to handle cases like 2-ая
        endings = pynini.string_file(get_abs_path("data/numbers/ordinal_endings.tsv"))
        not_dash = pynini.closure(pynini.difference(NEMO_SIGMA, "-"))
        del_ending = pynini.cdrewrite(pynini.cross("-" + not_dash, ""), "", "[EOS]", NEMO_SIGMA)
        ordinal_numbers_marked = (
            ((separators @ ordinal).optimize() + pynini.accep("-") + not_dash).optimize()
            @ (NEMO_SIGMA + endings).optimize()
            @ del_ending
        ).optimize()

        self.ordinal_numbers = ordinal_numbers
        # "03" -> remove leading zeros and verbalize
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.ordinal_numbers_with_leading_zeros = (leading_zeros + ordinal_numbers).optimize()

        final_graph = (ordinal_numbers | ordinal_numbers_marked).optimize()
        final_graph = pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
