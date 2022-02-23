# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_preserve_order,
)
from nemo_text_processing.text_normalization.es.graph_utils import ones, shift_cardinal_gender
from nemo_text_processing.text_normalization.es.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    unit_plural_fem = pynini.string_file(get_abs_path("data/measures/measurements_plural_fem.tsv"))
    unit_plural_masc = pynini.string_file(get_abs_path("data/measures/measurements_plural_masc.tsv"))

    unit_singular_fem = pynini.project(unit_plural_fem, "input")
    unit_singular_masc = pynini.project(unit_plural_masc, "input")

    unit_plural_fem = pynini.project(unit_plural_fem, "output")
    unit_plural_masc = pynini.project(unit_plural_masc, "output")

    PYNINI_AVAILABLE = True

except (ModuleNotFoundError, ImportError):
    unit_plural_fem = None
    unit_plural_masc = None

    unit_singular_fem = None
    unit_singular_masc = None

    PYNINI_AVAILABLE = False


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { cardinal { integer: "dos" units: "gramos" } } -> "dos gramos"
        measure { cardinal { integer_part: "dos" quantity: "millones" units: "gramos" } } -> "dos millones de gramos"

    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst, fraction: GraphFst, deterministic: bool):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        graph_decimal = decimal.fst
        graph_cardinal = cardinal.fst
        graph_fraction = fraction.fst

        unit_masc = (unit_plural_masc | unit_singular_masc) + pynini.closure(
            NEMO_WHITE_SPACE + "por" + pynini.closure(NEMO_NOT_QUOTE, 1), 0, 1
        )
        unit_masc |= "por" + pynini.closure(NEMO_NOT_QUOTE, 1)
        unit_masc = pynutil.delete("units: \"") + (pynini.closure(NEMO_NOT_QUOTE) @ unit_masc) + pynutil.delete("\"")

        unit_fem = (unit_plural_fem | unit_singular_fem) + pynini.closure(
            NEMO_WHITE_SPACE + "por" + pynini.closure(NEMO_NOT_QUOTE, 1), 0, 1
        )
        unit_fem = pynutil.delete("units: \"") + (pynini.closure(NEMO_NOT_QUOTE) @ unit_fem) + pynutil.delete("\"")

        graph_masc = (graph_cardinal | graph_decimal | graph_fraction) + NEMO_WHITE_SPACE + unit_masc
        graph_fem = (
            shift_cardinal_gender(graph_cardinal | graph_decimal | graph_fraction) + NEMO_WHITE_SPACE + unit_fem
        )
        graph = graph_masc | graph_fem

        graph = (
            pynini.cdrewrite(
                pynutil.insert(" de"), "quantity: \"" + pynini.closure(NEMO_NOT_QUOTE, 1), "\"", NEMO_SIGMA
            )
            @ graph
        )  # billones de xyz

        graph @= pynini.cdrewrite(pynini.cross(ones, "uno"), "", NEMO_WHITE_SPACE + "por", NEMO_SIGMA)

        # To manage alphanumeric combonations ("a-8, 5x"), we let them use a weighted default path.
        alpha_num_unit = pynutil.delete("units: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        graph_alpha_num = pynini.union(
            (graph_cardinal | graph_decimal) + NEMO_SPACE + alpha_num_unit,
            alpha_num_unit + delete_extra_space + (graph_cardinal | graph_decimal),
        )

        graph |= pynutil.add_weight(graph_alpha_num, 0.01)

        graph += delete_preserve_order

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
