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
import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SPACE, GraphFst, delete_extra_space
from nemo_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

articles = pynini.union("de", "del", "el", "del año")
delete_leading_zero = (pynutil.delete("0") | (NEMO_DIGIT - "0")) + NEMO_DIGIT
month_numbers = pynini.string_file(get_abs_path("data/dates/months.tsv"))


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "01.04.2010" -> date { day: "un" month: "enero" year: "dos mil diez" preserve_order: true }
        "marzo 4 2000" -> date { month: "marzo" day: "cuatro" year: "dos mil" }
        "1990-20-01" -> date { year: "mil novecientos noventa" day: "veinte" month: "enero" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        number_to_month = month_numbers.optimize()
        month_graph = pynini.project(number_to_month, "output")

        numbers = cardinal.graph
        optional_leading_zero = delete_leading_zero | NEMO_DIGIT

        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ numbers
        day = (pynutil.insert("day: \"") + digit_day + pynutil.insert("\"")).optimize()

        digit_month = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        number_to_month = digit_month @ number_to_month

        month_name = (pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")).optimize()
        month_number = (pynutil.insert("month: \"") + number_to_month + pynutil.insert("\"")).optimize()

        # prefer cardinal over year
        year = (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 1, 3)  # 90, 990, 1990
        year @= numbers
        self.year = year

        year_only = pynutil.insert("year: \"") + year + pynutil.insert("\"")
        year_with_articles = (
            pynutil.insert("year: \"") + pynini.closure(articles + NEMO_SPACE, 0, 1) + year + pynutil.insert("\"")
        )

        graph_dmy = (
            day
            + pynini.closure(pynutil.delete(" de"))
            + NEMO_SPACE
            + month_name
            + pynini.closure(NEMO_SPACE + year_with_articles, 0, 1)
        )

        graph_mdy = (  # English influences on language
            month_name + delete_extra_space + day + pynini.closure(NEMO_SPACE + year_with_articles, 0, 1)
        )

        separators = [".", "-", "/"]
        for sep in separators:
            year_optional = pynini.closure(pynini.cross(sep, NEMO_SPACE) + year_only, 0, 1)
            new_graph = day + pynini.cross(sep, NEMO_SPACE) + month_number + year_optional
            graph_dmy |= new_graph
            if not deterministic:
                new_graph = month_number + pynini.cross(sep, NEMO_SPACE) + day + year_optional
                graph_mdy |= new_graph

        dash = "-"
        day_optional = pynini.closure(pynini.cross(dash, NEMO_SPACE) + day, 0, 1)
        graph_ymd = NEMO_DIGIT ** 4 @ year_only + pynini.cross(dash, NEMO_SPACE) + month_number + day_optional

        final_graph = graph_dmy + pynutil.insert(" preserve_order: true")
        final_graph |= graph_ymd
        final_graph |= graph_mdy

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
