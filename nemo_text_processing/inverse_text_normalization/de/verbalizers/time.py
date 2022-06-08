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

import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, NEMO_DIGIT, GraphFst, delete_space
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "8" minutes: "30" zone: "e s t" } -> 08:30 Uhr est
        time { hours: "8" } -> 8 Uhr
        time { hours: "8" minutes: "30" seconds: "10" } -> 08:30:10 Uhr 
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")

        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_DIGIT, 1) + pynutil.delete("\"")
        zone = (
            pynutil.delete("zone: \"") + pynini.closure(NEMO_ALPHA + delete_space) + NEMO_ALPHA + pynutil.delete("\"")
        )
        optional_zone = pynini.closure(pynini.accep(" ") + zone, 0, 1)
        graph = (
            delete_space
            + pynutil.insert(":")
            + (minute @ add_leading_zero_to_double_digit)
            + pynini.closure(delete_space + pynutil.insert(":") + (second @ add_leading_zero_to_double_digit), 0, 1)
            + pynutil.insert(" Uhr")
            + optional_zone
        )
        graph_h = hour + pynutil.insert(" Uhr") + optional_zone
        graph_hm = hour @ add_leading_zero_to_double_digit + graph
        graph_hms = hour @ add_leading_zero_to_double_digit + graph
        final_graph = graph_hm | graph_hms | graph_h
        self.fst = self.delete_tokens(final_graph).optimize()
