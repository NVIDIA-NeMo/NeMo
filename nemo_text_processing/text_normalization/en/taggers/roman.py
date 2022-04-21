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


from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, GraphFst
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying roman numbers:
        e.g. "IV" -> tokens { roman { integer: "four" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        roman_dict = load_labels(get_abs_path("data/roman/roman_to_spoken.tsv"))
        default_graph = pynini.string_map(roman_dict).optimize()
        default_graph = pynutil.insert("integer: \"") + default_graph + pynutil.insert("\"")
        graph_teens = pynini.string_map([x[0] for x in roman_dict[:19]]).optimize()

        # up to five digit roman numerals with a preceding name are converted to ordinal form
        names = get_names()
        graph = (
            pynutil.insert("key_the_ordinal: \"")
            + names
            + pynutil.insert("\"")
            + pynini.accep(" ")
            + graph_teens @ default_graph
        ).optimize()

        # single symbol roman numerals with preceding key words are converted to cardinal form
        key_words = pynini.string_map(load_labels(get_abs_path("data/roman/key_word.tsv"))).optimize()
        graph |= (
            pynutil.insert("key_cardinal: \"") + key_words + pynutil.insert("\"") + pynini.accep(" ") + default_graph
        ).optimize()

        if deterministic:
            # two digit roman numerals up to 49
            roman_to_cardinal = pynini.compose(
                pynini.closure(NEMO_ALPHA, 2),
                (
                    pynutil.insert("default_cardinal: \"default\" ")
                    + (pynini.string_map([x[0] for x in roman_dict[:50]]).optimize()) @ default_graph
                ),
            )
        elif not lm:
            # two or more digit roman numerals
            roman_to_cardinal = pynini.compose(
                pynini.closure(NEMO_ALPHA, 2),
                (
                    pynutil.insert("default_cardinal: \"default\" ")
                    + (pynini.string_map([x[0] for x in roman_dict[:50]]).optimize()) @ default_graph
                ),
            )

        # convert three digit roman or up with suffix to ordinal
        roman_to_ordinal = pynini.compose(
            pynini.closure(NEMO_ALPHA, 3),
            (pynutil.insert("default_ordinal: \"default\" ") + graph_teens @ default_graph + pynutil.delete("th")),
        )

        graph |= roman_to_cardinal | roman_to_ordinal

        # # add a higher weight when roman number consists of a single symbol
        # graph = pynini.compose(pynini.closure(NEMO_CHAR, 2), graph) | pynutil.add_weight(
        #     pynini.compose(NEMO_CHAR, graph), 101
        # )
        # graph = graph.optimize() + pynini.closure(pynutil.delete("."), 0, 1)

        # graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        graph = self.add_tokens(graph)

        self.fst = graph.optimize()


def get_names():
    """
    Returns the graph that matched common male and female names.
    """
    male_labels = load_labels(get_abs_path("data/roman/male.tsv"))
    female_labels = load_labels(get_abs_path("data/roman/female.tsv"))
    male_labels.extend([[x[0].upper()] for x in male_labels])
    female_labels.extend([[x[0].upper()] for x in female_labels])
    names = pynini.string_map(male_labels).optimize()
    names |= pynini.string_map(female_labels).optimize()
    return names
