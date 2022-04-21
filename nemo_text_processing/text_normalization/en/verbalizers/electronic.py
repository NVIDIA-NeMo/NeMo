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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil
    from pynini.examples import plurals

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "cdf1" domain: "abc.edu" } } -> c d f one at a b c dot e d u

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)
        graph_digit_no_zero = pynini.invert(pynini.string_file(get_abs_path("data/cardinal/digit.tsv"))).optimize()
        graph_zero = pynini.cross("0", "zero")

        if not deterministic:
            graph_zero |= pynini.cross("0", "o") | pynini.cross("0", "oh")

        graph_digit = graph_digit_no_zero | graph_zero
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbol.tsv")).optimize()
        chars = pynini.difference(NEMO_NOT_QUOTE, pynini.project(graph_symbols, "input"))
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + (
                pynini.closure(
                    pynutil.add_weight(graph_digit + insert_space, 0.00009)
                    | pynutil.add_weight(pynini.closure(graph_symbols + insert_space), 0.00009)
                    | pynutil.add_weight(chars + insert_space, 0.0001)
                )
            )
            + pynutil.delete("\"")
        )

        server_common = pynini.string_file(get_abs_path("data/electronic/server.tsv"))
        domain_common = pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        default_chars_symbols = (
            (chars | graph_symbols) + pynini.closure(insert_space + (chars | graph_symbols))
        ).optimize()

        # nvidia.com
        common_server_common_domain = server_common + insert_space + domain_common
        common_server_common_domain |= common_server_common_domain + pynini.closure(
            insert_space + pynini.cross(".", "dot ") + default_chars_symbols,
        )

        # unknown.com
        default_server_common_domain_input = (
            pynini.difference(NEMO_SIGMA, pynini.project(server_common, "input"))
            + pynini.project(domain_common, "input")
            + NEMO_SIGMA
        )
        default_server_common_domain = pynini.compose(
            default_server_common_domain_input,
            default_chars_symbols
            + insert_space
            + domain_common
            + pynini.closure(insert_space + (chars | graph_symbols)),
        ).optimize()

        # nvidia.unknown
        common_server_default_domain_input = (
            pynini.project(server_common, "input")
            + pynini.difference(NEMO_SIGMA, pynini.project(domain_common, "input")).optimize()
        )
        common_server_default_domain = pynini.compose(
            common_server_default_domain_input,
            server_common + insert_space + pynini.compose(pynini.accep(".") + NEMO_SIGMA, default_chars_symbols),
        ).optimize()

        # unknown.unknown
        non_common_input = pynini.difference(
            NEMO_SIGMA,
            pynini.project(server_common, "input") + pynini.project(domain_common, "input")
            | default_server_common_domain_input
            | common_server_default_domain_input,
        ).optimize()
        default_domain = pynini.compose(
            non_common_input,
            default_chars_symbols
            + insert_space
            + pynini.compose(pynini.accep(".") + NEMO_SIGMA, default_chars_symbols),
        ).optimize()

        domain = plurals._priority_union(
            common_server_common_domain,
            plurals._priority_union(
                default_server_common_domain | common_server_default_domain, default_domain, NEMO_SIGMA
            ),
            NEMO_SIGMA,
        ).optimize()

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + domain
            + delete_space
            + pynutil.delete("\"")
        ).optimize()

        domain @= pynini.cdrewrite(pynutil.add_weight(graph_digit, -0.0001), "", "", NEMO_SIGMA)

        protocol = pynutil.delete("protocol: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        graph = (
            pynini.closure(protocol + delete_space, 0, 1)
            + pynini.closure(user_name + delete_space + pynutil.insert("at ") + delete_space, 0, 1)
            + domain
            + delete_space
        ).optimize()

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
