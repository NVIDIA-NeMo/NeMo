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


import string

import numpy as np
import pytest

from nemo.collections.common.tokenizers.column_coder import ColumnCodes
from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer


class TestTabularTokenizer:
    def setup_method(self, test_method):
        column_configs = [
            {
                "name": "col_a",
                "code_type": "float",
                "args": {"code_len": 4, "base": 16, "fillall": False, "hasnan": True, "transform": 'yeo-johnson'},
            },
            {
                "name": "col_b",
                "code_type": "float",
                "args": {"code_len": 4, "base": 177, "fillall": True, "hasnan": True, "transform": 'quantile'},
            },
            {
                "name": "col_c",
                "code_type": "int",
                "args": {"code_len": 3, "base": 12, "fillall": True, "hasnan": True},
            },
            {"name": "col_d", "code_type": "category",},
        ]

        example_arrays = {}
        np.random.seed(1234)

        array = np.random.random(100)
        example_arrays['col_a'] = array

        array = np.random.random(100)
        example_arrays['col_b'] = array

        array = np.random.randint(3, 1000, 100)
        example_arrays['col_c'] = array

        ALPHABET = np.array(list(string.ascii_lowercase + ' '))
        array = np.char.add(np.random.choice(ALPHABET, 1000), np.random.choice(ALPHABET, 1000))
        example_arrays['col_d'] = array

        self.cc = ColumnCodes.get_column_codes(column_configs, example_arrays)

    @pytest.mark.unit
    def test_tabular_tokenizer(self):
        tab = TabularTokenizer(self.cc, delimiter=',')
        text = "0.323, 0.1, 232, xy\n0.323, 0.1, 232, xy<|endoftext|>"
        r = tab.text_to_tokens(text)
        assert len(r) == 10
        assert tab.eod == 1351
        assert tab.eor == 1352
        assert tab.num_columns == 4
        assert self.cc.vocab_size == 1351
        assert tab.vocab_size == 1353
        r = tab.text_to_ids(text)
        assert (sum(self.cc.sizes) + 1) * 2 == len(r)
        assert np.array_equal(
            np.array(r[0:13]), np.array([49, 32, 29, 15, 584, 417, 305, 76, 787, 780, 773, 1313, 1352])
        )
        assert np.array_equal(
            np.array(r[13:]), np.array([49, 32, 29, 15, 584, 417, 305, 76, 787, 780, 773, 1313, 1351])
        )
        reversed_text = tab.ids_to_text(r)
        assert reversed_text == '0.3230,0.0999998,232,xy\n0.3230,0.0999998,232,xy<|endoftext|>'

        text = "xy\n0.323, 0.1, 232, xy<|endoftext|>"
        r = tab.text_to_tokens(text)
        assert len(r) == 7
        r = tab.text_to_ids(text)
        assert sum(self.cc.sizes) + 1 + 2 == len(r)
        assert np.array_equal(np.array(r[0:2]), np.array([1313, 1352]))
        assert np.array_equal(
            np.array(r[2:15]), np.array([49, 32, 29, 15, 584, 417, 305, 76, 787, 780, 773, 1313, 1351])
        )
        reversed_text = tab.ids_to_text(r)
        assert reversed_text == 'xy\n0.3230,0.0999998,232,xy<|endoftext|>'

        text = "\n0.323, 0.1, 232, xy<|endoftext|>"
        r = tab.text_to_tokens(text)
        assert len(r) == 5
        r = tab.text_to_ids(text)
        assert sum(self.cc.sizes) + 1 == len(r)
        assert np.array_equal(
            np.array(r[0:13]), np.array([49, 32, 29, 15, 584, 417, 305, 76, 787, 780, 773, 1313, 1351])
        )
        reversed_text = tab.ids_to_text(r)
        assert reversed_text == '0.3230,0.0999998,232,xy<|endoftext|>'

        text = "232, xy\n0.323, 0.1, 232, xy<|endoftext|>"
        r = tab.text_to_tokens(text)
        assert len(r) == 8
        r = tab.text_to_ids(text)
        assert sum(self.cc.sizes) + 1 + 5 == len(r)
        assert np.array_equal(np.array(r[0:5]), np.array([787, 780, 773, 1313, 1352]))
        assert np.array_equal(
            np.array(r[5:18]), np.array([49, 32, 29, 15, 584, 417, 305, 76, 787, 780, 773, 1313, 1351])
        )
        reversed_text = tab.ids_to_text(r)
        assert reversed_text == '232,xy\n0.3230,0.0999998,232,xy<|endoftext|>'

        text = "0.1, 232, xy\n0.323, 0.1, 232, xy<|endoftext|>"
        r = tab.text_to_tokens(text)
        assert len(r) == 9
        r = tab.text_to_ids(text)
        assert sum(self.cc.sizes) + 1 + 9 == len(r)
        assert np.array_equal(np.array(r[0:9]), np.array([584, 417, 305, 76, 787, 780, 773, 1313, 1352]))
        assert np.array_equal(
            np.array(r[9:22]), np.array([49, 32, 29, 15, 584, 417, 305, 76, 787, 780, 773, 1313, 1351])
        )
        reversed_text = tab.ids_to_text(r)
        assert reversed_text == '0.0999998,232,xy\n0.3230,0.0999998,232,xy<|endoftext|>'
