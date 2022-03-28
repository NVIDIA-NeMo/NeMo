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

from nemo.collections.common.tokenizers.column_coder import CategoryCode, ColumnCodes, FloatCode, IntCode


class TestColumnCoder:
    @pytest.mark.unit
    def test_float(self):
        np.random.seed(1234)
        series = np.random.random(100)
        float_coder = FloatCode('t', 5, 0, False, 10, True)
        float_coder.compute_code(data_series=series)
        r = float_coder.encode('0.323')
        assert np.array_equal(np.array(r), np.array([40, 32, 25, 17, 3]))
        decoded = float_coder.decode(r)
        assert decoded == '0.32290'
        r = float_coder.encode('1.323')
        assert np.array_equal(np.array(r), np.array([41, 30, 20, 10, 0]))
        decoded = float_coder.decode(r)
        assert decoded == '0.99208'
        r = float_coder.encode('nan')
        assert np.array_equal(np.array(r), np.array([42, 39, 29, 19, 9]))
        decoded = float_coder.decode(r)
        assert decoded == 'nan'

        float_coder = FloatCode('t', 5, 0, False, 10, True, 'yeo-johnson')
        float_coder.compute_code(data_series=series)
        r = float_coder.encode('0.323')
        assert np.array_equal(np.array(r), np.array([41, 30, 25, 14, 5]))
        decoded = float_coder.decode(r)
        assert decoded == '0.32300'
        r = float_coder.encode('1.323')
        assert np.array_equal(np.array(r), np.array([43, 39, 21, 16, 3]))
        decoded = float_coder.decode(r)
        assert decoded == '1.08064'
        r = float_coder.encode('nan')
        assert np.array_equal(np.array(r), np.array([44, 39, 29, 19, 9]))
        decoded = float_coder.decode(r)
        assert decoded == 'nan'

        float_coder = FloatCode('t', 5, 0, False, 10, True, 'robust')
        float_coder.compute_code(data_series=series)
        r = float_coder.encode('0.323')
        assert np.array_equal(np.array(r), np.array([40, 37, 24, 10, 8]))
        decoded = float_coder.decode(r)
        assert decoded == '0.32299'
        r = float_coder.encode('1.323')
        assert np.array_equal(np.array(r), np.array([42, 30, 27, 19, 3]))
        decoded = float_coder.decode(r)
        assert decoded == '0.89536'
        r = float_coder.encode('nan')
        assert np.array_equal(np.array(r), np.array([43, 39, 29, 19, 9]))
        decoded = float_coder.decode(r)
        assert decoded == 'nan'

        float_coder = FloatCode('t', 5, 0, True, 377, True)
        float_coder.compute_code(data_series=series)
        r = float_coder.encode('0.323')
        assert np.array_equal(np.array(r), np.array([1508, 1228, 765, 663, 194]))
        decoded = float_coder.decode(r)
        assert decoded == '0.32299999994'
        r = float_coder.encode('nan')
        assert np.array_equal(np.array(r), np.array([1885, 1507, 1130, 753, 376]))
        decoded = float_coder.decode(r)
        assert decoded == 'nan'
        assert float_coder.end_id == 1886
        assert float_coder.code_range[0] == (1508, 1886)
        assert float_coder.code_range[1] == (1131, 1508)
        assert float_coder.code_range[2] == (754, 1131)
        assert float_coder.code_range[3] == (377, 754)
        assert float_coder.code_range[4] == (0, 377)

        float_coder = FloatCode('t', 5, 0, True, 377, False)
        float_coder.compute_code(data_series=series)
        assert float_coder.end_id == 1885
        assert float_coder.code_range[0] == (1508, 1885)
        assert float_coder.code_range[1] == (1131, 1508)
        assert float_coder.code_range[2] == (754, 1131)
        assert float_coder.code_range[3] == (377, 754)
        assert float_coder.code_range[4] == (0, 377)
        try:
            float_coder.encode('nan')
        except ValueError as e:
            assert str(e) == 'colum t cannot handle nan, please set hasnan=True'

    @pytest.mark.unit
    def test_int(self):
        np.random.seed(1234)
        array = np.random.randint(3, 1000, 100)
        int_coder = IntCode('i', 3, 0, False, 16, True)
        int_coder.compute_code(array)

        r = int_coder.encode('232')
        assert np.array_equal(np.array(r), np.array([32, 30, 2]))
        decoded = int_coder.decode(r)
        assert decoded == '232'

        r = int_coder.encode('nan')
        assert np.array_equal(np.array(r), np.array([36, 31, 15]))
        decoded = int_coder.decode(r)
        assert decoded == 'nan'

    @pytest.mark.unit
    def test_category(self):
        np.random.seed(1234)
        ALPHABET = np.array(list(string.ascii_lowercase + ' '))
        array = np.char.add(np.random.choice(ALPHABET, 1000), np.random.choice(ALPHABET, 1000))

        int_coder = CategoryCode('c', 0)
        int_coder.compute_code(array)

        r = int_coder.encode('xy')
        assert np.array_equal(np.array(r), np.array([509]))
        decoded = int_coder.decode(r)
        assert decoded == 'xy'

    @pytest.mark.unit
    def test_column_coder(self):
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

        cc = ColumnCodes.get_column_codes(column_configs, example_arrays)

        rr = cc.encode('col_a', '0.323')
        assert np.array_equal(np.array(rr), np.array([49, 32, 29, 15]))
        decoded = cc.decode('col_a', rr)
        assert decoded == '0.3230'

        rr = cc.encode('col_b', '0.323')
        assert np.array_equal(np.array(rr), np.array([584, 457, 235, 110]))
        decoded = cc.decode('col_b', rr)
        assert decoded == '0.3229999'

        rr = cc.encode('col_c', '232')
        assert np.array_equal(np.array(rr), np.array([787, 780, 773]))
        decoded = cc.decode('col_c', rr)
        assert decoded == '232'

        rr = cc.encode('col_d', 'xy')
        assert np.array_equal(np.array(rr), np.array([1313]))
        decoded = cc.decode('col_d', rr)
        assert decoded == 'xy'

        # assert cc.vocab_size == 1343
