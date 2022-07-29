# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import tempfile

import pytest

from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer

DEFAULT_REGEX = (
    r"""\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"""
)


class TestRegexTokenizer:
    def create_test_data(self):
        data_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        data_file.writelines(
            """zinc_id,smiles
            ZINC000510438538,FC(F)Oc1ccc([C@H](NCc2cnc3ccccn23)C(F)(F)F)cc1
            """
        )
        data_file_path = str(data_file.name)
        data_file.close()

        return data_file_path

    @pytest.mark.unit
    def test_create_vocab(self):
        data_file_path = self.create_test_data()
        tokenizer = RegExTokenizer(regex=DEFAULT_REGEX)
        tokenizer.build_vocab_from_csv(data_csv_file=data_file_path)

        assert len(tokenizer.vocab) == 18

    @pytest.mark.unit
    def test_text_2_tokens(self):
        tokenizer = RegExTokenizer(regex=DEFAULT_REGEX)

        tokens = tokenizer.text_to_tokens("Zc")
        assert ''.join(tokens) == 'Zc'

    @pytest.mark.unit
    def test_text_2_ids(self):
        data_file_path = self.create_test_data()
        tokenizer = RegExTokenizer(regex=DEFAULT_REGEX)
        tokenizer.build_vocab_from_csv(data_csv_file=data_file_path)

        ids = tokenizer.text_to_ids("Zc")
        assert ','.join(list(map(lambda x: str(x), ids))) == '1,11'

    @pytest.mark.unit
    def test_tokens_2_text(self):
        tokenizer = RegExTokenizer(regex=DEFAULT_REGEX)

        tokens = tokenizer.tokens_to_text(['^', 'Z', 'c', '&'])
        assert ''.join(tokens) == 'Zc'
