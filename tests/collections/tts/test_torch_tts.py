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

import os

import pytest
import torch

from nemo.collections.tts.torch.data import TTSDataset
from nemo.collections.tts.torch.g2ps import EnglishG2p
from nemo.collections.tts.torch.tts_tokenizers import EnglishPhonemesTokenizer


class TestTTSDataset:
    @pytest.mark.unit
    @pytest.mark.torch_tts
    def test_dataset(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')

        dataset = TTSDataset(
            manifest_filepath=manifest_path,
            sample_rate=22050,
            sup_data_types=["pitch"],
            sup_data_path=sup_path,
            text_tokenizer=EnglishPhonemesTokenizer(
                punct=True,
                stresses=True,
                chars=True,
                space=' ',
                apostrophe=True,
                pad_with_space=True,
                g2p=EnglishG2p(),
            ),
        )

        dataloader = torch.utils.data.DataLoader(dataset, 2, collate_fn=dataset._collate_fn)
        data, _, _, _, _, _ = next(iter(dataloader))

    @pytest.mark.unit
    @pytest.mark.torch_tts
    def test_raise_exception_on_not_supported_sup_data_types(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')
        with pytest.raises(NotImplementedError):
            dataset = TTSDataset(
                manifest_filepath=manifest_path,
                sample_rate=22050,
                sup_data_types=["not_supported_sup_data_type"],
                sup_data_path=sup_path,
                text_tokenizer=EnglishPhonemesTokenizer(
                    punct=True,
                    stresses=True,
                    chars=True,
                    space=' ',
                    apostrophe=True,
                    pad_with_space=True,
                    g2p=EnglishG2p(),
                ),
            )

    @pytest.mark.unit
    @pytest.mark.torch_tts
    def test_raise_exception_on_not_supported_window(self, test_data_dir):
        manifest_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/manifest.json')
        sup_path = os.path.join(test_data_dir, 'tts/mini_ljspeech/sup')
        with pytest.raises(NotImplementedError):
            dataset = TTSDataset(
                manifest_filepath=manifest_path,
                sample_rate=22050,
                sup_data_types=["pitch"],
                sup_data_path=sup_path,
                window="not_supported_window",
                text_tokenizer=EnglishPhonemesTokenizer(
                    punct=True,
                    stresses=True,
                    chars=True,
                    space=' ',
                    apostrophe=True,
                    pad_with_space=True,
                    g2p=EnglishG2p(),
                ),
            )
