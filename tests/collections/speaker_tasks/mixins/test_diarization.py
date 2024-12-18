# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig, SpkDiarizationMixin


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1)

        self.execution_count = 0
        self.flag_begin = False

    def forward(self, x):
        # Input: [1, 1] Output = [1, 1
        out = self.encoder(x)
        return out


@pytest.mark.with_downloads()
@pytest.fixture()
def audio_files(test_data_dir):
    """
    Returns a list of audio files for testing.
    """
    import soundfile as sf

    audio_file1 = os.path.join(test_data_dir, "an4_speaker", "an4", "wav", "an4_clstk", "fash", "an251-fash-b.wav")
    audio_file2 = os.path.join(test_data_dir, "an4_speaker", "an4", "wav", "an4_clstk", "ffmm", "cen1-ffmm-b.wav")

    audio1, _ = sf.read(audio_file1, dtype='float32')
    audio2, _ = sf.read(audio_file2, dtype='float32')

    return audio1, audio2


class DiarizableDummy(DummyModel, SpkDiarizationMixin):
    def _diarize_on_begin(self, audio, diarcfg: DiarizeConfig):
        super()._diarize_on_begin(audio, diarcfg)
        self.flag_begin = True

    def _diarize_input_manifest_processing(self, audio_files: List[str], temp_dir: str, diarcfg: DiarizeConfig):
        # Create a dummy manifest
        manifest_path = os.path.join(temp_dir, 'dummy_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as fp:
            for audio_file in audio_files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                fp.write(json.dumps(entry) + '\n')

        ds_config = {
            'paths2audio_files': audio_files,
            'batch_size': diarcfg.batch_size,
            'temp_dir': temp_dir,
            'session_len_sec': diarcfg.session_len_sec,
            'num_workers': diarcfg.num_workers,
        }
        return ds_config

    def _setup_diarize_dataloader(self, config: Dict) -> DataLoader:
        class DummyDataset(Dataset):
            def __init__(self, audio_files: List[str], config: Dict):
                self.audio_files = audio_files
                self.config = config

            def __getitem__(self, index):
                data = self.audio_files[index]
                data = torch.tensor([float(data)]).view(1)
                return data

            def __len__(self):
                return len(self.audio_files)

        dataset = DummyDataset(config['paths2audio_files'], config)

        return DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=False,
            drop_last=False,
        )

    def _diarize_forward(self, batch: Any):
        output = self(batch)
        return output

    def _diarize_output_processing(self, outputs, uniq_ids, diarcfg: DiarizeConfig):
        self.execution_count += 1

        result = []
        for output in outputs:
            result.append(float(output.item()))

        if hasattr(diarcfg, 'output_type') and diarcfg.output_type == 'dict':
            results = {'output': result}
            return results

        if hasattr(diarcfg, 'output_type') and diarcfg.output_type == 'dict2':
            results = [{'output': res} for res in result]
            return results

        if hasattr(diarcfg, 'output_type') and diarcfg.output_type == 'tuple':
            result = tuple(result)
            return result

        # Pass list of results by default
        return result


class DummyDataset(Dataset):
    def __init__(self, audio_tensors: List[str], config: Dict = None):
        self.audio_tensors = audio_tensors
        self.config = config

    def __getitem__(self, index):
        data = self.audio_tensors[index]
        samples = torch.tensor(data)
        # Calculate seq length
        seq_len = torch.tensor(samples.shape[0], dtype=torch.long)

        # Dummy text tokens
        targets = torch.tensor([0], dtype=torch.long)
        targets_len = torch.tensor(1, dtype=torch.long)
        return (samples, seq_len, targets, targets_len)

    def __len__(self):
        return len(self.audio_tensors)


@pytest.fixture()
def dummy_model():
    return DiarizableDummy()


class TestSpkDiarizationMixin:
    @pytest.mark.unit
    def test_constructor_non_instance(self):
        model = DummyModel()
        assert not isinstance(model, SpkDiarizationMixin)
        assert not hasattr(model, 'diarize')

    @pytest.mark.unit
    def test_diarize(self, dummy_model):
        dummy_model = dummy_model.eval()
        dummy_model.encoder.weight.data.fill_(1.0)
        dummy_model.encoder.bias.data.fill_(0.0)

        audio = ['1.0', '2.0', '3.0']
        outputs = dummy_model.diarize(audio, batch_size=1)
        assert len(outputs) == 3
        assert outputs[0] == 1.0
        assert outputs[1] == 2.0
        assert outputs[2] == 3.0

    @pytest.mark.unit
    def test_diarize_generator(self, dummy_model):
        dummy_model = dummy_model.eval()
        dummy_model.encoder.weight.data.fill_(1.0)
        dummy_model.encoder.bias.data.fill_(0.0)

        audio = ['1.0', '2.0', '3.0']

        diarize_config = DiarizeConfig(batch_size=1)
        generator = dummy_model.diarize_generator(audio, override_config=diarize_config)

        outputs = []
        index = 1
        for result in generator:
            outputs.extend(result)
            assert len(result) == 1
            assert len(outputs) == index
            index += 1

        assert len(outputs) == 3
        assert outputs[0] == 1.0
        assert outputs[1] == 2.0
        assert outputs[2] == 3.0

    @pytest.mark.unit
    def test_diarize_generator_explicit_stop_check(self, dummy_model):
        dummy_model = dummy_model.eval()
        dummy_model.encoder.weight.data.fill_(1.0)
        dummy_model.encoder.bias.data.fill_(0.0)

        audio = ['1.0', '2.0', '3.0']

        diarize_config = DiarizeConfig(batch_size=1)
        generator = dummy_model.diarize_generator(audio, override_config=diarize_config)

        outputs = []
        index = 1
        while True:
            try:
                result = next(generator)
            except StopIteration:
                break
            outputs.extend(result)
            assert len(result) == 1
            assert len(outputs) == index
            index += 1

        assert len(outputs) == 3
        assert outputs[0] == 1.0
        assert outputs[1] == 2.0
        assert outputs[2] == 3.0

    @pytest.mark.unit
    def test_diarize_check_flags(self, dummy_model):
        dummy_model = dummy_model.eval()

        audio = ['1.0', '2.0', '3.0']
        dummy_model.diarize(audio, batch_size=1)
        assert dummy_model.flag_begin

    @pytest.mark.unit
    def test_transribe_override_config_incorrect(self, dummy_model):
        # Not subclassing DiarizeConfig
        @dataclass
        class OverrideConfig:
            batch_size: int = 1
            output_type: str = 'dict'

        dummy_model = dummy_model.eval()

        audio = [1.0, 2.0, 3.0]
        override_cfg = OverrideConfig(batch_size=1, output_type='dict')
        with pytest.raises(ValueError):
            _ = dummy_model.diarize(audio, override_config=override_cfg)

    @pytest.mark.unit
    def test_transribe_override_config_correct(self, dummy_model):
        @dataclass
        class OverrideConfig(DiarizeConfig):
            output_type: str = 'dict'
            verbose: bool = False

        dummy_model = dummy_model.eval()
        dummy_model.encoder.weight.data.fill_(1.0)
        dummy_model.encoder.bias.data.fill_(0.0)

        audio = ['1.0', '2.0', '3.0']
        override_cfg = OverrideConfig(batch_size=1, output_type='list')
        outputs = dummy_model.diarize(audio, override_config=override_cfg)

        assert isinstance(outputs, list)
        assert len(outputs) == 3
        assert outputs[0] == 1.0
        assert outputs[1] == 2.0
        assert outputs[2] == 3.0
