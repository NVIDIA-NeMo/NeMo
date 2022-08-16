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

from nemo.collections.asr.data.data_simulation import MultiSpeakerSimulator, RIRMultiSpeakerSimulator
from nemo.core.config import hydra_runner


"""
This script creates a synthetic diarization session using the LibriSpeech dataset.
Usage:
  python create_diarization_dataset_librispeech.py
Check out parameters in ./conf/data_simulator.yaml.
"""


@hydra_runner(config_path="conf", config_name="data_simulator.yaml")
def main(cfg):
    if cfg.data_simulator.rir_generation.use_rir:
        lg = RIRMultiSpeakerSimulator(cfg=cfg)
    else:
        lg = MultiSpeakerSimulator(cfg=cfg)
    lg.generate_sessions()


if __name__ == "__main__":
    main()
