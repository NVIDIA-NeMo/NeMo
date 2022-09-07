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
#
# Download the AMI test dataset used to evaluate Speaker Diarization
# More information here: https://groups.inf.ed.ac.uk/ami/corpus/
# USAGE: python get_ami_data.py
import argparse
import os

from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

# todo: once https://github.com/tango4j/diarization_annotation/pull/1 merged, we can use the same repo
test_rttm_url = (
    "https://raw.githubusercontent.com/tango4j/diarization_annotation/main/AMI_corpus/test/split_rttms.tar.gz"
)
dev_rttm_url = (
    "https://raw.githubusercontent.com/SeanNaren/diarization_annotation/dev/AMI_corpus/dev/split_rttms.tar.gz"
)

test_set_ids = [
    "EN2002a",
    "EN2002b",
    "EN2002c",
    "EN2002d",
    "ES2004a",
    "ES2004b",
    "ES2004c",
    "ES2004d",
    "ES2014a",
    "ES2014b",
    "ES2014c",
    "ES2014d",
    "IS1009a",
    "IS1009b",
    "IS1009c",
    "IS1009d",
    "TS3003a",
    "TS3003b",
    "TS3003c",
    "TS3003d",
    "TS3007a",
    "TS3007b",
    "TS3007c",
    "TS3007d",
]

dev_set_ids = [
    "IS1008a",
    "IS1008b",
    "IS1008c",
    "IS1008d",
    "ES2011a",
    "ES2011b",
    "ES2011c",
    "ES2011d",
    "TS3004a",
    "TS3004b",
    "TS3004c",
    "TS3004d",
    "IB4001",
    "IB4002",
    "IB4003",
    "IB4004",
    "IB4010",
    "IB4011",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the AMI Test Corpus Dataset for Speaker Diarization")
    parser.add_argument(
        "--test_manifest_filepath",
        help="path to output test manifest file",
        type=str,
        default='AMItest_input_manifest.json',
    )
    parser.add_argument(
        "--dev_manifest_filepath",
        help="path to output test manifest file",
        type=str,
        default='AMIdev_input_manifest.json',
    )
    parser.add_argument("--data_root", help="path to output data directory", type=str, default="ami_dataset")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_root)
    os.makedirs(data_path, exist_ok=True)

    for ids, manifest_path, split, rttm_url in (
        (test_set_ids, args.test_manifest_filepath, 'test', test_rttm_url),
        (dev_set_ids, args.dev_manifest_filepath, 'dev', dev_rttm_url),
    ):
        split_path = os.path.join(data_path, split)
        audio_path = os.path.join(split_path, "audio")
        os.makedirs(split_path, exist_ok=True)
        rttm_path = os.path.join(split_path, "split_rttms")

        for id in ids:
            os.system(
                f"wget -P {audio_path} https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/{id}/audio/{id}.Mix-Headset.wav"
            )

        if not os.path.exists(f"{split_path}/split_rttms.tar.gz"):
            os.system(f"wget -P {split_path} {rttm_url}")
        os.system(f"tar -xzvf {split_path}/split_rttms.tar.gz -C {split_path}")

        audio_files_path = os.path.join(split_path, 'audio_files.txt')
        rttm_files_path = os.path.join(split_path, 'rttm_files.txt')
        with open(audio_files_path, 'w') as f:
            f.write('\n'.join(os.path.join(audio_path, p) for p in os.listdir(audio_path)))
        with open(rttm_files_path, 'w') as f:
            f.write('\n'.join(os.path.join(rttm_path, p) for p in os.listdir(rttm_path)))

        create_manifest(audio_files_path, manifest_path, rttm_path=rttm_files_path)
