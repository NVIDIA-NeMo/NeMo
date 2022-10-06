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

"""
Usage:

python process_slurp_data.py \
    --data_dir=<directory to store the data> \
    --text_key=<data to store in the 'text' field of manifests, choices=['semantics', 'transcript']> \
    --suffix=<suffix to be added to manifest filenames, e.g., 'slu' or 'asr'> \ 

Note that use text_key=semantics for end-to-end SLU, use text_key=transcript for trainng ASR models on SLURP
"""

import argparse
import json
import multiprocessing
import os
import tarfile
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import wget
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

sampling_rate = 16000

AUDIO_URLS = [
    "https://zenodo.org/record/4274930/files/slurp_real.tar.gz",
    "https://zenodo.org/record/4274930/files/slurp_synth.tar.gz",
]

ANNO_URLS = [
    "https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/test.jsonl",
    "https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/devel.jsonl",
    "https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/train_synthetic.jsonl",
    "https://github.com/pswietojanski/slurp/raw/master/dataset/slurp/train.jsonl",
]

FIELD_AUDIO = "audio_filepath"
FIELD_TEXT = "text"
FIELD_DATA_DIR = "data_dir"


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource

    Returns:

    """
    if not os.path.exists(destination):
        print(f"{destination} does not exist. Downloading ...")
        wget.download(source, destination)
        print(f"Downloaded {destination}.")
    else:
        print(f"Destination {destination} exists. Skipping.")
    return destination


def __extract_all_files(filepath: str, data_dir: str):
    tar = tarfile.open(filepath)
    tar.extractall(data_dir)
    tar.close()


def download_slurp(data_dir: str, anno_dir: str):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    anno_dir = Path(anno_dir)
    anno_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading and extracting audio files, this may take a long time...")
    for url in AUDIO_URLS:
        target_file = url.split("/")[-1]
        destination = str(data_dir / Path(target_file))
        print(f"Getting {target_file}")
        __maybe_download_file(destination, url)
        print(f"Extracting {target_file}")
        __extract_all_files(destination, data_dir)

    print("Downloading annotation files...")
    for url in ANNO_URLS:
        target_file = url.split("/")[-1]
        destination = str(anno_dir / Path(target_file))
        print(f"Getting {target_file}")
        __maybe_download_file(destination, url)

    print("Finished downloading data.")


def process_raw_annotations(anno_dir: str, text_key: str = "semantics", suffix: str = "slu"):
    anno_dir = Path(anno_dir)

    splits = [
        "train",
        "train_synthetic",
        "devel",
        "test",
    ]
    id = 0
    for split in splits:
        tag = "_" + suffix if suffix else ""
        new_filename = f"{os.path.join(anno_dir, split)}{tag}.json"
        print(f"Preparing {new_filename}...")

        IDs = []
        slurp_id = []
        audio = []
        audio_format = []
        audio_opts = []

        semantics = []
        semantics_format = []
        semantics_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        jsonl_path = os.path.join(anno_dir, split + ".jsonl")

        with open(jsonl_path, "r") as fin:
            for line in fin.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                obj = json.loads(line)
                sid = obj["slurp_id"]
                scenario = obj["scenario"]
                action = obj["action"]
                sentence_annotation = obj["sentence_annotation"]
                num_entities = sentence_annotation.count("[")
                entities = []
                for slot in range(num_entities):
                    type = sentence_annotation.split("[")[slot + 1].split("]")[0].split(":")[0].strip()
                    filler = sentence_annotation.split("[")[slot + 1].split("]")[0].split(":")[1].strip()
                    entities.append({"type": type.lower(), "filler": filler.lower()})
                for recording in obj["recordings"]:
                    IDs.append(id)
                    slurp_id.append(sid)
                    if "synthetic" in split:
                        audio_folder = "slurp_synth/"
                    else:
                        audio_folder = "slurp_real/"

                    path = os.path.join(audio_folder, recording["file"])

                    audio.append(path)
                    audio_format.append("flac")
                    audio_opts.append(None)

                    transcript.append(obj["sentence"])
                    transcript_format.append("string")
                    transcript_opts.append(None)

                    semantics_dict = {
                        "scenario": scenario,
                        "action": action,
                        "entities": entities,
                    }

                    semantics_ = str(semantics_dict)
                    semantics.append(semantics_)
                    semantics_format.append("string")
                    semantics_opts.append(None)
                    id += 1

        df = pd.DataFrame(
            {"ID": IDs, "slurp_id": slurp_id, "audio": audio, "semantics": semantics, "transcript": transcript,}
        )

        if text_key not in ["transcript", "semantics"]:
            text_key = "transcript"

        with open(new_filename, "w") as fout:
            for idx in tqdm(range(len(df))):
                item = {
                    "id": str(df["ID"][idx]),
                    "slurp_id": str(df["slurp_id"][idx]),
                    "audio_filepath": df["audio"][idx],
                    "transcript": df["transcript"][idx],
                    "semantics": df["semantics"][idx],
                    "text": df[text_key][idx],
                }
                fout.write(json.dumps(item) + "\n")
        print(f"Saved output to: {new_filename}")


def process(x: dict) -> dict:
    if not isinstance(x[FIELD_TEXT], str):
        x[FIELD_TEXT] = ''
    else:
        x[FIELD_TEXT] = x[FIELD_TEXT].lower().strip()

    data_dir = x[FIELD_DATA_DIR]
    input_file = Path(x[FIELD_AUDIO])
    if not input_file.is_absolute():
        input_file_path = str(data_dir / input_file)
    else:
        input_file_path = str(input_file)

    output_file = Path(input_file.stem + ".wav")

    if "slurp_real" in input_file_path:
        output_dir = Path("wavs/slurp_real")
    else:
        output_dir = Path("wavs/slurp_synth")

    output_file_path = str(data_dir / output_dir / output_file)

    if not os.path.exists(output_file_path):
        y, _ = librosa.load(input_file_path, sr=sampling_rate)
        sf.write(output_file_path, y, sampling_rate)

    y, _ = librosa.load(output_file_path, sr=sampling_rate)
    x['duration'] = librosa.get_duration(y=y, sr=sampling_rate)
    x[FIELD_AUDIO] = str(output_dir / output_file)
    del x[FIELD_DATA_DIR]
    return x


def load_data(manifest: str, data_dir: str):
    data = []
    with open(manifest, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            item[FIELD_DATA_DIR] = Path(data_dir)
            data.append(item)
    return data


def decode_resample_slurp(data_dir: str, anno_dir: str):
    wavs_dir = Path(data_dir) / Path("wavs")
    wavs_dir.mkdir(parents=True, exist_ok=True)
    wavs_real_dir = wavs_dir / Path("slurp_real")
    wavs_real_dir.mkdir(parents=True, exist_ok=True)
    wavs_synth_dir = wavs_dir / Path("slurp_synth")
    wavs_synth_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(anno_dir)
    if manifest_path.is_dir():
        manifest_list = list(manifest_path.glob("*.json"))
    else:
        manifest_list = [str(manifest_path)]

    print(f"Found {len(manifest_list)} manifests to be processed.")
    for manifest in manifest_list:
        print(f"Processing manifest: {manifest}")
        data = load_data(str(manifest), data_dir)

        data_new = process_map(process, data, max_workers=multiprocessing.cpu_count(), chunksize=100)

        output_file = Path(data_dir) / Path(manifest.name)
        with output_file.open("w") as f:
            for item in tqdm(data_new):
                f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="slurp_data", help="Root directory for dataset")
    parser.add_argument(
        "--text_key",
        type=str,
        default="semantics",
        help="Data to be put in the text field, choices=[semantics,transcript]",
    )
    parser.add_argument("--suffix", type=str, default="slu", help="Suffix to be added to the manifest filenames")

    args = parser.parse_args()

    data_dir = args.data_dir
    anno_dir = str(Path(data_dir) / Path("raw_annotations"))

    download_slurp(data_dir=data_dir, anno_dir=anno_dir)

    process_raw_annotations(anno_dir=anno_dir, text_key=args.text_key, suffix=args.suffix)

    decode_resample_slurp(data_dir=data_dir, anno_dir=anno_dir)

    print("All done!")
