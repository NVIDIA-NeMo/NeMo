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

import argparse
import json
import multiprocessing
import os
import random

import librosa as l
import numpy as np
import soundfile as sf
import sox
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.contrib.concurrent import process_map

random.seed(42)

"""
This scipt converts a scp file where each line contains  
<absolute path of wav file> 
to a manifest json file. 
Args: 
--scp: scp file name
--manifest(optional): if you already have manifest file, but would like to process it for creating chunks and splitting then use manifest ignoring scp
--id: index of speaker label in filename present in scp file that is separated by '/'
--out: output manifest file name
--split: True / False if you would want to split the  manifest file for training purposes
        you may not need this for test set. output file names is <out>_<train/dev>.json
        Defaults to False
--create_chunks: bool if you would want to chunk each manifest line to chunks of 3 sec or less
        you may not need this for test set, Defaults to False
--write_chunks: writes chunked files based on offset to {current working directory}/chunks/{label}/{original_file_name}_{offset}_{duration}.wav
--min_spkrs_count: min number of samples per speaker to consider and ignore otherwise
"""

DURATIONS = [1.5, 2, 3]
MIN_ENERGY = 0.01
CWD = './'


def filter_manifest_line(manifest_line):
    split_manifest = []
    audio_path = manifest_line['audio_filepath']
    start = manifest_line.get('offset', 0)
    dur = manifest_line['duration']

    if dur >= min(DURATIONS):
        signal, sr = l.load(audio_path, sr=None)
        remaining_dur = dur
        temp_dur = random.choice(DURATIONS)
        remaining_dur = remaining_dur - temp_dur
        while remaining_dur >= 0:
            segment_audio = signal[int(start * sr) : int(start * sr + temp_dur * sr)]
            if l.feature.rms(y=segment_audio).mean() > MIN_ENERGY:
                meta = manifest_line.copy()
                meta['offset'] = start
                meta['duration'] = temp_dur
                split_manifest.append(meta)
            start = start + temp_dur
            temp_dur = random.choice(DURATIONS)
            remaining_dur = remaining_dur - temp_dur

    return split_manifest


def count_and_consider_only(speakers, lines, min_count=10):
    """
    consider speakers only if samples per speaker is at least min_count
    """
    uniq_speakers, indices, counts = np.unique(speakers, return_index=True, return_counts=True)
    print("speaker count before filtering minimum number of speaker counts: ", len(uniq_speakers))
    required_speakers = {}
    for idx, count in enumerate(counts):
        if count >= min_count:
            required_speakers[uniq_speakers[idx]] = count

    print("speaker count after filtering minimum number of speaker counts: ", len(required_speakers))
    required_lines = []
    speakers_only = []
    for idx, speaker in enumerate(speakers):
        if speaker in required_speakers:
            required_lines.append(lines[idx])
            speakers_only.append(speaker)

    return speakers_only, required_lines


def write_file(name, lines, idx):
    with open(name, 'w', encoding='utf-8') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')
    print("wrote", name)


def read_file(scp_file, id=-1):
    json_lines = []
    with open(scp_file, 'r') as fo:
        lines = fo.readlines()
        lines = sorted(lines)
        for line in lines:
            line = line.strip()
            speaker = line.split('/')[id]
            speaker = list(speaker)
            speaker = ''.join(speaker)
            meta = {"audio_filepath": line, "offset": 0, "duration": None, "label": speaker}
            json_lines.append(meta)
    return json_lines


def read_manifest(manifest):
    data = []
    with open(manifest, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def get_duration(json_line):
    dur = json_line['duration']
    if dur is not None:
        wav_path = json_line['audio_filepath']
        json_line['duration'] = sox.file_info.duration(wav_path)
    return json_line


def get_labels(lines):
    labels = []
    for line in lines:
        label = line['label']
        labels.append(label)
    return labels


def write_audio_file(line):
    filename = line['audio_filepath']
    label = line['label']
    offset = line['offset']
    duration = line['duration']
    basename = os.path.basename(filename).replace('.wav', '')
    to_path = os.path.join(CWD, 'chunks', label)
    os.makedirs(to_path, exist_ok=True)
    to_path = os.path.join(to_path, basename)
    final_string = '_' + str(offset) + '_' + str(duration)
    final_string = final_string.replace('.', '-')
    samples, sr = sf.read(filename)
    start = int(float(offset * sr))
    end = start + int(float(duration * sr))
    chunk = samples[start:end]
    to_file = to_path + final_string + '.wav'
    sf.write(to_file, chunk, sr)

    line['offset'] = 0
    line['audio_filepath'] = to_file
    return line


def main(scp, manifest, id, out, split=False, create_chunks=False, write_chunks=False, min_count=10, workers=4):
    if os.path.exists(out):
        os.remove(out)
    if scp:
        lines = read_file(scp_file=scp, id=id)
    else:
        lines = read_manifest(manifest)

    lines = process_map(get_duration, lines, chunksize=100)

    if create_chunks:
        print("creating chunk")
        lines = process_map(filter_manifest_line, lines, chunksize=100)
        temp = []
        for line in lines:
            temp.extend(line)
        del lines
        lines = temp

    if create_chunks and write_chunks:
        print("writing chunks created before as new wav files")
        lines = process_map(write_audio_file, lines, chunksize=100)

    speakers = [x['label'] for x in lines]

    if min_count:
        speakers, lines = count_and_consider_only(speakers, lines, min_count)

    write_file(out, lines, range(len(lines)))
    path = os.path.dirname(out)
    if split:
        speakers = [x['label'] for x in lines]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        for train_idx, test_idx in sss.split(speakers, speakers):
            print("number of train samples after split: ", len(train_idx))

        out = os.path.join(path, 'train.json')
        write_file(out, lines, train_idx)
        out = os.path.join(path, 'dev.json')
        write_file(out, lines, test_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scp", help="scp file name", type=str, required=False, default=None)
    parser.add_argument("--manifest", help="manifest file name", type=str, required=False, default=None)
    parser.add_argument(
        "--id",
        help="field num seperated by '/' to be considered as speaker label from scp file, can be ignored if manifest file is already provided with labels",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument("--out", help="manifest_file name", type=str, required=True)
    parser.add_argument(
        "--split",
        help="bool if you would want to split the manifest file for training purposes",
        required=False,
        action='store_true',
    )
    parser.add_argument(
        "--create_chunks",
        help="bool if you would want to chunk each manifest line to chunks of 3 sec or less",
        required=False,
        action='store_true',
    )
    parser.add_argument(
        "--write_chunks",
        help="bool if you would want to write the chunks created with --create_chunk to CWD ",
        required=False,
        action='store_true',
    )
    parser.add_argument(
        "--min_spkrs_count",
        default=10,
        type=int,
        help="min number of samples per speaker to consider and ignore otherwise",
    )
    parser.add_argument(
        "--num_workers", default=multiprocessing.cpu_count(), type=int, help="Workers to process dataset."
    )

    args = parser.parse_args()

    main(
        args.scp,
        args.manifest,
        args.id,
        args.out,
        args.split,
        args.create_chunks,
        args.write_chunks,
        args.min_spkrs_count,
        args.num_workers,
    )
