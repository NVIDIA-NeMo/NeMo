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
import logging
import os
import random
from collections import Counter
from collections import OrderedDict as od

import librosa

from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels

random.seed(42)

"""
This script creates manifest file for speaker diarization inference purposes. 
Useful to get manifest when you have list of audio files and optionally rttm and uem files for evaluation

Note: make sure basename for each file is unique and rttm files also has the corresponding base name for mapping
"""


def write_file(name, lines, idx):
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')
    logging.info("wrote", name)


def read_file(pathlist):
    pathlist = open(pathlist, 'r').readlines()
    return sorted(pathlist)


def get_dict_from_wavlist(pathlist):
    path_dict = od()
    pathlist = sorted(pathlist)
    for line_path in pathlist:
        uniq_id = os.path.splitext(os.path.basename(line_path))[0]
        path_dict[uniq_id] = line_path
    return path_dict


def get_dict_from_list(data_pathlist, uniqids):
    path_dict = {}
    for line_path in data_pathlist:
        uniq_id = os.path.splitext(os.path.basename(line_path))[0]
        if uniq_id in uniqids:
            path_dict[uniq_id] = line_path
        else:
            raise ValueError(f'uniq id {uniq_id} is not in wav filelist')
    return path_dict


def get_path_dict(data_path, uniqids, len_wavs=None):
    if data_path is not None:
        data_pathlist = read_file(data_path)
        if len_wavs is not None:
            assert len(data_pathlist) == len_wavs
            data_pathdict = get_dict_from_list(data_pathlist, uniqids)
    elif len_wavs is not None:
        data_pathdict = {uniq_id: None for uniq_id in uniqids}
    return data_pathdict


def main(
    wav_path, text_path=None, rttm_path=None, uem_path=None, ctm_path=None, manifest_filepath=None, add_duration=False
):
    if os.path.exists(manifest_filepath):
        os.remove(manifest_filepath)

    wav_pathlist = read_file(wav_path)
    wav_pathdict = get_dict_from_wavlist(wav_pathlist)
    len_wavs = len(wav_pathlist)
    uniqids = sorted(wav_pathdict.keys())

    text_pathdict = get_path_dict(text_path, uniqids, len_wavs)
    rttm_pathdict = get_path_dict(rttm_path, uniqids, len_wavs)
    uem_pathdict = get_path_dict(uem_path, uniqids, len_wavs)
    ctm_pathdict = get_path_dict(ctm_path, uniqids, len_wavs)

    lines = []
    for uid in uniqids:
        wav, text, rttm, uem, ctm = (
            wav_pathdict[uid],
            text_pathdict[uid],
            rttm_pathdict[uid],
            uem_pathdict[uid],
            ctm_pathdict[uid],
        )

        audio_line = wav.strip()
        if rttm is not None:
            rttm = rttm.strip()
            labels = rttm_to_labels(rttm)
            num_speakers = Counter([l.split()[-1] for l in labels]).keys().__len__()
        else:
            num_speakers = None

        if uem is not None:
            uem = uem.strip()

        if text is not None:
            text = open(text.strip()).readlines()[0].strip()
        else:
            text = "-"

        if ctm is not None:
            ctm = ctm.strip()

        duration = None
        if add_duration:
            y, sr = librosa.get_duration(filename=audio_line, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
        meta = [
            {
                "audio_filepath": audio_line,
                "offset": 0,
                "duration": duration,
                "label": "infer",
                "text": text,
                "num_speakers": num_speakers,
                "rttm_filepath": rttm,
                "uem_filepath": uem,
                "ctm_filepath": ctm,
            }
        ]
        lines.extend(meta)

    write_file(manifest_filepath, lines, range(len(lines)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths2audio_files", help="path to text file containing list of audio files", type=str, required=True
    )
    parser.add_argument("--paths2txt_files", help="path to text file containing list of transcription files", type=str)
    parser.add_argument("--paths2rttm_files", help="path to text file containing list of rttm files", type=str)
    parser.add_argument("--paths2uem_files", help="path to uem files", type=str)
    parser.add_argument("--paths2ctm_files", help="path to ctm files", type=str)
    parser.add_argument("--manifest_filepath", help="path to output manifest file", type=str, required=True)
    parser.add_argument(
        "--add_duration", help="add duration of audio files to output manifest files.", action='store_true',
    )
    args = parser.parse_args()

    main(
        args.paths2audio_files,
        args.paths2txt_files,
        args.paths2rttm_files,
        args.paths2uem_files,
        args.paths2ctm_files,
        args.manifest_filepath,
        args.add_duration,
    )
