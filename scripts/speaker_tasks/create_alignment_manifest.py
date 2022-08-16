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

import argparse
import os
import random
import shutil

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_ctm, write_manifest
from nemo.utils import logging

random.seed(42)


def get_unaligned_examples(unaligned_path, dataset):
    """
    Get librispeech examples without alignments for the desired dataset in
    order to filter them out (as they cannot be used for data simulation))

    Args:
        unaligned_path (str): Path to the file containing unaligned examples
        dataset (str): LibriSpeech data split being used

    Returns:
        skip_files (list): Unaligned file names to skip
    """
    with open(unaligned_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        skip_files = []
        while i < len(lines):
            l = lines[i].strip('\n')
            if l[0] == '#':
                unaligned_dataset = l.split(" ")[1]
            elif unaligned_dataset == dataset:
                unaligned_file = l.split(" ")[0]
                skip_files.append(unaligned_file)
            i += 1
    return skip_files


def create_new_ctm_entry(session_name, speaker_id, wordlist, alignments, output_precision=3):
    """
    Create new CTM entry (to write to output ctm file)

    Args:
        session_name (str): Current session name.
        speaker_id (int): LibriSpeech speaker ID for the current entry.
        wordlist (list): List of words
        alignments (list): List of alignments
        output_precision (int): Precision for CTM outputs
    Returns:
        arr (list): List of ctm entries
    """
    arr = []
    for i in range(len(wordlist)):
        word = wordlist[i]
        if word != "":
            # note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
            align1 = float(round(alignments[i - 1], output_precision))
            align2 = float(round(alignments[i] - alignments[i - 1], output_precision,))
            text = f"{session_name} {speaker_id} {align1} {align2} {word} 0\n"
            arr.append((align1, text))
    return arr


def create_ctm_alignments(input_manifest_filepath, base_alignment_path, ctm_directory, dataset):
    """
    Create new CTM alignments using input LibriSpeech word alignments. 

    Args:
        input_manifest_filepath (str): Path to the input LibriSpeech manifest file
        base_alignment_path (str): Path to the base directory containing the LibriSpeech word alignments
        ctm_directory (str): Directory to write the CTM files to
        dataset (str): Which split of the LibriSpeech dataset is being used
    """
    manifest = read_manifest(input_manifest_filepath)
    unaligned_path = os.path.join(base_alignment_path, "unaligned.txt")
    dataset = dataset.replace("_", "-")
    unaligned = get_unaligned_examples(unaligned_path, dataset)

    # delete output directory if it exists or throw warning
    if os.path.isdir(ctm_directory):
        shutil.rmtree(ctm_directory)
    os.mkdir(ctm_directory)

    idx = 0
    while idx < len(manifest):
        f = manifest[idx]
        # get speaker_id
        fn = f['audio_filepath'].split('/')[-1]
        speaker_id = fn.split('-')[0]
        book_id = fn.split('-')[1]

        book_dir = os.path.join(base_alignment_path, "LibriSpeech", dataset, speaker_id, book_id)
        alignment_fpath = os.path.join(book_dir, f"{speaker_id}-{book_id}.alignment.txt")

        # Parse each utterance present in the file
        alignment_file = open(alignment_fpath, "r")
        for line in alignment_file:

            # filter out unaligned examples
            f = manifest[idx]
            fn = f['audio_filepath'].split('/')[-1]
            line_id = fn.split('.')[0]
            while line_id in unaligned:
                idx += 1
                f = manifest[idx]
                fn = f['audio_filepath'].split('/')[-1]
                line_id = fn.split('.')[0]

            # from https://github.com/CorentinJ/librispeech-alignments/blob/master/parser_example.py
            # Retrieve the utterance id, the words as a list and the end_times as a list
            utterance_id, words, end_times = line.strip().split(' ')
            if utterance_id != line_id:
                raise Exception("Mismatch between source and target utterance id")
            words = words.replace('\"', '').lower().split(',')
            end_times = [float(e) for e in end_times.replace('\"', '').split(',')]

            ctm_list = create_new_ctm_entry(line_id, speaker_id, words, end_times)
            write_ctm(os.path.join(ctm_directory, line_id + '.ctm'), ctm_list)
            idx += 1
        alignment_file.close()


def create_manifest_with_alignments(
    input_manifest_filepath, ctm_directory, output_manifest_filepath, output_precision=3
):
    """
    Create new manifest file with word alignments using CTM files

    Args:
        input_manifest_filepath (str): Path to the input manifest file
        ctm_directory (str): Directory to read the CTM files from
        output_manifest_filepath (str): Path to the output manifest file containing word alignments
        precision (int): How many decimal places to keep in the manifest file
    """
    manifest = read_manifest(input_manifest_filepath)
    target_manifest = []
    src_i = 0
    tgt_i = 0
    while src_i < len(manifest):
        f = manifest[src_i]
        fn = f['audio_filepath'].split('/')[-1]
        filename = fn.split('.')[0]  # assuming that there is only one period in the input filenames

        ctm_filepath = os.path.join(ctm_directory, filename + '.ctm')

        if not os.path.isfile(ctm_filepath):
            logging.info(f"Skipping {filename}.wav as there is no corresponding CTM file")
            src_i += 1
            continue

        with open(ctm_filepath, 'r') as ctm_file:
            lines = ctm_file.readlines()

        words = []
        end_times = []
        i = 0
        prev_end = 0
        for i in range(len(lines)):
            ctm = lines[i].split(' ')
            speaker_id = ctm[1]
            start = float(ctm[2])
            end = float(ctm[2]) + float(ctm[3])
            start = round(start, output_precision)
            end = round(end, output_precision)
            if start > prev_end:  # insert silence
                words.append("")
                end_times.append(start)

            words.append(ctm[4])
            end_times.append(end)

            i += 1
            prev_end = end

        # append last end
        if f['duration'] > prev_end:
            words.append("")
            end_times.append(f['duration'])

        # build target manifest entry
        target_manifest.append({})
        target_manifest[tgt_i]['audio_filepath'] = f['audio_filepath']
        target_manifest[tgt_i]['duration'] = f['duration']
        target_manifest[tgt_i]['text'] = f['text']
        target_manifest[tgt_i]['words'] = words
        target_manifest[tgt_i]['alignments'] = end_times
        target_manifest[tgt_i]['speaker_id'] = speaker_id

        src_i += 1
        tgt_i += 1

    logging.info(f"Writing output manifest file to {output_manifest_filepath}")
    write_manifest(output_manifest_filepath, target_manifest)


def main():
    """
    Create a combined manifest file including word alignments and speaker IDs
    """
    input_manifest_filepath = args.input_manifest_filepath
    base_alignment_path = args.base_alignment_path
    output_manifest_filepath = args.output_manifest_filepath
    ctm_directory = args.ctm_directory
    dataset = args.dataset
    use_ctm = args.use_ctm
    output_precision = args.output_precision

    if not use_ctm:
        create_ctm_alignments(input_manifest_filepath, base_alignment_path, ctm_directory, dataset)

    create_manifest_with_alignments(
        input_manifest_filepath, ctm_directory, output_manifest_filepath, output_precision=output_precision
    )


if __name__ == "__main__":
    """
    This script creates a manifest file to be used for generating synthetic
    multispeaker audio sessions. The script takes in the default manifest file
    for a LibriSpeech dataset and corresponding word alignments and produces
    a combined manifest file that contains word alignments and speaker IDs
    per example. It can also be used to produce a manifest file for a different
    dataset if alignments are passed in CTM files.

    The alignments are obtained from: https://github.com/CorentinJ/librispeech-alignments

    Args:
        input_manifest_filepath (str): Path to input manifest file
        base_alignment_path (str): Path to the base directory for the LibriSpeech alignment dataset (specifically to the LibriSpeech-Alignments directory containing both the LibriSpeech folder as well as the unaligned.txt file) or to a directory containing the requisite CTM files
        output_manifest_filepath (str): Path to output manifest file
        ctm_directory (str): Path to output CTM directory (only used for LibriSpeech)
        dataset (str): Which dataset split to create a combined manifest file for
        use_ctm (bool): If true, base_alignment_path points to a directory containing ctm files
    """
    parser = argparse.ArgumentParser(description="LibriSpeech Alignment Manifest Creator")
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--base_alignment_path", help="path to alignments (LibriSpeech)", type=str, required=False)
    parser.add_argument("--output_manifest_filepath", help="path to output manifest file", type=str, required=True)
    parser.add_argument(
        "--ctm_directory",
        help="path to output ctm directory for LibriSpeech (or to input CTM directory)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        help="which test/dev/training set to create a manifest for (only used for LibriSpeech)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--use_ctm",
        help="if true, base_alignment_path points to a directory containing ctm files",
        action='store_true',
        required=False,
    )
    parser.add_argument(
        "--output_precision", help="precision for output alignments", type=int, required=False, default=3
    )
    args = parser.parse_args()

    main()
