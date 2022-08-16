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

import os
import shutil
import warnings
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import convolve
from scipy.signal.windows import cosine, hamming, hann
from scipy.stats import halfnorm
from tqdm import trange

from nemo.collections.asr.parts.utils.manifest_utils import (
    create_manifest,
    create_segment_manifest,
    read_manifest,
    write_ctm,
    write_manifest,
    write_text,
)
from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile
from nemo.utils import logging

try:
    import pyroomacoustics as pra
    from pyroomacoustics.directivities import CardioidFamily, DirectionVector, DirectivityPattern

    PRA = True
except ImportError:
    PRA = False
try:
    from gpuRIR import att2t_SabineEstimator, beta_SabineEstimation, simulateRIR, t2n

    GPURIR = True
except ImportError:
    GPURIR = False


def clamp_min_list(target_list: List[float], min_val: float) -> List[float]:
    """
    Clamp numbers in the given list with `min_val`.
    Args:
        target_list (list):
            List containing floating point numbers
        min_val (float):
            Desired minimum value to clamp the numbers in `target_list`

    Returns:
        (list) List containing clamped numbers
    """
    return [max(x, min_val) for x in target_list]


def clamp_max_list(target_list: List[float], max_val: float) -> List[float]:
    """
    Clamp numbers in the given list with `max_val`.
    Args:
        target_list (list):
            List containing floating point numbers
        min_val (float):
            Desired maximum value to clamp the numbers in `target_list`

    Returns:
        (list) List containing clamped numbers
    """
    return [min(x, max_val) for x in target_list]


class MultiSpeakerSimulator(object):
    """
    Multispeaker Audio Session Simulator - Simulates multispeaker audio sessions using single-speaker audio files and 
    corresponding word alignments.

    Args:
        cfg: OmegaConf configuration loaded from yaml file.

    Parameters:
    manifest_filepath (str): Manifest file with paths to single speaker audio files
    sr (int): Sampling rate of the input audio files from the manifest
    random_seed (int): Seed to random number generator
    session_config:
      num_speakers (int): Number of unique speakers per multispeaker audio session
      num_sessions (int): Number of sessions to simulate
      session_length (int): Length of each simulated multispeaker audio session (seconds)
    session_params:
      sentence_length_params (list): k,p values for a negative_binomial distribution which is sampled to get the 
                                     sentence length (in number of words)
      dominance_var (float): Variance in speaker dominance (where each speaker's dominance is sampled from a normal 
                             distribution centered on 1/`num_speakers`, and then the dominance values are together 
                             normalized to 1)
      min_dominance (float): Minimum percentage of speaking time per speaker (note that this can cause the dominance of 
                             the other speakers to be slightly reduced)
      turn_prob (float): Probability of switching speakers after each utterance
      mean_overlap (float): Mean proportion of overlap in the overall speaking time (overlap lengths are sampled from 
                            half normal distribution)
      mean_silence (float): Mean proportion of silence to speaking time in the audio session (overlap lengths are 
                            sampled from half normal distribution)
      overlap_prob (float): Proportion of overlap occurrences versus silence between utterances (used to balance the 
                            length of silence gaps and overlapping segments, so a value close to 
                            `mean_overlap`/(`mean_silence`+`mean_overlap`) is suggested)
      start_window (bool): Whether to window the start of sentences to smooth the audio signal (and remove silence at 
                            the start of the clip)
      window_type (str): Type of windowing used when segmenting utterances ("hamming", "hann", "cosine")
      window_size (float): Length of window at the start or the end of segmented utterance (seconds)
      start_buffer (float): Buffer of silence before the start of the sentence (to avoid cutting off speech or starting 
                            abruptly)
      split_buffer (float): Split RTTM labels if greater than twice this amount of silence (to avoid long gaps between 
                            utterances as being labelled as speech)
      release_buffer (float): Buffer before window at end of sentence (to avoid cutting off speech or ending abruptly)
      normalize (bool): Normalize speaker volumes
      normalization_type (str): Normalizing speakers ("equal" - same volume per speaker, "var" - variable volume per 
                                speaker)
      normalization_var (str): Variance in speaker volume (sample from standard deviation centered at 1)
      min_volume (float): Minimum speaker volume (only used when variable normalization is used)
      max_volume (float): Maximum speaker volume (only used when variable normalization is used)
      end_buffer (float): Buffer at the end of the session to leave blank
    outputs:
      output_dir (str): Output directory for audio sessions and corresponding label files
      output_filename (str): Output filename for the wav and rttm files
      overwrite_output (bool): If true, delete the output directory if it exists
      output_precision (int): Number of decimal places in output files
    background_noise: 
      add_bg (bool): Add ambient background noise if true
      background_manifest (str): Path to background noise manifest file
      snr (int): SNR for background noise (using average speaker power)
    speaker_enforcement:
      enforce_num_speakers (bool): Enforce that all requested speakers are present in the output wav file
      enforce_time (list): Percentage of the way through the audio session that enforcement mode is triggered (sampled 
                           between time 1 and 2)
    segment_manifest: (parameters for regenerating the segment manifest file)
      window (float): Window length for segmentation
      shift (float): Shift length for segmentation 
      step_count (int): Number of the unit segments you want to create per utterance
      deci (int): Rounding decimals for segment manifest file
    """

    def __init__(self, cfg):
        self._params = cfg
        # internal params
        self._manifest = read_manifest(self._params.data_simulator.manifest_filepath)
        self._sentence = None
        self._text = ""
        self._words = []
        self._alignments = []
        # keep track of furthest sample per speaker to avoid overlapping same speaker
        self._furthest_sample = [0 for n in range(self._params.data_simulator.session_config.num_speakers)]
        # use to ensure overlap percentage is correct
        self._missing_overlap = 0
        # creating manifests during online data simulation
        self.base_manifest_filepath = None
        self.segment_manifest_filepath = None
        # variable speaker volume
        self._volume = None
        self._check_args()  # error check arguments

    def _check_args(self):
        """
        Checks YAML arguments to ensure they are within valid ranges.
        """
        if self._params.data_simulator.session_config.num_speakers < 2:
            raise Exception("At least two speakers are required for multispeaker audio sessions (num_speakers < 2)")
        if (
            self._params.data_simulator.session_params.turn_prob < 0
            or self._params.data_simulator.session_params.turn_prob > 1
        ):
            raise Exception("Turn probability is outside of [0,1]")
        if (
            self._params.data_simulator.session_params.overlap_prob < 0
            or self._params.data_simulator.session_params.overlap_prob > 1
        ):
            raise Exception("Overlap probability is outside of [0,1]")

        if (
            self._params.data_simulator.session_params.mean_overlap < 0
            or self._params.data_simulator.session_params.mean_overlap > 1
        ):
            raise Exception("Mean overlap is outside of [0,1]")
        if (
            self._params.data_simulator.session_params.mean_silence < 0
            or self._params.data_simulator.session_params.mean_silence > 1
        ):
            raise Exception("Mean silence is outside of [0,1]")
        if (
            self._params.data_simulator.session_params.min_dominance < 0
            or self._params.data_simulator.session_params.min_dominance > 1
        ):
            raise Exception("Minimum dominance is outside of [0,1]")
        if (
            self._params.data_simulator.speaker_enforcement.enforce_time[0] < 0
            or self._params.data_simulator.speaker_enforcement.enforce_time[0] > 1
        ):
            raise Exception("Speaker enforcement start is outside of [0,1]")
        if (
            self._params.data_simulator.speaker_enforcement.enforce_time[1] < 0
            or self._params.data_simulator.speaker_enforcement.enforce_time[1] > 1
        ):
            raise Exception("Speaker enforcement end is outside of [0,1]")

        if (
            self._params.data_simulator.session_params.min_dominance
            * self._params.data_simulator.session_config.num_speakers
            > 1
        ):
            raise Exception("Number of speakers times minimum dominance is greater than 1")

        if (
            self._params.data_simulator.session_params.overlap_prob
            / self._params.data_simulator.session_params.turn_prob
            > 1
        ):
            raise Exception("Overlap probability / turn probability is greater than 1")
        if (
            self._params.data_simulator.session_params.overlap_prob
            / self._params.data_simulator.session_params.turn_prob
            == 1
            and self._params.data_simulator.session_params.mean_silence > 0
        ):
            raise Exception("Overlap probability / turn probability is equal to 1 and mean silence is greater than 0")

        if (
            self._params.data_simulator.session_params.window_type not in ['hamming', 'hann', 'cosine']
            and self._params.data_simulator.session_params.window_type != None
        ):
            raise Exception("Incorrect window type provided")

    def _get_speaker_ids(self) -> List[str]:
        """
        Randomly select speaker IDs from the loaded manifest file.

        Returns:
            speaker_ids (list): Speaker IDs
        """
        speaker_ids = []
        s = 0
        while s < self._params.data_simulator.session_config.num_speakers:
            file = self._manifest[np.random.randint(0, len(self._manifest) - 1)]
            speaker_id = file['speaker_id']
            if speaker_id not in speaker_ids:  # ensure speaker IDs are not duplicated
                speaker_ids.append(speaker_id)
                s += 1
        return speaker_ids

    def _get_speaker_samples(self, speaker_ids: List[str]) -> Dict[str, list]:
        """
        Get a list of the samples for each of the specified speakers.

        Args:
            speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
        Returns:
            speaker_lists (dict): Dictionary of manifest lines per speaker
        """
        speaker_lists = {}
        for i in range(self._params.data_simulator.session_config.num_speakers):
            speaker_lists[str(speaker_ids[i])] = []
        # loop over manifest and add files corresponding to each speaker to each sublist
        for file in self._manifest:
            new_speaker_id = file['speaker_id']
            if new_speaker_id in speaker_ids:
                speaker_lists[str(new_speaker_id)].append(file)
        return speaker_lists

    def _load_speaker_sample(self, speaker_lists: List[dict], speaker_ids: List[str], speaker_turn: int) -> str:
        """
        Load a sample for the selected speaker ID.

        Args:
            speaker_lists (list): List of samples for each speaker in the session.
            speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
            speaker_turn (int): Current speaker turn.
        Returns:
            file_path (str): Path to the desired audio file
        """
        speaker_id = speaker_ids[speaker_turn]
        file_id = np.random.randint(0, len(speaker_lists[str(speaker_id)]) - 1)
        file_path = speaker_lists[str(speaker_id)][file_id]
        return file_path

    def _get_speaker_dominance(self) -> List[float]:
        """
        Get the dominance value for each speaker, accounting for the dominance variance and
        the minimum per-speaker dominance.

        Returns:
            dominance (list): Per-speaker dominance
        """
        dominance_mean = 1.0 / self._params.data_simulator.session_config.num_speakers
        dominance = np.random.normal(
            loc=dominance_mean,
            scale=self._params.data_simulator.session_params.dominance_var,
            size=self._params.data_simulator.session_config.num_speakers,
        )
        dominance = clamp_min_list(dominance, 0)
        # normalize while maintaining minimum dominance
        total = np.sum(dominance)
        if total == 0:
            for i in range(len(dominance)):
                dominance[i] += min_dominance
        # scale accounting for min_dominance which has to be added after
        dominance = (dominance / total) * (
            1
            - self._params.data_simulator.session_params.min_dominance
            * self._params.data_simulator.session_config.num_speakers
        )
        for i in range(len(dominance)):
            dominance[i] += self._params.data_simulator.session_params.min_dominance
            if (
                i > 0
            ):  # dominance values are cumulative to make it easy to select the speaker using a random value in [0,1]
                dominance[i] = dominance[i] + dominance[i - 1]
        return dominance

    def _increase_speaker_dominance(
        self, base_speaker_dominance: List[float], factor: int
    ) -> Tuple[List[float], bool]:
        """
        Increase speaker dominance for unrepresented speakers (used only in enforce mode).
        Increases the dominance for these speakers by the input factor (and then renormalizes the probabilities to 1).

        Args:
            base_speaker_dominance (list): Dominance values for each speaker.
            factor (int): Factor to increase dominance of unrepresented speakers by.
        Returns:
            dominance (list): Per-speaker dominance
            enforce (bool): Whether to keep enforce mode turned on
        """
        increase_percent = []
        for i in range(self._params.data_simulator.session_config.num_speakers):
            if self._furthest_sample[i] == 0:
                increase_percent.append(i)
        # ramp up enforce counter until speaker is sampled, then reset once all speakers have spoken
        if len(increase_percent) > 0:
            # extract original per-speaker probabilities
            dominance = np.copy(base_speaker_dominance)
            for i in range(len(dominance) - 1, 0, -1):
                dominance[i] = dominance[i] - dominance[i - 1]
            # increase specified speakers by the desired factor
            for i in increase_percent:
                dominance[i] = dominance[i] * factor
            # renormalize
            dominance = dominance / np.sum(dominance)
            for i in range(1, len(dominance)):
                dominance[i] = dominance[i] + dominance[i - 1]
            enforce = True
        else:  # no unrepresented speakers, so enforce mode can be turned off
            dominance = base_speaker_dominance
            enforce = False
        return dominance, enforce

    def _set_speaker_volume(self):
        """
        Set the volume for each speaker (either equal volume or variable speaker volume).
        """
        if self._params.data_simulator.session_params.normalization_type == 'equal':
            self._volume = np.ones(self._params.data_simulator.session_config.num_speakers)
        elif self._params.data_simulator.session_params.normalization_type == 'variable':
            self._volume = np.random.normal(
                loc=1.0,
                scale=self._params.data_simulator.session_params.normalization_var,
                size=self._params.data_simulator.session_config.num_speakers,
            )
            self._volume = clamp_min_list(self._volume, self._params.data_simulator.session_params.min_volume)
            self._volume = clamp_max_list(self._volume, self._params.data_simulator.session_params.max_volume)

    def _get_next_speaker(self, prev_speaker: int, dominance: List[float]) -> int:
        """
        Get the next speaker (accounting for turn probability and dominance distribution).

        Args:
            prev_speaker (int): Previous speaker turn.
            dominance (list): Dominance values for each speaker.
        Returns:
            prev_speaker/speaker_turn (int): Speaker turn
        """
        if np.random.uniform(0, 1) > self._params.data_simulator.session_params.turn_prob and prev_speaker != None:
            return prev_speaker
        else:
            speaker_turn = prev_speaker
            while speaker_turn == prev_speaker:  # ensure another speaker goes next
                rand = np.random.uniform(0, 1)
                speaker_turn = 0
                while rand > dominance[speaker_turn]:
                    speaker_turn += 1
            return speaker_turn

    def _get_window(self, window_amount: int, start: bool = False):
        """
        Get window curve to alleviate abrupt change of time-series signal when segmenting audio samples.

        Args:
            window_amount (int): Window length (in terms of number of samples).
            start (bool): If true, return the first half of the window.

        Returns:
            window (tensor): Half window (either first half or second half)
        """
        if self._params.data_simulator.session_params.window_type == 'hamming':
            window = hamming(window_amount * 2)
        elif self._params.data_simulator.session_params.window_type == 'hann':
            window = hann(window_amount * 2)
        elif self._params.data_simulator.session_params.window_type == 'cosine':
            window = cosine(window_amount * 2)
        else:
            raise Exception("Incorrect window type provided")
        # return the first half or second half of the window
        if start:
            return window[:window_amount]
        else:
            return window[window_amount:]

    def _get_start_buffer_and_window(self, first_alignment: int) -> Tuple[int, int]:
        """
        Get the start cutoff and window length for smoothing the start of the sentence.

        Args:
            first_alignment (int): Start of the first word (in terms of number of samples).
        Returns:
            start_cutoff (int): Amount into the audio clip to start
            window_amount (int): Window length
        """
        window_amount = int(self._params.data_simulator.session_params.window_size * self._params.data_simulator.sr)
        start_buffer = int(self._params.data_simulator.session_params.start_buffer * self._params.data_simulator.sr)

        if first_alignment < start_buffer:
            window_amount = 0
            start_cutoff = 0
        elif first_alignment < start_buffer + window_amount:
            window_amount = first_alignment - start_buffer
            start_cutoff = 0
        else:
            start_cutoff = first_alignment - start_buffer - window_amount

        return start_cutoff, window_amount

    def _get_end_buffer_and_window(
        self, current_sr: int, remaining_duration_sr: int, remaining_len_audio_file: int
    ) -> Tuple[int, int]:
        """
        Get the end buffer and window length for smoothing the end of the sentence.

        Args:
            current_sr (int): Current location in the target file (in terms of number of samples).
            remaining_duration_sr (int): Remaining duration in the target file (in terms of number of samples).
            remaining_len_audio_file (int): Length remaining in audio file (in terms of number of samples).
        Returns:
            release_buffer (int): Amount after the end of the last alignment to include
            window_amount (int): Window length
        """

        window_amount = int(self._params.data_simulator.session_params.window_size * self._params.data_simulator.sr)
        release_buffer = int(
            self._params.data_simulator.session_params.release_buffer * self._params.data_simulator.sr
        )

        if current_sr + release_buffer > remaining_duration_sr:
            release_buffer = remaining_duration_sr - current_sr
            window_amount = 0
        elif current_sr + window_amount + release_buffer > remaining_duration_sr:
            window_amount = remaining_duration_sr - current_sr - release_buffer

        if remaining_len_audio_file < release_buffer:
            release_buffer = remaining_len_audio_file
            window_amount = 0
        elif remaining_len_audio_file < release_buffer + window_amount:
            window_amount = remaining_len_audio_file - release_buffer

        return release_buffer, window_amount

    def _add_file(
        self,
        file: dict,
        audio_file: torch.Tensor,
        sentence_duration: int,
        max_sentence_duration: int,
        max_sentence_duration_sr: int,
    ) -> Tuple[int, torch.Tensor]:
        """
        Add audio file to current sentence (up to the desired number of words). 
        Uses the alignments to segment the audio file.

        Args:
            file (dict): Line from manifest file for current audio file
            audio_file (tensor): Current loaded audio file
            sentence_duration (int): Running count for number of words in sentence
            max_sentence_duration (int): Maximum count for number of words in sentence
            max_sentence_duration_sr (int): Maximum length for sentence in terms of samples
        Returns:
            sentence_duration+nw (int): Running word count
            len(self._sentence) (tensor): Current length of the audio file
        """
        if (
            sentence_duration == 0
        ) and self._params.data_simulator.session_params.start_window:  # cut off the start of the sentence
            first_alignment = int(file['alignments'][0] * self._params.data_simulator.sr)
            start_cutoff, start_window_amount = self._get_start_buffer_and_window(first_alignment)
        else:
            start_cutoff = 0

        # ensure the desired number of words are added and the length of the output session isn't exceeded
        sentence_duration_sr = len(self._sentence)
        remaining_duration_sr = max_sentence_duration_sr - sentence_duration_sr
        remaining_duration = max_sentence_duration - sentence_duration
        prev_dur_sr, dur_sr = 0, 0
        nw, i = 0, 0
        while nw < remaining_duration and dur_sr < remaining_duration_sr and i < len(file['words']):
            dur_sr = int(file['alignments'][i] * self._params.data_simulator.sr) - start_cutoff
            if dur_sr > remaining_duration_sr:
                break

            word = file['words'][i]
            self._words.append(word)
            self._alignments.append(
                float(sentence_duration_sr * 1.0 / self._params.data_simulator.sr)
                - float(start_cutoff * 1.0 / self._params.data_simulator.sr)
                + file['alignments'][i]
            )

            if word == "":
                i += 1
                continue
            elif self._text == "":
                self._text += word
            else:
                self._text += " " + word
            i += 1
            nw += 1
            prev_dur_sr = dur_sr

        # add audio clip up to the final alignment
        if (
            sentence_duration == 0
        ) and self._params.data_simulator.session_params.window_type != None:  # cut off the start of the sentence
            if start_window_amount > 0:  # include window
                window = self._get_window(start_window_amount, start=True)
                self._sentence = torch.cat(
                    (
                        self._sentence,
                        np.multiply(audio_file[start_cutoff : start_cutoff + start_window_amount], window),
                    ),
                    0,
                )
            self._sentence = torch.cat(
                (self._sentence, audio_file[start_cutoff + start_window_amount : start_cutoff + prev_dur_sr]), 0
            )
        else:
            self._sentence = torch.cat((self._sentence, audio_file[:prev_dur_sr]), 0)

        # windowing at the end of the sentence
        if (i < len(file['words'])) and self._params.data_simulator.session_params.window_type != None:
            release_buffer, end_window_amount = self._get_end_buffer_and_window(
                prev_dur_sr, remaining_duration_sr, len(audio_file[start_cutoff + prev_dur_sr :])
            )
            self._sentence = torch.cat(
                (self._sentence, audio_file[start_cutoff + prev_dur_sr : start_cutoff + prev_dur_sr + release_buffer]),
                0,
            )
            if end_window_amount > 0:  # include window
                window = self._get_window(end_window_amount, start=False)
                self._sentence = torch.cat(
                    (
                        self._sentence,
                        np.multiply(
                            audio_file[
                                start_cutoff
                                + prev_dur_sr
                                + release_buffer : start_cutoff
                                + prev_dur_sr
                                + release_buffer
                                + end_window_amount
                            ],
                            window,
                        ),
                    ),
                    0,
                )

        return sentence_duration + nw, len(self._sentence)

    def _build_sentence(
        self, speaker_turn: int, speaker_ids: List[str], speaker_lists: List[dict], max_sentence_duration_sr: int
    ):
        """
        Build a new sentence by attaching utterance samples together until the sentence has reached a desired length. 
        While generating the sentence, alignment information is used to segment the audio.

        Args:
            speaker_turn (int): Current speaker turn.
            speaker_ids (list): LibriSpeech speaker IDs for each speaker in the current session.
            speaker_lists (list): List of samples for each speaker in the session.
            max_sentence_duration_sr (int): Maximum length for sentence in terms of samples
        """
        # select speaker length
        sl = (
            np.random.negative_binomial(
                self._params.data_simulator.session_params.sentence_length_params[0],
                self._params.data_simulator.session_params.sentence_length_params[1],
            )
            + 1
        )

        # initialize sentence, text, words, alignments
        self._sentence = torch.zeros(0)
        self._text = ""
        self._words = []
        self._alignments = []
        sentence_duration = sentence_duration_sr = 0

        # build sentence
        while sentence_duration < sl and sentence_duration_sr < max_sentence_duration_sr:
            file = self._load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
            audio_file, sr = sf.read(file['audio_filepath'])
            audio_file = torch.from_numpy(audio_file)
            if audio_file.ndim > 1:
                audio_file = torch.mean(audio_file, 1, False)
            sentence_duration, sentence_duration_sr = self._add_file(
                file, audio_file, sentence_duration, sl, max_sentence_duration_sr
            )

        # look for split locations
        splits = []
        new_start = 0
        for i in range(len(self._words)):
            if self._words[i] == "" and i != 0 and i != len(self._words) - 1:
                silence_length = self._alignments[i] - self._alignments[i - 1]
                if (
                    silence_length > 2 * self._params.data_simulator.session_params.split_buffer
                ):  # split utterance on silence
                    new_end = self._alignments[i - 1] + self._params.data_simulator.session_params.split_buffer
                    splits.append(
                        [
                            int(new_start * self._params.data_simulator.sr),
                            int(new_end * self._params.data_simulator.sr),
                        ]
                    )
                    new_start = self._alignments[i] - self._params.data_simulator.session_params.split_buffer
        splits.append([int(new_start * self._params.data_simulator.sr), len(self._sentence)])

        # per-speaker normalization (accounting for active speaker time)
        if self._params.data_simulator.session_params.normalize:
            if torch.max(torch.abs(self._sentence)) > 0:
                split_length = split_sum = 0
                for split in splits:
                    split_length += len(self._sentence[split[0] : split[1]])
                    split_sum += torch.sum(self._sentence[split[0] : split[1]] ** 2)
                average_rms = torch.sqrt(split_sum * 1.0 / split_length)
                self._sentence = self._sentence / (1.0 * average_rms) * self._volume[speaker_turn]

    # returns new overlapped (or shifted) start position
    def _add_silence_or_overlap(
        self,
        speaker_turn: int,
        prev_speaker: int,
        start: int,
        length: int,
        session_length_sr: int,
        prev_length_sr: int,
        enforce: bool,
    ) -> int:
        """
        Returns new overlapped (or shifted) start position after inserting overlap or silence.

        Args:
            speaker_turn (int): Current speaker turn.
            prev_speaker (int): Previous speaker turn.
            start (int): Current start of the audio file being inserted.
            length (int): Length of the audio file being inserted.
            session_length_sr (int): Running length of the session in terms of number of samples
            prev_length_sr (int): Length of previous sentence (in terms of number of samples)
            enforce (bool): Whether speaker enforcement mode is being used
        Returns:
            new_start (int): New starting position in the session accounting for overlap or silence
        """
        overlap_prob = self._params.data_simulator.session_params.overlap_prob / (
            self._params.data_simulator.session_params.turn_prob
        )  # accounting for not overlapping the same speaker
        mean_overlap_percent = (
            self._params.data_simulator.session_params.mean_overlap
            / (1 + self._params.data_simulator.session_params.mean_overlap)
        ) / self._params.data_simulator.session_params.overlap_prob
        mean_silence_percent = self._params.data_simulator.session_params.mean_silence / (
            1 - self._params.data_simulator.session_params.overlap_prob
        )

        # overlap
        if prev_speaker != speaker_turn and prev_speaker != None and np.random.uniform(0, 1) < overlap_prob:
            overlap_percent = halfnorm(loc=0, scale=mean_overlap_percent * np.sqrt(np.pi) / np.sqrt(2)).rvs()
            desired_overlap_amount = int(prev_length_sr * overlap_percent)
            new_start = start - desired_overlap_amount

            # reinject missing overlap to ensure desired overlap percentage is met
            if self._missing_overlap > 0 and overlap_percent < 1:
                rand = int(
                    prev_length_sr
                    * np.random.uniform(
                        0, 1 - overlap_percent / (1 + self._params.data_simulator.session_params.mean_overlap)
                    )
                )
                if rand > self._missing_overlap:
                    new_start -= self._missing_overlap
                    desired_overlap_amount += self._missing_overlap
                    self._missing_overlap = 0
                else:
                    new_start -= rand
                    desired_overlap_amount += rand
                    self._missing_overlap -= rand

            # avoid overlap at start of clip
            if new_start < 0:
                desired_overlap_amount -= 0 - new_start
                self._missing_overlap += 0 - new_start
                new_start = 0

            # if same speaker ends up overlapping from any previous clip, pad with silence instead
            if new_start < self._furthest_sample[speaker_turn]:
                desired_overlap_amount -= self._furthest_sample[speaker_turn] - new_start
                self._missing_overlap += self._furthest_sample[speaker_turn] - new_start
                new_start = self._furthest_sample[speaker_turn]

            prev_start = start - prev_length_sr
            prev_end = start
            new_end = new_start + length
            overlap_amount = 0
            if prev_start < new_start and new_end > prev_end:
                overlap_amount = prev_end - new_start
            elif prev_start < new_start and new_end < prev_end:
                overlap_amount = new_end - new_start
            elif prev_start > new_start and new_end < prev_end:
                overlap_amount = new_end - prev_start
            elif prev_start > new_start and new_end > prev_end:
                overlap_amount = prev_end - prev_start

            overlap_amount = max(overlap_amount, 0)
            if overlap_amount < desired_overlap_amount:
                self._missing_overlap += desired_overlap_amount - overlap_amount

        else:
            # add silence
            silence_percent = halfnorm(loc=0, scale=mean_silence_percent * np.sqrt(np.pi) / np.sqrt(2)).rvs()
            silence_amount = int(length * silence_percent)

            if start + length + silence_amount > session_length_sr and not enforce:
                new_start = session_length_sr - length
            else:
                new_start = start + silence_amount

        return new_start

    def _get_background(self, len_array: int, power_array: float) -> torch.Tensor:
        """
        Augment with background noise (inserting ambient background noise up to the desired SNR for the full clip).

        Args:
            len_array (int): Length of background noise required.
            avg_power_array (float): Average power of the audio file.
        Returns:
            bg_array (tensor): Tensor containing background noise
        """

        manifest = read_manifest(self._params.data_simulator.background_noise.background_manifest)
        bg_array = torch.zeros(len_array)
        desired_snr = self._params.data_simulator.background_noise.snr
        ratio = 10 ** (desired_snr / 20)
        desired_avg_power_noise = power_array / ratio
        running_len = 0
        while running_len < len_array:  # build background audio stream (the same length as the full file)
            file_id = np.random.randint(0, len(manifest) - 1)
            file = manifest[file_id]
            audio_file, sr = sf.read(file['audio_filepath'])
            audio_file = torch.from_numpy(audio_file)
            if audio_file.ndim > 1:
                audio_file = torch.mean(audio_file, 1, False)

            if running_len + len(audio_file) < len_array:
                end_audio_file = running_len + len(audio_file)
            else:
                end_audio_file = len_array

            pow_audio_file = torch.mean(audio_file[: end_audio_file - running_len] ** 2)
            scaled_audio_file = audio_file[: end_audio_file - running_len] * torch.sqrt(
                desired_avg_power_noise / pow_audio_file
            )

            bg_array[running_len:end_audio_file] = scaled_audio_file
            running_len = end_audio_file

        return bg_array

    def _create_new_rttm_entry(self, start: int, end: int, speaker_id: int) -> List[str]:
        """
        Create new RTTM entries (to write to output rttm file)

        Args:
            start (int): Current start of the audio file being inserted.
            end (int): End of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
        Returns:
            rttm_list (list): List of rttm entries
        """
        rttm_list = []
        new_start = start
        # look for split locations
        for i in range(len(self._words)):
            if self._words[i] == "" and i != 0 and i != len(self._words) - 1:
                silence_length = self._alignments[i] - self._alignments[i - 1]
                if (
                    silence_length > 2 * self._params.data_simulator.session_params.split_buffer
                ):  # split utterance on silence
                    new_end = start + self._alignments[i - 1] + self._params.data_simulator.session_params.split_buffer
                    s = float(round(new_start, self._params.data_simulator.outputs.output_precision))
                    e = float(round(new_end, self._params.data_simulator.outputs.output_precision))
                    rttm_list.append(f"{s} {e} {speaker_id}")
                    new_start = start + self._alignments[i] - self._params.data_simulator.session_params.split_buffer

        s = float(round(new_start, self._params.data_simulator.outputs.output_precision))
        e = float(round(end, self._params.data_simulator.outputs.output_precision))
        rttm_list.append(f"{s} {e} {speaker_id}")
        return rttm_list

    def _create_new_json_entry(
        self, wav_filename: str, start: int, length: int, speaker_id: int, rttm_filepath: str, ctm_filepath: str
    ) -> dict:
        """
        Create new JSON entries (to write to output json file).

        Args:
            wav_filename (str): Output wav filepath.
            start (int): Current start of the audio file being inserted.
            length (int): Length of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
            rttm_filepath (str): Output rttm filepath.
            ctm_filepath (str): Output ctm filepath.
        Returns:
            dict (dict): JSON entry
        """
        start = float(round(start, self._params.data_simulator.outputs.output_precision))
        length = float(round(length, self._params.data_simulator.outputs.output_precision))
        meta = {
            "audio_filepath": wav_filename,
            "offset": start,
            "duration": length,
            "label": speaker_id,
            "text": self._text,
            "num_speakers": self._params.data_simulator.session_config.num_speakers,
            "rttm_filepath": rttm_filepath,
            "ctm_filepath": ctm_filepath,
            "uem_filepath": None,
        }
        return meta

    def _create_new_ctm_entry(self, session_name: str, speaker_id: int, start: int) -> List[str]:
        """
        Create new CTM entry (to write to output ctm file)

        Args:
            session_name (str): Current session name.
            start (int): Current start of the audio file being inserted.
            speaker_id (int): LibriSpeech speaker ID for the current entry.
        Returns:
            arr (list): List of ctm entries
        """
        arr = []
        start = float(round(start, self._params.data_simulator.outputs.output_precision))
        for i in range(len(self._words)):
            word = self._words[i]
            if (
                word != ""
            ):  # note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
                align1 = float(
                    round(self._alignments[i - 1] + start, self._params.data_simulator.outputs.output_precision)
                )
                align2 = float(
                    round(
                        self._alignments[i] - self._alignments[i - 1],
                        self._params.data_simulator.outputs.output_precision,
                    )
                )
                text = f"{session_name} {speaker_id} {align1} {align2} {word} 0\n"
                arr.append((align1, text))
        return arr

    def create_base_manifest_ds(self) -> str:
        """
        Create base diarization manifest file for online data simulation.

        Returns:
            self.base_manifest_filepath (str): Path to manifest file
        """
        basepath = self._params.data_simulator.outputs.output_dir
        wav_path = os.path.join(basepath, 'synthetic_wav.list')
        text_path = os.path.join(basepath, 'synthetic_txt.list')
        rttm_path = os.path.join(basepath, 'synthetic_rttm.list')
        ctm_path = os.path.join(basepath, 'synthetic_ctm.list')
        manifest_filepath = os.path.join(basepath, 'base_manifest.json')

        create_manifest(
            wav_path,
            manifest_filepath,
            text_path=text_path,
            rttm_path=rttm_path,
            ctm_path=ctm_path,
            add_duration=False,
        )

        self.base_manifest_filepath = manifest_filepath
        return self.base_manifest_filepath

    def create_segment_manifest_ds(self) -> str:
        """
        Create segmented diarization manifest file for online data simulation.

        Returns:
            self.segment_manifest_filepath (str): Path to manifest file
        """
        basepath = self._params.data_simulator.outputs.output_dir
        output_manifest_filepath = os.path.join(basepath, 'segment_manifest.json')
        input_manifest_filepath = self.base_manifest_filepath
        window = self._params.data_simulator.segment_manifest.window
        shift = self._params.data_simulator.segment_manifest.shift
        step_count = self._params.data_simulator.segment_manifest.step_count
        deci = self._params.data_simulator.segment_manifest.deci

        create_segment_manifest(input_manifest_filepath, output_manifest_filepath, window, shift, step_count, deci)

        self.segment_manifest_filepath = output_manifest_filepath
        return self.segment_manifest_filepath

    def _generate_session(self, idx: int, basepath: str, filename: str, enforce_counter: int = 2):
        """
        Generate a multispeaker audio session and corresponding label files.

        Args:
            idx (int): Index for current session (out of total number of sessions).
            basepath (str): Path to output directory.
            filename (str): Filename for output files.
            enforce_counter (int): In enforcement mode, dominance is increased by a factor of enforce_counter for unrepresented speakers
        """
        speaker_ids = self._get_speaker_ids()  # randomly select speaker IDs
        speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
        base_speaker_dominance = np.copy(speaker_dominance)
        speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker
        self._set_speaker_volume()

        running_length_sr, prev_length_sr = 0, 0
        prev_speaker = None
        rttm_list, json_list, ctm_list = [], [], []
        self._furthest_sample = [0 for n in range(self._params.data_simulator.session_config.num_speakers)]
        self._missing_overlap = 0

        # hold enforce until all speakers have spoken
        enforce_time = np.random.uniform(
            self._params.data_simulator.speaker_enforcement.enforce_time[0],
            self._params.data_simulator.speaker_enforcement.enforce_time[1],
        )
        enforce = self._params.data_simulator.speaker_enforcement.enforce_num_speakers

        session_length_sr = int(
            (self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr)
        )
        array = torch.zeros(session_length_sr)
        is_bg = torch.zeros(session_length_sr)

        while running_length_sr < session_length_sr or enforce:
            # enforce num_speakers
            if running_length_sr > enforce_time * session_length_sr and enforce:
                speaker_dominance, enforce = self._increase_speaker_dominance(base_speaker_dominance, enforce_counter)
                if enforce:
                    enforce_counter += 1

            # select speaker
            speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

            # build sentence (only add if remaining length >  specific time)
            max_sentence_duration_sr = session_length_sr - running_length_sr
            if enforce:
                max_sentence_duration_sr = float('inf')
            elif (
                max_sentence_duration_sr
                < self._params.data_simulator.session_params.end_buffer * self._params.data_simulator.sr
            ):
                break
            self._build_sentence(speaker_turn, speaker_ids, speaker_lists, max_sentence_duration_sr)

            length = len(self._sentence)
            start = self._add_silence_or_overlap(
                speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr, enforce
            )
            end = start + length
            if end > len(array):  # only occurs in enforce mode
                array = torch.nn.functional.pad(array, (0, end - len(array)))
                is_bg = torch.nn.functional.pad(is_bg, (0, end - len(is_bg)))
            array[start:end] += self._sentence
            is_bg[start:end] = 1

            # build entries for output files
            new_rttm_entries = self._create_new_rttm_entry(
                start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn]
            )
            for entry in new_rttm_entries:
                rttm_list.append(entry)
            new_json_entry = self._create_new_json_entry(
                os.path.join(basepath, filename + '.wav'),
                start / self._params.data_simulator.sr,
                length / self._params.data_simulator.sr,
                speaker_ids[speaker_turn],
                os.path.join(basepath, filename + '.rttm'),
                os.path.join(basepath, filename + '.ctm'),
            )
            json_list.append(new_json_entry)
            new_ctm_entries = self._create_new_ctm_entry(
                filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr
            )
            for entry in new_ctm_entries:
                ctm_list.append(entry)

            running_length_sr = np.maximum(running_length_sr, end)
            self._furthest_sample[speaker_turn] = running_length_sr
            prev_speaker = speaker_turn
            prev_length_sr = length

        # background noise augmentation
        if self._params.data_simulator.background_noise.add_bg:
            avg_power_array = torch.mean(array[is_bg == 1] ** 2)
            bg = self._get_background(len(array), avg_power_array)
            array += bg

        array = array / (1.0 * torch.max(torch.abs(array)))  # normalize wav file to avoid clipping
        sf.write(os.path.join(basepath, filename + '.wav'), array, self._params.data_simulator.sr)
        labels_to_rttmfile(rttm_list, filename, self._params.data_simulator.outputs.output_dir)
        write_manifest(os.path.join(basepath, filename + '.json'), json_list)
        write_ctm(os.path.join(basepath, filename + '.ctm'), ctm_list)
        write_text(os.path.join(basepath, filename + '.txt'), ctm_list)

    def generate_sessions(self):
        """
        Generate several multispeaker audio sessions and corresponding list files.
        """
        logging.info(f"Generating Diarization Sessions")
        np.random.seed(self._params.data_simulator.random_seed)
        output_dir = self._params.data_simulator.outputs.output_dir

        # delete output directory if it exists or throw warning
        if os.path.isdir(output_dir) and os.listdir(output_dir):
            if self._params.data_simulator.outputs.overwrite_output:
                shutil.rmtree(output_dir)
                os.mkdir(output_dir)
            else:
                raise Exception("Output directory is nonempty and overwrite_output = false")
        elif not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # only add root if paths are relative
        if not os.path.isabs(output_dir):
            ROOT = os.getcwd()
            basepath = os.path.join(ROOT, output_dir)
        else:
            basepath = output_dir

        wavlist = open(os.path.join(basepath, "synthetic_wav.list"), "w")
        rttmlist = open(os.path.join(basepath, "synthetic_rttm.list"), "w")
        jsonlist = open(os.path.join(basepath, "synthetic_json.list"), "w")
        ctmlist = open(os.path.join(basepath, "synthetic_ctm.list"), "w")
        textlist = open(os.path.join(basepath, "synthetic_txt.list"), "w")

        for i in trange(self._params.data_simulator.session_config.num_sessions):
            self._furthest_sample = [0 for n in range(self._params.data_simulator.session_config.num_speakers)]
            self._missing_overlap = 0

            filename = self._params.data_simulator.outputs.output_filename + f"_{i}"
            self._generate_session(i, basepath, filename)

            wavlist.write(os.path.join(basepath, filename + '.wav\n'))
            rttmlist.write(os.path.join(basepath, filename + '.rttm\n'))
            jsonlist.write(os.path.join(basepath, filename + '.json\n'))
            ctmlist.write(os.path.join(basepath, filename + '.ctm\n'))
            textlist.write(os.path.join(basepath, filename + '.txt\n'))

            # throw error if number of speakers is less than requested
            num_missing = 0
            for k in range(len(self._furthest_sample)):
                if self._furthest_sample[k] == 0:
                    num_missing += 1
            if num_missing != 0:
                warnings.warn(
                    f"{self._params.data_simulator.session_config.num_speakers-num_missing} speakers were included in the clip instead of the requested amount of {self._params.data_simulator.session_config.num_speakers}"
                )

        wavlist.close()
        rttmlist.close()
        jsonlist.close()
        ctmlist.close()
        textlist.close()


class RIRMultiSpeakerSimulator(MultiSpeakerSimulator):
    """
    RIR Augmented Multispeaker Audio Session Simulator - simulates multispeaker audio sessions using single-speaker 
    audio files and corresponding word alignments, as well as simulated RIRs for augmentation.

    Args:
        cfg: OmegaConf configuration loaded from yaml file.

    Parameters (in addition to the base MultiSpeakerSimulator parameters):
    rir_generation:
      use_rir (bool): Whether to generate synthetic RIR
      toolkit (str): Which toolkit to use ("pyroomacoustics", "gpuRIR")
      room_config:
        room_sz (list): Size of the shoebox room environment (1d array for specific, 2d array for random range to be 
                        sampled from)
        pos_src (list): Positions of the speakers in the simulated room environment (2d array for specific, 3d array 
                        for random ranges to be sampled from)
        noise_src_pos (list): Position in room for the ambient background noise source
      mic_config:
        num_channels (int): Number of output audio channels
        pos_rcv (list): Microphone positions in the simulated room environment (1d/2d array for specific, 2d/3d array 
                        for range assuming num_channels is 1/2+)
        orV_rcv (list or null): Microphone orientations (needed for non-omnidirectional microphones)
        mic_pattern (str): Microphone type ("omni" - omnidirectional) - currently only omnidirectional microphones are 
                           supported for pyroomacoustics
      absorbtion_params: (Note that only `T60` is used for pyroomacoustics simulations)
        abs_weights (list): Absorption coefficient ratios for each surface
        T60 (float): Room reverberation time (`T60` is the time it takes for the RIR to decay by 60DB)
        att_diff (float): Starting attenuation (if this is different than att_max, the diffuse reverberation model is
                          used by gpuRIR)
        att_max (float): End attenuation when using the diffuse reverberation model (gpuRIR)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._check_args_rir()

    def _check_args_rir(self):
        """
        Checks RIR YAML arguments to ensure they are within valid ranges
        """

        if not (self._params.data_simulator.rir_generation.toolkit in ['pyroomacoustics', 'gpuRIR']):
            raise Exception("Toolkit must be pyroomacoustics or gpuRIR")
        if self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics' and not PRA:
            raise ImportError("pyroomacoustics should be installed to run this simulator with RIR augmentation")

        if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR' and not GPURIR:
            raise ImportError("gpuRIR should be installed to run this simulator with RIR augmentation")

        if len(self._params.data_simulator.rir_generation.room_config.room_sz) != 3:
            raise Exception("Incorrect room dimensions provided")
        if self._params.data_simulator.rir_generation.mic_config.num_channels == 0:
            raise Exception("Number of channels should be greater or equal to 1")
        if len(self._params.data_simulator.rir_generation.room_config.pos_src) < 2:
            raise Exception("Less than 2 provided source positions")
        for sublist in self._params.data_simulator.rir_generation.room_config.pos_src:
            if len(sublist) != 3:
                raise Exception("Three coordinates must be provided for sources positions")
        if len(self._params.data_simulator.rir_generation.mic_config.pos_rcv) == 0:
            raise Exception("No provided mic positions")
        for sublist in self._params.data_simulator.rir_generation.room_config.pos_src:
            if len(sublist) != 3:
                raise Exception("Three coordinates must be provided for mic positions")

        if self._params.data_simulator.session_config.num_speakers != len(
            self._params.data_simulator.rir_generation.room_config.pos_src
        ):
            raise Exception("Number of speakers is not equal to the number of provided source positions")
        if self._params.data_simulator.rir_generation.mic_config.num_channels != len(
            self._params.data_simulator.rir_generation.mic_config.pos_rcv
        ):
            raise Exception("Number of channels is not equal to the number of provided microphone positions")

        if (
            not self._params.data_simulator.rir_generation.mic_config.orV_rcv
            and self._params.data_simulator.rir_generation.mic_config.mic_pattern != 'omni'
        ):
            raise Exception("Microphone orientations must be provided if mic_pattern != omni")
        if self._params.data_simulator.rir_generation.mic_config.orV_rcv != None:
            if len(self._params.data_simulator.rir_generation.mic_config.orV_rcv) != len(
                self._params.data_simulator.rir_generation.mic_config.pos_rcv
            ):
                raise Exception("A different number of microphone orientations and microphone positions were provided")
            for sublist in self._params.data_simulator.rir_generation.mic_config.orV_rcv:
                if len(sublist) != 3:
                    raise Exception("Three coordinates must be provided for orientations")

    def _generate_rir_gpuRIR(self) -> Tuple[torch.Tensor, int]:
        """
        Create simulated RIR using the gpuRIR library

        Returns:
            RIR (tensor): Generated RIR
            RIR_pad (int): Length of padding added when convolving the RIR with an audio file
        """
        room_sz_tmp = np.array(self._params.data_simulator.rir_generation.room_config.room_sz)
        if room_sz_tmp.ndim == 2:  # randomize
            room_sz = np.zeros(room_sz_tmp.shape[0])
            for i in range(room_sz_tmp.shape[0]):
                room_sz[i] = np.random.uniform(room_sz_tmp[i, 0], room_sz_tmp[i, 1])
        else:
            room_sz = room_sz_tmp

        pos_src_tmp = np.array(self._params.data_simulator.rir_generation.room_config.pos_src)
        if pos_src_tmp.ndim == 3:  # randomize
            pos_src = np.zeros((pos_src_tmp.shape[0], pos_src_tmp.shape[1]))
            for i in range(pos_src_tmp.shape[0]):
                for j in range(pos_src_tmp.shape[1]):
                    pos_src[i] = np.random.uniform(pos_src_tmp[i, j, 0], pos_src_tmp[i, j, 1])
        else:
            pos_src = pos_src_tmp

        if self._params.data_simulator.background_noise.add_bg:
            pos_src = np.vstack((pos_src, self._params.data_simulator.rir_generation.room_config.noise_src_pos))

        mic_pos_tmp = np.array(self._params.data_simulator.rir_generation.mic_config.pos_rcv)
        if mic_pos_tmp.ndim == 3:  # randomize
            mic_pos = np.zeros((mic_pos_tmp.shape[0], mic_pos_tmp.shape[1]))
            for i in range(mic_pos_tmp.shape[0]):
                for j in range(mic_pos_tmp.shape[1]):
                    mic_pos[i] = np.random.uniform(mic_pos_tmp[i, j, 0], mic_pos_tmp[i, j, 1])
        else:
            mic_pos = mic_pos_tmp

        orV_rcv = self._params.data_simulator.rir_generation.mic_config.orV_rcv
        if orV_rcv:  # not needed for omni mics
            orV_rcv = np.array(orV_rcv)
        mic_pattern = self._params.data_simulator.rir_generation.mic_config.mic_pattern
        abs_weights = self._params.data_simulator.rir_generation.absorbtion_params.abs_weights
        T60 = self._params.data_simulator.rir_generation.absorbtion_params.T60
        att_diff = self._params.data_simulator.rir_generation.absorbtion_params.att_diff
        att_max = self._params.data_simulator.rir_generation.absorbtion_params.att_max
        sr = self._params.data_simulator.sr

        beta = beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
        Tdiff = att2t_SabineEstimator(att_diff, T60)  # Time to start the diffuse reverberation model [s]
        Tmax = att2t_SabineEstimator(att_max, T60)  # Time to stop the simulation [s]
        nb_img = t2n(Tdiff, room_sz)  # Number of image sources in each dimension
        RIR = simulateRIR(
            room_sz, beta, pos_src, mic_pos, nb_img, Tmax, sr, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern
        )
        RIR_pad = RIR.shape[2] - 1
        return RIR, RIR_pad

    def _generate_rir_pyroomacoustics(self) -> Tuple[torch.Tensor, int]:
        """
        Create simulated RIR using the pyroomacoustics library

        Returns:
            RIR (tensor): Generated RIR
            RIR_pad (int): Length of padding added when convolving the RIR with an audio file
        """

        rt60 = self._params.data_simulator.rir_generation.absorbtion_params.T60  # The desired reverberation time
        sr = self._params.data_simulator.sr

        room_sz_tmp = np.array(self._params.data_simulator.rir_generation.room_config.room_sz)
        if room_sz_tmp.ndim == 2:  # randomize
            room_sz = np.zeros(room_sz_tmp.shape[0])
            for i in range(room_sz_tmp.shape[0]):
                room_sz[i] = np.random.uniform(room_sz_tmp[i, 0], room_sz_tmp[i, 1])
        else:
            room_sz = room_sz_tmp

        pos_src_tmp = np.array(self._params.data_simulator.rir_generation.room_config.pos_src)
        if pos_src_tmp.ndim == 3:  # randomize
            pos_src = np.zeros((pos_src_tmp.shape[0], pos_src_tmp.shape[1]))
            for i in range(pos_src_tmp.shape[0]):
                for j in range(pos_src_tmp.shape[1]):
                    pos_src[i] = np.random.uniform(pos_src_tmp[i, j, 0], pos_src_tmp[i, j, 1])
        else:
            pos_src = pos_src_tmp

        # We invert Sabine's formula to obtain the parameters for the ISM simulator
        e_absorption, max_order = pra.inverse_sabine(rt60, room_sz)
        room = pra.ShoeBox(room_sz, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)

        if self._params.data_simulator.background_noise.add_bg:
            pos_src = np.vstack((pos_src, self._params.data_simulator.rir_generation.room_config.noise_src_pos))
        for pos in pos_src:
            room.add_source(pos)

        # currently only supports omnidirectional microphones
        mic_pattern = self._params.data_simulator.rir_generation.mic_config.mic_pattern
        if self._params.data_simulator.rir_generation.mic_config.mic_pattern == 'omni':
            mic_pattern = DirectivityPattern.OMNI
            dir_vec = DirectionVector(azimuth=0, colatitude=90, degrees=True)
        dir_obj = CardioidFamily(orientation=dir_vec, pattern_enum=mic_pattern,)

        mic_pos_tmp = np.array(self._params.data_simulator.rir_generation.mic_config.pos_rcv)
        if mic_pos_tmp.ndim == 3:  # randomize
            mic_pos = np.zeros((mic_pos_tmp.shape[0], mic_pos_tmp.shape[1]))
            for i in range(mic_pos_tmp.shape[0]):
                for j in range(mic_pos_tmp.shape[1]):
                    mic_pos[i] = np.random.uniform(mic_pos_tmp[i, j, 0], mic_pos_tmp[i, j, 1])
        else:
            mic_pos = mic_pos_tmp

        room.add_microphone_array(mic_pos.T, directivity=dir_obj)

        room.compute_rir()
        rir_pad = 0
        for channel in room.rir:
            for pos in channel:
                if pos.shape[0] - 1 > rir_pad:
                    rir_pad = pos.shape[0] - 1
        return room.rir, rir_pad

    def _convolve_rir(self, input, speaker_turn: int, RIR: torch.Tensor) -> Tuple[list, int]:
        """
        Augment one sentence (or background noise segment) using a synthetic RIR.

        Args:
            input (torch.tensor): Input audio.
            speaker_turn (int): Current speaker turn.
            RIR (torch.tensor): Room Impulse Response.
        Returns:
            output_sound (list): List of tensors containing augmented audio
            length (int): Length of output audio channels (or of the longest if they have different lengths)
        """
        output_sound = []
        length = 0
        for channel in range(self._params.data_simulator.rir_generation.mic_config.num_channels):
            if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR':
                out_channel = convolve(input, RIR[speaker_turn, channel, : len(input)]).tolist()
            elif self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics':
                out_channel = convolve(input, RIR[channel][speaker_turn][: len(input)]).tolist()
            if len(out_channel) > length:
                length = len(out_channel)
            output_sound.append(torch.tensor(out_channel))
        return output_sound, length

    def _generate_session(self, idx: int, basepath: str, filename: str, enforce_counter: int = 2):
        """
        Generate a multispeaker audio session and corresponding label files.

        Args:
            idx (int): Index for current session (out of total number of sessions).
            basepath (str): Path to output directory.
            filename (str): Filename for output files.
            enforce_counter (int): In enforcement mode, dominance is increased by a factor of enforce_counter for unrepresented speakers
        """
        speaker_ids = self._get_speaker_ids()  # randomly select speaker IDs
        speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
        base_speaker_dominance = np.copy(speaker_dominance)
        speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker
        self._set_speaker_volume()

        running_length_sr, prev_length_sr = 0, 0  # starting point for each sentence
        prev_speaker = None
        rttm_list, json_list, ctm_list = [], [], []
        self._furthest_sample = [0 for n in range(self._params.data_simulator.session_config.num_speakers)]
        self._missing_overlap = 0

        # Room Impulse Response Generation (performed once per batch of sessions)
        if self._params.data_simulator.rir_generation.toolkit == 'gpuRIR':
            RIR, RIR_pad = self._generate_rir_gpuRIR()
        elif self._params.data_simulator.rir_generation.toolkit == 'pyroomacoustics':
            RIR, RIR_pad = self._generate_rir_pyroomacoustics()
        else:
            raise Exception("Toolkit must be pyroomacoustics or gpuRIR")

        # hold enforce until all speakers have spoken
        enforce_time = np.random.uniform(
            self._params.data_simulator.speaker_enforcement.enforce_time[0],
            self._params.data_simulator.speaker_enforcement.enforce_time[1],
        )
        enforce = self._params.data_simulator.speaker_enforcement.enforce_num_speakers

        session_length_sr = int(
            (self._params.data_simulator.session_config.session_length * self._params.data_simulator.sr)
        )
        array = torch.zeros((session_length_sr, self._params.data_simulator.rir_generation.mic_config.num_channels))
        is_bg = torch.zeros(session_length_sr)

        while running_length_sr < session_length_sr or enforce:
            # enforce num_speakers
            if running_length_sr > enforce_time * session_length_sr and enforce:
                speaker_dominance, enforce = self._increase_speaker_dominance(base_speaker_dominance, enforce_counter)
                if enforce:
                    enforce_counter += 1

            # select speaker
            speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

            # build sentence (only add if remaining length >  specific time)
            max_sentence_duration_sr = (
                session_length_sr - running_length_sr - RIR_pad
            )  # sentence will be RIR_len - 1 longer than the audio was pre-augmentation
            if enforce:
                max_sentence_duration_sr = float('inf')
            elif (
                max_sentence_duration_sr
                < self._params.data_simulator.session_params.end_buffer * self._params.data_simulator.sr
            ):
                break
            self._build_sentence(speaker_turn, speaker_ids, speaker_lists, max_sentence_duration_sr)
            augmented_sentence, length = self._convolve_rir(self._sentence, speaker_turn, RIR)

            start = self._add_silence_or_overlap(
                speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr, enforce
            )
            end = start + length
            if end > len(array):
                array = torch.nn.functional.pad(array, (0, 0, 0, end - len(array)))
                is_bg = torch.nn.functional.pad(is_bg, (0, end - len(is_bg)))

            is_bg[start:end] = 1

            for channel in range(self._params.data_simulator.rir_generation.mic_config.num_channels):
                len_ch = len(augmented_sentence[channel])  # accounts for how channels are slightly different lengths
                array[start : start + len_ch, channel] += augmented_sentence[channel]

            # build entries for output files
            new_rttm_entries = self._create_new_rttm_entry(
                start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn]
            )
            for entry in new_rttm_entries:
                rttm_list.append(entry)
            new_json_entry = self._create_new_json_entry(
                os.path.join(basepath, filename + '.wav'),
                start / self._params.data_simulator.sr,
                length / self._params.data_simulator.sr,
                speaker_ids[speaker_turn],
                os.path.join(basepath, filename + '.rttm'),
                os.path.join(basepath, filename + '.ctm'),
            )
            json_list.append(new_json_entry)
            new_ctm_entries = self._create_new_ctm_entry(
                filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr
            )
            for entry in new_ctm_entries:
                ctm_list.append(entry)

            running_length_sr = np.maximum(running_length_sr, end)
            self._furthest_sample[speaker_turn] = running_length_sr
            prev_speaker = speaker_turn
            prev_length_sr = length

        # background noise augmentation
        if self._params.data_simulator.background_noise.add_bg:
            avg_power_array = torch.mean(array[is_bg == 1] ** 2)
            length = array.shape[0]
            bg = self._get_background(length, avg_power_array)
            augmented_bg, _ = self._convolve_rir(bg, -1, RIR)
            for channel in range(self._params.data_simulator.rir_generation.mic_config.num_channels):
                array[:, channel] += augmented_bg[channel][:length]

        array = array / (1.0 * torch.max(torch.abs(array)))  # normalize wav file to avoid clipping
        sf.write(os.path.join(basepath, filename + '.wav'), array, self._params.data_simulator.sr)
        labels_to_rttmfile(rttm_list, filename, self._params.data_simulator.outputs.output_dir)
        write_manifest(os.path.join(basepath, filename + '.json'), json_list)
        write_ctm(os.path.join(basepath, filename + '.ctm'), ctm_list)
        write_text(os.path.join(basepath, filename + '.txt'), ctm_list)
