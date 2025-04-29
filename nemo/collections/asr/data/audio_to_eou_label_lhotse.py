# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.utils.data
from lhotse.cut import Cut, CutSet, MixedCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

EOU_LABEL = 2
EOB_LABEL = 3
EOU_STRING = '<EOU>'
EOB_STRING = '<EOB>'


EOU_LENGTH_PERTURBATION = ['speed', 'time_stretch']
EOU_PROHIBITED_AUGMENTATIONS = ['random_segment']


@dataclass
class AudioToTextEOUBatch:
    sample_ids: List | None = None
    audio_filepaths: List | None = None
    audio_signal: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    text_token_lengths: torch.Tensor | None = None
    eou_targets: torch.Tensor | None = None
    eou_target_lengths: torch.Tensor | None = None


class LhotseSpeechToTextBpeEOUDataset(torch.utils.data.Dataset):
    """
    This dataset processes the audio data and the corresponding text data to generate the ASR labels,
    along with EOU labels for each frame. The audios used in this dataset should only contain speech with
    NO precedding or following silence. The dataset also randomly pads non-speech frames before and after
    the audio signal for training EOU prediction task.

    To generate EOU labels, the last frame of utterance will be marked as "end of utterance" (labeled as `2`),
    while if it's a backchannel utterance it'll be marked asd "end of backchannel" (labeled as `3`).
    The rest of the speech frames will be marked as "speech" (labeled as `1`).
    The padded non-speech signals will be marked as "non-speech" (labeled as 0).

    Args:
        cfg: DictConfig object container following keys, usually taken from your `model.train_ds`
            or `model.validation_ds` config:
        ```
            sample_rate: # int, Sample rate of the audio signal
            window_stride: # float, Window stride for audio encoder
            subsampling_factor: # Subsampling factor for audio encoder
            random_padding:  # Random padding configuration
                prob: 0.9  # probability of applying padding
                min_pad_duration: 0.5  # minimum duration of pre/post padding in seconds
                max_total_duration: 30.0  # maximum total duration of the padded audio in seconds
                pad_distribution: 'uniform'  # distribution of padding duration, 'uniform' or 'normal'
                normal_mean: 0.5  # mean of normal distribution for padding duration
                normal_std: 2.0  # standard deviation of normal distribution for padding duration
        ```

    Returns:
        audio: torch.Tensor of audio signal
        audio_lens: torch.Tensor of audio signal length
        text_tokens: torch.Tensor of text text_tokens
        text_token_lens: torch.Tensor of text token length
        eou_targets (optional): torch.Tensor of EOU labels
        eou_target_lens (optional): torch.Tensor of EOU label length

    The input manifest should be a jsonl file where each line is a python dictionary.
    Example manifest sample:
    {
        "audio_filepath": "/path/to/audio.wav",
        "offset": 0.0,
        "duration": 6.0,
        "sou_time": [0.3, 4.0],
        "eou_time": [1.3, 4.5],
        "utterances": ["Tell me a joke", "Ah-ha"],
        "is_backchannel": [False, True],
    }

    Padding logic:
    0. Don't pad when `random_padding` is None or during validation/test
    1. randomly draw a probability to decide whether to apply padding
    2. if not padding or audio duration is longer than the maximum duration,
        1) return the original audio and EOU labels
    3. if apply padding,
        1) get the max padding duration based on the maximum total duration and the audio duration
        2) randomly draw a total padding duration based on the given distribution
        3) randomly split the total padding duration into pre-padding and post-padding
        4) randomly generate the non-speech signal (audio signal=0) for pre-padding and post-padding
        5) concatenate the pre-padding, audio, and post-padding to get the padded audio signal
        6) update the EOU labels accordingly

    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define the output types of the dataset."""
        return {
            'audio': NeuralType(('B', 'T'), AudioSignal()),
            'audio_lens': NeuralType(tuple('B'), LengthsType()),
            'eou_targets': NeuralType(('B', 'T'), LabelsType()),
            'eou_target_lens': NeuralType(tuple('B'), LengthsType()),
            'text_tokens': NeuralType(tuple('B', 'T'), LengthsType(), optional=True),
            'text_token_lens': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, cfg: DictConfig, tokenizer: TokenizerSpec, return_cuts: bool = False):
        super().__init__()
        self.cfg = cfg
        self.return_cuts = return_cuts
        self.eou_string = self.cfg.get('eou_string', EOU_STRING)
        self.eob_string = self.cfg.get('eob_string', EOB_STRING)

        if cfg.get('check_tokenizer', True):
            self._check_special_tokens(tokenizer)

        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.sample_rate = self.cfg.get('sample_rate', 16000)
        self.num_sample_per_mel_frame = int(
            self.cfg.get('window_stride', 0.01) * self.sample_rate
        )  # 160 samples for every 1ms by default
        self.num_mel_frame_per_target_frame = int(self.cfg.get('subsampling_factor', 8))
        self.add_sep_before_eou = self.cfg.get('add_sep_before_eou', False)
        self.padding_cfg = self.cfg.get('random_padding', None)
        self.augmentor = None
        self.len_augmentor = None
        if self.cfg.get('augmentor', None) is not None:
            augmentor = {}
            len_augmentor = {}
            aug_cfg = OmegaConf.to_container(self.cfg.augmentor, resolve=True)
            for k, v in aug_cfg.items():
                if k in EOU_PROHIBITED_AUGMENTATIONS:
                    logging.warning(f"EOU dataset does not support {k} augmentation, skipping.")
                    continue
                if k in EOU_LENGTH_PERTURBATION:
                    len_augmentor[k] = v
                else:
                    augmentor[k] = v

            if len(augmentor) > 0:
                logging.info(f"EOU dataset will apply augmentations: {augmentor}")
                self.augmentor = process_augmentations(augmentor)
            if len(len_augmentor) > 0:
                logging.info(f"EOU dataset will apply length augmentations: {len_augmentor}")
                self.len_augmentor = process_augmentations(len_augmentor)

    def _check_special_tokens(self, tokenizer: TokenizerSpec):
        """
        Check if the special tokens are in the tokenizer vocab.
        """
        special_tokens = set([self.eou_string, self.eob_string])
        vocab_size = tokenizer.vocab_size
        special_tokens_in_vocab = set([tokenizer.ids_to_text(vocab_size - 1), tokenizer.ids_to_text(vocab_size - 2)])
        if special_tokens != special_tokens_in_vocab:
            raise ValueError(
                f"Input special tokens {special_tokens} don't match with the tokenizer vocab {special_tokens_in_vocab}. "
                f"Please add them to tokenizer or change input `eou_string` and/or `eob_string` accordingly. "
                "Special tokens should be added as the last two tokens in the new tokenizer. "
                "Please refer to scripts/asr_end_of_utterance/tokenizers/add_special_tokens_to_sentencepiece.py for details."
            )

    def __getitem__(self, cuts: CutSet) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        audio_signals = []
        audio_lengths = []
        eou_targets = []
        text_tokens = []
        sample_ids = []
        audio_filepaths = []
        for i in range(len(cuts)):
            c = cuts[i]
            if isinstance(c, MixedCut):
                c = c.first_non_padding_cut

            sample_ids.append(c.id)
            audio_filepaths.append(c.recording.sources[0].source)

            audio_i = audio[i]
            audio_len_i = audio_lens[i]

            # Maybe apply speed perturbation, this has to be done before getting the EOU labels
            audio_i, audio_len_i = self._maybe_augment_length(audio_i, audio_len_i)

            # Get EOU labels and text tokens
            eou_targets_i = self._get_frame_labels(c, audio_len_i)
            text_tokens_i = self._get_text_tokens(c)

            # Maybe apply random padding to both sides of the audio
            audio_i, audio_len_i, eou_targets_i = self._random_pad_audio(audio_i, audio_len_i, eou_targets_i)

            # Maybe apply augmentations to the audio signal after padding
            audio_i, audio_len_i = self._maybe_augment_audio(audio_i, audio_len_i)

            # Append the processed audio, EOU labels, and text tokens to the lists
            audio_signals.append(audio_i)
            audio_lengths.append(audio_len_i)
            eou_targets.append(eou_targets_i)
            text_tokens.append(text_tokens_i)

        audio_signals = collate_vectors(audio_signals, padding_value=0)
        audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)
        eou_target_lens = torch.tensor([t.size(0) for t in eou_targets], dtype=torch.long)
        eou_targets = collate_vectors(eou_targets, padding_value=0)
        text_token_lens = torch.tensor([t.size(0) for t in text_tokens], dtype=torch.long)
        text_tokens = collate_vectors(text_tokens, padding_value=0)

        if self.return_cuts:
            return audio_signals, audio_lengths, cuts

        return AudioToTextEOUBatch(
            sample_ids=sample_ids,
            audio_filepaths=audio_filepaths,
            audio_signal=audio_signals,
            audio_lengths=audio_lengths,
            text_tokens=text_tokens,
            text_token_lengths=text_token_lens,
            eou_targets=eou_targets,
            eou_target_lengths=eou_target_lens,
        )

    def _audio_len_to_frame_len(self, num_samples: int):
        """
        Convert the raw audio length to the number of frames after audio encoder.

        self.num_sample_per_mel_frame = int(
            self.cfg.get('window_stride', 0.01) * self.cfg.get('sample_rate', 16000)
        )  # 160 samples for every 1ms by default
        self.num_mel_frame_per_target_frame = int(self.cfg.get('subsampling_factor', 8))
        """
        mel_frame_count = math.ceil((num_samples + 1) / self.num_sample_per_mel_frame)
        hidden_length = math.ceil(mel_frame_count / self.num_mel_frame_per_target_frame)
        return hidden_length

    def _get_frame_labels(self, cut: Cut, num_samples: int):
        hidden_length = self._audio_len_to_frame_len(num_samples)
        if not "sou_time" in cut.custom or not "eou_time" in cut.custom:
            # assume only single speech segment
            text = cut.supervisions[0].text
            if not text:
                # skip empty utterances
                return torch.zeros(hidden_length).long()
            eou_targets = torch.ones(hidden_length).long()  # speech label
            eou_targets[-1] = EOU_LABEL  # by default it's end of utterance
            if cut.has_custom("is_backchannel") and cut.custom["is_backchannel"]:
                eou_targets[-1] = EOB_LABEL  # end of backchannel
            return eou_targets

        sou_time = cut.custom["sou_time"]
        eou_time = cut.custom["eou_time"]
        if not isinstance(sou_time, list):
            sou_time = [sou_time]
        if not isinstance(eou_time, list):
            eou_time = [eou_time]

        assert len(sou_time) == len(
            eou_time
        ), f"Number of SOU time and EOU time do not match: SOU ({len(sou_time)}) vs EOU ({len(eou_time)})"

        if cut.has_custom("is_backchannel"):
            is_backchannel = cut.custom["is_backchannel"]
            if not isinstance(is_backchannel, list):
                is_backchannel = [is_backchannel]
            assert len(sou_time) == len(
                cut.custom["is_backchannel"]
            ), f"Number of SOU and backchannel do not match: SOU ({len(sou_time)}) vs backchannel ({len(is_backchannel)})"
        else:
            is_backchannel = [False] * len(sou_time)

        eou_targets = torch.zeros(hidden_length).long()
        for i in range(len(sou_time)):
            if sou_time[i] is None or eou_time[i] is None or sou_time[i] < 0 or eou_time[i] < 0:
                # skip empty utterances
                continue
            sou_idx = self._audio_len_to_frame_len(int((sou_time[i] - cut.start) * self.sample_rate))
            seg_len_in_secs = eou_time[i] - sou_time[i]
            seg_len = self._audio_len_to_frame_len(int(seg_len_in_secs * self.sample_rate))
            eou_targets[sou_idx : sou_idx + seg_len] = 1
            last_idx = min(sou_idx + seg_len - 1, hidden_length - 1)
            if is_backchannel[i]:
                eou_targets[last_idx] = EOB_LABEL  # end of backchannel
            else:
                eou_targets[last_idx] = EOU_LABEL  # end of utterance

        return eou_targets

    def _get_text_tokens(self, cut: Cut):
        if not cut.has_custom("sou_time") or not cut.has_custom("eou_time") or not cut.has_custom("utterances"):
            # assume only single speech segment
            utterances = [cut.supervisions[0].text]
        else:
            utterances = cut.custom["utterances"]

        if not isinstance(utterances, list):
            utterances = [utterances]

        if cut.has_custom("is_backchannel"):
            is_backchannel = cut.custom["is_backchannel"]
            if not isinstance(is_backchannel, list):
                is_backchannel = [is_backchannel]
            assert len(utterances) == len(
                cut.custom["is_backchannel"]
            ), f"Number of utterances and backchannel do not match: utterance ({len(utterances)}) vs backchannel ({len(is_backchannel)})"
        else:
            is_backchannel = [False] * len(utterances)

        total_text = ""
        for i, text in enumerate(utterances):
            if not text:
                # skip empty utterances
                continue
            eou_string = self.eob_string if is_backchannel[i] else self.eou_string
            if self.add_sep_before_eou:
                eou_string = " " + eou_string
            total_text += text + eou_string + " "
        total_text = total_text.strip()
        return torch.as_tensor(self.tokenizer(total_text))

    def _random_pad_audio(self, audio: torch.Tensor, audio_len: torch.Tensor, eou_targets: torch.Tensor):
        """
        Randomly pad the audio signal with non-speech signal before and after the audio signal.
        Args:
            audio: torch.Tensor of a single audio signal, shape [T]
            audio_len: torch.Tensor of audio signal length, shape [1]
            eou_targets: torch.Tensor of EOU labels, shape [T]
        Returns:
            padded_audio: torch.Tensor of padded audio signal, shape [T+padding]
            padded_audio_len: torch.Tensor of padded audio signal length, shape [1]
            padded_eou_targets: torch.Tensor of padded EOU labels, shape [T+padding]
            padded_eou_targets_len: torch.Tensor of padded EOU label length, shape [1]
        """
        p = np.random.rand()
        if self.padding_cfg is None or p > self.padding_cfg.prob:
            return audio, audio_len, eou_targets

        duration = audio_len.item() / self.cfg.sample_rate
        # if already longer than the maximum duration, return the original audio
        if duration >= self.padding_cfg.max_total_duration:
            return audio, audio_len, eou_targets

        # apply padding
        audio = audio[:audio_len]
        max_padding_duration = max(0, self.padding_cfg.max_total_duration - duration)
        if max_padding_duration <= self.padding_cfg.min_pad_duration:
            min_padding_duration = 0
        else:
            min_padding_duration = self.padding_cfg.min_pad_duration

        if self.padding_cfg.pad_distribution == 'uniform':
            total_padding_duration = np.random.uniform(min_padding_duration, max_padding_duration)
        elif self.padding_cfg.pad_distribution == 'normal':
            total_padding_duration = np.random.normal(self.padding_cfg.normal_mean, self.padding_cfg.normal_std)
            total_padding_duration = max(min_padding_duration, min(max_padding_duration, total_padding_duration))
        else:
            raise ValueError(f"Unknown padding distribution: {self.padding_cfg.pad_distribution}")

        pre_padding_duration = np.random.uniform(0, total_padding_duration)
        post_padding_duration = total_padding_duration - pre_padding_duration

        pre_padding_len = math.ceil(pre_padding_duration * self.cfg.sample_rate)
        post_padding_len = math.ceil(post_padding_duration * self.cfg.sample_rate)

        # pad the audio signal
        pre_padding = torch.zeros(pre_padding_len, dtype=audio.dtype)
        post_padding = torch.zeros(post_padding_len, dtype=audio.dtype)
        padded_audio = torch.cat((pre_padding, audio, post_padding), dim=0)
        padded_audio_len = audio_len + pre_padding_len + post_padding_len

        # pad the EOU labels
        pre_padding_eou_len = self._audio_len_to_frame_len(pre_padding_len)
        post_padding_eou_len = self._audio_len_to_frame_len(post_padding_len)
        pre_padding_eou = torch.zeros(pre_padding_eou_len, dtype=eou_targets.dtype)
        post_padding_eou = torch.zeros(post_padding_eou_len, dtype=eou_targets.dtype)
        padded_eou_targets = torch.cat((pre_padding_eou, eou_targets, post_padding_eou), dim=0)

        return padded_audio, padded_audio_len, padded_eou_targets

    def _maybe_augment_audio(self, audio: torch.Tensor, audio_len: torch.Tensor):
        """
        Apply augmentation to the audio signal if augmentor is provided.
        Args:
            audio: torch.Tensor of a single audio signal, shape [T]
            audio_len: torch.Tensor of audio signal length, shape [1]
        Returns:
            augmented_audio: torch.Tensor of augmented audio signal, shape [T]
            augmented_audio_len: torch.Tensor of augmented audio signal length, shape [1]
        """
        if self.augmentor is None:
            return audio, audio_len

        # Cast to AudioSegment
        audio_segment = AudioSegment(
            samples=audio[:audio_len].numpy(),
            sample_rate=self.sample_rate,
            offset=0,
            duration=audio_len.item() / self.sample_rate,
        )
        # Apply augmentation
        self.augmentor.perturb(audio_segment)
        audio = torch.from_numpy(audio_segment.samples).float()
        audio_len = audio.size(0)

        return audio, audio_len

    def _maybe_augment_length(self, audio: torch.Tensor, audio_len: torch.Tensor):
        """
        Apply length augmentation (e.g., speed perturb) to the audio signal if augmentor is provided.
        Args:
            audio: torch.Tensor of a single audio signal, shape [T]
            audio_len: torch.Tensor of audio signal length, shape [1]
        Returns:
            augmented_audio: torch.Tensor of augmented audio signal, shape [T]
            augmented_audio_len: torch.Tensor of augmented audio signal length, shape [1]
        """
        if self.len_augmentor is None:
            return audio, audio_len

        # Cast to AudioSegment
        audio_segment = AudioSegment(
            samples=audio[:audio_len].numpy(),
            sample_rate=self.sample_rate,
            offset=0,
            duration=audio_len.item() / self.sample_rate,
        )
        # Apply augmentation
        self.len_augmentor.perturb(audio_segment)
        audio = torch.from_numpy(audio_segment.samples).float()
        audio_len = audio.size(0)

        return audio, audio_len
