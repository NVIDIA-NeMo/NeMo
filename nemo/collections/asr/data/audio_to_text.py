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
import io
import math
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import braceexpand
import librosa
import numpy as np
import torch
import webdataset as wd
from scipy.stats import betabinom
from torch.nn import functional as F
from torch.utils.data import ChainDataset

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common import tokenizers
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.core.neural_types.elements import ProbsType
from nemo.utils import logging
from nemo.utils.decorators import deprecated

__all__ = [
    'AudioToCharDataset',
    'AudioToCharWithDursF0Dataset',
    'AudioToCharWithPriorDataset',
    'AudioToBPEDataset',
    'TarredAudioToCharDataset',
    'TarredAudioToBPEDataset',
]


def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, audio_lengths, _, tokens_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 4:
        sample_ids = None
        _, audio_lengths, _, tokens_lengths = packed_batch
    else:
        raise ValueError("Expects 4 or 5 tensors in the batch!")
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for b in batch:
        if len(b) == 5:
            sig, sig_len, tokens_i, tokens_i_len, _ = b
        else:
            sig, sig_len, tokens_i, tokens_i_len = b
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)
    if sample_ids is None:
        return audio_signal, audio_lengths, tokens, tokens_lengths
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return audio_signal, audio_lengths, tokens, tokens_lengths, sample_ids


class ASRManifestProcessor:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        bos_id: Id of beginning of sequence symbol to append if not None.
        eos_id: Id of end of sequence symbol to append if not None.
        pad_id: Id of pad symbol. Defaults to 0.
    """

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        index_by_file_id: bool = False,
    ):
        self.parser = parser

        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
        )

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text_by_id(self, index: int) -> (List[int], int):
        sample = self.collection[index]
        return self.process_text_by_sample(sample)

    def process_text_by_file_id(self, file_id: str) -> (List[int], int):
        manifest_idx = self.collection.mapping[file_id]
        sample = self.collection[manifest_idx]
        return self.process_text_by_sample(sample)

    def process_text_by_sample(self, sample: collections.ASRAudioText.OUTPUT_TYPE) -> (List[int], int):
        t, tl = sample.text_tokens, len(sample.text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl


def expand_audio_filepaths(audio_tar_filepaths, shard_strategy: str, world_size: int, global_rank: int):
    valid_shard_strategies = ['scatter', 'replicate']
    if shard_strategy not in valid_shard_strategies:
        raise ValueError(f"`shard_strategy` must be one of {valid_shard_strategies}")

    if isinstance(audio_tar_filepaths, str):
        # Replace '(' and '[' with '{'
        brace_keys_open = ['(', '[', '<', '_OP_']
        for bkey in brace_keys_open:
            if bkey in audio_tar_filepaths:
                audio_tar_filepaths = audio_tar_filepaths.replace(bkey, "{")

        # Replace ')' and ']' with '}'
        brace_keys_close = [')', ']', '>', '_CL_']
        for bkey in brace_keys_close:
            if bkey in audio_tar_filepaths:
                audio_tar_filepaths = audio_tar_filepaths.replace(bkey, "}")

    if isinstance(audio_tar_filepaths, str):
        # Brace expand
        audio_tar_filepaths = list(braceexpand.braceexpand(audio_tar_filepaths))

    # Check for distributed and partition shards accordingly
    if world_size > 1:
        if shard_strategy == 'scatter':
            logging.info("All tarred dataset shards will be scattered evenly across all nodes.")

            if len(audio_tar_filepaths) % world_size != 0:
                logging.warning(
                    f"Number of shards in tarred dataset ({len(audio_tar_filepaths)}) is not divisible "
                    f"by number of distributed workers ({world_size})."
                )

            begin_idx = (len(audio_tar_filepaths) // world_size) * global_rank
            end_idx = begin_idx + len(audio_tar_filepaths) // world_size
            audio_tar_filepaths = audio_tar_filepaths[begin_idx:end_idx]
            logging.info(
                "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
            )

        elif shard_strategy == 'replicate':
            logging.info("All tarred dataset shards will be replicated across all nodes.")
        else:
            raise ValueError(f"Invalid shard strategy ! Allowed values are : {valid_shard_strategies}")

    return audio_tar_filepaths


class _AudioTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        trim: whether or not to trim silence. Defaults to False
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        pad_id: Id of pad symbol. Defaults to 0
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
    ):
        if type(manifest_filepath) == str:
            manifest_filepath = manifest_filepath.split(",")

        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim, orig_sr=sample.orig_sr
        )
        f, fl = features, torch.tensor(features.shape[0]).long()

        t, tl = self.manifest_processor.process_text_by_sample(sample=sample)

        if self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        return output

    def __len__(self):
        return len(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id)


class AudioToCharDataset(_AudioTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        labels: Union[str, List[str]],
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        parser: Union[str, Callable] = 'en',
        return_sample_id: bool = False,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
        )


@deprecated(version="1.8", explanation="Please, use ``nemo.tts.collections.torch.data.TTSDataset`` instead.")
class AudioToCharWithDursF0Dataset(AudioToCharDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio_1.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82}

    Additionally, user provides path to precomputed durations via "durs_file" arg, which is a pickled python dict,
    mapping example tag to it's durations tensor, and name of method that was received durations via "durs_type".
    Tag is a unique example identifier, which is a wav filename without suffix. Durations depend on "durs_type".
    If durs_type == "asr_based" then durations are an additional dict of two tensors: graphemes/phonemes durations and blanks durations with
    "blanks" and "tokens" keys respectively.
    Example below:
    {'LJ050-0234': {'blanks': `blanks_durs_tensor`, 'tokens': `tokens_durs_tensor`},
    ...}
    If durs_type == "aligner_based" then durations are just tensor graphemes/phonemes durations.
    Example below:
    {'LJ050-0234': `tokens_durs_tensor`},
    ...}

    Additionally, F0 statistics is passed precomputed along mel length via "f0_file" arg, which is a pickled python
    dict, mapping example tag to it's f0 tensor.
    Example below:
    {'LJ050-0234': `f0_tensor`,
    ...}

    Args:
        **kwargs: Passed to AudioToCharDataset constructor.
        durs_file (str): String path to pickled durations location.
        durs_type (str): Type of durations. Currently supported durations are "asr-based" and "aligned-based".
        f0_file (str): String path to pickled f0 statistics location.
        blanking (bool): Boolean flag indicate whether add or not blanks between text graphemes.
        load_audio(bool): Boolean flag indicate whether do or not load audio.
        vocab: Vocabulary config (parser + set of graphemes to use). Constructor propagates these to
            `self.make_vocab` function call to build a complete vocabulary.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio': NeuralType(('B', 'T'), AudioSignal()),
            'audio_len': NeuralType(('B',), LengthsType()),
            'text': NeuralType(('B', 'T'), LabelsType()),
            'text_len': NeuralType(('B',), LengthsType()),
            'durs': NeuralType(('B', 'T'), LengthsType()),
            'f0': NeuralType(('B', 'T'), FloatType()),
            'f0_mask': NeuralType(('B', 'T'), MaskType()),
        }

    @staticmethod
    def make_vocab(
        notation='chars',
        punct=True,
        spaces=False,
        stresses=False,
        chars=False,
        add_blank_at="last_but_one",
        pad_with_space=False,
        improved_version_g2p=False,
        phoneme_dict_path=None,
    ):
        """Constructs vocabulary from given parameters.

        Args:
            notation (str): Either 'chars' or 'phonemes' as general notation.
            punct (bool): True if reserve grapheme for basic punctuation.
            spaces (bool): True if prepend spaces to every punctuation symbol.
            stresses (bool): True if use phonemes codes with stresses (0-2).
            chars (bool): True if additionally use chars together with phonemes.
            add_blank_at: add blank to labels in the specified order ("last" or "last_but_one"),
             if None then no blank in labels.
            pad_with_space (bool): True if pad text with spaces at the beginning and at the end.
            improved_version_g2p (bool): True if use new version of g2p.
            phoneme_dict_path (str): path to phoneme dict file (like CMU Pronouncing dictionary). If it's None then cmudict.dict() will be used.

        Returns:
            (vocabs.Base) Vocabulary
        """
        from nemo.collections.common.data import vocabs

        if notation == 'chars':
            vocab = vocabs.Chars(punct=punct, spaces=spaces, add_blank_at=add_blank_at)
        elif notation == 'phonemes':
            vocab = vocabs.Phonemes(
                punct=punct,
                stresses=stresses,
                spaces=spaces,
                chars=chars,
                add_blank_at=add_blank_at,
                pad_with_space=pad_with_space,
                improved_version_g2p=improved_version_g2p,
                phoneme_dict_path=phoneme_dict_path,
            )
        else:
            raise ValueError("Unsupported vocab type.")
        return vocab

    def __init__(self, **kwargs):
        durs_file = kwargs.pop('durs_file', None)
        durs_type = kwargs.pop('durs_type', "asr-based")
        f0_file = kwargs.pop('f0_file', None)
        self.blanking = kwargs.pop('blanking', False)
        self.load_audio = kwargs.pop('load_audio', True)
        self.vocab = self.make_vocab(**kwargs.pop('vocab', {}))
        kwargs.setdefault('labels', [])  # For compatibility.
        super().__init__(**kwargs)

        tags = []
        self.id2enc_text = {}
        for i, e in enumerate(self.manifest_processor.collection):
            tag = os.path.splitext(os.path.basename(e.audio_file))[0]
            tags.append(tag)
            # cache vocab encoding
            self.id2enc_text[i] = self.vocab.encode(e.text_raw)

        if durs_file:
            tag2durs = torch.load(durs_file)
            durs = []
            for tag in tags:
                tag_durs = tag2durs[tag]
                if durs_type == "asr-based":
                    durs.append(self.interleave(tag_durs['blanks'], tag_durs['tokens']))
                elif durs_type == "aligner-based":
                    durs.append(tag_durs)
                else:
                    raise NotImplementedError(
                        f"{durs_type} duration type is not supported. Use asr-based or align-based."
                    )
            self.durs = durs
        if f0_file:
            tag2f0 = torch.load(f0_file)
            self.f0 = [tag2f0[tag] for tag in tags]

    def __getitem__(self, item):
        audio, audio_len = None, None
        if self.load_audio:
            audio, audio_len, _, _ = super().__getitem__(item)  # noqa

        text = self.id2enc_text[item]
        text, text_len = torch.tensor(text).long(), torch.tensor(len(text)).long()
        durs, f0 = self.durs[item], self.f0[item]
        return (
            audio,
            audio_len,
            text,
            text_len,
            durs,
            f0,
        )

    @staticmethod
    def merge(tensors, dim=0, value=0, dtype=None):
        """Merges list of tensors into one."""
        tensors = [tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) for tensor in tensors]
        dim = dim if dim != -1 else len(tensors[0].shape) - 1
        dtype = tensors[0].dtype if dtype is None else dtype
        max_len = max(tensor.shape[dim] for tensor in tensors)
        new_tensors = []
        for tensor in tensors:
            pad = (2 * len(tensor.shape)) * [0]
            pad[-2 * dim - 1] = max_len - tensor.shape[dim]
            new_tensors.append(F.pad(tensor, pad=pad, value=value))
        return torch.stack(new_tensors).to(dtype=dtype)

    @classmethod
    def repeat_merge(cls, x, reps, pad):
        """Repeats `x` values according to `reps` tensor and merges."""
        return cls.merge(
            tensors=[torch.repeat_interleave(text1, durs1) for text1, durs1 in zip(x, reps)], value=pad, dtype=x.dtype,
        )

    @staticmethod
    def make_mask(lengths, max_length=None):
        """Makes mask from list of lengths."""
        device = lengths.device if torch.is_tensor(lengths) else 'cpu'
        lengths = lengths if torch.is_tensor(lengths) else torch.tensor(lengths)
        max_length = max_length or torch.max(lengths)
        start = torch.tensor(0).int()
        indices = torch.arange(start=start, end=max_length, device=device)  # noqa
        mask = indices.lt(lengths.view(-1, 1))

        return mask

    @staticmethod
    def interleave(x, y):
        """Interleave two tensors."""
        xy = torch.stack([x[:-1], y], dim=1).view(-1)
        xy = F.pad(xy, pad=[0, 1], value=x[-1])
        return xy

    def _collate_fn(self, batch):
        batch = list(zip(*batch))

        asr_batch = _speech_collate_fn(list(zip(*batch[:4])), pad_id=self.vocab.pad)
        audio, audio_len, text, text_len = asr_batch

        if self.blanking:
            text = [
                self.interleave(
                    x=torch.empty(len(t) + 1, dtype=torch.long, device=t.device).fill_(self.vocab.blank), y=t,
                )
                for t in text
            ]
            text = self.merge(text, value=self.vocab.pad, dtype=torch.long)
            text_len = text_len * 2 + 1

        durs, f0 = batch[4:]
        durs = self.merge(durs, dtype=torch.long)
        f0_mask = self.make_mask([f.shape[-1] for f in f0])  # noqa
        f0 = self.merge(f0, dtype=torch.float)

        return (
            audio,
            audio_len,
            text,
            text_len,
            durs,
            f0,
            f0_mask,
        )


@deprecated(version="1.8", explanation="Please, use ``nemo.tts.collections.torch.data.TTSDataset`` instead.")
class AudioToCharWithPriorDataset(AudioToCharDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio_1.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82}

    Additionally, user provides path to folder with precomputed attention priors via "attn_prior_folder" arg.
    This folder should contain saved numpy array as attention prior along mel and text lengths.
    If folder is empty, attention priors will be calculated and saved in specified folder.
    Every name of saved numpy array is a unique example identifier, which is a wav filename without suffix.

    Args:
        **kwargs: Passed to AudioToCharDataset constructor.
        attn_prior_folder (str): String path to folder with precomputed attention priors.
        n_window_stride (int): Stride of window for fft in samples. Need to generate prior.
        vocab: Vocabulary config (parser + set of graphemes to use). Constructor propagates these to
            `self.make_vocab` function call to build a complete vocabulary.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio': NeuralType(('B', 'T'), AudioSignal()),
            'audio_len': NeuralType(('B',), LengthsType()),
            'text': NeuralType(('B', 'T'), LabelsType()),
            'text_len': NeuralType(('B',), LengthsType()),
            'attn_prior': NeuralType(('B', 'T', 'D'), ProbsType()),
        }

    def __init__(self, attn_prior_folder, n_window_stride=256, **kwargs):
        self.vocab = AudioToCharWithDursF0Dataset.make_vocab(**kwargs.pop('vocab', {}))
        kwargs.setdefault('labels', [])  # For compatibility.
        super().__init__(**kwargs)

        Path(attn_prior_folder).mkdir(parents=True, exist_ok=True)
        self.attn_prior_folder = attn_prior_folder

        self.n_window_stride = n_window_stride
        self.id2enc_text = {}
        for i, e in enumerate(self.manifest_processor.collection):
            # cache vocab encoding
            self.id2enc_text[i] = self.vocab.encode(e.text_raw)

    @staticmethod
    def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling_factor=1.0):
        x = np.arange(0, phoneme_count)
        mel_text_probs = []
        for i in range(1, mel_count + 1):
            a, b = scaling_factor * i, scaling_factor * (mel_count + 1 - i)
            mel_i_prob = betabinom(phoneme_count, a, b).pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)

    def __getitem__(self, item):
        audio, audio_len, _, _ = super().__getitem__(item)  # noqa

        text = self.id2enc_text[item]

        attn_prior_path = (
            Path(self.attn_prior_folder)
            / f"ap_tl{len(text)}_al_{math.ceil((audio_len.item() + 1) / self.n_window_stride)}.npy"
        )

        if attn_prior_path.exists():
            attn_prior = np.load(attn_prior_path)
        else:
            attn_prior = self.beta_binomial_prior_distribution(
                len(text), math.ceil((audio_len.item() + 1) / self.n_window_stride)
            )
            np.save(attn_prior_path, attn_prior)

        text, text_len, attn_prior = (
            torch.tensor(text).long(),
            torch.tensor(len(text)).long(),
            torch.tensor(attn_prior),
        )

        return audio, audio_len, text, text_len, attn_prior

    def _collate_fn(self, batch):
        batch = list(zip(*batch))

        asr_batch = _speech_collate_fn(list(zip(*batch[:4])), pad_id=self.vocab.pad)
        audio, audio_len, text, text_len = asr_batch
        attn_prior_list = batch[4]

        attn_prior = torch.zeros(
            len(attn_prior_list),
            max([attn_prior_i.shape[0] for attn_prior_i in attn_prior_list]),
            max([attn_prior_i.shape[1] for attn_prior_i in attn_prior_list]),
        )

        for i, attn_prior_i in enumerate(attn_prior_list):
            attn_prior[i, : attn_prior_i.shape[0], : attn_prior_i.shape[1]] = attn_prior_i

        return audio, audio_len, text, text_len, attn_prior


@deprecated(version="1.8", explanation="Please, use ``nemo.tts.collections.torch.data.TTSDataset`` instead.")
class AudioToCharWithPriorAndPitchDataset(AudioToCharWithPriorDataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio': NeuralType(('B', 'T'), AudioSignal()),
            'audio_len': NeuralType(('B',), LengthsType()),
            'text': NeuralType(('B', 'T'), LabelsType()),
            'text_len': NeuralType(('B',), LengthsType()),
            'attn_prior': NeuralType(('B', 'T', 'D'), ProbsType()),
            'pitch': NeuralType(('B', 'T'), RegressionValuesType()),
            'speakers': NeuralType(('B',), Index(), optional=True),
        }

    def __init__(self, sup_data_path, pitch_fmin, pitch_fmax, n_window_size, pitch_avg, pitch_std, **kwargs):
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        self.n_window_size = n_window_size
        self.pitch_avg = pitch_avg
        self.pitch_std = pitch_std
        super().__init__(attn_prior_folder=sup_data_path, **kwargs)

    def __getitem__(self, item):
        audio, audio_len, text, text_len, attn_prior = super().__getitem__(item)
        tag = Path(self.manifest_processor.collection[item].audio_file).stem
        pitch_path = (
            Path(self.attn_prior_folder)
            / f"{tag}_pitch_pyin_fmin{self.pitch_fmin}_fmax{self.pitch_fmax}_fl{self.n_window_size}.npy"
        )

        if pitch_path.exists():
            pitch = np.load(pitch_path)
        else:
            pitch, _, _ = librosa.pyin(
                audio.numpy(),
                fmin=self.pitch_fmin,
                fmax=self.pitch_fmax,
                frame_length=self.n_window_size,
                sr=self.featurizer.sample_rate,
                fill_na=0.0,
            )
            np.save(pitch_path, pitch)
        pitch -= self.pitch_avg
        pitch[pitch == -self.pitch_avg] = 0.0  # Zero out values that were perviously zero
        pitch /= self.pitch_std

        speaker = None
        if self.manifest_processor.collection[item].speaker is not None:
            speaker = torch.zeros_like(text_len).fill_(self.manifest_processor.collection[item].speaker)

        return audio, audio_len, text, text_len, attn_prior, torch.tensor(pitch), speaker

    def _collate_fn(self, batch):
        batch = list(zip(*batch))
        audio, audio_len, text, text_len, attn_prior = super()._collate_fn(list(zip(*batch[:5])))
        pitch_list = batch[5]
        speaker_list = batch[6]

        pitch = torch.zeros(len(pitch_list), max([pitch.shape[0] for pitch in pitch_list]))

        for i, pitch_i in enumerate(pitch_list):
            pitch[i, : pitch_i.shape[0]] = pitch_i

        speakers = []
        for i, speaker_i in enumerate(speaker_list):
            speakers.append(speaker_i)

        speakers = torch.stack(speakers).to(text_len.dtype) if speakers[0] is not None else None

        return audio, audio_len, text, text_len, attn_prior, pitch, speakers


@deprecated(version="1.8", explanation="Please, use ``nemo.tts.collections.torch.data.TTSDataset`` instead.")
class FastPitchDataset(_AudioTextDataset):
    """
    Dataset used for FastPitch that has both duration and pitch information per input char.
    See https://github.com/NVIDIA/NeMo/pull/1799 for information on how to extract duration and pitch information.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            'audio': NeuralType(('B', 'T'), AudioSignal()),
            'audio_len': NeuralType(('B',), LengthsType()),
            'text': NeuralType(('B', 'T'), LabelsType()),
            'text_len': NeuralType(('B',), LengthsType()),
            'durs': NeuralType(('B', 'T'), TokenDurationType()),
            'pitch': NeuralType(('B', 'T'), RegressionValuesType()),
            'speakers': NeuralType(('B',), Index()),
        }

    def __getitem__(self, item):
        audio, audio_len, text, text_len = super().__getitem__(item)  # noqa

        audio_path = self.manifest_processor.collection[item].audio_file
        durs_path = audio_path.replace('/wavs/', '/fastpitch/durations/').replace('.wav', '.pt')
        pitch_path = audio_path.replace('/wavs/', '/fastpitch/pitch_char/').replace('.wav', '.pt')
        speaker = None
        if self.manifest_processor.collection[item].speaker is not None:
            speaker = torch.zeros_like(text_len).fill_(self.manifest_processor.collection[item].speaker)

        return (audio, audio_len, text, text_len, torch.load(durs_path), torch.load(pitch_path), speaker, audio_path)

    def _collate_fn(self, batch):
        pad_id = self.manifest_processor.pad_id
        asr_batch = list(zip(*batch))[:4]
        asr_batch = _speech_collate_fn(list(zip(*asr_batch)), pad_id=pad_id)
        audio, audio_len, text, text_len = asr_batch

        max_tokens_len = text.size(1)
        durs, pitch, speakers = [], [], []
        for _, _, _, tokens_i_len, durs_i, pitch_i, speaker_i, path_i in batch:
            pad = (0, max_tokens_len - tokens_i_len.item())
            assert len(durs_i) == len(pitch_i), f"{len(durs_i)}, {len(pitch_i)}: {path_i}"
            assert len(durs_i) == tokens_i_len, f"{len(durs_i)}, {tokens_i_len}:  {path_i}"
            durs.append(F.pad(durs_i, pad, value=pad_id))
            pitch.append(F.pad(pitch_i, pad, value=pad_id))
            speakers.append(speaker_i)

        return (
            audio,
            audio_len,
            text,
            text_len,
            torch.stack(durs).to(audio.dtype),
            torch.stack(pitch).to(audio.dtype),
            torch.stack(speakers).to(audio.dtype) if speakers[0] is not None else None,
        )


class AudioToBPEDataset(_AudioTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}

    In practice, the dataset and manifest used for character encoding and byte pair encoding
    are exactly the same. The only difference lies in how the dataset tokenizes the text in
    the manifest.

    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        tokenizer: A subclass of the Tokenizer wrapper found in the common collection,
            nemo.collections.common.tokenizers.TokenizerSpec. ASR Models support a subset of
            all available tokenizers.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        trim: Whether to trim silence segments
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        return_sample_id: bool = False,
    ):
        if use_start_end_token and hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            trim=trim,
            return_sample_id=return_sample_id,
        )


class _TarredAudioToTextDataset(IterableDataset):
    """
    A similar Dataset to the AudioToCharDataset/AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToCharDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        parser (callable): A callable which is used to pre-process the text output.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        parser: Callable,
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
    ):
        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id
        self.return_sample_id = return_sample_id

        audio_tar_filepaths = expand_audio_filepaths(
            audio_tar_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )

        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=audio_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = (
            self._dataset.rename(audio='wav;ogg', key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .map(f=self._build_sample)
        )

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRAudioText already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection):
                self.iterator = iterator
                self.collection = collection

            def __iter__(self):
                return self

            def __next__(self):
                while True:
                    audio_bytes, audio_filename = next(self.iterator)
                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if file_id in self.collection.mapping:
                        return audio_bytes, audio_filename

        return TarredAudioFilter(self.manifest_processor.collection)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, self.pad_id)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename = tup

        # Grab manifest entry from self.manifest_preprocessor.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.manifest_processor.collection.mapping[file_id]
        manifest_entry = self.manifest_processor.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream,
            offset=offset,
            duration=manifest_entry.duration,
            trim=self.trim,
            orig_sr=manifest_entry.orig_sr,
        )
        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        # Text features
        t, tl = manifest_entry.text_tokens, len(manifest_entry.text_tokens)

        self.manifest_processor.process_text_by_sample(sample=manifest_entry)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        if self.return_sample_id:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), manifest_idx
        else:
            return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.manifest_processor.collection)


class TarredAudioToCharDataset(_TarredAudioToTextDataset):
    """
    A similar Dataset to the AudioToCharDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToCharDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        labels: List[str],
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        parser: Optional[str] = 'en',
        pad_id: int = 0,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            shard_strategy=shard_strategy,
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=return_sample_id,
        )


class TarredAudioToBPEDataset(_TarredAudioToTextDataset):
    """
    A similar Dataset to the AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        tokenizer (TokenizerSpec): Either a Word Piece Encoding tokenizer (BERT),
            or a Sentence Piece Encoding tokenizer (BPE). The CTC blank
            symbol is automatically added later for models using ctc.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        max_utts (int): Limit number of utterances. 0 means no maximum.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        use_start_end_token: Boolean which dictates whether to add [BOS] and [EOS]
            tokens to beginning and ending of speech respectively.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_utts: int = 0,
        trim: bool = False,
        use_start_end_token: bool = True,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        return_sample_id: bool = False,
    ):
        if use_start_end_token and hasattr(tokenizer, 'bos_token'):
            bos_id = tokenizer.bos_id
        else:
            bos_id = None

        if use_start_end_token and hasattr(tokenizer, 'eos_token'):
            eos_id = tokenizer.eos_id
        else:
            eos_id = None

        if hasattr(tokenizer, 'pad_token'):
            pad_id = tokenizer.pad_id
        else:
            pad_id = 0

        class TokenizerWrapper:
            def __init__(self, tokenizer):
                if isinstance(tokenizer, tokenizers.aggregate_tokenizer.AggregateTokenizer):
                    self.is_aggregate = True
                else:
                    self.is_aggregate = False
                self._tokenizer = tokenizer

            def __call__(self, *args):
                t = self._tokenizer.text_to_ids(*args)
                return t

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            parser=TokenizerWrapper(tokenizer),
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            shard_strategy=shard_strategy,
            global_rank=global_rank,
            world_size=world_size,
            return_sample_id=return_sample_id,
        )


class BucketingDataset(IterableDataset):
    """
    A Dataset which wraps another IterableDataset and adopts it for bucketing
    Args:
        dataset (IterableDataset): The IterableDataset to get wrapped
        bucketing_batch_size (int): Number of samples to build a batch
    """

    def __init__(
        self, dataset: IterableDataset, bucketing_batch_size: int,
    ):
        self.wrapped_dataset = dataset
        self.bucketing_batch_size = bucketing_batch_size
        super().__init__()

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch[0], self.wrapped_dataset.pad_id)

    def __iter__(self):
        return BucketingIterator(
            wrapped_iter=self.wrapped_dataset._dataset.__iter__(), bucketing_batch_size=self.bucketing_batch_size
        ).__iter__()

    def __len__(self):
        return int(math.ceil(len(self.wrapped_dataset) / float(self.bucketing_batch_size)))


class BucketingIterator:
    def __init__(self, wrapped_iter, bucketing_batch_size):
        self.wrapped_iter = wrapped_iter
        self.bucketing_batch_size = bucketing_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        batches = []
        for idx in range(self.bucketing_batch_size):
            try:
                sample = next(self.wrapped_iter)
            except StopIteration:
                break
            batches.append(sample)
        if len(batches) == 0:
            raise StopIteration
        return batches


class RandomizedChainDataset(ChainDataset):
    def __init__(self, datasets: Iterable[Dataset], rnd_seed=0) -> None:
        super(RandomizedChainDataset, self).__init__(list(datasets))
        self.rnd_gen = np.random.RandomState(rnd_seed)

    def __iter__(self):
        shuffled_order = self.rnd_gen.permutation(len(self.datasets))
        for dataset_idx in shuffled_order:
            d = self.datasets[dataset_idx]
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x
