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

import collections
import json
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from nemo.collections.common.parts.preprocessing import manifest, parsers
from nemo.utils import logging


class _Collection(collections.UserList):
    """List of parsed and preprocessed data."""

    OUTPUT_TYPE = None  # Single element output type.


class Text(_Collection):
    """Simple list of preprocessed text entries, result in list of tokens."""

    OUTPUT_TYPE = collections.namedtuple('TextEntity', 'tokens')

    def __init__(self, texts: List[str], parser: parsers.CharParser):
        """Instantiates text manifest and do the preprocessing step.

        Args:
            texts: List of raw texts strings.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        data, output_type = [], self.OUTPUT_TYPE
        for text in texts:
            tokens = parser(text)

            if tokens is None:
                logging.warning("Fail to parse '%s' text line.", text)
                continue

            data.append(output_type(tokens))

        super().__init__(data)


class FromFileText(Text):
    """Another form of texts manifest with reading from file."""

    def __init__(self, file: str, parser: parsers.CharParser):
        """Instantiates text manifest and do the preprocessing step.

        Args:
            file: File path to read from.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        texts = self.__parse_texts(file)

        super().__init__(texts, parser)

    @staticmethod
    def __parse_texts(file: str) -> List[str]:
        if not os.path.exists(file):
            raise ValueError('Provided texts file does not exists!')

        _, ext = os.path.splitext(file)
        if ext == '.csv':
            texts = pd.read_csv(file)['transcript'].tolist()
        elif ext == '.json':  # Not really a correct json.
            texts = list(item['text'] for item in manifest.item_iter(file))
        else:
            with open(file, 'r') as f:
                texts = f.readlines()

        return texts


class AudioText(_Collection):
    """List of audio-transcript text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='AudioTextEntity',
        field_names='id audio_file duration text_tokens offset text_raw speaker orig_sr lang',
    )

    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        durations: List[float],
        texts: List[str],
        offsets: List[str],
        speakers: List[Optional[int]],
        orig_sampling_rates: List[Optional[int]],
        langs: List[Optional[str]],
        parser: parsers.CharParser,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-text manifest with filters and preprocessing.

        Args:
            ids: List of examples positions.
            audio_files: List of audio files.
            durations: List of float durations.
            texts: List of raw text transcripts.
            offsets: List of duration offsets or None.
            speakers: List of optional speakers ids.
            orig_sampling_rates: List of original sampling rates of audio files.
            langs: List of language ids, one for eadh sample, or None.
            parser: Instance of `CharParser` to convert string to tokens.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for id_, audio_file, duration, offset, text, speaker, orig_sr, lang in zip(
            ids, audio_files, durations, offsets, texts, speakers, orig_sampling_rates, langs
        ):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if text != '':
                if hasattr(parser, "is_aggregate") and parser.is_aggregate:
                    if lang is not None:
                        text_tokens = parser(text, lang)
                    else:
                        raise ValueError("lang required in manifest when using aggregate tokenizers")
                else:
                    text_tokens = parser(text)
            else:
                text_tokens = []

            if text_tokens is None:
                duration_filtered += duration
                num_filtered += 1
                continue

            total_duration += duration

            data.append(output_type(id_, audio_file, duration, text_tokens, offset, text, speaker, orig_sr, lang))
            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)

        super().__init__(data)


class ASRAudioText(AudioText):
    """`AudioText` collector from asr structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `AudioText` constructor.
            **kwargs: Kwargs to pass to `AudioText` constructor.
        """

        ids, audio_files, durations, texts, offsets, speakers, orig_srs, langs = [], [], [], [], [], [], [], []
        for item in manifest.item_iter(manifests_files):
            ids.append(item['id'])
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            texts.append(item['text'])
            offsets.append(item['offset'])
            speakers.append(item['speaker'])
            orig_srs.append(item['orig_sr'])
            langs.append(item['lang'])

        super().__init__(ids, audio_files, durations, texts, offsets, speakers, orig_srs, langs, *args, **kwargs)


class SpeechLabel(_Collection):
    """List of audio-label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='SpeechLabelEntity', field_names='audio_file duration label offset',)

    def __init__(
        self,
        audio_files: List[str],
        durations: List[float],
        labels: List[Union[int, str]],
        offsets: List[Optional[float]],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-label manifest with filters and preprocessing.

        Args:
            audio_files: List of audio files.
            durations: List of float durations.
            labels: List of labels.
            offsets: List of offsets or None.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        if index_by_file_id:
            self.mapping = {}
        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0
        for audio_file, duration, command, offset in zip(audio_files, durations, labels, offsets):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                continue

            data.append(output_type(audio_file, duration, command, offset))

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info(
            "Filtered duration for loading collection is %f.", duration_filtered,
        )
        self.uniq_labels = sorted(set(map(lambda x: x.label, data)))
        logging.info("# {} files loaded accounting to # {} labels".format(len(data), len(self.uniq_labels)))

        super().__init__(data)


class ASRSpeechLabel(SpeechLabel):
    """`SpeechLabel` collector from structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], is_regression_task=False, *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            is_regression_task: It's a regression task
            *args: Args to pass to `SpeechLabel` constructor.
            **kwargs: Kwargs to pass to `SpeechLabel` constructor.
        """
        audio_files, durations, labels, offsets = [], [], [], []

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item):
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            if not is_regression_task:
                labels.append(item['label'])
            else:
                labels.append(float(item['label']))

            offsets.append(item['offset'])

        super().__init__(audio_files, durations, labels, offsets, *args, **kwargs)

    def __parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Audio file
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper audio file key."
            )
        item['audio_file'] = os.path.expanduser(item['audio_file'])

        # Duration.
        if 'duration' not in item:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper duration key.")

        # Label.
        if 'command' in item:
            item['label'] = item.pop('command')
        elif 'target' in item:
            item['label'] = item.pop('target')
        elif 'label' in item:
            pass
        else:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper label key.")

        item = dict(
            audio_file=item['audio_file'],
            duration=item['duration'],
            label=item['label'],
            offset=item.get('offset', None),
        )

        return item


class FeatureSequenceLabel(_Collection):
    """List of feature sequence of label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='FeatureSequenceLabelEntity', field_names='feature_file seq_label',)

    def __init__(
        self,
        feature_files: List[str],
        seq_labels: List[str],
        max_number: Optional[int] = None,
        index_by_file_id: bool = False,
    ):
        """Instantiates feature-SequenceLabel manifest with filters and preprocessing.

        Args:
            feature_files: List of feature files.
            seq_labels: List of sequences of abels.
            max_number: Maximum number of samples to collect.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, num_filtered = (
            [],
            0.0,
        )
        self.uniq_labels = set()

        if index_by_file_id:
            self.mapping = {}

        for feature_file, seq_label in zip(feature_files, seq_labels):

            label_tokens, uniq_labels_in_seq = self.relative_speaker_parser(seq_label)

            data.append(output_type(feature_file, label_tokens))
            self.uniq_labels |= uniq_labels_in_seq

            if label_tokens is None:
                num_filtered += 1
                continue

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(feature_file))
                self.mapping[feature_file] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        logging.info("# {} files loaded including # {} unique labels".format(len(data), len(self.uniq_labels)))
        super().__init__(data)

    def relative_speaker_parser(self, seq_label):
        """ Convert sequence of speaker labels to relative labels.
        Convert sequence of absolute speaker to sequence of relative speaker [E A C A E E C] -> [0 1 2 1 0 0 2]
        In this seq of label , if label do not appear before, assign new relative labels len(pos); else reuse previous assigned relative labels.
        Args:
            seq_label (str): A string of a sequence of labels.

        Return:
            relative_seq_label (List) : A list of relative sequence of labels
            unique_labels_in_seq (Set): A set of unique labels in the sequence
        """
        seq = seq_label.split()
        conversion_dict = dict()
        relative_seq_label = []

        for seg in seq:
            if seg in conversion_dict:
                converted = conversion_dict[seg]
            else:
                converted = len(conversion_dict)
                conversion_dict[seg] = converted

            relative_seq_label.append(converted)

        unique_labels_in_seq = set(conversion_dict.keys())
        return relative_seq_label, unique_labels_in_seq


class ASRFeatureSequenceLabel(FeatureSequenceLabel):
    """`FeatureSequenceLabel` collector from asr structured json files."""

    def __init__(
        self, manifests_files: Union[str, List[str]], max_number: Optional[int] = None, index_by_file_id: bool = False,
    ):

        """Parse lists of feature files and sequences of labels.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            max_number:  Maximum number of samples to collect; pass to `FeatureSequenceLabel` constructor.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data; pass to `FeatureSequenceLabel` constructor.
        """

        feature_files, seq_labels = [], []
        for item in manifest.item_iter(manifests_files, parse_func=self._parse_item):
            feature_files.append(item['feature_file'])
            seq_labels.append(item['seq_label'])

        super().__init__(feature_files, seq_labels, max_number, index_by_file_id)

    def _parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Feature file
        if 'feature_filename' in item:
            item['feature_file'] = item.pop('feature_filename')
        elif 'feature_filepath' in item:
            item['feature_file'] = item.pop('feature_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper feature file key."
            )
        item['feature_file'] = os.path.expanduser(item['feature_file'])

        # Seq of Label.
        if 'seq_label' in item:
            item['seq_label'] = item.pop('seq_label')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper seq_label key."
            )

        item = dict(feature_file=item['feature_file'], seq_label=item['seq_label'],)

        return item
