# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map, get_uniqname_from_filepath
from nemo.collections.asr.parts.utils.vad_utils import PostProcessingParams, load_postprocessing_from_yaml
from nemo.collections.common.data.utils import move_data_to_device
from nemo.utils import logging

GenericDiarizationType = Union[List[Any], List[List[Any]], Tuple[Any], Tuple[List[Any]]]


@dataclass
class InternalDiarizeConfig:
    """Internal diarization configuration parameters for diarization inference."""

    # Internal values
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    training_mode: bool = False
    logging_level: Optional[Any] = None

    # Preprocessor values
    dither_value: float = 0.0
    pad_to_value: int = 0

    # Scratch space
    temp_dir: Optional[str] = None
    manifest_filepath: Optional[str] = None


@dataclass
class DiarizeConfig:
    """Configuration parameters for diarization inference."""

    session_len_sec: float = -1  # End-to-end diarization session length limit in seconds
    batch_size: int = 1
    num_workers: int = 1
    postprocessing_yaml: Optional[str] = None  # Path to a yaml file for postprocessing configurations
    verbose: bool = True
    include_tensor_outputs: bool = False
    postprocessing_params: PostProcessingParams = None

    # Utility
    _internal: Optional[InternalDiarizeConfig] = None


def get_value_from_diarization_config(diarcfg, key, default):
    """
    Utility function to get a value from the diarization config.
    If the value is not present in the diarization config, the default value is returned.

    Args:
        diarcfg: A dataclass that represents the diarization config.
        key: The name of the arg to retrieve.
        default: The default value to return if the key is not present in the diarization config.

    Returns:
        The value of the key in the diarization config or the default value.
    """
    if hasattr(diarcfg, key):
        return getattr(diarcfg, key)
    else:
        logging.debug(
            f"Using default value of {default} for {key} because it is not present \
                in the diarization config {diarcfg}."
        )
        return default


class SpkDiarizationMixin(ABC):
    """
    An abstract class for diarize-able models.

    Creates a template function `diarize()` that provides an interface to perform transcription of audio tensors or
    filepaths.

    The following abstract classes must be implemented by the subclass:

        - `_setup_diarize_dataloader()`:
            Setup the dataloader for diarization. Receives the output from
            `_diarize_input_manifest_processing()`.

        - `_diarize_forward()`:
            Implements the model's custom forward pass to return outputs that are processed by
            `_diarize_output_processing()`.

        - `_diarize_output_processing()`:
            Implements the post processing of the model's outputs to return the results to
            the user. The result can be a list of objects, list of list of objects, tuple of objects, tuple of list of
            objects, or a dict of list of objects.

    """

    def __init__(self):
        self._diarize_audio_rttm_map = {}

    @torch.inference_mode()
    def diarize(
        self,
        audio: Union[str, List[str], np.ndarray, DataLoader],
        batch_size: int = 1,
        include_tensor_outputs: bool = False,
        postprocessing_yaml: Optional[str] = None,
        num_workers: int = 1,
        verbose: bool = False,
        override_config: Optional[DiarizeConfig] = None,
        **config_kwargs,
    ) -> GenericDiarizationType:
        """
        Takes paths to audio files and returns speaker labels
        """

        if override_config is None:
            postprocessing_params = load_postprocessing_from_yaml(postprocessing_yaml)
            diarize_cfg = DiarizeConfig(
                batch_size=batch_size,
                num_workers=num_workers,
                verbose=verbose,
                include_tensor_outputs=include_tensor_outputs,
                postprocessing_yaml=postprocessing_yaml,
                postprocessing_params=postprocessing_params,
                **config_kwargs,
            )
        else:
            if not hasattr(override_config, '_internal'):
                raise ValueError(
                    "`diarize_cfg must have an `_internal` argument, which must be of an object of type "
                    "InternalDiarizeConfig or its subclass."
                )

            if override_config._internal is None:
                override_config._internal = InternalDiarizeConfig()

            diarize_cfg = override_config

        # Add new internal config
        if diarize_cfg._internal is None:
            diarize_cfg._internal = InternalDiarizeConfig()
        else:
            # Check if internal config is valid
            if not isinstance(diarize_cfg._internal, InternalDiarizeConfig):
                raise ValueError(
                    "`diarize_cfg._internal` must be of an object of type InternalDiarizeConfig or " "its subclass"
                )

        # Hold the results here
        results = None

        try:
            generator = self.diarize_generator(audio, override_config=diarize_cfg)

            for processed_outputs in generator:
                # Store results
                if isinstance(processed_outputs, list):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = []

                    results.extend(processed_outputs)

                elif isinstance(processed_outputs, tuple):
                    # Create a results of the same type as each element in processed_outputs
                    if results is None:
                        results = tuple([[] for _ in processed_outputs])

                    # If nested list structure
                    if isinstance(processed_outputs[0], list):
                        for i, processed_output in enumerate(processed_outputs):
                            results[i].extend(processed_output)
                    else:
                        # If flat list structure
                        if len(processed_outputs) != len(results):
                            raise RuntimeError(
                                f"The number of elements in the result ({len(results)}) does not "
                                f"match the results of the current batch ({len(processed_outputs)})."
                            )

                        for i, processed_output in enumerate(processed_outputs):
                            results[i].append(processed_output)

        except StopIteration:
            pass

        return results

    def diarize_generator(self, audio, override_config: Optional[DiarizeConfig]):
        """
        A generator version of `diarize` function.
        """
        if override_config is None:
            override_config = DiarizeConfig()

        if not hasattr(override_config, '_internal'):
            raise ValueError(
                "`diarize_cfg must have an `_internal` argument, which must be of an object of type "
                "InternalDiarizeConfig or its subclass."
            )

        # Add new internal config
        if override_config._internal is None:
            override_config._internal = InternalDiarizeConfig()
        else:
            # Check if internal config is valid
            if not isinstance(override_config._internal, InternalDiarizeConfig):
                raise ValueError(
                    "`diarize_cfg._internal` must be of an object of type InternalDiarizeConfig or " "its subclass"
                )

        diarize_cfg = override_config

        try:
            # Initialize and assert the diarization environment
            self._diarize_on_begin(audio, diarize_cfg)

            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                diarize_cfg._internal.temp_dir = tmpdir

                # Create a DataLoader if not already present
                if not isinstance(audio, DataLoader):
                    dataloader = self._diarize_input_processing(audio, diarize_cfg)
                else:
                    dataloader = audio

                if hasattr(diarize_cfg, 'verbose'):
                    verbose = diarize_cfg.verbose
                else:
                    verbose = True

                for batch_idx, test_batch in enumerate(tqdm(dataloader, desc="Diarizing", disable=not verbose)):
                    # Move batch to device
                    test_batch = move_data_to_device(test_batch, diarize_cfg._internal.device)
                    uniq_ids = list(self._diarize_audio_rttm_map.keys())[
                        batch_idx * diarize_cfg.batch_size : (batch_idx + 1) * diarize_cfg.batch_size
                    ]

                    # Run forward pass
                    pred_outputs = self._diarize_forward(test_batch)
                    processed_outputs = self._diarize_output_processing(pred_outputs, uniq_ids, diarize_cfg)

                    # Yield results if generator
                    yield processed_outputs

                    # clear up memory
                    del test_batch, pred_outputs, processed_outputs
                    torch.cuda.empty_cache()

        finally:
            # set mode back to its original value
            self._diarize_on_end(diarize_cfg)

    def _input_audio_to_rttm_processing(self, audio_files: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Generate manifest style dict if `audio` is a list of paths to audio files.

        Args:
            audio_files: A list of paths to audio files.

        Returns:
            audio_rttm_map_dict A list of manifest style dicts.
        """
        audio_rttm_map_dict = {}
        for audio_file in audio_files:
            uniq_id = get_uniqname_from_filepath(audio_file)
            entry = {
                'uniq_id': uniq_id,
                'audio_filepath': audio_file,
                'offset': 0.0,
                'duration': None,
                'text': '-',
                'label': 'infer',
            }
            audio_rttm_map_dict[uniq_id] = entry
        return audio_rttm_map_dict

    def _diarize_on_begin(self, audio: Union[str, List[str]], diarcfg: DiarizeConfig):
        """
        Internal function to setup the model for diarization. Perform all setup and pre-checks here.

        Args:
            audio (Union[str, List[str]]): Of type `GenericDiarizationType`
            diarcfg (DiarizeConfig): An instance of `DiarizeConfig`.
        """
        if audio is None:
            return {}

        if isinstance(audio, str):
            audio = [audio]

        if isinstance(audio, list) and len(audio) == 0:
            return {}

        # Set num_workers
        num_workers = get_value_from_diarization_config(diarcfg, 'num_workers', default=1)

        if num_workers is None:
            _batch_size = get_value_from_diarization_config(diarcfg, 'batch_size', default=1)
            num_workers = min(_batch_size, os.cpu_count() - 1)

        # Assign num_workers if available as key in diarcfg
        if hasattr(diarcfg, 'num_workers'):
            diarcfg.num_workers = num_workers

        # Model's mode and device
        diarcfg._internal.training_mode = self.training

        # Switch model to evaluation mode
        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                diarcfg._internal.dither_value = self.preprocessor.featurizer.dither
                self.preprocessor.featurizer.dither = 0.0

            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                diarcfg._internal.pad_to_value = self.preprocessor.featurizer.pad_to
                self.preprocessor.featurizer.pad_to = 0

        # Switch model to evaluation mode
        self.eval()

        # Disable logging
        diarcfg._internal.logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

    def _diarize_input_processing(self, audio, diarcfg: DiarizeConfig):
        """
        Internal function to process the input audio data and return a DataLoader. This function is called by
        `diarize()` and `diarize_generator()` to setup the input data for diarization.

        Args:
            audio: Of type `GenericDiarizationType`
            diarcfg: The diarization config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        if isinstance(audio, (list, tuple)):
            if len(audio) == 0:
                raise ValueError("Input `audio` is empty")
        else:
            # Assume it is a single variable, so wrap it in a list
            audio = [audio]

        # Check if audio is a list of strings (filepaths or manifests)
        if isinstance(audio[0], str):
            if len(audio) == 1 and audio[0].endswith('.json') or audio[0].endswith('.jsonl'):
                # Assume it is a path to a manifest file
                diarcfg._internal.manifest_filepath = audio[0]
                self._diarize_audio_rttm_map = audio_rttm_map(audio[0])
                audio_files = []
                for uniq_id, meta_dict in self._diarize_audio_rttm_map.items():
                    audio_files.append(meta_dict['audio_filepath'])
            else:
                # Make `audio_files` a list of audio file paths
                audio_files = list(audio)
                self._diarize_audio_rttm_map = self._input_audio_to_rttm_processing(audio_files=audio_files)

            tmp_dir = diarcfg._internal.temp_dir
            ds_config = self._diarize_input_manifest_processing(audio_files, tmp_dir, diarcfg)

            temp_dataloader = self._setup_diarize_dataloader(ds_config)
            return temp_dataloader

        else:
            raise ValueError(
                f"Input `audio` is of type {type(audio[0])}. " "Only `str` (path to audio file) is supported as input."
            )

    def _diarize_input_manifest_processing(
        self, audio_files: List[str], temp_dir: str, diarcfg: DiarizeConfig
    ) -> Dict[str, Any]:
        """
        Internal function to process the input audio filepaths and return a config dict for the dataloader.

        Args:
            audio_files: A list of string filepaths for audio files.
            temp_dir: A temporary directory to store intermediate files.
            diarcfg: The diarization config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            A config dict that is used to setup the dataloader for diarization.
        """
        with open(os.path.join(temp_dir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in audio_files:
                if isinstance(audio_file, str):
                    entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                    fp.write(json.dumps(entry) + '\n')
                elif isinstance(audio_file, dict):
                    fp.write(json.dumps(audio_file) + '\n')
                else:
                    raise ValueError(
                        f"Input `audio` is of type {type(audio_file)}. "
                        "Only `str` (path to audio file) or `dict` are supported as input."
                    )

        ds_config = {
            'paths2audio_files': audio_files,
            'batch_size': get_value_from_diarization_config(diarcfg, 'batch_size', 1),
            'temp_dir': temp_dir,
            'session_len_sec': get_value_from_diarization_config(diarcfg, 'session_len_sec', diarcfg.session_len_sec),
            'num_workers': get_value_from_diarization_config(diarcfg, 'num_workers', 1),
        }

        return ds_config

    @abstractmethod
    def _setup_diarize_dataloader(self, config: Dict) -> DataLoader:
        """
        Internal function to setup the dataloader for diarization. This function is called by
        `diarize()` and `diarize_generator()` to setup the input data for diarization.

        Args:
            config: A config dict that is used to setup the dataloader for diarization.
                It can be generated by `_diarize_input_manifest_processing()`.

        Returns:
            A DataLoader object that is used to iterate over the input audio data.
        """
        pass

    @abstractmethod
    def _diarize_forward(self, batch: Any):
        """
        Internal function to perform the model's custom forward pass to return outputs that are processed by
        `_diarize_output_processing()`.
        This function is called by `diarize()` and `diarize_generator()` to perform the model's forward pass.

        Args:
            batch: A batch of input data from the data loader that is used to perform the model's forward pass.

        Returns:
            The model's outputs that are processed by `_diarize_output_processing()`.
        """
        pass

    @abstractmethod
    def _diarize_output_processing(self, outputs, uniq_ids, diarcfg: DiarizeConfig) -> GenericDiarizationType:
        """
        Internal function to process the model's outputs to return the results to the user. This function is called by
        `diarize()` and `diarize_generator()` to process the model's outputs.

        Args:
            outputs: The model's outputs that are processed by `_diarize_forward()`.
            uniq_ids: List of unique recording identificators in batch
            diarcfg: The diarization config dataclass. Subclasses can change this to a different dataclass if needed.

        Returns:
            The output can be a list of
            objects, list of list of objects, tuple of objects, tuple of list of objects.
            Its type is defined in `GenericDiarizationType`.
        """
        pass

    def _diarize_on_end(self, diarcfg: DiarizeConfig):
        """
        Internal function to teardown the model after transcription. Perform all teardown and post-checks here.

        Args:
            diarcfg: The diarization config dataclass. Subclasses can change this to a different dataclass if needed.
        """
        # set mode back to its original value
        self.train(mode=diarcfg._internal.training_mode)

        if hasattr(self, 'preprocessor'):
            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'dither'):
                self.preprocessor.featurizer.dither = diarcfg._internal.dither_value

            if hasattr(self.preprocessor, 'featurizer') and hasattr(self.preprocessor.featurizer, 'pad_to'):
                self.preprocessor.featurizer.pad_to = diarcfg._internal.pad_to_value

        if diarcfg._internal.logging_level is not None:
            logging.set_verbosity(diarcfg._internal.logging_level)
