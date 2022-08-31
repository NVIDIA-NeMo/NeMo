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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import MISSING

import nemo.core.classes.dataset
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMFCCPreprocessorConfig,
    CropOrPadSpectrogramAugmentationConfig,
    SpectrogramAugmentationConfig,
)
from nemo.collections.asr.modules.conv_asr import ConvASRDecoderClassificationConfig, ConvASREncoderConfig
from nemo.core.config import modelPT as model_cfg


@dataclass
class EncDecClassificationDatasetConfig(nemo.core.classes.dataset.DatasetConfig):
    manifest_filepath: Optional[str] = None
    sample_rate: int = MISSING
    labels: List[str] = MISSING
    trim_silence: bool = False

    # Tarred dataset support
    is_tarred: bool = False
    tarred_audio_filepaths: Optional[str] = None
    tarred_shard_strategy: str = "scatter"
    shuffle_n: int = 0

    # Optional
    int_values: Optional[int] = None
    augmentor: Optional[Dict[str, Any]] = None
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None
    cal_labels_occurrence: Optional[bool] = False

    # VAD Optional
    vad_stream: Optional[bool] = None
    window_length_in_sec: float = 0.31
    shift_length_in_sec: float = 0.01
    normalize_audio: bool = False
    is_regression_task: bool = False

    # bucketing params
    bucketing_strategy: str = "synced_randomized"
    bucketing_batch_size: Optional[Any] = None
    bucketing_weights: Optional[List[int]] = None


@dataclass
class EncDecClassificationConfig(model_cfg.ModelConfig):
    # Model global arguments
    sample_rate: int = 16000
    repeat: int = 1
    dropout: float = 0.0
    separable: bool = True
    kernel_size_factor: float = 1.0
    labels: List[str] = MISSING
    timesteps: int = MISSING

    # Dataset configs
    train_ds: EncDecClassificationDatasetConfig = EncDecClassificationDatasetConfig(
        manifest_filepath=None, shuffle=True, trim_silence=False
    )
    validation_ds: EncDecClassificationDatasetConfig = EncDecClassificationDatasetConfig(
        manifest_filepath=None, shuffle=False
    )
    test_ds: EncDecClassificationDatasetConfig = EncDecClassificationDatasetConfig(
        manifest_filepath=None, shuffle=False
    )

    # Optimizer / Scheduler config
    optim: Optional[model_cfg.OptimConfig] = model_cfg.OptimConfig(sched=model_cfg.SchedConfig())

    # Model component configs
    preprocessor: AudioToMFCCPreprocessorConfig = AudioToMFCCPreprocessorConfig()
    spec_augment: Optional[SpectrogramAugmentationConfig] = SpectrogramAugmentationConfig()
    crop_or_pad_augment: Optional[CropOrPadSpectrogramAugmentationConfig] = CropOrPadSpectrogramAugmentationConfig(
        audio_length=timesteps
    )

    encoder: ConvASREncoderConfig = ConvASREncoderConfig()
    decoder: ConvASRDecoderClassificationConfig = ConvASRDecoderClassificationConfig()


@dataclass
class EncDecClassificationModelConfig(model_cfg.NemoConfig):
    model: EncDecClassificationConfig = EncDecClassificationConfig()
