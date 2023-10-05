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

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from omegaconf import MISSING

from nemo.collections.asr.models.configs import asr_models_config as ctc_cfg
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorConfig,
    SpectrogramAugmentationConfig,
)
from nemo.collections.asr.modules.conv_asr import ConvASRDecoderConfig, ConvASREncoderConfig, JasperEncoderConfig
from nemo.core.config import modelPT as model_cfg


# fmt: off
def qn_15x5():
    config = [
        JasperEncoderConfig(filters=256, repeat=1, kernel=[33], stride=[2], dilation=[1], dropout=0.0,
                            residual=False, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[33], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[33], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[33], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[39], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[39], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[39], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[51], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[51], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[51], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[63], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[63], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[63], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[75], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[75], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[75], stride=[1], dilation=[1], dropout=0.0,
                            residual=True, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=1, kernel=[87], stride=[1], dilation=[2], dropout=0.0,
                            residual=False, groups=1, separable=True, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=1024, repeat=1, kernel=[1], stride=[1], dilation=[1], dropout=0.0,
                            residual=False, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False)
    ]
    return config


def jasper_10x5_dr():
    config = [
        JasperEncoderConfig(filters=256, repeat=1, kernel=[11], stride=[2], dilation=[1], dropout=0.2,
                            residual=False, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[11], stride=[1], dilation=[1], dropout=0.2,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=256, repeat=5, kernel=[11], stride=[1], dilation=[1], dropout=0.2,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=384, repeat=5, kernel=[13], stride=[1], dilation=[1], dropout=0.2,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=384, repeat=5, kernel=[13], stride=[1], dilation=[1], dropout=0.2,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[17], stride=[1], dilation=[1], dropout=0.2,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=512, repeat=5, kernel=[17], stride=[1], dilation=[1], dropout=0.2,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=640, repeat=5, kernel=[21], stride=[1], dilation=[1], dropout=0.3,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=640, repeat=5, kernel=[21], stride=[1], dilation=[1], dropout=0.3,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=768, repeat=5, kernel=[25], stride=[1], dilation=[1], dropout=0.3,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=768, repeat=5, kernel=[25], stride=[1], dilation=[1], dropout=0.3,
                            residual=True, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=True, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=896, repeat=1, kernel=[29], stride=[1], dilation=[2], dropout=0.4,
                            residual=False, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False),
        JasperEncoderConfig(filters=1024, repeat=1, kernel=[1], stride=[1], dilation=[1], dropout=0.4,
                            residual=False, groups=1, separable=False, heads=-1, residual_mode='add',
                            residual_dense=False, se=False, se_reduction_ratio=8, se_context_size=-1,
                            se_interpolation_mode='nearest', kernel_size_factor=1.0, stride_last=False)
    ]
    return config
# fmt: on


@dataclass
class JasperModelConfig(ctc_cfg.EncDecCTCConfig):
    # Model global arguments
    sample_rate: int = 16000
    repeat: int = 1
    dropout: float = 0.0
    separable: bool = False
    labels: List[str] = MISSING

    # Dataset configs
    train_ds: ctc_cfg.ASRDatasetConfig = field(
        default_factory=lambda: ctc_cfg.ASRDatasetConfig(manifest_filepath=None, shuffle=True, trim_silence=True)
    )
    validation_ds: ctc_cfg.ASRDatasetConfig = field(
        default_factory=lambda: ctc_cfg.ASRDatasetConfig(manifest_filepath=None, shuffle=False)
    )
    test_ds: ctc_cfg.ASRDatasetConfig = field(
        default_factory=lambda: ctc_cfg.ASRDatasetConfig(manifest_filepath=None, shuffle=False)
    )

    # Optimizer / Scheduler config
    optim: Optional[model_cfg.OptimConfig] = field(
        default_factory=lambda: model_cfg.OptimConfig(sched=model_cfg.SchedConfig())
    )

    # Model general component configs
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = field(
        default_factory=lambda: AudioToMelSpectrogramPreprocessorConfig()
    )
    spec_augment: Optional[SpectrogramAugmentationConfig] = field(
        default_factory=lambda: SpectrogramAugmentationConfig()
    )
    encoder: ConvASREncoderConfig = field(default_factory=lambda: ConvASREncoderConfig(activation="relu"))
    decoder: ConvASRDecoderConfig = field(default_factory=lambda: ConvASRDecoderConfig())


@dataclass
class QuartzNetModelConfig(JasperModelConfig):
    separable: bool = True


class EncDecCTCModelConfigBuilder(model_cfg.ModelConfigBuilder):
    VALID_CONFIGS = ['quartznet_15x5', 'quartznet_15x5_zh', 'jasper_10x5dr']

    def __init__(self, name: str = 'quartznet_15x5', encoder_cfg_func: Optional[Callable[[], List[Any]]] = None):
        if name not in EncDecCTCModelConfigBuilder.VALID_CONFIGS:
            raise ValueError("`name` must be one of : \n" f"{EncDecCTCModelConfigBuilder.VALID_CONFIGS}")

        self.name = name

        if 'quartznet_15x5' in name:
            if encoder_cfg_func is None:
                encoder_cfg_func = qn_15x5

            model_cfg = QuartzNetModelConfig(
                repeat=5,
                separable=True,
                spec_augment=SpectrogramAugmentationConfig(rect_masks=5, rect_freq=50, rect_time=120),
                encoder=ConvASREncoderConfig(jasper=encoder_cfg_func(), activation="relu"),
                decoder=ConvASRDecoderConfig(),
            )

        elif 'jasper_10x5' in name:
            if encoder_cfg_func is None:
                encoder_cfg_func = jasper_10x5_dr

            model_cfg = JasperModelConfig(
                repeat=5,
                separable=False,
                spec_augment=SpectrogramAugmentationConfig(rect_masks=5, rect_freq=50, rect_time=120),
                encoder=ConvASREncoderConfig(jasper=encoder_cfg_func(), activation="relu"),
                decoder=ConvASRDecoderConfig(),
            )

        else:
            raise ValueError(f"Invalid config name submitted to {self.__class__.__name__}")

        super(EncDecCTCModelConfigBuilder, self).__init__(model_cfg)
        self.model_cfg: ctc_cfg.EncDecCTCConfig = model_cfg  # enable type hinting

        if 'zh' in name:
            self.set_dataset_normalize(normalize=False)

    def set_labels(self, labels: List[str]):
        self.model_cfg.labels = labels

    def set_separable(self, separable: bool):
        self.model_cfg.separable = separable

    def set_repeat(self, repeat: int):
        self.model_cfg.repeat = repeat

    def set_sample_rate(self, sample_rate: int):
        self.model_cfg.sample_rate = sample_rate

    def set_dropout(self, dropout: float = 0.0):
        self.model_cfg.dropout = dropout

    def set_dataset_normalize(self, normalize: bool):
        self.model_cfg.train_ds.normalize = normalize
        self.model_cfg.validation_ds.normalize = normalize
        self.model_cfg.test_ds.normalize = normalize

    # Note: Autocomplete for users wont work without these overrides
    # But practically it is not needed since python will infer at runtime

    # def set_train_ds(self, cfg: Optional[ctc_cfg.ASRDatasetConfig] = None):
    #     super().set_train_ds(cfg)
    #
    # def set_validation_ds(self, cfg: Optional[ctc_cfg.ASRDatasetConfig] = None):
    #     super().set_validation_ds(cfg)
    #
    # def set_test_ds(self, cfg: Optional[ctc_cfg.ASRDatasetConfig] = None):
    #     super().set_test_ds(cfg)

    def _finalize_cfg(self):
        # propagate labels
        self.model_cfg.train_ds.labels = self.model_cfg.labels
        self.model_cfg.validation_ds.labels = self.model_cfg.labels
        self.model_cfg.test_ds.labels = self.model_cfg.labels
        self.model_cfg.decoder.vocabulary = self.model_cfg.labels

        # propagate num classes
        self.model_cfg.decoder.num_classes = len(self.model_cfg.labels)

        # propagate sample rate
        self.model_cfg.sample_rate = self.model_cfg.sample_rate
        self.model_cfg.preprocessor.sample_rate = self.model_cfg.sample_rate
        self.model_cfg.train_ds.sample_rate = self.model_cfg.sample_rate
        self.model_cfg.validation_ds.sample_rate = self.model_cfg.sample_rate
        self.model_cfg.test_ds.sample_rate = self.model_cfg.sample_rate

        # propagate filters
        self.model_cfg.encoder.feat_in = self.model_cfg.preprocessor.features
        self.model_cfg.decoder.feat_in = self.model_cfg.encoder.jasper[-1].filters

        # propagate separable
        for layer in self.model_cfg.encoder.jasper[:-1]:  # type: JasperEncoderConfig
            layer.separable = self.model_cfg.separable

        # propagate repeat
        for layer in self.model_cfg.encoder.jasper[1:-2]:  # type: JasperEncoderConfig
            layer.repeat = self.model_cfg.repeat

        # propagate dropout
        for layer in self.model_cfg.encoder.jasper:  # type: JasperEncoderConfig
            layer.dropout = self.model_cfg.dropout

    def build(self) -> ctc_cfg.EncDecCTCConfig:
        return super().build()
