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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.speechlm.utils import to_dict_config
from nemo.core.classes.common import Serialization
from nemo.core.classes.module import NeuralModule
from nemo.lightning import io
from nemo.utils import logging, model_utils


class MCoreASRModule(MegatronModule):
    """
    Wrapper class for ASR encoder from `nemo.collections.asr.models.ASRModel`.

    `TransformerConfig` is a dummy config to satisfy the `MegatronModule` constructor.
    `num_attention_heads` is set to 16 such that it's divisible by the value of TP.
    `num_layers` and `hidden_size` are set to 1 since not used.
    """

    def __init__(
        self,
        encoder: NeuralModule,
        preprocessor: Optional[nn.Module] = None,
        spec_augment: Optional[nn.Module] = None,
    ):
        super().__init__(config=TransformerConfig(num_layers=1, hidden_size=1, num_attention_heads=16))
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.spec_augmentation = spec_augment

    def maybe_preprocess_audio(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) is False:
            raise ValueError(
                f"{self.__class__} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )
        return processed_signal, processed_signal_length

    def forward(
        self,
        input_signal: Optional[torch.Tensor],
        input_signal_length: Optional[torch.Tensor],
        processed_signal: Optional[torch.Tensor],
        processed_signal_length: Optional[torch.Tensor],
    ):
        processed_signal, processed_signal_length = self.maybe_preprocess_audio(
            input_signal, input_signal_length, processed_signal, processed_signal_length
        )

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        return encoded, encoded_len


@dataclass
class ASRModuleConfig(ModelParallelConfig, io.IOMixin):
    _target_: Optional[str] = None
    pretrained_model: Optional[str] = "stt_en_fastconformer_transducer_large"
    config: Optional[dict] = None
    preprocessor_config: Optional[dict] = None
    spec_augment_config: Optional[dict] = None
    init_from_pretrained_model: Optional[str] = None
    init_from_nemo_model: Optional[str] = None
    init_from_ptl_ckpt: Optional[str] = None
    target_module: Optional[str] = "encoder"

    def configure_model(self):
        if self._target_ is not None:
            imported_cls = model_utils.import_class_by_path(self._target_)
        else:
            imported_cls = nemo_asr.models.ASRModel
        if self.pretrained_model is not None and self.config is None:
            if str(self.pretrained_model).endswith(".nemo"):
                asr_model = imported_cls.restore_from(self.pretrained_model)  # type: nemo_asr.models.ASRModel
            else:
                asr_model = imported_cls.from_pretrained(
                    model_name=self.pretrained_model
                )  # type: nemo_asr.models.ASRModel
            self.config = OmegaConf.to_container(asr_model.cfg, resolve=True)
        else:
            cfg = OmegaConf.create(self.config)
            asr_model = imported_cls(cfg=cfg)  # type: nemo_asr.models.ASRModel
            asr_model.maybe_init_from_pretrained_checkpoint(self)

        model = asr_model
        if self.target_module is not None:
            model = getattr(asr_model, self.target_module, None)  # type: NeuralModule
        if model is None:
            raise ValueError(f"Model {self._target_} does not have attribute {self.target_module}")

        if self.preprocessor_config is not None:
            preprocessor = Serialization.from_config_dict(to_dict_config(self.preprocessor_config))
        elif hasattr(asr_model, "preprocessor"):
            preprocessor = asr_model.preprocessor
        else:
            preprocessor = None
            logging.warning(f"Model {self._target_} does not have a preprocessor, use with caution.")

        if self.spec_augment_config is not None:
            spec_augment = Serialization.from_config_dict(to_dict_config(self.spec_augment_config))
        else:
            spec_augment = None

        return MCoreASRModule(encoder=model, preprocessor=preprocessor, spec_augment=spec_augment)
