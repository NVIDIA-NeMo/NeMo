# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from torch.nn import Linear

from nemo.collections.common.losses.cross_entropy import CrossEntropyLoss
from nemo.collections.nlp.models.token_classification.punctuation_capitalization_model import (
    PunctuationCapitalizationModel,
)
from nemo.collections.nlp.modules.common.transformer import TransformerDecoder
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import adapter_mixins

try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer

    ASR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ASR_AVAILABLE = False

__all__ = ['PunctuationCapitalizationLexicalAudioModel']


def update_model_config_to_support_adapter(model_cfg):
    with open_dict(model_cfg):
        adapter_metadata = adapter_mixins.get_registered_adapter(model_cfg.encoder._target_)
        if adapter_metadata is not None:
            model_cfg.encoder._target_ = adapter_metadata.adapter_class_path

    return model_cfg


class PunctuationCapitalizationLexicalAudioModel(PunctuationCapitalizationModel):
    """
        A model for restoring punctuation and capitalization in text using lexical and audio features.

        The model consists of a language model and two multilayer perceptrons (MLP) on top the fusion of LM and AM. The first
        MLP serves for punctuation prediction and the second is for capitalization prediction. You can use only BERT-like
        HuggingFace language models (model ``forward`` method accepts ``input_ids``, ``token_types_ids``,
        ``attention_mask`` arguments). See more about model config options :ref:`here<model-config-label>`.
        And any :class:`~nemo.collections.asr.models.EncDecCTCModel` which has encoder module which is used as an AM.

        For training and testing use dataset
        :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset` with parameter ``use_audio`` set to ``True``,
        for training on huge amounts of data which cannot be loaded into memory simultaneously use
        :class:`~nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.BertPunctuationCapitalizationTarredDataset` with parameter ``use_audio`` set to ``True``.

        Args:
            cfg: a model configuration. It should follow dataclass
                :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_config.PunctuationCapitalizationLexicalAudioModelConfig`
                See an example of full config in
                `nemo/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml
                <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml>`_
            trainer: an instance of a PyTorch Lightning trainer
        """

    def __init__(self, cfg: DictConfig, trainer: Trainer = None) -> None:
        super().__init__(cfg, trainer)
        if not ASR_AVAILABLE:
            raise ModuleNotFoundError(
                'Nemo ASR was not installed, see https://github.com/NVIDIA/NeMo#installation for installation instructions'
            )
        audio_cfg = nemo_asr.models.ASRModel.from_pretrained(cfg.audio_encoder.pretrained_model, return_config=True)

        if cfg.audio_encoder.get('adapter', None):
            if cfg.audio_encoder.adapter.enable:
                audio_cfg = update_model_config_to_support_adapter(audio_cfg)

        self.audio_encoder = nemo_asr.models.ASRModel.from_pretrained(
            cfg.audio_encoder.pretrained_model, override_config_path=audio_cfg
        )
        if cfg.audio_encoder.adapter.get('enable', False):
            with open_dict(cfg):
                cfg.audio_encoder.adapter.config.in_features = self.audio_encoder.cfg.decoder.feat_in
            self.audio_encoder.add_adapter(name='audio_adapter', cfg=cfg.audio_encoder.adapter.config)
            self.audio_encoder.set_enabled_adapters(enabled=True)
            self.audio_encoder.freeze()
            self.audio_encoder.unfreeze_enabled_adapters()

        self.fusion = TransformerDecoder(
            num_layers=cfg.audio_encoder.fusion.num_layers,
            hidden_size=self.bert_model(**self.bert_model.input_example()[0]).size()[-1],
            inner_size=cfg.audio_encoder.fusion.inner_size,
            num_attention_heads=cfg.audio_encoder.fusion.num_attention_heads,
        )
        self.audio_proj = Linear(
            self.audio_encoder.cfg.decoder.feat_in, self.bert_model(**self.bert_model.input_example()[0]).size()[-1]
        )

        if cfg.audio_encoder.freeze.get('is_enabled', False):
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            for i in range(cfg.audio_encoder.fusion.get('num_layers')):
                self.audio_encoder.add_module(
                    f'conf_encoder_{i}',
                    ConformerLayer(
                        d_model=cfg.audio_encoder.freeze.get('d_model'), d_ff=cfg.audio_encoder.freeze.get('d_ff')
                    ),
                )

        if cfg.get('restore_lexical_encoder_from', None) and not self._is_model_being_restored():
            if os.path.exists(cfg.get('restore_lexical_encoder_from')):
                self.bert_model = (
                    PunctuationCapitalizationModel.restore_from(cfg.restore_lexical_encoder_from)
                    .to(self.device)
                    .bert_model
                )
            else:
                raise ValueError(f'Provided path {cfg.get("restore_lexical_encoder_from")} does not exists')

        del self.audio_encoder.decoder
        del self.audio_encoder._wer
        del self.audio_encoder.loss

        if cfg.get('use_weighted_loss', False):
            punct_freq = torch.tensor(
                list(self.train_dataloader().dataset.punct_label_frequencies.values()), dtype=torch.float
            )
            punct_weight = 1 - (punct_freq - punct_freq.min()) / punct_freq.max()

            capit_freq = torch.tensor(
                list(self.train_dataloader().dataset.capit_label_frequencies.values()), dtype=torch.float
            )
            capit_weight = 1 - (capit_freq - capit_freq.min()) / capit_freq.max()

            self.loss_punct = CrossEntropyLoss(logits_ndim=3, weight=punct_weight)
            self.loss_capit = CrossEntropyLoss(logits_ndim=3, weight=capit_weight)
        else:
            self.loss_punct = self.loss
            self.loss_capit = self.loss

        self.set_max_audio_length(1024)

    def _make_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        punct_logits, capit_logits = self(
            input_ids=batch['input_ids'],
            token_type_ids=batch['segment_ids'],
            attention_mask=batch['input_mask'],
            features=batch['features'],
            features_length=batch['features_length'],
        )

        punct_loss = self.loss_punct(logits=punct_logits, labels=batch['punct_labels'], loss_mask=batch['loss_mask'])
        capit_loss = self.loss_capit(logits=capit_logits, labels=batch['capit_labels'], loss_mask=batch['loss_mask'])
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        return loss, punct_logits, capit_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        features: torch.Tensor = None,
        features_length: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
                Executes a forward pass through the model. For more details see ``forward`` method of :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_config.PunctuationCapitalizationLexicalAudioModelConfig`
                and ``forward`` method of :class:'~nemo.collections.asr.models.EncDecCTCModel'

                Args:
                    input_ids (:obj:`torch.Tensor`): an integer torch tensor of shape ``[Batch, Time]``. Contains encoded
                        source tokens.
                    attention_mask (:obj:`torch.Tensor`): a boolean torch tensor of shape ``[Batch, Time]``. Contains an
                        attention mask for excluding paddings.
                    token_type_ids (:obj:`torch.Tensor`): an integer torch Tensor of shape ``[Batch, Time]``. Contains an index
                        of segment to which a token belongs. If ``token_type_ids`` is not ``None``, then it should be a zeros
                        tensor.
                    features (:obj:`torch.Tensor`): tensor that represents a batch of raw audio signals,
                        of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                        sample_rate number of floating point values.
                    features_length (:obj:`torch.Tensor`): Vector of length B, that contains the individual lengths of the audio
                        sequences.

                Returns:
                    :obj:`Tuple[torch.Tensor, torch.Tensor]`: a tuple containing

                        - ``punct_logits`` (:obj:`torch.Tensor`): a float torch tensor of shape
                          ``[Batch, Time, NumPunctuationLabels]`` containing punctuation logits
                        - ``capit_logits`` (:obj:`torch.Tensor`): a float torch tensor of shape
                          ``[Batch, Time, NumCapitalizationLabels]`` containing capitalization logits
                """
        self.update_max_seq_length(seq_length=features.size(1), device=features.device)
        lexical_hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        if isinstance(lexical_hidden_states, tuple):
            lexical_hidden_states = lexical_hidden_states[0]

        processed_signal, processed_signal_length = self.audio_encoder.preprocessor(
            input_signal=features, length=features_length,
        )

        if self.audio_encoder.spec_augmentation is not None and self.training:
            processed_signal = self.audio_encoder.spec_augmentation(
                input_spec=processed_signal, length=processed_signal_length
            )

        audio_hidden_states, audio_hidden_states_length = self.audio_encoder.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        audio_hidden_states = audio_hidden_states.permute(0, 2, 1)
        audio_hidden_states = self.audio_proj(audio_hidden_states)

        fused = self.fusion(
            lexical_hidden_states,
            attention_mask,
            audio_hidden_states,
            self.make_pad_mask(audio_hidden_states.size(1), audio_hidden_states_length),
        )

        punct_logits = self.punct_classifier(hidden_states=fused)
        capit_logits = self.capit_classifier(hidden_states=fused)

        return punct_logits, capit_logits

    def make_pad_mask(self, max_audio_length, seq_lens):
        """Make masking for padding."""
        mask = self.seq_range[:max_audio_length].expand(seq_lens.size(0), -1) < seq_lens.unsqueeze(-1)
        return mask

    def update_max_seq_length(self, seq_length: int, device):
        if torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)

            # Update across all ranks in the distributed system
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)

            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_audio_length, device=device)
        if hasattr(self, 'seq_range'):
            self.seq_range = seq_range
        else:
            self.register_buffer('seq_range', seq_range, persistent=False)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
