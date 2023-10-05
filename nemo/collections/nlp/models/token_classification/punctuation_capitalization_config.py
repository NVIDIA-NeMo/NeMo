# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import Any, Dict, Optional

from omegaconf.omegaconf import MISSING, DictConfig, OmegaConf, open_dict

from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    PunctuationCapitalizationEvalDataConfig,
    PunctuationCapitalizationTrainDataConfig,
    legacy_data_config_to_new_data_config,
)
from nemo.core.config import TrainerConfig
from nemo.core.config.modelPT import NemoConfig
from nemo.utils.exp_manager import ExpManagerConfig


@dataclass
class FreezeConfig:
    is_enabled: bool = False
    """Freeze audio encoder weight and add Conformer Layers on top of it"""
    d_model: Optional[int] = 256
    """`d_model` parameter of ``ConformerLayer``"""
    d_ff: Optional[int] = 1024
    """``d_ff`` parameter of ``ConformerLayer``"""
    num_layers: Optional[int] = 8
    """``num_layers`` number of ``ConformerLayer`` modules to add on top of audio encoder"""


@dataclass
class AdapterConfig:
    config: Optional[LinearAdapterConfig] = None
    """Linear adapter config see ``collections.common.parts.LinearAdapterConfig``"""
    enable: bool = False
    """Use adapters for audio encoder"""


@dataclass
class FusionConfig:
    num_layers: Optional[int] = 4
    """"Number of layers to use in fusion"""
    num_attention_heads: Optional[int] = 4
    """Number of attention heads to use in fusion"""
    inner_size: Optional[int] = 2048
    """Fusion inner size"""


@dataclass
class AudioEncoderConfig:
    pretrained_model: str = MISSING
    """A configuration for restoring pretrained audio encoder"""
    freeze: Optional[FreezeConfig] = None
    adapter: Optional[AdapterConfig] = None
    fusion: Optional[FusionConfig] = None


@dataclass
class TokenizerConfig:
    """A structure and default values of source text tokenizer."""

    vocab_file: Optional[str] = None
    """A path to vocabulary file which is used in ``'word'``, ``'char'``, and HuggingFace tokenizers"""

    tokenizer_name: str = MISSING
    """A name of the tokenizer used for tokenization of source sequences. Possible options are ``'sentencepiece'``,
    ``'word'``, ``'char'``, HuggingFace tokenizers (e.g. ``'bert-base-uncased'``). For more options see function
    ``nemo.collections.nlp.modules.common.get_tokenizer``. The tokenizer must have properties ``cls_id``, ``pad_id``,
    ``sep_id``, ``unk_id``."""

    special_tokens: Optional[Dict[str, str]] = None
    """A dictionary with special tokens passed to constructors of ``'char'``, ``'word'``, ``'sentencepiece'``, and
    various HuggingFace tokenizers."""

    tokenizer_model: Optional[str] = None
    """A path to a tokenizer model required for ``'sentencepiece'`` tokenizer."""


@dataclass
class LanguageModelConfig:
    """
    A structure and default values of language model configuration of punctuation and capitalization model. BERT like
    HuggingFace models are supported. Provide a valid ``pretrained_model_name`` and, optionally, you may
    reinitialize model via ``config_file`` or ``config``.

    Alternatively you can initialize the language model using ``lm_checkpoint``.

    This config is a part of :class:`PunctuationCapitalizationModelConfig` config.
    """

    pretrained_model_name: str = MISSING
    """A mandatory parameter containing name of HuggingFace pretrained model. For example, ``'bert-base-uncased'``."""

    config_file: Optional[str] = None
    """A path to a file with HuggingFace model config which is used to reinitialize language model."""

    config: Optional[Dict] = None
    """A HuggingFace config which is used to reinitialize language model."""

    lm_checkpoint: Optional[str] = None
    """A path to a ``torch`` checkpoint of a language model."""


@dataclass
class HeadConfig:
    """
    A structure and default values of configuration of capitalization or punctuation model head. This config defines a
    multilayer perceptron which is applied to output of a language model. Number of units in the hidden layer is equal
    to the dimension of the language model.

    This config is a part of :class:`PunctuationCapitalizationModelConfig` config.
    """

    num_fc_layers: int = 1
    """A number of hidden layers in a multilayer perceptron."""

    fc_dropout: float = 0.1
    """A dropout used in an MLP."""

    activation: str = 'relu'
    """An activation used in hidden layers."""

    use_transformer_init: bool = True
    """Whether to initialize the weights of the classifier head with the approach that was used for language model
    initialization."""


@dataclass
class ClassLabelsConfig:
    """
    A structure and default values of a mandatory part of config which contains names of files which are saved in .nemo
    checkpoint. These files can also be used for passing label vocabulary to the model. For using them as label
    vocabularies you will need to provide path these files in parameter
    ``model.common_dataset_parameters.label_vocab_dir``. Each line in labels files
    contains 1 label. The values are sorted, ``<line number>==<label id>``, starting from ``0``. A label with ``0`` id
    must contain neutral label which must be equal to ``model.common_dataset_parameters.pad_label``.

    This config is a part of :class:`~CommonDatasetParametersConfig`.
    """

    punct_labels_file: str = MISSING
    """A name of punctuation labels file."""

    capit_labels_file: str = MISSING
    """A name of capitalization labels file."""


@dataclass
class CommonDatasetParametersConfig:
    """
    A structure and default values of common dataset parameters config which includes label and loss mask information.
    If you omit parameters ``punct_label_ids``, ``capit_label_ids``, ``label_vocab_dir``, then labels will be inferred
    from a training dataset or loaded from a checkpoint.

    Parameters ``ignore_extra_tokens`` and ``ignore_start_end`` are responsible for forming loss mask. A loss mask
    defines on which tokens loss is computed.

    This parameter is a part of config :class:`~PunctuationCapitalizationModelConfig`.
    """

    pad_label: str = MISSING
    """A mandatory parameter which should contain label used for punctuation and capitalization label padding. It
    also serves as a neutral label for both punctuation and capitalization. If any of ``punct_label_ids``,
    ``capit_label_ids`` parameters is provided, then ``pad_label`` must have ``0`` id in them. In addition, if ``label_vocab_dir``
    is provided, then ``pad_label`` must be on the first lines in files ``class_labels.punct_labels_file`` and
    ``class_labels.capit_labels_file``."""

    ignore_extra_tokens: bool = False
    """Whether to compute loss on not first tokens in words. If this parameter is ``True``, then loss mask is ``False``
    for all tokens in a word except the first."""

    ignore_start_end: bool = True
    """If ``False``, then loss is computed on [CLS] and [SEP] tokens."""

    punct_label_ids: Optional[Dict[str, int]] = None
    """A dictionary with punctuation label ids. ``pad_label`` must have ``0`` id in this dictionary. You can omit this
    parameter and pass label ids through ``class_labels.punct_labels_file`` or let the model to infer label ids from
    dataset or load them from checkpoint."""

    capit_label_ids: Optional[Dict[str, int]] = None
    """A dictionary with capitalization label ids. ``pad_label`` must have ``0`` id in this dictionary. You can omit
    this parameter and pass label ids through ``class_labels.capit_labels_file`` or let model to infer label ids from
    dataset or load them from checkpoint."""

    label_vocab_dir: Optional[str] = None
    """A path to directory which contains class labels files. See :class:`ClassLabelsConfig`. If this parameter is
    provided, then labels will be loaded from files which are located in ``label_vocab_dir`` and have names specified
    in ``model.class_labels`` configuration section. A label specified in ``pad_label`` has to be on the first lines
    of ``model.class_labels`` files."""


@dataclass
class PunctuationCapitalizationModelConfig:
    """
    A configuration of
    :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel`
    model.

    See an example of model config in
    `nemo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml
    <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`_

    This config is a part of :class:`~PunctuationCapitalizationConfig`.
    """

    class_labels: ClassLabelsConfig = field(default_factory=lambda: ClassLabelsConfig())
    """A mandatory parameter containing a dictionary with names of label id files used in .nemo checkpoints.
    These file names can also be used for passing label vocabularies to the model. If you wish to use ``class_labels``
    for passing vocabularies, please provide path to vocabulary files in
    ``model.common_dataset_parameters.label_vocab_dir`` parameter."""

    common_dataset_parameters: Optional[CommonDatasetParametersConfig] = field(
        default_factory=lambda: CommonDatasetParametersConfig()
    )
    """Label ids and loss mask information information."""

    train_ds: Optional[PunctuationCapitalizationTrainDataConfig] = None
    """A configuration for creating training dataset and data loader."""

    validation_ds: Optional[PunctuationCapitalizationEvalDataConfig] = None
    """A configuration for creating validation datasets and data loaders."""

    test_ds: Optional[PunctuationCapitalizationEvalDataConfig] = None
    """A configuration for creating test datasets and data loaders."""

    punct_head: HeadConfig = field(default_factory=lambda: HeadConfig())
    """A configuration for creating punctuation MLP head that is applied to a language model outputs."""

    capit_head: HeadConfig = field(default_factory=lambda: HeadConfig())
    """A configuration for creating capitalization MLP head that is applied to a language model outputs."""

    tokenizer: Any = field(default_factory=lambda: TokenizerConfig())
    """A configuration for source text tokenizer."""

    language_model: LanguageModelConfig = field(default_factory=lambda: LanguageModelConfig())
    """A configuration of a BERT-like language model which serves as a model body."""

    optim: Optional[Any] = None
    """A configuration of optimizer and learning rate scheduler. There is much variability in such config. For
    description see `Optimizers
    <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/core/core.html#optimizers>`_ section in
    documentation and `primer <https://github.com/NVIDIA/NeMo/blob/main/tutorials/00_NeMo_Primer.ipynb>_ tutorial."""


@dataclass
class PunctuationCapitalizationLexicalAudioModelConfig(PunctuationCapitalizationModelConfig):
    """
    A configuration of
    :class:`~nemo.collections.nlp.models.token_classification.punctuation_lexical_audio_capitalization_model.PunctuationCapitalizationLexicalAudioModel`
    model.

    See an example of model config in
    `nemo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml
    <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_lexical_audio_config.yaml>`_

    Audio encoder can be frozen during training with ``freeze_audio_encoder`` parameter.
    Adapter can be added to audio encoder with ``use_adapters`` and ``adapter_config`` parameters.
    More conformer layers can be added on top of pretrained audio encoder with ``frozen_conf_d_model``, ``frozen_conf_d_ff`` and ``frozen_conf_num_layers`` parameters.
    """

    train_ds: Optional[PunctuationCapitalizationTrainDataConfig] = None
    """A configuration for creating training dataset and data loader."""

    validation_ds: Optional[PunctuationCapitalizationEvalDataConfig] = None
    """A configuration for creating validation datasets and data loaders."""

    test_ds: Optional[PunctuationCapitalizationEvalDataConfig] = None
    """A configuration for creating test datasets and data loaders."""

    audio_encoder: Optional[AudioEncoderConfig] = None

    restore_lexical_encoder_from: Optional[str] = None
    """"Path to .nemo checkpoint to load weights from"""  # add more comments

    use_weighted_loss: Optional[bool] = False
    """If set to ``True`` CrossEntropyLoss will be weighted"""


@dataclass
class PunctuationCapitalizationConfig(NemoConfig):
    """
    A config for punctuation model training and testing.

    See an example of full config in
    `nemo/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml
    <https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/token_classification/conf/punctuation_capitalization_config.yaml>`_
    """

    pretrained_model: Optional[str] = None
    """Can be an NVIDIA's NGC cloud model or a path to a .nemo checkpoint. You can get list of possible cloud options
    by calling method
    :func:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel.list_available_models`.
    """

    name: Optional[str] = 'Punctuation_and_Capitalization'
    """A name of the model. Used for naming output directories and ``.nemo`` checkpoints."""

    do_training: bool = True
    """Whether to perform training of the model."""

    do_testing: bool = False
    """Whether ot perform testing of the model."""

    model: PunctuationCapitalizationModelConfig = field(default_factory=lambda: PunctuationCapitalizationModelConfig())
    """A configuration for the
    :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_model.PunctuationCapitalizationModel`
    model."""

    trainer: Optional[TrainerConfig] = field(default_factory=lambda: TrainerConfig())
    """Contains ``Trainer`` Lightning class constructor parameters."""

    exp_manager: Optional[ExpManagerConfig] = field(
        default_factory=lambda: ExpManagerConfig(name=None, files_to_copy=[])
    )
    """A configuration with various NeMo training options such as output directories, resuming from checkpoint,
    tensorboard and W&B logging, and so on. For possible options see :ref:`exp-manager-label`."""

    def __post_init__(self):
        if self.exp_manager is not None:
            self.exp_manager.name = self.name


@dataclass
class PunctuationCapitalizationLexicalAudioConfig(PunctuationCapitalizationConfig):
    model: PunctuationCapitalizationLexicalAudioModelConfig = field(
        default_factory=lambda: PunctuationCapitalizationLexicalAudioModelConfig()
    )


def is_legacy_model_config(model_cfg: DictConfig) -> bool:
    """
    Test if model config is old style config. Old style configs are configs which were used before
    ``common_dataset_parameters`` item was added. Old style datasets use ``dataset`` instead of
    ``common_dataset_parameters``, ``batch_size`` instead of ``tokens_in_batch``. Old style configs do not support
    tarred datasets.

    Args:
        model_cfg: model configuration

    Returns:
        whether ``model_config`` is legacy
    """
    return 'common_dataset_parameters' not in model_cfg


def legacy_model_config_to_new_model_config(model_cfg: DictConfig) -> DictConfig:
    """
    Transform old style config into
    :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_config.PunctuationCapitalizationModelConfig`.
    Old style configs are configs which were used before ``common_dataset_parameters`` item was added. Old style
    datasets use ``dataset`` instead of ``common_dataset_parameters``, ``batch_size`` instead of ``tokens_in_batch``.
    Old style configs do not support tarred datasets.

    Args:
        model_cfg: old style config

    Returns:
        model config which follows dataclass
            :class:`~nemo.collections.nlp.models.token_classification.punctuation_capitalization_config.PunctuationCapitalizationModelConfig`
    """
    train_ds = model_cfg.get('train_ds')
    validation_ds = model_cfg.get('validation_ds')
    test_ds = model_cfg.get('test_ds')
    dataset = model_cfg.dataset
    punct_head_config = model_cfg.get('punct_head', {})
    capit_head_config = model_cfg.get('capit_head', {})
    omega_conf = OmegaConf.structured(
        PunctuationCapitalizationModelConfig(
            class_labels=model_cfg.class_labels,
            common_dataset_parameters=CommonDatasetParametersConfig(
                pad_label=dataset.pad_label,
                ignore_extra_tokens=dataset.get(
                    'ignore_extra_tokens', CommonDatasetParametersConfig.ignore_extra_tokens
                ),
                ignore_start_end=dataset.get('ignore_start_end', CommonDatasetParametersConfig.ignore_start_end),
                punct_label_ids=model_cfg.punct_label_ids,
                capit_label_ids=model_cfg.capit_label_ids,
            ),
            train_ds=None
            if train_ds is None
            else legacy_data_config_to_new_data_config(train_ds, dataset, train=True),
            validation_ds=None
            if validation_ds is None
            else legacy_data_config_to_new_data_config(validation_ds, dataset, train=False),
            test_ds=None if test_ds is None else legacy_data_config_to_new_data_config(test_ds, dataset, train=False),
            punct_head=HeadConfig(
                num_fc_layers=punct_head_config.get('punct_num_fc_layers', HeadConfig.num_fc_layers),
                fc_dropout=punct_head_config.get('fc_dropout', HeadConfig.fc_dropout),
                activation=punct_head_config.get('activation', HeadConfig.activation),
                use_transformer_init=punct_head_config.get('use_transformer_init', HeadConfig.use_transformer_init),
            ),
            capit_head=HeadConfig(
                num_fc_layers=capit_head_config.get('capit_num_fc_layers', HeadConfig.num_fc_layers),
                fc_dropout=capit_head_config.get('fc_dropout', HeadConfig.fc_dropout),
                activation=capit_head_config.get('activation', HeadConfig.activation),
                use_transformer_init=capit_head_config.get('use_transformer_init', HeadConfig.use_transformer_init),
            ),
            tokenizer=model_cfg.tokenizer,
            language_model=model_cfg.language_model,
            optim=model_cfg.optim,
        )
    )
    with open_dict(omega_conf):
        retain_during_legacy_conversion = model_cfg.get('retain_during_legacy_conversion', {})
        for key in retain_during_legacy_conversion.keys():
            omega_conf[key] = retain_during_legacy_conversion[key]
    return omega_conf
