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

import os
from abc import ABC, abstractmethod
from typing import List

from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.parts.utils import asr_module_utils
from nemo.collections.common import tokenizers
from nemo.utils import logging


class ASRBPEMixin(ABC):
    """ ASR BPE Mixin class that sets up a Tokenizer via a config

    This mixin class adds the method `_setup_tokenizer(...)`, which can be used by ASR models
    which depend on subword tokenization.

    The setup_tokenizer method adds the following parameters to the class -
        -   tokenizer_cfg: The resolved config supplied to the tokenizer (with `dir` and `type` arguments).
        -   tokenizer_dir: The directory path to the tokenizer vocabulary + additional metadata.
        -   tokenizer_type: The type of the tokenizer. Currently supports `bpe` and `wpe`, as well as `agg`.
        -   vocab_path: Resolved path to the vocabulary text file.

    In addition to these variables, the method will also instantiate and preserve a tokenizer
    (subclass of TokenizerSpec) if successful, and assign it to self.tokenizer.

    The mixin also supports aggregate tokenizers, which consist of ordinary, monolingual tokenizers.
    If a conversion between a monolongual and an aggregate tokenizer (or vice versa) is detected,
    all registered artifacts will be cleaned up.
    """

    # this will be used in configs and nemo artifacts
    AGGREGATE_TOKENIZERS_DICT_PREFIX = 'langs'

    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        tokenizer_type = tokenizer_cfg.get('type')
        if tokenizer_type is None:
            raise ValueError("`tokenizer.type` cannot be None")
        elif tokenizer_type.lower() == 'agg':
            self._setup_aggregate_tokenizer(tokenizer_cfg)
        else:
            self._setup_monolingual_tokenizer(tokenizer_cfg)

    def _setup_monolingual_tokenizer(self, tokenizer_cfg: DictConfig):
        # Prevent tokenizer parallelism (unless user has explicitly set it)
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.tokenizer_cfg = OmegaConf.to_container(tokenizer_cfg, resolve=True)  # type: dict
        self.tokenizer_dir = self.tokenizer_cfg.pop('dir')  # Remove tokenizer directory
        self.tokenizer_type = self.tokenizer_cfg.pop('type').lower()  # Remove tokenizer_type

        self.hf_tokenizer_kwargs = self.tokenizer_cfg.pop("hf_kwargs", {})  # Remove HF tokenizer kwargs

        # just in case the previous tokenizer was an aggregate
        self._cleanup_aggregate_config_and_artifacts_if_needed()

        # Preserve config
        if hasattr(self, 'cfg') and 'tokenizer' in self.cfg:
            self.cfg.tokenizer.dir = self.tokenizer_dir
            self.cfg.tokenizer.type = self.tokenizer_type

            if 'hf_kwargs' in tokenizer_cfg:
                with open_dict(self.cfg.tokenizer):
                    self.cfg.tokenizer.hf_kwargs = tokenizer_cfg.get('hf_kwargs')

        if self.tokenizer_type not in ['bpe', 'wpe']:
            raise ValueError(
                "`tokenizer.type` must be either `bpe` for SentencePiece tokenizer or "
                "`wpe` for BERT based tokenizer"
            )

        if self.tokenizer_type == 'bpe':
            # This is a BPE Tokenizer
            if 'model_path' in self.tokenizer_cfg:
                model_path = self.tokenizer_cfg.get('model_path')
            else:
                model_path = os.path.join(self.tokenizer_dir, 'tokenizer.model')
            model_path = self.register_artifact('tokenizer.model_path', model_path)
            self.model_path = model_path

            if 'special_tokens' in self.tokenizer_cfg:
                special_tokens = self.tokenizer_cfg['special_tokens']

                if special_tokens is not None:
                    raise ValueError("`special_tokens` are no longer supported for SentencePiece based tokenizers.")

            # Update special tokens
            self.tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path)

            if 'vocab_path' in self.tokenizer_cfg:
                vocab_path = self.tokenizer_cfg.get('vocab_path')
            else:
                vocab_path = os.path.join(self.tokenizer_dir, 'vocab.txt')
            vocab_path = self.register_artifact('tokenizer.vocab_path', vocab_path)
            self.vocab_path = vocab_path

            try:
                if 'spe_tokenizer_vocab' in self.tokenizer_cfg:
                    spe_vocab_path = self.tokenizer_cfg.get('spe_tokenizer_vocab')
                else:
                    spe_vocab_path = os.path.join(self.tokenizer_dir, 'tokenizer.vocab')
                spe_vocab_path = self.register_artifact('tokenizer.spe_tokenizer_vocab', spe_vocab_path)
                self.spe_vocab_path = spe_vocab_path
            except FileNotFoundError:
                # fallback case for older checkpoints that did not preserve the tokenizer.vocab
                self.spe_vocab_path = None

            vocabulary = {}
            for i in range(self.tokenizer.vocab_size):
                piece = self.tokenizer.ids_to_tokens([i])
                piece = piece[0]
                vocabulary[piece] = i + 1

            # wrapper method to get vocabulary conveniently
            def get_vocab():
                return vocabulary

            # attach utility values to the tokenizer wrapper
            self.tokenizer.tokenizer.vocab_size = len(vocabulary)
            self.tokenizer.tokenizer.get_vocab = get_vocab
            self.tokenizer.tokenizer.all_special_tokens = self.tokenizer.special_token_to_id

        else:
            # This is a WPE Tokenizer
            # If path from previous registration exists, remove it
            if 'vocab_path' in self.tokenizer_cfg:
                vocab_path = self.tokenizer_cfg.get('vocab_path')
            else:
                vocab_path = os.path.join(self.tokenizer_dir, 'vocab.txt')
            vocab_path = self.register_artifact('tokenizer.vocab_path', vocab_path)
            self.vocab_path = vocab_path

            # If path from previous registration exists, remove it
            if 'vocab_path' in self.tokenizer_cfg:
                self.tokenizer_cfg.pop('vocab_path')

            self.tokenizer = tokenizers.AutoTokenizer(
                pretrained_model_name='bert-base-cased',
                vocab_file=self.vocab_path,
                mask_token=self.hf_tokenizer_kwargs.get('mask_token', None),
                bos_token=self.hf_tokenizer_kwargs.get('bos_token', None),
                eos_token=self.hf_tokenizer_kwargs.get('eos_token', None),
                pad_token=self.hf_tokenizer_kwargs.get('pad_token', None),
                sep_token=self.hf_tokenizer_kwargs.get('sep_token', None),
                cls_token=self.hf_tokenizer_kwargs.get('cls_token', None),
                unk_token=self.hf_tokenizer_kwargs.get('unk_token', None),
                use_fast=self.hf_tokenizer_kwargs.get('use_fast', False),
            )

        logging.info(
            "Tokenizer {} initialized with {} tokens".format(
                self.tokenizer.__class__.__name__, self.tokenizer.vocab_size
            )
        )

    def _setup_aggregate_tokenizer(self, tokenizer_cfg: DictConfig):
        # Prevent tokenizer parallelism (unless user has explicitly set it)
        if 'TOKENIZERS_PARALLELISM' not in os.environ:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self.tokenizer_cfg = OmegaConf.to_container(tokenizer_cfg, resolve=True)  # type: dict

        # the aggregate tokenizer does not have one tokenizer_dir but multiple ones
        self.tokenizer_dir = None

        self.tokenizer_cfg.pop('dir', None)  # Remove tokenizer directory, if any
        # Remove tokenizer_type -- obviously if we are here, the type is 'agg'
        self.tokenizer_type = self.tokenizer_cfg.pop('type').lower()

        # the aggregate tokenizer should not have these
        self.hf_tokenizer_kwargs = {}
        self.tokenizer_cfg.pop("hf_kwargs", {})  # Remove HF tokenizer kwargs, if any

        logging.info('_setup_tokenizer: detected an aggregate tokenizer')
        # need to de-register any monolingual config items if they exist
        self._cleanup_monolingual_and_aggregate_config_and_artifacts_if_needed()

        # overwrite tokenizer type
        if hasattr(self, 'cfg') and 'tokenizer' in self.cfg:
            self.cfg.tokenizer.type = self.tokenizer_type

        tokenizers_dict = {}
        # init each of the monolingual tokenizers found in the config and assemble into  AggregateTokenizer
        for lang, tokenizer_config in self.tokenizer_cfg[self.AGGREGATE_TOKENIZERS_DICT_PREFIX].items():
            (tokenizer, model_path, vocab_path, spe_vocab_path,) = self._make_tokenizer(tokenizer_config, lang)

            tokenizers_dict[lang] = tokenizer
            if hasattr(self, 'cfg'):
                with open_dict(self.cfg.tokenizer):
                    self.cfg.tokenizer[self.AGGREGATE_TOKENIZERS_DICT_PREFIX][lang]['dir'] = self.tokenizer_cfg[
                        self.AGGREGATE_TOKENIZERS_DICT_PREFIX
                    ][lang]['dir']
                    self.cfg.tokenizer[self.AGGREGATE_TOKENIZERS_DICT_PREFIX][lang]['type'] = self.tokenizer_cfg[
                        self.AGGREGATE_TOKENIZERS_DICT_PREFIX
                    ][lang]['type']

        self.tokenizer = tokenizers.AggregateTokenizer(tokenizers_dict)

    def _make_tokenizer(self, tokenizer_cfg: DictConfig, lang=None):

        tokenizer_type = tokenizer_cfg.get('type').lower()
        tokenizer_dir = tokenizer_cfg.get('dir')

        if tokenizer_type not in ['bpe', 'wpe']:
            raise ValueError(
                '`tokenizer.type` must be either `bpe` for SentencePiece tokenizer or' '`wpe` for BERT based tokenizer'
            )

        # defaults
        model_path = None
        vocab_path = None
        spe_vocab_path = None

        if tokenizer_type == 'bpe':
            # This is a BPE Tokenizer
            if 'model_path' in tokenizer_cfg:
                model_path = tokenizer_cfg.get('model_path')
            else:
                model_path = os.path.join(tokenizer_dir, 'tokenizer.model')

            model_path = self.register_artifact(
                'tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.' + lang + '.model_path', model_path
            )

            if 'special_tokens' in tokenizer_cfg:
                special_tokens = tokenizer_cfg['special_tokens']
                if special_tokens is not None:
                    raise ValueError('`special_tokens` are no longer supported for SentencePiece based tokenizers.')

            # Update special tokens
            tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path)

            if 'vocab_path' in tokenizer_cfg:
                vocab_path = tokenizer_cfg.get('vocab_path')
            else:
                vocab_path = os.path.join(tokenizer_dir, 'vocab.txt')

            vocab_path = self.register_artifact(
                'tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.' + lang + '.vocab_path', vocab_path
            )

            try:
                if 'spe_tokenizer_vocab' in tokenizer_cfg:
                    spe_vocab_path = tokenizer_cfg.get('spe_tokenizer_vocab')
                else:
                    spe_vocab_path = os.path.join(tokenizer_dir, 'tokenizer.vocab')

                spe_vocab_path = self.register_artifact(
                    'tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.' + lang + '.spe_tokenizer_vocab',
                    spe_vocab_path,
                )

            except FileNotFoundError:
                # fallback case for older checkpoints that did not preserve the tokenizer.vocab
                spe_vocab_path = None

            vocabulary = {}
            for i in range(tokenizer.vocab_size):
                piece = tokenizer.ids_to_tokens([i])
                piece = piece[0]
                vocabulary[piece] = i + 1

            # wrapper method to get vocabulary conveniently
            def get_vocab():
                return vocabulary

            # attach utility values to the tokenizer wrapper
            tokenizer.tokenizer.vocab_size = len(vocabulary)
            tokenizer.tokenizer.get_vocab = get_vocab
            tokenizer.tokenizer.all_special_tokens = tokenizer.special_token_to_id

        else:
            # This is a WPE Tokenizer
            # If path from previous registration exists, remove it
            if 'vocab_path' in tokenizer_cfg:
                vocab_path = tokenizer_cfg.get('vocab_path')
            else:
                vocab_path = os.path.join(tokenizer_dir, 'vocab.txt')

            vocab_path = self.register_artifact(
                'tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.' + lang + '.vocab_path', vocab_path
            )

            # If path from previous registration exists, remove it
            if 'vocab_path' in tokenizer_cfg:
                tokenizer_cfg.pop('vocab_path')

            hf_tokenizer_kwargs = tokenizer_cfg.get('hf_kwargs', {})
            tokenizer = tokenizers.AutoTokenizer(
                pretrained_model_name='bert-base-cased',
                vocab_file=vocab_path,
                mask_token=hf_tokenizer_kwargs.get('mask_token', None),
                bos_token=hf_tokenizer_kwargs.get('bos_token', None),
                eos_token=hf_tokenizer_kwargs.get('eos_token', None),
                pad_token=hf_tokenizer_kwargs.get('pad_token', None),
                sep_token=hf_tokenizer_kwargs.get('sep_token', None),
                cls_token=hf_tokenizer_kwargs.get('cls_token', None),
                unk_token=hf_tokenizer_kwargs.get('unk_token', None),
                use_fast=hf_tokenizer_kwargs.get('use_fast', False),
            )

        logging.info(
            'Tokenizer {} initialized with {} tokens'.format(tokenizer.__class__.__name__, tokenizer.vocab_size)
        )

        return tokenizer, model_path, vocab_path, spe_vocab_path

    def _cleanup_monolingual_and_aggregate_config_and_artifacts_if_needed(self):
        """
        Clean ups any monolingual and some aggregate config items and artifacts.
        We need to do this when we switch from a monolingual tokenizer to an aggregate one
        or go between aggregate tokenizers which could have a different number of languages
        """
        if hasattr(self, 'cfg'):
            with open_dict(self.cfg.tokenizer):
                self.cfg.tokenizer.pop('dir', None)
                self.cfg.tokenizer.pop('model_path', None)
                self.cfg.tokenizer.pop('vocab_path', None)
                self.cfg.tokenizer.pop('spe_tokenizer_vocab', None)
                self.cfg.tokenizer.pop('hf_kwargs', None)

        # need to de-register any monolingual artifacts if they exist
        if hasattr(self, 'artifacts'):
            self.artifacts.pop('tokenizer.model_path', None)
            self.artifacts.pop('tokenizer.vocab_path', None)
            self.artifacts.pop('tokenizer.spe_tokenizer_vocab', None)

            # just in case we are replacing one aggregate tokenizer with another one, we better
            # clean up the old aggregate artifacts as well
            for akey in list(self.artifacts.keys()):
                if akey.startswith('tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.'):
                    self.artifacts.pop(akey)

    def _cleanup_aggregate_config_and_artifacts_if_needed(self):
        """
        Clean ups any aggregate config items and artifacts.
        We need to do this when we switch from an aggregate tokenizer to a monolingual one
        """
        if hasattr(self, 'cfg'):
            with open_dict(self.cfg.tokenizer):
                self.cfg.tokenizer.pop(self.AGGREGATE_TOKENIZERS_DICT_PREFIX, None)

        # clean up the old aggregate artifacts as well
        if hasattr(self, 'artifacts'):
            for akey in list(self.artifacts.keys()):
                if akey.startswith('tokenizer.' + self.AGGREGATE_TOKENIZERS_DICT_PREFIX + '.'):
                    self.artifacts.pop(akey)


class ASRModuleMixin(ABC):
    """
    ASRModuleMixin is a mixin class added to ASR models in order to add methods that are specific
    to a particular instantiation of a module inside of an ASRModel.

    Each method should first check that the module is present within the subclass, and support additional
    functionality if the corresponding module is present.
    """

    def change_conv_asr_se_context_window(self, context_window: int, update_config: bool = True):
        """
        Update the context window of the SqueezeExcitation module if the provided model contains an
        `encoder` which is an instance of `ConvASREncoder`.

        Args:
            context_window:  An integer representing the number of input timeframes that will be used
                to compute the context. Each timeframe corresponds to a single window stride of the
                STFT features.

                Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
                of context to compute the Squeeze step.
            update_config: Whether to update the config or not with the new context window.
        """
        asr_module_utils.change_conv_asr_se_context_window(
            self, context_window=context_window, update_config=update_config
        )


class DiarizationMixin(ABC):
    @abstractmethod
    def diarize(self, paths2audio_files: List[str], batch_size: int = 1) -> List[str]:
        """
        Takes paths to audio files and returns speaker labels
        Args:
            paths2audio_files: paths to audio fragment to be transcribed

        Returns:
            Speaker labels
        """
        pass
