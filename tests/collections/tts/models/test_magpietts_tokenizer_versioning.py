# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""
Unit tests for the versioned tokenizer defaults that MagpieTTS configs carry.

``HindiCharsTokenizer`` / ``ArabicCharsTokenizer`` (``charset_version``, ``punct_version``) and the
pt-BR ``IPATokenizer`` (``locale_specific_punct``) each gained a second character/punctuation set after
models had already been trained and released. A config that omits those fields is therefore ambiguous:
it is either an archive that predates them (and whose vocabulary was built with the v1 values) or a
fresh training config that should get today's defaults. These tests pin both readings:

1. ``setup_tokenizers`` uses the current defaults for fresh configs, and the pre-versioning values only
   when ``use_legacy_defaults=True`` -- so new training is never silently downgraded to v1.
2. ``is_restored_model_config`` is what tells the two apart, via the ``nemo_version`` stamp that
   ``ModelPT`` writes into every config it saves.
3. Whatever the resolved values are, they are written back into the config, so a model saved today
   restores to the same token-to-ID mapping no matter how the class defaults evolve later.
"""

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.tts.data.text_to_speech_dataset_lhotse import (
    is_restored_model_config,
    persist_versioned_tokenizer_defaults,
    setup_tokenizers,
)
from nemo.core.classes import ModelPT
from nemo.utils.app_state import AppState

_HINDI_CHARS = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.HindiCharsTokenizer"
_ARABIC_CHARS = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.ArabicCharsTokenizer"
_IPA = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer"


def _tokenizers_cfg(**tokenizer_fields):
    """A single-entry ``text_tokenizers`` config for the Hindi character tokenizer."""
    return OmegaConf.create({"hindi_chartokenizer": {"_target_": _HINDI_CHARS, **tokenizer_fields}})


@pytest.fixture(autouse=True)
def _clear_restore_flag():
    """Keep ``AppState``'s process-global restore flag from leaking between tests."""
    AppState().is_model_being_restored = False
    yield
    AppState().is_model_being_restored = False


class _TokenizerVersionModel(ModelPT):
    """Minimal stand-in for MagpieTTS that reproduces only its tokenizer/embedding sizing.

    It mirrors the two lines that matter -- resolve the tokenizer defaults according to whether the
    config came from a checkpoint, then size the text embedding from the resulting vocabulary -- so a
    ``save_to`` / ``restore_from`` round-trip exercises the real vocabulary-drift failure without
    building a codec, encoders, or downloading a released archive.
    """

    def __init__(self, cfg, trainer=None):
        restored_model_config = is_restored_model_config(cfg)
        self.tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            use_legacy_defaults=restored_model_config,
        )
        super().__init__(cfg=cfg, trainer=trainer)
        self.text_embedding = torch.nn.Embedding(len(self.tokenizer.tokens) + 2, 4)

    def setup_training_data(self, train_data_config):
        self._train_dl = None

    def setup_validation_data(self, val_data_config):
        self._validation_dl = None

    def setup_test_data(self, test_data_config):
        self._test_dl = None

    @classmethod
    def list_available_models(cls):
        return []


class TestSetupTokenizersDefaults:
    """Covers that the flag reaches the tokenizer config. What each version *means* for the resulting
    vocabulary is already pinned by ``test_tts_tokenizers.py`` at the tokenizer-class level."""

    @pytest.mark.unit
    @pytest.mark.parametrize("use_legacy_defaults, expected", [(False, 2), (True, 1)])
    def test_missing_versions_follow_the_legacy_flag(self, use_legacy_defaults, expected):
        """Fresh training must get the current charsets; restoring an older archive must get v1."""
        cfg = _tokenizers_cfg()

        setup_tokenizers(cfg, use_legacy_defaults=use_legacy_defaults)

        assert cfg.hindi_chartokenizer.charset_version == expected
        assert cfg.hindi_chartokenizer.punct_version == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("use_legacy_defaults", [False, True])
    def test_explicit_values_are_never_overridden(self, use_legacy_defaults):
        """An explicitly configured version wins over both defaults -- that is what pins v2607."""
        cfg = _tokenizers_cfg(charset_version=1, punct_version=2)

        setup_tokenizers(cfg, use_legacy_defaults=use_legacy_defaults)

        assert cfg.hindi_chartokenizer.charset_version == 1
        assert cfg.hindi_chartokenizer.punct_version == 2


class TestPersistVersionedTokenizerDefaults:
    @pytest.mark.unit
    @pytest.mark.parametrize("use_legacy_defaults, expected", [(False, 2), (True, 1)])
    def test_arabic_charset_version(self, use_legacy_defaults, expected):
        cfg = OmegaConf.create({"_target_": _ARABIC_CHARS})

        persist_versioned_tokenizer_defaults(cfg, use_legacy_defaults=use_legacy_defaults)

        assert cfg.charset_version == expected

    @pytest.mark.unit
    @pytest.mark.parametrize("use_legacy_defaults, expected", [(False, True), (True, False)])
    def test_pt_br_locale_specific_punct(self, use_legacy_defaults, expected):
        cfg = OmegaConf.create({"_target_": _IPA, "locale": "pt-BR"})

        persist_versioned_tokenizer_defaults(cfg, use_legacy_defaults=use_legacy_defaults)

        assert cfg.locale_specific_punct is expected

    @pytest.mark.unit
    def test_non_default_punct_list_suppresses_pt_br_backfill(self):
        """An explicit punctuation list already fixes the vocabulary; adding the flag would fight it."""
        cfg = OmegaConf.create({"_target_": _IPA, "locale": "pt-BR", "non_default_punct_list": [".", ","]})

        persist_versioned_tokenizer_defaults(cfg, use_legacy_defaults=True)

        assert "locale_specific_punct" not in cfg

    @pytest.mark.unit
    def test_other_ipa_locales_are_untouched(self):
        """Only pt-BR's punctuation set diverged from DEFAULT_PUNCTUATION, so only it is backfilled."""
        cfg = OmegaConf.create({"_target_": _IPA, "locale": "es-ES"})

        persist_versioned_tokenizer_defaults(cfg, use_legacy_defaults=True)

        assert "locale_specific_punct" not in cfg

    @pytest.mark.unit
    def test_config_without_target_is_ignored(self):
        cfg = OmegaConf.create({"pretrained_model": "google-t5/t5-small"})

        persist_versioned_tokenizer_defaults(cfg, use_legacy_defaults=True)

        assert cfg == OmegaConf.create({"pretrained_model": "google-t5/t5-small"})


class TestIsRestoredModelConfig:
    @pytest.mark.unit
    def test_nemo_version_marks_a_saved_config(self):
        """``ModelPT.__init__`` stamps ``nemo_version``, so its presence means the config was saved."""
        assert is_restored_model_config(OmegaConf.create({"nemo_version": "2.6.0rc0"}))

    @pytest.mark.unit
    def test_restore_in_flight_marks_a_saved_config(self):
        """Covers archives whose ``nemo_version`` was stripped by an override config."""
        AppState().is_model_being_restored = True

        assert is_restored_model_config(OmegaConf.create({"text_tokenizers": {}}))

    @pytest.mark.unit
    def test_fresh_struct_config_is_not_restored(self):
        """Model configs reach ``__init__`` in struct mode, where a plain attribute read would raise."""
        cfg = OmegaConf.create({"text_tokenizers": {}})
        OmegaConf.set_struct(cfg, True)

        assert not is_restored_model_config(cfg)


class TestNemoRoundTrip:
    """End-to-end ``save_to``/``restore_from`` coverage of the vocabulary-drift failure."""

    @staticmethod
    def _save_legacy_archive(tmp_path):
        """Write a .nemo that looks like a pre-versioning release (v2512/v2602).

        Such archives were trained with the v1 charset/punctuation but their configs name neither, so
        the versioned fields are stripped back out after the model is built.
        """
        model = _TokenizerVersionModel(OmegaConf.create({"text_tokenizers": _tokenizers_cfg()}))
        model.text_embedding = torch.nn.Embedding(
            len(setup_tokenizers(_tokenizers_cfg(), use_legacy_defaults=True).tokens) + 2, 4
        )
        with open_dict(model.cfg):
            del model.cfg.text_tokenizers.hindi_chartokenizer.charset_version
            del model.cfg.text_tokenizers.hindi_chartokenizer.punct_version
        path = str(tmp_path / "legacy.nemo")
        model.save_to(path)
        return path, model.text_embedding.num_embeddings

    @pytest.mark.unit
    def test_archive_without_version_fields_restores_with_v1_vocab(self, tmp_path):
        """Regression: a released checkpoint that predates the fields must not pick up the v2 charsets.

        Picking them up rebuilds a vocabulary of a different size than the checkpoint was trained with
        (v2 collapses case, so it is the *smaller* of the two), and ``restore_from`` then dies on a
        text_embedding size mismatch -- exactly how v2602 broke.
        """
        path, expected_num_embeddings = self._save_legacy_archive(tmp_path)

        restored = _TokenizerVersionModel.restore_from(path, map_location="cpu")

        assert restored.text_embedding.num_embeddings == len(restored.tokenizer.tokens) + 2
        assert restored.text_embedding.num_embeddings == expected_num_embeddings

    @pytest.mark.unit
    def test_newly_trained_model_round_trips_on_current_defaults(self, tmp_path):
        """A model trained today keeps its v2 vocabulary through a save/restore cycle.

        The restore takes the legacy branch, so this pins that the versions persisted at save time --
        not the branch -- are what decides the vocabulary.
        """
        model = _TokenizerVersionModel(OmegaConf.create({"text_tokenizers": _tokenizers_cfg()}))
        assert model.cfg.text_tokenizers.hindi_chartokenizer.charset_version == 2
        path = str(tmp_path / "current.nemo")
        model.save_to(path)

        restored = _TokenizerVersionModel.restore_from(path, map_location="cpu")

        assert restored.cfg.text_tokenizers.hindi_chartokenizer.charset_version == 2
        assert restored.cfg.text_tokenizers.hindi_chartokenizer.punct_version == 2
        assert restored.text_embedding.num_embeddings == model.text_embedding.num_embeddings
