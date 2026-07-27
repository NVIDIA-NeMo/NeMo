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
2. ``predates_versioned_tokenizer_fields`` is what tells the two apart, via the ``nemo_version`` stamp that
   ``ModelPT`` writes into every config it saves.
3. Whatever the resolved values are, they are written back into the config, so a model saved today
   restores to the same token-to-ID mapping no matter how the class defaults evolve later.
"""

import inspect
from unittest.mock import MagicMock, patch

import hydra
import pytest
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.tts.data.text_to_speech_dataset_lhotse import (
    persist_versioned_tokenizer_defaults,
    predates_versioned_tokenizer_fields,
    setup_tokenizers,
)
from nemo.core.classes import ModelPT

_HINDI_CHARS = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.HindiCharsTokenizer"
_ARABIC_CHARS = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.ArabicCharsTokenizer"
_IPA = "nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.IPATokenizer"


def _tokenizers_cfg(**tokenizer_fields):
    """A single-entry ``text_tokenizers`` config for the Hindi character tokenizer."""
    return OmegaConf.create({"hindi_chartokenizer": {"_target_": _HINDI_CHARS, **tokenizer_fields}})


class _TokenizerVersionModel(ModelPT):
    """Minimal stand-in for MagpieTTS that reproduces only its tokenizer/embedding sizing.

    It mirrors the two lines that matter -- resolve the tokenizer defaults according to the config's
    provenance, then size the text embedding from the resulting vocabulary -- so a ``save_to`` /
    ``restore_from`` round-trip exercises the real vocabulary-drift failure without building a codec,
    encoders, or downloading a released archive.
    """

    def __init__(self, cfg, trainer=None):
        self.tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            use_legacy_defaults=predates_versioned_tokenizer_fields(cfg),
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

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "target, field",
        [
            (_HINDI_CHARS, "charset_version"),
            (_HINDI_CHARS, "punct_version"),
            (_ARABIC_CHARS, "charset_version"),
        ],
    )
    def test_non_legacy_backfill_matches_the_tokenizer_class_default(self, target, field):
        """What gets persisted for a fresh config must equal what the tokenizer would have chosen itself.

        Otherwise bumping a version in the tokenizer signature would silently leave this backfill writing
        the old value, and every newly trained model would be pinned a version behind. Read off the real
        signature rather than a literal, so this keeps holding when the defaults move.
        """
        class_default = inspect.signature(hydra.utils.get_class(target)).parameters[field].default
        cfg = OmegaConf.create({"_target_": target})

        persist_versioned_tokenizer_defaults(cfg, use_legacy_defaults=False)

        assert cfg[field] == class_default

    @pytest.mark.unit
    def test_non_legacy_pt_br_backfill_matches_the_tokenizer_class_default(self):
        """Same invariant for the pt-BR flag, whose default lives on ``IPATokenizer``."""
        class_default = inspect.signature(hydra.utils.get_class(_IPA)).parameters["locale_specific_punct"].default
        cfg = OmegaConf.create({"_target_": _IPA, "locale": "pt-BR"})

        persist_versioned_tokenizer_defaults(cfg, use_legacy_defaults=False)

        assert cfg.locale_specific_punct == class_default

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg_fields, use_legacy_defaults, should_warn",
        [
            ({}, True, True),
            ({"_target_": _IPA, "locale": "pt-BR"}, True, True),
            ({}, False, False),
            ({"punct_version": 1, "charset_version": 1}, True, False),
        ],
        ids=["legacy-hindi", "legacy-pt-br", "fresh", "already-explicit"],
    )
    def test_legacy_backfill_is_never_silent(self, cfg_fields, use_legacy_defaults, should_warn):
        """Falling back to a pre-versioning vocabulary must say so.

        The tokenizers cannot be relied on for this: the pt-BR path emits nothing at all, and the
        Hindi/Arabic ``DeprecationWarning`` is swallowed by Python's default filters outside pytest.
        Warning only on the legacy direction keeps ordinary training runs quiet.
        """
        cfg = OmegaConf.create({"_target_": _HINDI_CHARS, **cfg_fields})

        with patch("nemo.collections.tts.data.text_to_speech_dataset_lhotse.logging.warning") as mock_warning:
            persist_versioned_tokenizer_defaults(cfg, use_legacy_defaults=use_legacy_defaults)

        assert mock_warning.called is should_warn
        if should_warn:
            # The message has to name the field, otherwise it is not actionable.
            assert any(field in mock_warning.call_args.args[0] for field in ("punct_version", "locale_specific_punct"))


class TestPredatesVersionedTokenizerFields:
    @pytest.mark.unit
    def test_serialized_config_predates_the_fields(self):
        """``ModelPT.__init__`` stamps ``nemo_version``, so carrying one means the config was serialized;
        current code always writes the versioned fields, so a serialized config lacking them is older."""
        assert predates_versioned_tokenizer_fields(OmegaConf.create({"nemo_version": "2.6.0rc0"}))

    @pytest.mark.unit
    def test_hand_authored_struct_config_does_not(self):
        """Model configs reach ``__init__`` in struct mode, where a plain attribute read would raise."""
        cfg = OmegaConf.create({"text_tokenizers": {}})
        OmegaConf.set_struct(cfg, True)

        assert not predates_versioned_tokenizer_fields(cfg)


class _StopAtTokenizerSetup(Exception):
    """Raised from the patched ``setup_tokenizers`` to end ``__init__`` once the call site has run."""


def _mock_codec():
    """An AudioCodecModel stand-in with the numeric attributes ``__init__`` reads before tokenizer setup."""
    codec = MagicMock()
    codec.sample_rate = 22050
    codec.output_sample_rate = 22050
    codec.samples_per_frame = 1024
    codec.num_codebooks = 8
    codec.codebook_size = 1000
    return codec


class TestProductionCallSites:
    """The wiring in the real model constructors, which is the whole fix.

    Without this, mutating either call site (deleting the kwarg, or negating it) leaves the entire TTS
    unit suite green while reintroducing the v2602 restore failure -- the tests below are the only thing
    that fails on such a mutation, because every other test drives ``setup_tokenizers`` directly.

    Each constructor is stopped at the tokenizer-setup call rather than run to completion, so no codec,
    encoders, or downloads are needed.
    """

    @staticmethod
    def _captured_kwargs(model_cls, module_path, cfg_extra, codec_attr):
        cfg = OmegaConf.create(
            {
                "codecmodel_path": "nvidia/fake-codec",
                "text_tokenizers": _tokenizers_cfg(),
                **cfg_extra,
            }
        )
        captured = {}
        with (
            patch(f"{module_path}.AudioCodecModel") as mock_codec,
            patch(f"{module_path}.setup_tokenizers") as mock_setup,
        ):
            getattr(mock_codec, codec_attr).return_value = _mock_codec()

            def _record(*_args, **kwargs):
                captured.update(kwargs)
                raise _StopAtTokenizerSetup()

            mock_setup.side_effect = _record
            with pytest.raises(_StopAtTokenizerSetup):
                model_cls(cfg=cfg)
        return captured

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg_extra, expected", [({}, False), ({"nemo_version": "2.6.0rc0"}, True)], ids=["fresh", "serialized"]
    )
    def test_magpietts_passes_config_provenance(self, cfg_extra, expected):
        from nemo.collections.tts.models.magpietts import MagpieTTSModel

        captured = self._captured_kwargs(
            MagpieTTSModel, "nemo.collections.tts.models.magpietts", cfg_extra, "from_pretrained"
        )

        assert captured["use_legacy_defaults"] is expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "cfg_extra, expected", [({}, False), ({"nemo_version": "2.6.0rc0"}, True)], ids=["fresh", "serialized"]
    )
    def test_easy_magpietts_passes_config_provenance(self, cfg_extra, expected):
        from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel

        captured = self._captured_kwargs(
            EasyMagpieTTSInferenceModel,
            "nemo.collections.tts.models.easy_magpietts_inference",
            cfg_extra,
            "restore_from",
        )

        assert captured["use_legacy_defaults"] is expected


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
