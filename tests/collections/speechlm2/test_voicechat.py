# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""NemotronVoiceChat model tests (training/offline), matching NeMo main.

Streaming inference tests live under ``nemo_inference_pipelines/`` and load
the public 11B checkpoint instead of this toy-weight model.
"""

import os
import tempfile

import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.speechlm2 import DuplexSTTDataset
from nemo.collections.speechlm2.models import NemotronVoiceChat
from nemo.collections.speechlm2.models.nemotron_voicechat import (
    _apply_nemotron_labs_voicechat_release_config_shim,
    _is_nemotron_labs_voicechat_release,
)
from nemo.collections.speechlm2.streaming.duplex_stt_inference import DuplexSTTStreamingInference

if torch.cuda.is_available():
    torch.set_default_device('cuda')


pretrained_llm = "TinyLlama/TinyLlama_v1.1"
if os.path.exists("/home/TestData/speechlm/pretrained_models"):
    pretrained_llm = "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1"

# STT sampling rate
source_sample_rate = 16000
# TTS sampling rate
target_sample_rate = 22050


_RELEASE_BY_CONTENT = {
    "_rnnt_merge_info": {},
    "model": {
        "stt": {
            "model": {
                "pretrained_llm": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
                "use_function_head": True,
                "predict_user_text": False,
            }
        }
    },
}


@pytest.mark.parametrize(
    ("model_id", "cfg", "is_release"),
    [
        ("nvidia/NVIDIA-NemotronLabs-VoiceChat-11B", None, True),
        ("/checkpoints/NVIDIA-NemotronLabs-VoiceChat-11B", None, True),
        ("/checkpoints/NVIDIA-NemotronLabs-VoiceChat-11B/", None, True),
        # Recognised by config content once downloaded under another name.
        ("/checkpoints/arbitrary-name", _RELEASE_BY_CONTENT, True),
        ("/checkpoints/some-other-voicechat-export", None, False),
    ],
)
def test_identifies_the_nemotron_labs_voicechat_release(model_id, cfg, is_release):
    """The shim must apply to the public 11B and to nothing else."""
    assert _is_nemotron_labs_voicechat_release(model_id, cfg) is is_release


def test_release_config_shim_flattens_nested_keys_without_clobbering_current_ones():
    """Values already in the flat position win; only absent ones are filled."""
    stale = {
        "data": {"source_sample_rate": 8000},
        "exp_manager": {"explicit_log_dir": "/global"},
        "model": {
            "stt": {
                "model": {
                    "pretrained_llm": "dummy",
                    "base_model_name": "backbone",
                    "embed_tokens_name": "embeddings",
                },
                "data": {"source_sample_rate": 16000},
                "exp_manager": {"explicit_log_dir": "/stt"},
            },
            "speech_generation": {
                "model": {"tts_config": {"cas_config": {"pretrained_tokenizer_name": "legacy-tokenizer"}}}
            },
        },
    }

    shimmed = _apply_nemotron_labs_voicechat_release_config_shim(stale)
    normalized = shimmed["model"]["stt"]["model"]

    assert normalized["source_sample_rate"] == 16000
    assert normalized["validation_save_path"] == "/stt"
    assert normalized["llm_attr_name"] == "model"
    assert normalized["embed_tokens_attr_name"] == "embeddings"
    assert "pretrained_tokenizer_name" not in (
        shimmed["model"]["speech_generation"]["model"]["tts_config"]["cas_config"]
    )

    already_flat = {
        "model": {
            "stt": {
                "model": {
                    "source_sample_rate": 22050,
                    "validation_save_path": "/already-flat",
                    "llm_attr_name": "already-flat",
                    "embed_tokens_attr_name": "already-flat",
                    "base_model_name": "backbone",
                    "embed_tokens_name": "embeddings",
                },
                "data": {"source_sample_rate": 16000},
                "exp_manager": {"explicit_log_dir": "/nested"},
            }
        }
    }

    preserved = _apply_nemotron_labs_voicechat_release_config_shim(already_flat)["model"]["stt"]["model"]

    assert preserved["source_sample_rate"] == 22050
    assert preserved["validation_save_path"] == "/already-flat"
    assert preserved["llm_attr_name"] == "already-flat"
    assert preserved["embed_tokens_attr_name"] == "already-flat"


def create_model(
    predict_user_text=False,
    force_use_noise_augmentation=False,
    old_noise_prob=0.0,
    old_noise_min_snr=0.0,
    old_noise_max_snr=0.0,
):
    """Helper function to create a model with configurable settings."""
    log_dir = tempfile.mkdtemp(prefix="test_nemotron_voicechat_logs-")
    stt_log_dir = os.path.join(log_dir, "duplex_stt")
    test_stt_cfg = {
        "model": {
            "pretrained_llm": pretrained_llm,
            "pretrained_weights": False,
            "audio_loss_weight": 1,
            "text_loss_weight": 3,
            "source_sample_rate": source_sample_rate,
            "validation_save_path": stt_log_dir,
            "perception": {
                "_target_": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                "preprocessor": {
                    "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                    "features": 80,
                },
                "encoder": {
                    "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                    "feat_in": 80,
                    "d_model": 512,
                    "n_heads": 8,
                    "n_layers": 1,
                    "subsampling_factor": 8,
                },
                "modality_adapter": {
                    "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                    "d_model": 512,
                },
                "output_dim": 2048,
            },
            "predict_user_text": predict_user_text,
            "force_use_noise_augmentation": force_use_noise_augmentation,
            "old_noise_prob": old_noise_prob,
            "old_noise_min_snr": old_noise_min_snr,
            "old_noise_max_snr": old_noise_max_snr,
            "optimizer": {"_target_": "torch.optim.AdamW"},
        },
        "data": {
            "source_sample_rate": 16000,
        },
        "exp_manager": {
            "explicit_log_dir": stt_log_dir,
        },
    }

    test_tts_config = {
        "model": {
            "pretrained_lm_name": pretrained_llm,
            "pretrained_ae_dir": None,
            "pretrained_tts_model": None,
            "scoring_asr": "stt_en_fastconformer_transducer_large",
            "freeze_params": [
                r"^audio_codec\..+$",  # Keep audio codec frozen as it only provides supervision for training.
                r"^embed_tokens\..+$",  # Keep embed_tokens frozen as done in eartts
            ],
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<SPECIAL_12>",
            "audio_codec_run_dtype": "float32",
            "prevent_freeze_params": [],
            "audio_save_path": "",
            "inference_guidance_scale": 0.5,
            "inference_noise_scale": 0.8,
            "inference_top_p_or_k": 0.8,
            "inference_guidance_enabled": False,
            "subword_mask_exactly_as_eartts": False,
            "context_hidden_mask_exactly_as_eartts": False,
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 4e-5,
                "betas": [0.9, 0.98],
                "weight_decay": 0,
                "foreach": True,
            },
            "lr_scheduler": {
                "_target_": "nemo.core.optim.lr_scheduler.InverseSquareRootAnnealing",
                "warmup_steps": 2500,
                "min_lr": 1e-6,
                "max_steps": 100_000_000,
            },
            "codec_config": {
                "latent_size": 512,
                "n_fft": 16,
                "hop_length": 4,
                "base_hidden_size": 384,
                "channel_mult": [1, 2, 4],
                "rates": [7, 7, 9],
                "num_blocks": 3,
                "kernel_size": 7,
                "groups": 1,
                "codebook_size": 1024,
                "num_quantizers": 31,
                "wav_to_token_ratio": 1764,
            },
            "tts_config": {
                "use_gated_fusion_for_text_audio": True,
                "disable_eos_prediction": True,
                "use_bos_eos_emb": True,
                "use_subword_flag_emb": True,
                "num_delay_speech_tokens": 2,
                "backbone_type": "gemma3_text",
                "backbone_model_class": None,
                "backbone_config_class": None,
                "backbone_config": {
                    "hidden_size": 1152,
                    "intermediate_size": 4608,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 16,
                    "head_dim": 72,
                    "attention_dropout": 0.1,
                    "use_cache": False,
                },
                "latent_size": 512,
                "codebook_size": 1024,
                "num_quantizers": 31,
                "context_hidden_size": None,
                "cas_config": {
                    "backbone_type": "t5gemma",
                    "backbone_model_class": None,
                    "backbone_config_class": None,
                    "backbone_config": {
                        "is_encoder_decoder": False,
                        "encoder": {
                            "hidden_size": 1152,
                            "intermediate_size": 4608,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 16,
                            "num_key_value_heads": 16,
                            "head_dim": 72,
                            "use_cache": False,
                            "attention_dropout": 0.1,
                        },
                    },
                },
                "mog_head_config": {
                    "intermediate_size": 4608,
                    "num_layers": 3,
                    "low_rank": 64,
                    "num_predictions": 1024,
                    "min_log_std": -4.0,
                    "eps": 1e-6,
                },
                "p_uncond": 0.1,
                "label_smoothing": 0.01,
                "max_training_rate": 0.8,
                "quantizer_dropout": 0.5,
                "random_target_masking": False,
                "exponent": 3.0,
            },
        },
        "data": {
            "add_text_bos_and_eos_in_each_turn": True,
            "add_audio_prompt": True,
            "audio_prompt_duration": 3.0,
            "frame_length": 0.08,
            "source_sample_rate": source_sample_rate,
            "target_sample_rate": target_sample_rate,
        },
        "exp_manager": {
            "explicit_log_dir": stt_log_dir,
        },
    }

    test_config = {
        "model": {
            "scoring_asr": "stt_en_fastconformer_transducer_large",
            "stt": test_stt_cfg,
            "speech_generation": test_tts_config,
        },
        "data": {
            "frame_length": 0.08,
            "source_sample_rate": source_sample_rate,
            "target_sample_rate": target_sample_rate,
            "input_roles": ["user", "User"],
            "output_roles": ["agent", "Assistant", "assistant", "Agent"],
        },
        "exp_manager": {
            "explicit_log_dir": log_dir,
        },
    }
    model = NemotronVoiceChat(test_config)
    if torch.cuda.is_available():
        model.to("cuda")
    return model


@pytest.fixture(scope="session")
def model():
    return create_model(predict_user_text=False)


@pytest.fixture(scope="session")
def dataset(model):
    return DuplexSTTDataset(
        model.stt_model.tokenizer,
        frame_length=0.08,
        source_sample_rate=source_sample_rate,
        input_roles=["user"],
        output_roles=["assistant"],
    )


@pytest.fixture(scope="session")
def training_cutset_batch():
    cut = dummy_cut(0, recording=dummy_recording(0, with_data=True, duration=1.0, sampling_rate=22050))
    cut.target_audio = dummy_recording(1, with_data=True, duration=1.0, sampling_rate=22050)
    cut.supervisions = [
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0,
            duration=0.1,
            text='hi',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.3,
            duration=0.1,
            text='hello',
            speaker="assistant",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.5,
            duration=0.1,
            text='ok',
            speaker="user",
        ),
        SupervisionSegment(
            id=cut.id,
            recording_id=cut.recording_id,
            start=0.6,
            duration=0.1,
            text='okay',
            speaker="assistant",
        ),
    ]
    return CutSet([cut])


def test_forced_turn_taking():
    pad_id = 0
    bos_id = 1
    eos_id = 2
    non_special_user_token_id = 42

    threshold = 10
    pad_window = 3

    t = 4

    class DuplexSTTStreamingInferenceForTurnTakingTest(DuplexSTTStreamingInference):
        def __init__(self):
            class Tokenizer:
                @staticmethod
                def text_to_ids(token):
                    raise AssertionError(f"Unexpected legacy token lookup: {token}")

            Tokenizer.eos = eos_id

            self.text_pad_id = pad_id
            self.text_bos_id = bos_id
            self.text_eos_id = eos_id
            self.tokenizer = Tokenizer()
            self.cfg = {
                "force_turn_taking": True,
                "force_turn_taking_threshold": threshold,
                "force_turn_taking_pad_window": pad_window,
            }
            super().__init__(model=self)

    streaming_inference = DuplexSTTStreamingInferenceForTurnTakingTest()

    def new_tokens():
        return (
            torch.full((1, 6), pad_id, dtype=torch.long),
            torch.full((1, 6), pad_id, dtype=torch.long),
        )

    # The 1a/1b/2a labels match the cases documented in _maybe_apply_forced_turn_taking.
    # 1a. Silence-based "user stopped" trigger.
    # Negative cases: the rule should not fire when the token before the pad window is user BOS or PAD.
    # A pad window immediately after BOS is startup silence, not user speech.
    #   time step:      0    1      2       3       4
    #                        <- pad_window=3 ->
    #                   t-4  t-3    t-2     t-1     t
    #   ASR:            BOS  PAD    PAD     PAD     PAD
    #   expected text:  PAD  PAD    PAD     PAD     PAD
    gen_text, gen_asr = new_tokens()
    gen_asr[0, 0] = bos_id
    streaming_inference._maybe_apply_forced_turn_taking(t, gen_text, gen_asr)
    assert gen_text[0, t] == pad_id

    # Negative case: The rule also should not fire if user BOS is inside the pad window,
    # since the window is not all PAD.
    #   time step:      0    1      2       3       4
    #                        <- pad_window=3 ->
    #                   t-4  t-3    t-2     t-1     t
    #   ASR:            PAD  BOS    PAD     PAD     PAD
    #   expected text:  PAD  PAD    PAD     PAD     PAD
    gen_text, gen_asr = new_tokens()
    gen_asr[0, 1] = bos_id
    streaming_inference._maybe_apply_forced_turn_taking(t, gen_text, gen_asr)
    assert gen_text[0, t] == pad_id

    # 1a. Silence-based "user stopped" trigger.
    # Now check the positive case: the rule should fire after user speech.
    # A pad window after a real user token means the user stopped speaking.
    #   time step:      0                    1      2       3       4
    #                                        <- pad_window=3 ->
    #                   t-4                  t-3    t-2     t-1     t
    #   ASR:            non-special token    PAD    PAD     PAD     PAD
    #   expected text:  PAD                  PAD    PAD     PAD     BOS
    gen_text, gen_asr = new_tokens()
    gen_asr[0, 0] = non_special_user_token_id
    streaming_inference._maybe_apply_forced_turn_taking(t, gen_text, gen_asr)
    assert gen_text[0, t] == bos_id

    # 1b. Explicit user EOS means the user stopped speaking, so force agent BOS.
    #   time step:      0    1    2    3    4
    #   ASR:            PAD  PAD  PAD  PAD  EOS
    #   expected text:  PAD  PAD  PAD  PAD  BOS
    gen_text, gen_asr = new_tokens()
    gen_asr[0, t] = eos_id
    streaming_inference._maybe_apply_forced_turn_taking(t, gen_text, gen_asr)
    assert gen_text[0, t] == bos_id

    # Threshold negative case: do not force another agent BOS if one already
    # appears in the recent text-channel lookback.
    #   time step:      0    1    2    3    4
    #   ASR:            PAD  PAD  PAD  PAD  EOS
    #   text before:    BOS  PAD  PAD  PAD  PAD
    #   expected text:  BOS  PAD  PAD  PAD  PAD
    gen_text, gen_asr = new_tokens()
    gen_text[0, 0] = bos_id
    gen_asr[0, t] = eos_id
    streaming_inference._maybe_apply_forced_turn_taking(t, gen_text, gen_asr)
    assert gen_text[0, t] == pad_id

    # 2a. Explicit user BOS means the user started speaking, so force agent EOS.
    #   time step:      0    1    2    3    4
    #   ASR:            PAD  PAD  PAD  PAD  BOS
    #   expected text:  PAD  PAD  PAD  PAD  EOS
    gen_text, gen_asr = new_tokens()
    gen_asr[0, t] = bos_id
    streaming_inference._maybe_apply_forced_turn_taking(t, gen_text, gen_asr)
    assert gen_text[0, t] == eos_id

    # Threshold negative case: do not force another agent EOS if one already
    # appears in the recent text-channel lookback.
    #   time step:      0    1    2    3    4
    #   ASR:            PAD  PAD  PAD  PAD  BOS
    #   text before:    EOS  PAD  PAD  PAD  PAD
    #   expected text:  EOS  PAD  PAD  PAD  PAD
    gen_text, gen_asr = new_tokens()
    gen_text[0, 0] = eos_id
    gen_asr[0, t] = bos_id
    streaming_inference._maybe_apply_forced_turn_taking(t, gen_text, gen_asr)
    assert gen_text[0, t] == pad_id


def test_e2e_validation_step_feeds_its_metrics(model, dataset, training_cutset_batch):
    """One validation step must reach the metric and result buffers.

    ``validation_step`` returns nothing, so asserting on its return value
    cannot fail. What it is actually for is the chain behind it: offline
    inference -> ASR transcription -> BLEU/ASR-BLEU update -> result logging.
    Asserting that the buffers filled pins that chain, and fails if inference
    silently produces nothing.
    """
    model.eval()
    model.on_validation_epoch_start()
    batch = dataset[training_cutset_batch]
    batch = move_data_to_device(batch, device=model.device)
    model.validation_step(
        {"dummy_val_set": batch},
        batch_idx=0,
        speaker_audio=torch.randn(1, 22050, device=model.device),
        speaker_audio_lens=torch.tensor([22050], device=model.device),
    )

    assert model.results_logger.cached_results, "validation_step logged no results"


def test_e2s_offline_generation(model):
    model.eval()
    # 16000 samples == 1 second == 12.5 frames ~= 14 frames after encoder padding
    ans = model.offline_inference(
        input_signal=torch.randn(1, 16000, device=model.device),
        input_signal_lens=torch.tensor([16000], device=model.device),
        speaker_audio=torch.randn(1, 22050, device=model.device),
        speaker_audio_lens=torch.tensor([22050], device=model.device),
    )

    assert ans.keys() == {
        'text',
        'src_text',
        'tokens_text_src',
        'tokens_text',
        'tokens_len',
        'source_audio',
        'source_audio_len',
        "audio",
        "audio_len",
    }

    assert isinstance(ans["text"], list)
    assert isinstance(ans["text"][0], str)

    gen_text = ans["tokens_text"]
    assert gen_text.shape == (1, 14)
    assert gen_text.dtype == torch.long
    assert (gen_text >= 0).all()
    assert (gen_text < model.stt_model.text_vocab_size).all()
    # 14 tokens = 24696 audio frames
    gen_audio = ans["audio"]
    assert gen_audio.shape == (1, 24696)
