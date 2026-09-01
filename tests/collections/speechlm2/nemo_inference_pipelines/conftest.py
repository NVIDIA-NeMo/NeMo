# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared fixtures for streaming VoiceChat inference tests.

Most GPU tests build a tiny random-weight checkpoint (fast enough for CI).
A few integration tests load ``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B``,
downloading it into the Hugging Face cache if needed.

Toy-weight training/offline tests live in
``tests/collections/speechlm2/test_voicechat.py`` and do not use this
conftest.
"""

from __future__ import annotations

import gc
import json
import logging
import os

# nemotron_voicechat_pipeline_{parity,nocrash} tests set
# torch.use_deterministic_algorithms(True), which requires CuBLAS to have a
# deterministic workspace.  CuBLAS reads this env var only once — at
# initialization (first CUDA matmul in the process) — so it must be set here,
# before any fixture or test triggers CUDA work.  The setting is harmless for
# non-deterministic tests: it only reserves 32 KB of extra GPU workspace and
# has no effect unless deterministic mode is active.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from contextlib import ExitStack

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from nemo.collections.audio.parts.utils.transforms import resample
from nemo.collections.speechlm2.inference.factory.s2s_pipeline_builder import S2SPipelineBuilder
from nemo.collections.speechlm2.inference.model_wrappers.engine_selection import inference_precision_from_cfg
from nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline import StreamingS2SPipeline
from nemo.collections.speechlm2.models import NemotronVoiceChat
from nemo.collections.speechlm2.models.duplex_ear_tts import load_audio_librosa

_pretrained_llm = "TinyLlama/TinyLlama_v1.1"
if os.path.exists("/home/TestData/speechlm/pretrained_models"):
    _pretrained_llm = "/home/TestData/speechlm/pretrained_models/TinyLlama--TinyLlama_v1.1"

# The config the example launcher ships, so tests exercise what users run.
CONF_YAML = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../../examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml",
    )
)
_FORCE_ALIGN_AUDIO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "test_data", "force_align_test.mp3")
)
HF_VOICECHAT_11B = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
# Speaker name registered in the public 11B. The tiny checkpoint registers a
# random latent under the same name so every test passes the same
# ``s2s.speaker_name``.
DEFAULT_SPEAKER_NAME = "Aria"


def _merge_pipeline_cfg(model_path: str, audio_path: str, output_dir: str, *overrides: dict):
    """Shipped config with *overrides* merged on top, in order.

    OmegaConf merges nested dicts recursively, so ``{"s2s": {"top_p": 0.9}}``
    overrides only that key.
    """
    cfg = OmegaConf.merge(
        OmegaConf.load(CONF_YAML),
        {"audio_file": audio_path, "output_dir": output_dir, "s2s": {"model_path": model_path}},
    )
    for override in overrides:
        if override:
            cfg = OmegaConf.merge(cfg, override)
    return cfg


_EMPTY_CACHE_AFTER_GIB = 16


def _reclaim_gpu_after_large_load() -> None:
    """Return GPU memory to the driver after a large native load.

    ``pipeline.shutdown`` only tears down the vLLM runtime. Native weights stay
    on the wrapper until the pipeline is unreachable; PyTorch then keeps the
    blocks in this process. vLLM engine cores are child processes and treat
    that as used memory, so a native 11B test (~24 GiB) followed by vLLM/vLLM
    OOMs on an 80 GiB card.

    Tiny-model tests are a few GiB and rebuild from the same size, so the
    caching allocator is left warm. Both ``gc.collect`` and ``empty_cache``
    run only when reserved memory is still large — the 11B case, not the
    nocrash sweep. Call after the pipeline has gone out of scope; collecting
    while it is still a live local cannot free the weights.
    """
    if not torch.cuda.is_available():
        return
    reserved_gib = torch.cuda.memory_reserved() / 1024**3
    if reserved_gib >= _EMPTY_CACHE_AFTER_GIB:
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(
            "GPU reclaim: %.1f GiB reserved -> %.1f GiB",
            reserved_gib,
            torch.cuda.memory_reserved() / 1024**3,
        )


@pytest.fixture
def build_pipeline():
    """Factory fixture that builds a pipeline scoped to the test.

    Holds both scopes the production callers hold, on an ``ExitStack`` so they
    last to the end of the test: the precision globals (without which one
    ``deterministic=true`` test would leave the whole session in deterministic
    mode with seeded RNGs and the fast attention kernels off) and
    ``pipeline.shutdown``, which releases any vLLM runtime. After the stack
    unwinds the pipeline is unreachable, so a large leftover CUDA reservation
    can be returned to the driver (see ``_reclaim_gpu_after_large_load``).

    A fixture rather than an import because pytest loads these test modules as
    top-level modules with no parent package, so they cannot import from
    ``conftest`` directly.
    """
    with ExitStack() as stack:

        def build(model_path: str, audio_path: str, output_dir: str, *overrides: dict) -> StreamingS2SPipeline:
            cfg = _merge_pipeline_cfg(model_path, audio_path, output_dir, *overrides)
            stack.enter_context(inference_precision_from_cfg(cfg.s2s))
            pipeline = S2SPipelineBuilder.build_pipeline(cfg)
            stack.callback(pipeline.shutdown)
            return pipeline

        yield build
    _reclaim_gpu_after_large_load()


def _tiny_voicechat_config(
    *,
    log_dir: str,
    predict_user_text: bool = True,
    streaming_encoder: bool = False,
    use_function_head: bool = False,
) -> dict:
    """Return a minimal NemotronVoiceChat config with random weights.

    Args:
        log_dir: Base directory for the exp_manager and validation outputs.
            Pass a per-test temporary directory so runs cannot collide.
        predict_user_text: Enable ASR head for user text prediction.
        streaming_encoder: When True, configure the conformer encoder for
            cache-aware streaming (causal convolutions, chunked_limited
            attention) matching the real checkpoint.  When False, use
            default (non-causal) settings suitable for offline tests.
    """
    duplex_stt_log_dir = os.path.join(log_dir, "duplex_stt")
    parity_log_dir = os.path.join(log_dir, "parity")
    encoder_cfg: dict = {
        "_target_": "nemo.collections.asr.modules.ConformerEncoder",
        "feat_in": 80,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 1,
        "subsampling_factor": 8,
    }
    if streaming_encoder:
        encoder_cfg.update(
            {
                "subsampling": "dw_striding",
                "causal_downsampling": True,
                "att_context_size": [70, 0],
                "att_context_style": "chunked_limited",
                "conv_kernel_size": 9,
                "conv_context_size": "causal",
            }
        )

    return {
        "model": {
            "scoring_asr": "stt_en_fastconformer_transducer_large",
            "stt": {
                "model": {
                    "pretrained_llm": _pretrained_llm,
                    "pretrained_weights": False,
                    "predict_user_text": predict_user_text,
                    "use_function_head": use_function_head,
                    "audio_loss_weight": 1,
                    "text_loss_weight": 3,
                    "duplex_function_channel_weight": 2.0,
                    "source_sample_rate": 16000,
                    "validation_save_path": duplex_stt_log_dir,
                    "perception": {
                        "_target_": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                        "preprocessor": {
                            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                            "features": 80,
                        },
                        "encoder": encoder_cfg,
                        "modality_adapter": {
                            "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                            "d_model": 512,
                        },
                        "output_dim": 2048,
                    },
                    "optimizer": {"_target_": "torch.optim.AdamW"},
                },
                "data": {"source_sample_rate": 16000},
                "exp_manager": {"explicit_log_dir": duplex_stt_log_dir},
            },
            "speech_generation": {
                "model": {
                    "pretrained_lm_name": _pretrained_llm,
                    "pretrained_ae_dir": None,
                    "pretrained_tts_model": None,
                    "scoring_asr": "stt_en_fastconformer_transducer_large",
                    "freeze_params": [r"^audio_codec\..+$", r"^embed_tokens\..+$"],
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
                        # Required to construct audio_prompt_projection_W and
                        # register the fixture's speaker latent.
                        "use_audio_prompt_frozen_projection": True,
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
                    "source_sample_rate": 16000,
                    "target_sample_rate": 22050,
                },
                "exp_manager": {"explicit_log_dir": duplex_stt_log_dir},
            },
        },
        "data": {
            "frame_length": 0.08,
            "source_sample_rate": 16000,
            "target_sample_rate": 22050,
            "input_roles": ["user", "User"],
            "output_roles": ["agent", "Assistant", "assistant", "Agent"],
        },
        "exp_manager": {"explicit_log_dir": parity_log_dir},
    }


def _build_tiny_model_artifacts(base, *, predict_user_text: bool, use_function_head: bool):
    if not torch.cuda.is_available():
        pytest.skip("building the tiny checkpoint requires a GPU")

    audio_path = str(base / "test_audio.wav")
    sf.write(audio_path, np.random.RandomState(42).randn(3 * 16000).astype(np.float32), 16000)

    speaker_ref_path = str(base / "speaker_ref.wav")
    sf.write(speaker_ref_path, np.random.RandomState(99).randn(22050).astype(np.float32), 22050)

    cfg = _tiny_voicechat_config(
        log_dir=str(base / "logs"),
        predict_user_text=predict_user_text,
        streaming_encoder=True,
        use_function_head=use_function_head,
    )
    model = NemotronVoiceChat(cfg)
    model.to("cuda")
    model.eval()

    speaker_audio, sr = load_audio_librosa(speaker_ref_path)
    speaker_audio = resample(speaker_audio, sr, model.tts_model.target_sample_rate).to(model.device)
    speaker_audio_lens = torch.tensor([speaker_audio.size(1)]).long().repeat(speaker_audio.size(0)).to(model.device)
    with torch.no_grad():
        model.tts_model.set_audio_prompt_lantent(
            speaker_audio,
            speaker_audio_lens,
            system_prompt=None,
            batch_size=1,
            name=DEFAULT_SPEAKER_NAME,
        )

    model_dir = str(base / "model")
    model.save_pretrained(model_dir)

    # save_pretrained writes the tokenizer to llm_artifacts/, but config.json
    # still references the HF hub name (e.g. "TinyLlama/TinyLlama_v1.1").
    # Save the LLM model config alongside the tokenizer so llm_artifacts/
    # is a complete local model reference, then rewrite config.json to point
    # at it.  This avoids HuggingFace network requests on every from_pretrained.
    llm_artifacts = os.path.join(model_dir, "llm_artifacts")
    model.stt_model.llm.config.save_pretrained(llm_artifacts)
    cfg["model"]["stt"]["model"]["pretrained_llm"] = llm_artifacts
    cfg["model"]["speech_generation"]["model"]["pretrained_lm_name"] = llm_artifacts
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f)

    del model
    torch.cuda.empty_cache()

    return model_dir, audio_path, DEFAULT_SPEAKER_NAME


@pytest.fixture(scope="session")
def tiny_model_artifacts(tmp_path_factory):
    """Build the existing ASR-head tiny checkpoint used by pipeline tests."""
    return _build_tiny_model_artifacts(
        tmp_path_factory.mktemp("tiny_model"),
        predict_user_text=True,
        use_function_head=False,
    )


@pytest.fixture(scope="session")
def tiny_function_model_artifacts(tmp_path_factory):
    """Build a public-VoiceChat-style checkpoint: function head, no ASR head."""
    return _build_tiny_model_artifacts(
        tmp_path_factory.mktemp("tiny_function_model"),
        predict_user_text=False,
        use_function_head=True,
    )


@pytest.fixture(scope="session")
def hf_voicechat_11b():
    """``nvidia/NVIDIA-NemotronLabs-VoiceChat-11B``, downloaded into ``HF_HOME`` if needed."""
    from huggingface_hub import snapshot_download

    return snapshot_download(HF_VOICECHAT_11B)


@pytest.fixture(scope="session")
def voicechat_audio_path():
    return _FORCE_ALIGN_AUDIO


@pytest.fixture(scope="session")
def voicechat_speaker_name():
    return DEFAULT_SPEAKER_NAME


@pytest.fixture(scope="session")
def real_vllm_omni_wrapper(tmp_path_factory, hf_voicechat_11b):
    """Convert the public 11B snapshot for vLLM-Omni.

    ``NEMO_VLLM_WRAPPER_DIR`` is an optional prebuilt wrapper directory.
    ``build_wrapper_checkpoint`` reuses it when complete. Otherwise one is
    built under tmp.
    """
    pytest.importorskip("vllm_omni")
    if not torch.cuda.is_available():
        pytest.skip("converting the vLLM-Omni wrapper requires a GPU")

    from nemo.collections.speechlm2.inference.vllm_omni.checkpoint import build_wrapper_checkpoint

    wrapper_dir = os.environ.get("NEMO_VLLM_WRAPPER_DIR") or str(tmp_path_factory.mktemp("vllm_omni_wrapper"))
    build_wrapper_checkpoint(hf_voicechat_11b, wrapper_dir)
    return hf_voicechat_11b, wrapper_dir
