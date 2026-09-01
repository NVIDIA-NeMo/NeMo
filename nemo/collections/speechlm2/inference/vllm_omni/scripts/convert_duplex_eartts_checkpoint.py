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

"""Convert the DuplexEARTTS component of a NemotronVoiceChat checkpoint to vLLM format.

The converter expects the HuggingFace-format NemotronVoiceChat checkpoint layout:
``config.json`` contains ``model.speech_generation`` and ``model.stt`` entries,
and ``model.safetensors`` contains nested ``tts_model.tts_model.*`` weights.

The character-aware subword encoder (``embed_subword``) is collapsed into a
single pre-computed lookup table mapping ``token_id -> hidden_size`` embedding.
The character/transformer weights of that encoder are dropped, since the lookup
fully captures their deterministic per-token output (including the additive
subword-flag and BOS/EOS contributions).
"""

import argparse
import json
import os
import tqdm

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from nemo.collections.speechlm2.models.duplex_ear_tts import DuplexEARTTS
from nemo.utils import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--precompute-batch-size",
        type=int,
        default=256,
        help="Batch size for pre-computing per-token embeddings.",
    )
    return parser.parse_args()


def _precompute_subword_embeddings(model: DuplexEARTTS, batch_size: int) -> torch.Tensor:
    """Run ``embed_subword`` over the entire vocabulary to bake out a lookup table.

    The character-aware subword encoder is fully deterministic per token id
    (it takes ``subword_ids`` only and adds id-conditioned flag/BOS-EOS
    embeddings). Running it once per id and storing the result lets vLLM
    replace the whole encoder with a single ``nn.Embedding`` lookup.

    Returns:
        Tensor of shape ``[vocab_size, hidden_size]`` matching the dtype of the
        encoder's parameters.
    """
    embed_subword = model.tts_model.embed_subword
    embed_subword.eval()

    # Run the precomputation on GPU when available; the encoder is small but
    # the vocabulary loop is long, so this is a meaningful speedup. Only the
    # subword encoder is moved (not the full model) to keep peak memory low.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    embed_subword.to(device)
    logging.info(f"Precomputing subword embeddings on {device}")

    dtype = next(embed_subword.parameters()).dtype

    subword_ids_map = embed_subword.subword_id_to_char_ids
    vocab_size = max(int(k) for k in subword_ids_map.keys()) + 1
    hidden_size = embed_subword.proj_embedding.out_features

    table = torch.zeros((vocab_size, hidden_size), dtype=dtype, device=device)

    with torch.no_grad():
        for start in tqdm.tqdm(range(0, vocab_size, batch_size), desc="Precomputing subword embeddings"):
            end = min(start + batch_size, vocab_size)
            ids = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)
            mask = torch.ones_like(ids, dtype=torch.bool)
            embeds = embed_subword(ids, mask)
            table[start:end] = embeds.squeeze(0).to(dtype)

    return table.cpu()


def convert_to_vllm_format(outdir: str, config: str, model_path: str, precompute_batch_size: int = 256) -> None:
    """Convert DuplexEARTTS weights from a NemotronVoiceChat HF checkpoint for vLLM.

    Args:
        outdir: Directory where the vLLM-compatible checkpoint will be written.
        config: Path to the NemotronVoiceChat ``config.json`` file.
        model_path: Path to the NemotronVoiceChat ``model.safetensors`` file.
        precompute_batch_size: Batch size used while running the subword encoder
            once per token id to construct the lookup table.
    """
    os.makedirs(outdir, exist_ok=True)

    with open(config, "r") as f:
        full_config = json.load(f)

    # This converter builds a real DuplexEARTTS from the checkpoint's own
    # config, so it needs the same config normalization that
    # NemotronVoiceChat.from_pretrained applies to the published VoiceChat
    # release. Both helpers are no-ops for any other checkpoint.
    from nemo.collections.speechlm2.models.nemotron_voicechat import (
        _apply_nemotron_labs_voicechat_release_config_shim,
        _is_nemotron_labs_voicechat_release,
    )

    if _is_nemotron_labs_voicechat_release(os.path.dirname(os.path.abspath(config)), full_config):
        logging.info("Applying NemotronLabs VoiceChat release shim before building DuplexEARTTS")
        _apply_nemotron_labs_voicechat_release_config_shim(full_config)

    config_dict = full_config["model"]["speech_generation"]
    cfg = DictConfig(config_dict)
    # Inference-only overrides, so the model built here matches the settings
    # DuplexEARTTS.generate() uses when the embeddings are precomputed below.
    cfg.model.tts_config.use_unshifthed_prompt = True
    cfg.data.add_audio_prompt_after_description = True
    cfg.model.tts_config.use_unshifthed_prompt = True
    cfg.model.subword_mask_exactly_as_eartts = False
    cfg.model.context_hidden_mask_exactly_as_eartts = False
    cfg.model.tts_config.disable_eos_prediction = True
    cfg.model.inference_force_speech_silence_on_eos = True
    cfg.model.use_word_sep_tokenizer = False
    cfg.model.num_delay_speech_tokens = 0
    cfg.data.source_sample_rate = 22050
    cfg.data.target_sample_rate = 22050
    cfg.model.pretrained_model = None

    model = DuplexEARTTS(OmegaConf.to_container(cfg, resolve=True)).eval()
    hidden_size = cfg.model.tts_config.backbone_config.hidden_size

    # Load the HuggingFace-format NemotronVoiceChat safetensors checkpoint.
    raw_weights = load_file(model_path)
    # The checkpoint is wrapped by an outer module (NemotronVoiceChat) whose TTS
    # attribute is also called ``tts_model``. Strip a single ``tts_model.`` prefix
    # to land in the DuplexEARTTS state-dict namespace.
    weights = {k[len("tts_model.") :]: v for k, v in raw_weights.items() if k.startswith("tts_model.")}

    # Load the real weights into the DuplexEARTTS model so that running
    # ``embed_subword`` produces the trained per-token outputs (otherwise we
    # would just bake out random init values).
    missing, unexpected = model.load_state_dict(weights, strict=False)
    # Some keys (e.g. the unused language model / audio codec heads) may be
    # missing or unexpected; that is fine for the embedding sub-tree we care
    # about. Surface the diagnostics anyway.
    if missing:
        logging.info(f"load_state_dict missing keys (expected for unused submodules): {len(missing)}")
    if unexpected:
        logging.info(f"load_state_dict unexpected keys: {len(unexpected)}")

    # Pre-compute the subword lookup table once per token id. This collapses
    # the entire char-aware encoder (char embedding + transformer + projection
    # + subword/BOS-EOS flag adds) into a single ``nn.Embedding`` lookup that
    # vLLM can use directly.
    precomputed_subword_emb = _precompute_subword_embeddings(model, precompute_batch_size)
    vocab_size, _ = precomputed_subword_emb.shape

    # Codec silence tokens are produced once at training time by encoding a
    # zero waveform with the audio codec and picking the most common frame.
    # Bake the resulting per-codebook ids into the vLLM checkpoint so the
    # runtime does not need to load / run the codec to know what "silence"
    # looks like (used e.g. when forcing silence on EOS).
    codec_silence_tokens = model.codec_silence_tokens.detach().clone().cpu().to(torch.int32)

    # Strip the ``tts_model.`` prefix so the renaming below operates on
    # RVQEARTTSModel state-dict keys.
    weights = {k[len("tts_model.") :]: v for k, v in weights.items() if k.startswith("tts_model.")}

    # duplicate weights for rvq embeddings and embed code
    rvq_embs_weight = weights["rvq_embs"].clone()  # 31 x codebook_size x latent_size
    rvq_embs_weight_pad = torch.nn.functional.pad(
        rvq_embs_weight, [0, 0, 0, 1]
    )  # 31 x (codebook_size + 1) x latent_size
    embed_code_weight = weights["embed_code.weight"].clone()  # latent_size x hidden_size

    # ======================
    # embedding module weights
    bos_emb = weights["bos_emb"]
    null_emb = weights["null_emb"]

    embedding_module_weights = {}
    embedding_module_weights["bos_emb"] = bos_emb
    # CFG-only (unconditional text branch). Always exported so a wrapper
    # works with guidance on or off without reconverting.
    embedding_module_weights["null_emb"] = null_emb

    # Single pre-computed lookup replacing the entire char-aware encoder.
    embedding_module_weights["embed_subword.embed_subwords.weight"] = precomputed_subword_emb

    # Keep gated fusion + audio prompt projection: these depend on runtime
    # tensors, not on token id, so they cannot be pre-computed.
    for key, weight in weights.items():
        if key.startswith("gated_fusion_audio_text."):
            embedding_module_weights[key] = weight
    if "audio_prompt_projection_W" in weights:
        embedding_module_weights["audio_prompt_projection_W"] = weights["audio_prompt_projection_W"]

    for i in range(rvq_embs_weight_pad.shape[0]):
        embedding_module_weights[f"rvq_embs.{i}.weight"] = rvq_embs_weight_pad[i]
    embedding_module_weights["embed_code.weight"] = embed_code_weight
    embedding_module_weights = {f"total_emb.{k}": v for k, v in embedding_module_weights.items()}

    # ======================
    # gemma backbone weights
    backbone_module_weights = {k: v for k, v in weights.items() if k.startswith("backbone.")}
    backbone_module_weights["backbone.embed_tokens.weight"] = torch.randn(
        1, hidden_size, dtype=bos_emb.dtype, device=bos_emb.device
    )

    # ======================
    # sampler weights
    used_keys = ("rvq_embs", "embed_code", "mog_head")
    sampler_weights = {"sampler." + k: v for k, v in weights.items() if k.startswith(used_keys)}

    # combine embedding module and backbone module weights
    weights = {**embedding_module_weights, **backbone_module_weights, **sampler_weights}
    weights = {"model." + k: v for k, v in weights.items()}

    # Top-level silence token buffer (int32 tensor of shape [num_quantizers]).
    # Stored under ``model.sil_tokens`` so the vLLM model can register it as a
    # plain buffer at the top of its module tree.
    weights["model.sil_tokens"] = codec_silence_tokens

    # save weights
    safetensors_path = os.path.join(outdir, "model.safetensors")
    save_file(weights, safetensors_path)
    logging.info("Saved weights for vllm model")
    weight_map = {name: "model.safetensors" for name in weights.keys()}
    index = {
        "metadata": {"total_size": sum(w.numel() * w.element_size() for w in weights.values())},
        "weight_map": weight_map,
    }
    index_path = os.path.join(outdir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logging.info("Saved model index")

    # save config.json
    flat_config = {"architectures": ["EarTTSForCausalLM"], "model_type": "eartts"}
    # not using vocab size of the backbone model, but need 2 for dummy sampling to work
    flat_config["vocab_size"] = 2

    # Parse backbone config exactly as NeMo does to get all defaults from transformers
    backbone_type = cfg.model.tts_config.get("backbone_type", None)
    backbone_config_dict = (
        OmegaConf.to_container(cfg.model.tts_config.backbone_config, resolve=True)
        if cfg.model.tts_config.get("backbone_config")
        else {}
    )

    # Create AutoConfig the same way NeMo does - this fills in all defaults
    parsed_backbone_config = AutoConfig.for_model(backbone_type, **backbone_config_dict)

    # Store the backbone type for vllm to use
    flat_config["backbone_type"] = backbone_type

    # Forward all backbone configs from the parsed AutoConfig (includes defaults)
    for key in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rope_theta",
        "rope_local_base_freq",
        "sliding_window",
        "layer_types",
    ]:
        if hasattr(parsed_backbone_config, key):
            value = getattr(parsed_backbone_config, key)
            # convert to list if it's a tuple or other iterable (except str)
            if hasattr(value, '__iter__') and not isinstance(value, (str, dict)):
                value = list(value)
            flat_config[key] = value
    # forward overall configs
    for key in ["latent_size", "codebook_size", "num_quantizers", "exponent"]:
        flat_config[key] = cfg.model.tts_config[key]
    # forward mog head configs
    for key in ["num_layers", "low_rank", "num_predictions", "min_log_std", "eps"]:
        flat_config[f"mog_{key}"] = cfg.model.tts_config.mog_head_config[key]

    # forward inference configs (with name mapping for vLLM model)
    # MaskGIT unmasking iterations, fixed at 8 to match DuplexEARTTS inference.
    flat_config["num_iter"] = 8
    flat_config["noise_scale"] = cfg.model.get("inference_noise_scale", 0.8)
    flat_config["top_p_or_k"] = cfg.model.get("inference_top_p_or_k", 0.8)

    # Classifier-free guidance. ``guidance_scale`` is the checkpoint default;
    # individual requests may override it through ``additional_information["cfg_scale"]``.
    flat_config["guidance_scale"] = cfg.model.get("inference_guidance_scale", 0.5)
    flat_config["enable_guidance"] = True

    # Text-channel specials from the source tokenizer, not the Gemma backbone.
    flat_config["pad_token_id"] = int(model.text_pad_id)
    flat_config["eos_token_id"] = int(model.text_eos_id)

    # Embedding module configuration. The char-aware encoder is gone; vLLM only
    # needs to know the size of the pre-computed lookup table.
    flat_config["emb_vocab_size"] = vocab_size

    flat_config["use_gated_fusion_for_text_audio"] = cfg.model.tts_config.use_gated_fusion_for_text_audio
    flat_config["use_audio_prompt_frozen_projection"] = cfg.model.tts_config.use_audio_prompt_frozen_projection

    # configuring custom inputs/outputs
    flat_config["custom_input_specs"] = [
        {
            "name": "acoustic_tokens",
            "dim": flat_config["num_quantizers"],
            "dtype": "int32",
        },
        {"name": "text_tokens", "dtype": "int32"},
        {"name": "text_mask"},
        {"name": "bos_mask"},
        {"name": "speaker_latent", "dim": flat_config["hidden_size"]},
    ]
    flat_config["custom_outputs"] = ["acoustic_tokens"]

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(flat_config, f, indent=2)
    logging.info("Saved vllm config")

    # Extract and save pre-computed speaker latents (audio_prompt_latents.*)
    # from the NeMo checkpoint so they can be used at inference time.
    speaker_latents_dir = os.path.join(outdir, "speaker_latents")
    found_latents = False
    for key, tensor in raw_weights.items():
        if "audio_prompt_latents." in key:
            speaker_name = key.split("audio_prompt_latents.")[-1]
            os.makedirs(speaker_latents_dir, exist_ok=True)
            latent_path = os.path.join(speaker_latents_dir, f"{speaker_name}.pt")
            torch.save(tensor, latent_path)
            logging.info(f"Saved speaker latent '{speaker_name}' to {latent_path} (shape={tensor.shape})")
            found_latents = True
    if not found_latents:
        logging.warning(
            "No audio_prompt_latents found in checkpoint. " "speaker_name will not work unless latents are added."
        )


if __name__ == "__main__":
    args = parse_args()
    convert_to_vllm_format(args.outdir, args.config, args.model, args.precompute_batch_size)
