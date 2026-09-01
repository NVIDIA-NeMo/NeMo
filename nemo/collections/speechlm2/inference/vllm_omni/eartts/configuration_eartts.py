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
"""HuggingFace-style configuration for the EarTTS model.

The configuration mirrors the fields that
:class:`nemo.collections.speechlm2.inference.vllm_omni.eartts.eartts.EarTTSForCausalLM`
reads from ``vllm_config.model_config.hf_config``:

* Gemma3 backbone fields consumed by ``Gemma3Model`` (``hidden_size``,
  ``intermediate_size``, ``num_hidden_layers``, ``num_attention_heads``,
  ``num_key_value_heads``, ``head_dim``, ``vocab_size``,
  ``max_position_embeddings``, ``query_pre_attn_scalar``,
  ``attention_bias``, ``rms_norm_eps``, ``layer_types``,
  ``sliding_window``, ``rope_local_base_freq``, ``rope_theta``,
  ``rope_scaling``, ``hidden_activation``, ``tie_word_embeddings``,
  ``final_logit_softcapping``, ``attn_logits_soft_cap``,
  ``use_bidirectional_attention``, ``is_causal``).

* MaskGIT sampler fields (``num_quantizers``, ``codebook_size``,
  ``num_iter``, ``top_p_or_k``, ``noise_scale``, ``exponent``,
  ``latent_size``, ``mog_low_rank``, ``mog_num_layers``,
  ``mog_num_predictions``, ``mog_min_log_std``, ``mog_eps``).

* Subword embedding / fusion fields consumed by
  :class:`EarTTSInputEmbedding` (``emb_vocab_size``,
  ``use_gated_fusion_for_text_audio``,
  ``use_audio_prompt_frozen_projection``). The original NeMo model used
  a character-aware subword encoder + subword-flag + BOS/EOS additive
  embeddings; all of those operations are deterministic per token id
  and are baked out at checkpoint-conversion time into a single
  ``nn.Embedding`` of size ``(emb_vocab_size, hidden_size)``.

vLLM's ``patch_rope_parameters`` (transformers-v4 path) auto-populates
``config.rope_parameters`` from ``rope_scaling`` + ``rope_theta`` during
config loading, so the Gemma3 backbone (which expects
``config.rope_parameters``) works without any extra plumbing here.
"""

from typing import Optional

from transformers import AutoConfig, PretrainedConfig


class EarTTSConfig(PretrainedConfig):
    model_type = "eartts"

    def __init__(
        self,
        # Gemma 3 backbone
        hidden_size: int = 1152,
        context_hidden_size: int = 1536,
        intermediate_size: int = 4608,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 72,
        # ``vocab_size`` controls the width of the logits tensor returned
        # by ``EarTTSForCausalLM.compute_logits`` — vLLM's sampler and
        # ``LogitsProcessor`` size their working buffers from
        # ``config.vocab_size``, so it must match the dummy logits the
        # model produces. The model emits a 2-class placeholder
        # (``[0, -inf]``) so the sampler's argmax always picks index 0;
        # the real audio output is the codes tensor exposed via
        # ``make_omni_output``. ``2`` is the minimum that keeps vLLM's
        # sampler happy.
        vocab_size: int = 2,
        max_position_embeddings: int = 131072,
        # MaskGIT / MoG sampling
        num_quantizers: int = 31,
        codebook_size: int = 1024,
        num_iter: int = 8,
        top_p_or_k: float = 0.8,
        noise_scale: float = 0.8,
        exponent: float = 3.0,
        latent_size: int = 512,
        mog_low_rank: int = 64,
        mog_num_layers: int = 3,
        mog_num_predictions: int = 1024,
        mog_min_log_std: float = -4.0,
        mog_eps: float = 1e-6,
        # Classifier-free guidance. The converter exports both fields and
        # ``null_emb``; the runtime reads them unless the request overrides
        # ``cfg_scale``.
        enable_guidance: bool = False,
        guidance_scale: float = 0.5,
        # Gemma3-specific attributes required by Gemma3Model
        query_pre_attn_scalar: float = 256.0,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        layer_types: Optional[list] = None,
        sliding_window: Optional[int] = 4096,
        rope_local_base_freq: float = 10000.0,
        # NeMo / EarTTS uses 1M for the global-attention RoPE base.
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        hidden_activation: str = "gelu_pytorch_tanh",
        tie_word_embeddings: bool = True,
        final_logit_softcapping: Optional[float] = None,
        attn_logits_soft_cap: Optional[float] = None,
        use_bidirectional_attention: bool = False,
        is_causal: bool = True,
        # Subword encoding. The character-aware subword encoder /
        # subword-flag / BOS-EOS embedding tables that NeMo applied at
        # runtime are precomputed at checkpoint conversion time into a
        # single ``(emb_vocab_size, hidden_size)`` lookup, so only the
        # vocab size and the audio-side fusion / projection toggles
        # remain as runtime config.
        emb_vocab_size: int = 151936,
        use_gated_fusion_for_text_audio: bool = True,
        use_audio_prompt_frozen_projection: bool = False,
        # HF-canonical model dtype (replaces the deprecated
        # ``torch_dtype``). Forwarded to ``PretrainedConfig`` so it is
        # converted into a real ``torch.dtype`` and exposed as
        # ``config.dtype``.
        dtype: str = "float32",
        # Text-channel specials, copied from the source VoiceChat tokenizer
        # at conversion time. Used to pad prefill text and to force codec
        # silence when the incoming text token is EOS.
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ):
        # Gemma3 backbone
        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

        # MaskGIT / MoG sampling
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.num_iter = num_iter
        self.top_p_or_k = top_p_or_k
        self.noise_scale = noise_scale
        self.exponent = exponent
        self.latent_size = latent_size
        self.mog_low_rank = mog_low_rank
        self.mog_num_layers = mog_num_layers
        self.mog_num_predictions = mog_num_predictions
        self.mog_min_log_std = mog_min_log_std
        self.mog_eps = mog_eps
        self.enable_guidance = enable_guidance
        self.guidance_scale = guidance_scale

        # Gemma3-specific attributes
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        # Default all layers to global attention if not specified.
        self.layer_types = (
            layer_types if layer_types is not None else ["full_attention"] * num_hidden_layers
        )
        self.sliding_window = sliding_window
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_activation = hidden_activation
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.use_bidirectional_attention = use_bidirectional_attention
        self.is_causal = is_causal

        # Subword encoding (precomputed lookup; see class docstring).
        self.emb_vocab_size = emb_vocab_size
        self.use_gated_fusion_for_text_audio = use_gated_fusion_for_text_audio
        self.use_audio_prompt_frozen_projection = use_audio_prompt_frozen_projection

        # Forward HF-owned fields (``tie_word_embeddings`` and ``dtype``)
        # to ``PretrainedConfig`` so they round-trip through
        # save/load_pretrained and are visible as ``config.dtype`` /
        # ``config.tie_word_embeddings``. Without this, user-supplied
        # values silently fall back to PretrainedConfig's defaults.
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


def register_eartts_config() -> None:
    """Make ``model_type: "eartts"`` resolvable by ``AutoConfig``.

    Idempotent, because it has two callers by necessity and either may run
    first: :func:`register_nemo_voicechat` (the plugin entry point) and the
    import below. The import-time call is what covers subprocesses that reach
    this module without the plugin -- ``StageEngineCoreProc`` unpickles a
    config and calls ``AutoConfig.from_pretrained`` -- and mirrors the pattern
    used by other vllm-omni custom configs (voxcpm, fish_speech,
    mammoth_moda2, ...).
    """
    try:
        AutoConfig.register(EarTTSConfig.model_type, EarTTSConfig)
    except ValueError:
        # transformers raises when the model_type is already registered, which
        # is the expected outcome for whichever caller runs second.
        pass


register_eartts_config()
