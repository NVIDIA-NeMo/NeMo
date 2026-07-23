# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import json
from typing import Optional, Union, Iterable

import torch
from torch import nn
import numpy as np
from omegaconf import OmegaConf
from transformers.generation.logits_process import (
 TopPLogitsWarper,
 TopKLogitsWarper,
)
from transformers import AutoConfig, AutoTokenizer

from vllm.model_executor.models.gemma3 import Gemma3Model
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.cfg_ops import apply_cfg_logits, copy_columns_by_indices, set_uncond_embeddings

from .utils import AutoWeightsLoader
from .optimized_t5gemma import OptimizedT5GemmaEncoderModel


class RMSNorm(nn.Module):
 def __init__(self, dim: int, eps: float = 1e-6):
  super().__init__()
  self.eps = eps
  self.weight = nn.Parameter(torch.zeros(dim))

 def _norm(self, x):
  return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

 def forward(self, x):
  # TODO: is casting really needed?
  output = self._norm(x.float())
  # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
  output = output * (1.0 + self.weight.float())
  return output.type_as(x)


class MLP(nn.Module):
 def __init__(
  self,
  hidden_size: int,
  intermediate_size: int,
 ):
  super().__init__()
  self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
  self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
  self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
  self.act_fn = nn.GELU(approximate="tanh")

 def forward(self, x: torch.Tensor) -> torch.Tensor:
  down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  return down_proj


class MLPLayer(nn.Module):
 def __init__(
  self,
  hidden_size: int,
  intermediate_size: int,
  eps: float = 1e-6,
 ):
  super().__init__()
  self.pre_norm = RMSNorm(hidden_size, eps=eps)
  self.mlp = MLP(hidden_size, intermediate_size)
  self.post_norm = RMSNorm(hidden_size, eps=eps)

 def forward(self, x: torch.Tensor) -> torch.Tensor:
  y = self.pre_norm(x)
  y = self.mlp(y)
  y = self.post_norm(y)
  x = x + y
  return x


class GatedProjectedSumRMSNorm(nn.Module):
 def __init__(self, audio_dim, text_dim, hidden_dim, final_norm=True, num_codebooks=31, init_residual_scale=0.5):
  super().__init__()
  self.num_codebooks = num_codebooks

  self.audio_proj = nn.Linear(audio_dim, hidden_dim)
  self.text_proj = nn.Linear(text_dim, hidden_dim)

  nn.init.normal_(self.audio_proj.weight, mean=0.0, std=0.015)
  nn.init.zeros_(self.audio_proj.bias)
  nn.init.normal_(self.text_proj.weight, mean=0.0, std=0.015)
  nn.init.zeros_(self.text_proj.bias)

  # FP32 gate params
  self.gate = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32), requires_grad=False)
  self.residual_scale = nn.Parameter(torch.tensor(init_residual_scale, dtype=torch.float32), requires_grad=False)

  self.final_norm = RMSNorm(hidden_dim) if final_norm else nn.Identity()

 def forward(self, audio_emb, text_emb):
  audio_emb = audio_emb / self.num_codebooks

  # projections run in model dtype (BF16)
  audio_h = self.audio_proj(audio_emb)
  text_h = self.text_proj(text_emb)

  dtype = audio_h.dtype

  gate = torch.sigmoid(self.gate) # FP32
  res = torch.sigmoid(self.residual_scale) # FP32

  h = gate.to(dtype) * audio_h + (1 - gate).to(dtype) * text_h
  h = res.to(dtype) * h
  h = self.final_norm(h.float()).to(dtype)

  return h


class SubwordFlagEmbedding(nn.Module):
 """
 Adds a small continuation embedding for subwords (tokens without word-boundary marker).
 Automatically adds a custom padding token at index vocab_size.
 Ignores special tokens (starting with '<') when computing continuation flags.
 """

 def __init__(self, model_name: str, d_model: int):
  super().__init__()
  self.tokenizer = AutoTokenizer.from_pretrained(model_name)
  self.vocab_size = self.tokenizer.vocab_size
  self.d_model = d_model

  # Custom pad token at vocab_size
  self.pad_id = self.vocab_size
  # register pad_id as a tensor buffer to avoid device issues
  self.pad_tensor = nn.Parameter(torch.tensor(self.pad_id, dtype=torch.long), requires_grad=False)

  # Precompute continuation flags
  tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in range(self.vocab_size)]
  cont_flags = [
   1 if not (tok.startswith("Ġ") or tok.startswith("▁") or tok.startswith("<")) else 0 for tok in tokens
  ]
  cont_flags.append(0) # for the custom pad token
  self.is_continuation = nn.Parameter(torch.tensor(cont_flags, dtype=torch.long), requires_grad=False)

  # Continuation embedding
  init_std = self.d_model**-0.5
  self.cont_emb = nn.Embedding(2, self.d_model)
  nn.init.normal_(self.cont_emb.weight, mean=0.0, std=init_std)
  self.cont_emb.weight.data[0].zero_()

 def forward(self, subword_embeds: torch.Tensor, token_ids: torch.LongTensor):
  # Replace OOV token IDs with pad_id safely
  token_ids_clamped = torch.where(token_ids >= self.vocab_size, self.pad_tensor, token_ids)
  # Continuation flags
  cont_flags = self.is_continuation[token_ids_clamped]
  # Add continuation embedding
  cont_emb = self.cont_emb(cont_flags)
  return subword_embeds + cont_emb


class BOSEOSEmbedding(nn.Module):
 """
 Adds independent embeddings for BOS and EOS tokens using a single embedding table.
 Index 0 = regular token (ignored), 1 = BOS, 2 = EOS.
 Compatible with Hugging Face tokenizers that may or may not have BOS/EOS.
 """

 def __init__(self, model_name: str, d_model: int):
  super().__init__()
  self.tokenizer = AutoTokenizer.from_pretrained(model_name)
  # vocab size that includes special tokens
  vocab_dict = self.tokenizer.get_vocab()
  self.vocab_size = max(vocab_dict.values())
  self.d_model = d_model

  # Custom pad token for OOVs
  self.pad_id = self.vocab_size
  self.pad_tensor = nn.Parameter(torch.tensor(self.pad_id, dtype=torch.long), requires_grad=False)

  # Identify BOS and EOS tokens (may be None)
  tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in range(self.vocab_size)]

  if 'Qwen2.5' in model_name:
   # For Qwen, '<|im_start|>' is a common choice for a BOS token.
   # You can check your tokenizer's vocabulary for the best candidate.
   print("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.", flush=True)
   self.tokenizer.bos_token = '<|im_start|>'
   self.tokenizer.eos_token = '<|im_end|>'

  special_flags = []
  for tok in tokens:
   if self.tokenizer.bos_token is not None and tok == self.tokenizer.bos_token:
    special_flags.append(1)
   elif self.tokenizer.eos_token is not None and tok == self.tokenizer.eos_token:
    special_flags.append(2)
   else:
    special_flags.append(0)
  special_flags.append(0) # for custom pad token
  self.special_flags = nn.Parameter(torch.tensor(special_flags, dtype=torch.long), requires_grad=False)
  # Embedding table: 0 = regular, 1 = BOS, 2 = EOS
  init_std = self.d_model**-0.5
  self.special_emb = nn.Embedding(3, d_model)
  nn.init.normal_(self.special_emb.weight, mean=0.0, std=init_std)
  self.special_emb.weight.data[0].zero_() # regular tokens ignored

 def forward(self, token_embeds: torch.Tensor, token_ids: torch.LongTensor):
  """
  token_embeds: (B, T, d_model)
  token_ids: (B, T)
  """
  # Clamp OOVs to custom pad token
  safe_ids = torch.where(token_ids >= self.vocab_size, self.pad_tensor, token_ids)

  # Lookup flags (0=regular, 1=BOS, 2=EOS)
  flags = self.special_flags[safe_ids]
  return token_embeds + self.special_emb(flags)


class CharAwareSubwordEncoder(nn.Module):
 """
 An encoder that creates subword embeddings from character-level embeddings.
 This module replaces a standard subword embedding layer. It breaks down each
 subword into its constituent characters, embeds the characters, and then
 aggregates these character embeddings (e.g., via mean pooling) to form the
 final subword representation. This allows the model to handle rare or out-of-vocabulary
 subwords more gracefully.

 Args:
  out_size (int): The dimensionality of the output embedding vectors.
  vocab_size (int): Number of subword tokens in vocabulary
  char_vocab_size (int): Number of characters in vocabulary
  max_char_len (int): Maximum number of characters in a subword
  backbone_type (str | None): The type of backbone model from Hugging Face (e.g., "t5gemma").
  backbone_model_class (str | None): The class name of the backbone model if not using AutoModel.
  backbone_config_class (str | None): The class name of the backbone config.
  backbone_config (Config | None): A configuration for the backbone model.
 """

 def __init__(
  self,
  out_size: int,
  vocab_size: int,
  char_vocab_size: int,
  max_char_len: int,
  backbone_type: str,
  backbone_config: dict,
 ):
  super().__init__()
  self.max_char_len = max_char_len
  # 1. Initialize the backbone model for encoding characters
  config = AutoConfig.for_model(backbone_type, **backbone_config)
  self.backbone = OptimizedT5GemmaEncoderModel(config)
  self.backbone.eval()
  self.hidden_size = self.backbone.get_input_embeddings().weight.size(-1)
  delattr(self.backbone.encoder, "embed_tokens")
  # 2. Initialize embedding layer to embed characters
  self.embed_tokens = nn.Embedding(
   char_vocab_size + 1,
   self.hidden_size,
   padding_idx=char_vocab_size,
  )
  # 3. Initialize embedding layer to convert subword ids to char ids.
  # Also requires a layer which creates a mask for the backbone transformer
  self.embed_subwords = nn.Embedding(
   vocab_size,
   max_char_len,
  )
  self.embed_subwords_mask = nn.Embedding(
   vocab_size,
   max_char_len,
  )
  self.proj_embedding = nn.Linear(self.hidden_size, out_size, bias=False)

 def forward(self, subword_ids: torch.Tensor) -> torch.Tensor:
  """
  Performs the forward pass to get character-aware subword embeddings.
  Args:
   subword_ids (Tensor): A tensor of subword IDs. Shape: `[BT]`.

  Returns:
   Tensor: The final subword embeddings. Shape: `[BT, hidden_size]`.
  """
  char_ids = torch.round(self.embed_subwords(subword_ids)).to(torch.int32) # BT x 128
  char_ids_mask = self.embed_subwords_mask(subword_ids) # BT x 128
  char_embeds = self.embed_tokens(char_ids) # bt x 128 x hidden_size

  char_hidden_states = self.backbone(
   inputs_embeds=char_embeds, attention_mask=char_ids_mask
  ).last_hidden_state # BT x 128 x hidden_size
  # 3. Aggregate character embeddings to form subword embeddings (mean pooling)
  # We mask the padding characters before summing to get a correct mean.
  masked_sum = (char_hidden_states * char_ids_mask.unsqueeze(-1)).sum(dim=1) # BT x hidden_size
  # Avoid division by zero for empty sequences
  char_ids_lengths = char_ids_mask.sum(dim=1) # (bt,)
  mean_emb = masked_sum / (char_ids_lengths.unsqueeze(-1).clamp(min=1)) # BT x hidden_size
  # 4. Scatter the aggregated embeddings back to the original subword sequence shape
  out_emb = self.proj_embedding(mean_emb) # bt x hidden_size
  return out_emb


# module that takes text tokens, audio tokens and prepares input embedding for EarTTS model
class EarTTSInputEmbedding(nn.Module):
 def __init__(self, config):
  super().__init__()

  hidden_size = config.hidden_size
  vocab_size = config.emb_vocab_size
  char_vocab_size = config.emb_char_vocab_size
  max_char_len = config.max_char_len
  backbone_type = config.emb_backbone_type
  backbone_config = config.emb_backbone_config
  self.enable_guidance = config.enable_guidance


  # allows to embed acoustic tokens into a single embeddings
  self.rvq_embs = nn.ModuleList([
   nn.Embedding(config.codebook_size + 1, config.latent_size)
   for _ in range(config.num_quantizers)
  ])
  self.embed_code = nn.Linear(config.latent_size, hidden_size, bias=False)
  self.embed_subword = CharAwareSubwordEncoder(
   out_size=hidden_size,
   vocab_size=vocab_size,
   char_vocab_size=char_vocab_size,
   max_char_len=max_char_len,
   backbone_type=backbone_type,
   backbone_config=backbone_config,
  )
  self.bos_emb = nn.Parameter(torch.empty(hidden_size))
  # always have param created, so there is no problem loading the model
  self.null_emb = nn.Parameter(torch.empty(hidden_size))

  self.use_subword_flag_emb = config.use_subword_flag_emb
  pretrained_tokenizer_name = config.pretrained_tokenizer_name
  if self.use_subword_flag_emb:
   self.subword_flag_emb = SubwordFlagEmbedding(pretrained_tokenizer_name, hidden_size)
  self.use_bos_eos_emb = config.use_bos_eos_emb
  if self.use_bos_eos_emb:
   self.bos_eos_emb = BOSEOSEmbedding(pretrained_tokenizer_name, hidden_size)
  self.use_gated_fusion_for_text_audio = config.use_gated_fusion_for_text_audio
  if self.use_gated_fusion_for_text_audio:
   self.gated_fusion_audio_text = GatedProjectedSumRMSNorm(
    hidden_size, hidden_size, hidden_size, config.num_quantizers
   )

  self.use_audio_prompt_frozen_projection = config.use_audio_prompt_frozen_projection
  if self.use_audio_prompt_frozen_projection:
   self.audio_prompt_projection_W = nn.Parameter(
    torch.empty(hidden_size, hidden_size),
    requires_grad=False,
   )

 def forward(
  self,
  acoustic_tokens: torch.Tensor,
  text_tokens: torch.Tensor,
  text_mask: torch.Tensor,
  bos_mask: torch.Tensor,
  speaker_latent: Optional[torch.Tensor] = None,
 ) -> torch.Tensor:
  """
  Works for context and generation phases to prepare total input embeddings
  for EarTTS model.
  Inputs:
   acoustic_tokens: (BT x 31) - audio tokens
   text_tokens: (BT) - text token to embed
   text_mask: (BT) - masks text embeddings for prefill
   bos_mask: (BT) - specifies where BOS is applied (first frame of prefill)
   speaker_latent: (BT x hidden_size) - external speaker embedding.
                   Non-zero during prefill with speaker_name; zeros during
                   decode or when using speaker_reference fallback.
  Returns:
   embedding of shape (BT x dim)
  """

  # prepare bos emb that is applied to audio embedding
  bos_emb = bos_mask.unsqueeze(1) * self.bos_emb # BT x dim

  acoustic_tokens = acoustic_tokens.transpose(0, 1) # 31 x BT
  audio_emb = sum(emb(acoustic_tokens[i]) for i, emb in enumerate(self.rvq_embs)) # BT x latent_size
  audio_emb = self.embed_code(audio_emb) # BT x hidden_size

  if self.use_audio_prompt_frozen_projection:
   # Compute latent from acoustic tokens (used as fallback for speaker_reference)
   computed_latent = torch.nn.functional.linear(audio_emb, self.audio_prompt_projection_W.T)

   # When speaker_latent is non-zero (speaker_name path), use it.
   # When speaker_latent is all zeros (speaker_reference or decode), use computed_latent.
   # All ops are CUDA-graph safe (no Python branching on tensor values).
   if speaker_latent is not None:
    external_flag = (speaker_latent.abs().sum(-1, keepdim=True) > 0).float()
    latent = speaker_latent * external_flag + computed_latent * (1 - external_flag)
   else:
    latent = computed_latent

   pre_bos_mask = (bos_mask == 0).unsqueeze(-1) # BT x 1
   audio_emb = torch.where(pre_bos_mask, latent, audio_emb)

  audio_emb = audio_emb + bos_emb

  # embed text tokens by expanding them to chars and passing through transformer
  # apply the mask that turns this embedding to zeros for prefill tokens
  text_emb = self.embed_subword(text_tokens) * text_mask.unsqueeze(1) # BT x dim
  # update text embeddings with flags
  if self.use_subword_flag_emb:
   text_emb = self.subword_flag_emb(text_emb, text_tokens)
  if self.use_bos_eos_emb:
   text_emb = self.bos_eos_emb(text_emb, text_tokens)

  if self.enable_guidance:
   cfg_metadata = get_forward_context().cfg_metadata
   set_uncond_embeddings(
    text_emb,
    self.null_emb,
    cfg_metadata.uncond_token_mask,
    cfg_metadata.num_tokens,
    cfg_metadata.padded_num_tokens,
   )

  # prepare total embedding by adding all components
  if self.use_gated_fusion_for_text_audio:
   total_emb = self.gated_fusion_audio_text(audio_emb, text_emb)
  else:
   total_emb = audio_emb + text_emb # BT x dim
  return total_emb


def gumbel_like(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
 """
 Generates a tensor of Gumbel noise with the same shape as the input tensor.

 This is used for the Gumbel-Max trick, a technique to sample from a categorical
 distribution in a differentiable way (using a straight-through estimator).

 Args:
  tensor (torch.Tensor): The input tensor to match the shape of.
  eps (float): A small epsilon value for numerical stability.

 Returns:
  torch.Tensor: A tensor containing Gumbel noise.
 """
 # Sample from a uniform distribution
 u = torch.rand_like(tensor)
 # Apply the inverse CDF of the Gumbel distribution
 return -torch.log(-torch.log(u + eps) + eps)


def batch_matmul(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
 """
 Performs a batched matrix multiplication using PyTorch's native functions.
 In NeMo this is implemented as a custom kernel using triton.
 TODO: check vLLM kernels if there is one available, check if triton can be used here.

 Args:
  x (Tensor): The input tensor of shape `[batch_size, d_in]`.
  w (Tensor): The weight tensor of shape `[num_weights, d_out, d_in]`.
  y (Tensor): The index tensor of shape `[batch_size]`.

 Returns:
  Tensor: The result of the multiplication, shape `[batch_size, d_out]`.
 """
 # w[y] gathers the weight matrices for each item in the batch.
 # x.unsqueeze(2) reshapes x to [batch_size, d_in, 1] for bmm.
 # The result is squeezed to remove the trailing dimension of size 1.
 return torch.bmm(w[y], x.unsqueeze(2)).squeeze(2)


class MoGHead(nn.Module):
 """
 A Mixture of Gaussians (MoG) prediction head.

 This module takes a hidden state and predicts the parameters for a mixture of
 Gaussian distributions. It's suitable for modeling continuous, multi-modal data.

 Args:
  hidden_size (int): The dimensionality of the input hidden state.
  intermediate_size (int): The dimensionality of the MLP layers.
  out_size (int): The dimensionality of the output vectors (the mean of each Gaussian).
  num_layers (int): The number of MLP layers in the stack.
  num_predictions (int): The number of Gaussian components in the mixture.
  low_rank (int | None): The dimensionality used for compressing the hidden states.
  min_log_std (float): The minimum value for the logarithm of the standard deviation.
  eps (float): A small epsilon value for the RMSNorm layers.
 """

 def __init__(
  self,
  hidden_size: int,
  intermediate_size: int,
  out_size: int,
  num_layers: int,
  num_predictions: int,
  low_rank: Optional[int] = 64,
  top_p_or_k: Optional[float | int] = 1.0,
  min_log_std: float = -4.0,
  eps: float = 1e-6,
  enable_guidance: bool = False,
 ):
  super().__init__()
  self.out_size = out_size
  self.low_rank = low_rank
  self.num_predictions = num_predictions
  self.min_log_std = min_log_std
  self.top_p_or_k = top_p_or_k
  self.enable_guidance = enable_guidance

  self.logits_processor = (
   TopPLogitsWarper(self.top_p_or_k)
   if isinstance(self.top_p_or_k, float)
   else (
    TopKLogitsWarper(self.top_p_or_k)
    if isinstance(self.top_p_or_k, int)
    else None
   )
  )

  self.mlp_stack = nn.Sequential(
   *[
    MLPLayer(hidden_size, intermediate_size, eps=eps)
    for _ in range(num_layers)
   ],
   RMSNorm(hidden_size, eps=eps),
  )

  if low_rank is None:
   self.proj_logits = nn.Linear(
    hidden_size, num_predictions, bias=False
   ) # Predicts mixture weights
   self.proj_mus = nn.Linear(
    hidden_size, num_predictions * out_size, bias=False
   ) # Predicts means
   self.proj_logs = nn.Linear(
    hidden_size, 1, bias=False
   ) # Predicts log standard deviations
  else:
   assert low_rank < out_size
   self.proj_logits = nn.Linear(
    hidden_size, num_predictions, bias=False
   ) # Predicts mixture weights
   self.proj_mus = nn.Linear(
    hidden_size, num_predictions * low_rank, bias=False
   ) # Predicts means
   self.proj_logs = nn.Linear(
    hidden_size, 1, bias=False
   ) # Predicts log standard deviations
   self.proj_else = nn.Linear(hidden_size, out_size, bias=False)
   self.low_mat = nn.Parameter(
    torch.empty(num_predictions, out_size, low_rank)
   )

 def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """
  Performs inference by sampling from the predicted mixture distribution.

  Args:
   x (Tensor): The input hidden state.
   top_p_or_k (float | int): The value for top-p (nucleus) or top-k sampling of the mixture components.

  Returns:
   tuple[Tensor, Tensor]: A tuple containing the mean of the chosen component,
    and the log standard deviations.
  """
  bt = x.size(0)
  n, d = self.num_predictions, self.low_rank or self.out_size

  x = self.mlp_stack(x)

  # NOTE: in NeMo it is applied not to logits but before projection.
  # Always call the kernel when enable_guidance is True so the launch
  # is captured in CUDA graphs. The kernel reads num_cfg_pairs from a
  # GPU tensor and is a no-op when num_cfg_pairs=0.
  if self.enable_guidance:
   cfg_metadata = get_forward_context().cfg_metadata
   apply_cfg_logits(
    x,
    cfg_metadata.cond_logits_indices,
    cfg_metadata.uncond_logits_indices,
    cfg_metadata.guidance_scales,
    cfg_metadata.num_cfg_pairs,
    cfg_metadata.padded_num_tokens,
   )

  logits = self.proj_logits(x)

  # Apply top-p or top-k filtering to the mixture logits
  if self.logits_processor is not None:
   logits = self.logits_processor(None, logits.view(-1, n)).view_as(logits)

  # Sample a mixture component using the Gumbel-Max trick
  mixture_indices = (nn.functional.log_softmax(logits, dim=-1) + gumbel_like(logits)).argmax(-1)
  #mixture_indices = (nn.functional.log_softmax(logits, dim=-1)).argmax(-1)

  # Select the mean corresponding to the sampled component
  mu = batch_matmul(
   x.view(bt, -1),
   self.proj_mus.weight.detach().view(n, d, -1),
   mixture_indices.view(bt),
  ).view(bt, d)
  if self.proj_mus.bias is not None:
   mu += self.proj_mus.bias.detach().view(n, d)[mixture_indices]

  if self.low_rank:
   # assert math.log2(d).is_integer() and math.log2(self.out_size).is_integer()
   mu = batch_matmul(
    mu.view(bt, -1),
    self.low_mat.detach().view(n, self.out_size, -1),
    mixture_indices.view(bt),
    # TODO: these are the arguments for custom kernel impl
    # BLOCK_SIZE_DIN=d,
    # BLOCK_SIZE_DOUT=self.out_size,
   ).view(bt, self.out_size)
   mu_res = self.proj_else(x)
  else:
   mu_res = torch.zeros((bt, d), device=x.device)

  logs = self.proj_logs(x).clamp_min(self.min_log_std)
  return mu * torch.exp(logs) + mu_res, logs


class MaskGITSampler(nn.Module):
 def __init__(self, config):
  super().__init__()
  self.config = config
  # easy access of cruicial config params
  self.num_quantizers = self.config.num_quantizers
  self.codebook_size = self.config.codebook_size
  self.noise_scale = self.config.noise_scale

  # pre-compute how many tokens are unmasked at each iteration
  rates = np.linspace(0.0, 1.0, self.config.num_iter + 1)[:-1].reshape(-1, 1)
  masking_rates = np.power(
   1 - np.power(rates, self.config.exponent), 1 / self.config.exponent
  )
  num_maskings = np.ceil(masking_rates * self.num_quantizers).astype(int)
  num_maskings_shifted = np.pad(
   num_maskings[1:], ((0, 1), (0, 0)), constant_values=0
  )
  sampling_per_step = num_maskings - num_maskings_shifted
  sampling_per_step_flat = sampling_per_step.flatten()
  # Drop any values at the beginning that are 0
  first_nonzero = np.argmax(sampling_per_step_flat != 0)
  self.num_to_sample = sampling_per_step_flat[first_nonzero:].tolist()

  # create layers used for acoustic tokens embedding
  self.rvq_embs = nn.Parameter(torch.empty(
   self.config.num_quantizers,
   self.config.codebook_size,
   self.config.latent_size
  ))
  self.embed_code = nn.Linear(
   self.config.latent_size, self.config.hidden_size, bias=False
  )
  # MoG head for generation (uncompiled part)
  self.mog_head = MoGHead(
   hidden_size=self.config.hidden_size,
   intermediate_size=self.config.intermediate_size,
   out_size=self.config.latent_size,
   num_layers=self.config.mog_num_layers,
   num_predictions=self.config.mog_num_predictions,
   low_rank=self.config.mog_low_rank,
   top_p_or_k=self.config.top_p_or_k,
   min_log_std=self.config.mog_min_log_std,
   eps=self.config.mog_eps,
   enable_guidance=self.config.enable_guidance,
  )

 def _depthsum_embedding(self, code: torch.Tensor) -> torch.Tensor:
  """
  Embedds all codes into a single embedding.
  Args:
   code: Tensor (num_quantizers x BT) Acoustic codes to embed and add
   rvq_embeddings: Tensor (num_quantizers x codebook_size x latent_size) RVQ embeddings

  Returns:
   Tensor (BT x latent_size) - embedded codes
  """
  embs = nn.functional.pad(self.rvq_embs, [0, 0, 0, 1]) # num_quantizers x (codebook_size + 1) x latent_size
  res = nn.functional.embedding(code[0], embs[0])
  for i in range(1, len(embs)):
   res = res + nn.functional.embedding(
    code[i], embs[i]
   )
  return res

 def _depthsum_encoding_step_reshaped(
  self,
  r: torch.Tensor, # [B*T, hidden_size]
  code: torch.Tensor, # [num_quantizers, B*T]
  depth_str: int,
  k: int,
 ) -> torch.Tensor:
  """
  RVQ encoding with reshaped code tensor.

  Args:
   embs: [num_quantizers, vocab_size, hidden_size] - RVQ codebook embeddings
   r: [B*T, hidden_size] - residual to quantize
   code: [num_quantizers, B*T] - output code tensor
   depth_str: starting quantizer level
   k: number of quantizer levels to process
  """
  for i in range(depth_str, depth_str + k):
   # self.rvq_embeddings[i]: [vocab_size, latent_size]
   # r: [B*T, latent_size]

   # Compute distances: ||emb||² - 2⟨r, emb⟩
   idx_sel = (
    self.rvq_embs[i].pow(2).sum(-1) # [vocab_size]
    - 2 * (r @ self.rvq_embs[i].T) # [B*T, vocab_size]
   ).argmin(
    -1
   ) # [B*T]

   # Update residual
   emb_i = nn.functional.embedding(
    idx_sel, self.rvq_embs[i], #padding_idx=self.padding_idx
   ) # [B*T, latent_size]
   r = r - emb_i

   # Store selected indices
   code[i] = idx_sel

  return code

 def forward(
  self,
  hidden_states: torch.Tensor,
 ) -> torch.Tensor:
  """
  Performs the iterative unmasking process for a single generation step.
  This function takes the hidden state from the backbone transformer and generates
  codes through an iterative unmasking process.

  Args:
   hidden_states: Tensor (BT x hidden_size) - The hidden states from the backbone

  Returns:
   Tensor (BT x num_quantizers) - The generated codes
  """

  device = hidden_states.device
  # Initialize the full code tensor
  code = (
   torch.zeros(
    (self.num_quantizers, hidden_states.shape[0]),
    dtype=torch.long,
    device=device,
   )
   + self.codebook_size
  )
  # Iteratively unmask the continuous part of the code
  cnt = 0
  for k in self.num_to_sample:
   # Prepare input for the MoG head
   mog_input_embeds = self.embed_code(
    self._depthsum_embedding(code)
   ) # (BT x hidden_size)
   mog_input_embeds += hidden_states

   mog_mu, mog_logs = self.mog_head(
    mog_input_embeds,
   )
   z = (
    mog_mu
    + torch.exp(mog_logs) * torch.randn_like(mog_mu) * self.noise_scale
   )
   code = self._depthsum_encoding_step_reshaped(z, code, cnt, k)

   if self.config.enable_guidance:
    # next mog head iteration uses cond tokens as input, avoiding divergence
    cfg_metadata = get_forward_context().cfg_metadata
    copy_columns_by_indices(
     code,
     cfg_metadata.cond_logits_indices,
     cfg_metadata.uncond_logits_indices,
     cfg_metadata.num_cfg_pairs,
     cfg_metadata.padded_num_tokens,
    )

   cnt += k
  return code.transpose(0, 1) # BT x num_quantizers


@support_torch_compile
class EarTTSModel(nn.Module):
 """
 Wrapper module that combines the embedding preparation, backbone transformer and sampler.
 All components supports torch compile.
 """
 def __init__(
  self,
  *,
  vllm_config: VllmConfig,
  prefix: str = "",
 ):
  super().__init__()
  self.total_emb = EarTTSInputEmbedding(vllm_config.model_config.hf_config)
  self.backbone = Gemma3Model(vllm_config=vllm_config, prefix=prefix)
  self.sampler = MaskGITSampler(vllm_config.model_config.hf_config)

 def forward(
  self,
  input_ids: torch.Tensor,
  positions: torch.Tensor,
  intermediate_tensors: Optional[IntermediateTensors],
  acoustic_tokens: torch.Tensor,
  text_tokens: torch.Tensor,
  text_mask: torch.Tensor,
  bos_mask: torch.Tensor,
  speaker_latent: Optional[torch.Tensor] = None,
 ) -> torch.Tensor:
  """
  Forward pass through embeddings and backbone transformer.
  Returns hidden states to be used by the generation step.
  """
  total_emb = self.total_emb(
   acoustic_tokens=acoustic_tokens,
   text_tokens=text_tokens,
   text_mask=text_mask,
   bos_mask=bos_mask,
   speaker_latent=speaker_latent,
  )
  hidden_states = self.backbone(input_ids, positions, intermediate_tensors, inputs_embeds=total_emb)
  codes = self.sampler(hidden_states)
  return hidden_states, codes


class EarTTSForCausalLM(nn.Module):
 def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
  super().__init__()
  self.config = vllm_config.model_config.hf_config
  self.model = EarTTSModel(
   vllm_config=vllm_config,
   prefix=prefix,
  )

 def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
  return self.model.backbone.get_input_embeddings(input_ids)

 def forward(
  self,
  input_ids: torch.Tensor,
  positions: torch.Tensor,
  intermediate_tensors: Optional[IntermediateTensors] = None,
  inputs_embeds: Optional[torch.Tensor] = None,
  acoustic_tokens: Optional[torch.Tensor] = None,
  text_tokens: Optional[torch.Tensor] = None,
  text_mask: Optional[torch.Tensor] = None,
  bos_mask: Optional[torch.Tensor] = None,
  speaker_latent: Optional[torch.Tensor] = None,
 ) -> Union[torch.Tensor, IntermediateTensors]:
  """
  input_ids, positions, intermediate_tensors, inputs_embeds - not used,
  they are here for compatability with the way vllm model executed.
  """

  hidden_states, codes = self.model(
   input_ids=input_ids,
   positions=positions,
   intermediate_tensors=intermediate_tensors,
   acoustic_tokens=acoustic_tokens,
   text_tokens=text_tokens,
   text_mask=text_mask,
   bos_mask=bos_mask,
   speaker_latent=speaker_latent,
  )
  return hidden_states, codes

 def compute_logits(
  self,
  hidden_states: torch.Tensor,
 ) -> Optional[torch.Tensor]:
  return hidden_states

 def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
  # TODO: skip prefixes for embeddings, we dont use them
  loader = AutoWeightsLoader(
   self,
   skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
  )
  return loader.load_weights(weights)
