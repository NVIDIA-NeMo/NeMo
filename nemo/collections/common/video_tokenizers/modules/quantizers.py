# ******************************************************************************
# Copyright (C) 2024 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ******************************************************************************
"""Quantizers for discrete image and video tokenization."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from nemo.collections.common.video_tokenizers.modules.utils import (
    default,
    entropy,
    pack_one,
    rearrange,
    round_ste,
    unpack_one,
)


class ResidualFSQuantizer(nn.Module):
    """Residual Finite Scalar Quantization

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, levels: list[int], num_quantizers: int, **ignore_kwargs):
        super().__init__()
        self.dtype = ignore_kwargs.get("dtype", torch.float32)
        self.layers = nn.ModuleList([FSQuantizer(levels=levels) for _ in range(num_quantizers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices_stack = []
        residual = x
        quantized_out = 0
        loss_out = 0
        for i, layer in enumerate(self.layers):
            quant_indices, z, loss = layer(residual)
            indices_stack.append(quant_indices)
            residual = residual - z.detach()
            quantized_out = quantized_out + z
            loss_out = loss_out + loss
        self.residual = residual
        indices = torch.stack(indices_stack, dim=1)
        return indices, quantized_out.to(self.dtype), loss_out.to(self.dtype)

    def indices_to_codes(self, indices_stack: torch.Tensor) -> torch.Tensor:
        quantized_out = 0
        for layer, indices in zip(self.layers, indices_stack.transpose(0, 1)):
            quantized_out += layer.indices_to_codes(indices)
        return quantized_out


class FSQuantizer(nn.Module):
    """Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505

    Code adapted from Jax version in Appendix A.1.

    Adapted from: https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/
    vector_quantize_pytorch/finite_scalar_quantization.py
    [Copyright (c) 2020 Phil Wang]
    https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/LICENSE
    """

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        **ignore_kwargs,
    ):
        super().__init__()
        self.dtype = ignore_kwargs.get("dtype", torch.float32)
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out=False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor, project_out=True) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert z.shape[-1] == self.dim, f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")
            indices = unpack_one(indices, ps, "b * c")
            dummy_loss = torch.zeros_like(out.mean(dim=[1, 2, 3], keepdim=True))
        else:
            dummy_loss = torch.zeros_like(out.mean(dim=[1, 2], keepdim=True)).unsqueeze(1)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return (indices, out.to(self.dtype), dummy_loss)


class VectorQuantizer(nn.Module):
    """Improved version over VectorQuantizer. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.

    Adapted from: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/
    taming/modules/vqvae/quantize.py

    [Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer]
    https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/License.txt
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        remap: str = None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        legacy: bool = True,
        use_norm=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.n_e = num_embeddings
        self.e_dim = embedding_dim
        self.beta = beta
        self.legacy = legacy
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = num_embeddings

        self.sane_index_shape = sane_index_shape
        self.dtype = ignore_kwargs.get("dtype", torch.float32)

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits is False, "Only for interface compatible with Gumbel"
        assert return_logits is False, "Only for interface compatible with Gumbel"
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn",
                z_flattened,
                rearrange(self.embedding.weight, "n d -> d n"),
            )
        )

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_e, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        min_encodings = None

        z_q, z = self.norm(z_q), self.norm(z)

        # compute loss for embedding
        commit_loss = torch.mean((z_q - z.detach()) ** 2, dim=[1, 2, 3], keepdim=True)
        emb_loss = torch.mean((z_q.detach() - z) ** 2, dim=[1, 2, 3], keepdim=True)
        if not self.legacy:
            loss = self.beta * emb_loss + commit_loss
        else:
            loss = emb_loss + self.beta * commit_loss

        # preserve gradients
        z_q = z + (z_q - z).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # reshape back to match original input shape
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = encoding_indices.squeeze(1).reshape(z.shape[0], -1)  # add batch axis
            min_encoding_indices = self.remap_to_used(encoding_indices.squeeze(1))
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0], z_q.shape[2], z_q.shape[3])

        # TODO: return (indices, z_q, loss)
        return (
            z_q,
            loss,
            (
                encoding_indices.squeeze(1),
                min_encodings,
                commit_loss.mean().detach(),
                self.beta * emb_loss.mean().detach(),
                perplexity.mean().detach(),
            ),
        )

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class LFQuantizer(nn.Module):
    """Lookup-Free Quantization

    Adapted from: https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/
    vector_quantize_pytorch/lookup_free_quantization.py
    [Copyright (c) 2020 Phil Wang]
    https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/LICENSE
    """

    def __init__(
        self,
        *,
        codebook_size: int,
        codebook_dim: int,
        embed_dim: Optional[int] = None,  # if None, use codebook_dim
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        default_temp: float = 0.01,
        entropy_loss: bool = False,
        **ignore_kwargs,
    ):
        """Lookup-Free Quantization

        Args:
            codebook_size (int): The number of entries in the codebook.
            codebook_dim (int): The number of bits in each code.
            embed_dim (Optional[int], optional): The dimension of the input embedding. Defaults to None.
            entropy_loss_weight (float, optional): Whether to use entropy loss. Defaults to 0.1.
            commitment_loss_weight (float, optional): Weight for commitment loss. Defaults to 0.25.
            default_temp (float, optional): The temprature to use. Defaults to 0.01.
            entropy_loss (bool, optional): Flag for entropy loss. Defaults to False.
        """
        super().__init__()
        self.entropy_loss = entropy_loss
        self.codebook_dim = codebook_dim
        self.default_temp = default_temp
        self.entrop_loss_weight = entropy_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        embed_dim = embed_dim or codebook_dim

        has_projections = embed_dim != codebook_dim
        self.project_in = nn.Linear(embed_dim, codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, embed_dim) if has_projections else nn.Identity()

        self.dtype = ignore_kwargs.get("dtype", torch.float32)

        if entropy_loss:
            assert 2**codebook_dim == codebook_size, "codebook size must be 2 ** codebook_dim"
            self.codebook_size = codebook_size

            self.register_buffer(
                "mask",
                2 ** torch.arange(codebook_dim - 1, -1, -1),
                persistent=False,
            )
            self.register_buffer("zero", torch.tensor(0.0), persistent=False)

            all_codes = torch.arange(codebook_size)
            bits = ((all_codes[..., None].int() & self.mask) != 0).float()
            codebook = 2 * bits - 1.0

            self.register_buffer("codebook", codebook, persistent=False)  # [codebook_size, codebook_dim]

    def forward(self, z: torch.Tensor, temp: float = None) -> torch.Tensor:
        temp = temp or self.default_temp

        z = rearrange(z, "b d ... -> b ... d")
        z, ps = pack_one(z, "b * d")
        z = self.project_in(z)

        # split out number of codebooks
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        # quantization
        original_input = z

        codebook_value = torch.ones_like(z)
        z_q = torch.where(z > 0, codebook_value, -codebook_value)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # commit loss
        commit_loss = ((original_input - z_q.detach()) ** 2).mean(dim=[1, 2, 3])

        z_q = rearrange(z_q, "b n c d -> b n (c d)")
        z_q = self.project_out(z_q)

        # reshape
        z_q = unpack_one(z_q, ps, "b * d")
        z_q = rearrange(z_q, "b ... d -> b d ...")

        loss = self.commitment_loss_weight * commit_loss

        # entropy loss (eq-5)
        if self.entropy_loss:
            # indices
            indices = reduce((z > 0).int() * self.mask.int(), "b n c d -> b n c", "sum")
            indices = unpack_one(indices, ps, "b * c")
            indices = rearrange(indices, "... 1 -> ...")

            distance = -2 * torch.einsum(
                "... i d, j d -> ... i j",
                original_input,
                self.codebook.to(original_input.dtype),
            )
            prob = (-distance / temp).softmax(dim=-1)
            per_sample_entropy = entropy(prob).mean(dim=[1, 2])
            avg_prob = reduce(prob, "... c d -> c d", "mean")
            codebook_entropy = entropy(avg_prob).mean()
            entropy_aux_loss = per_sample_entropy - codebook_entropy

            loss += self.entrop_loss_weight * entropy_aux_loss

            # TODO: return (indices, z_q, loss)
            return (
                z_q,
                loss.unsqueeze(1).unsqueeze(1).unsqueeze(1),
                (
                    indices,
                    self.commitment_loss_weight * commit_loss.mean().detach(),
                    self.entrop_loss_weight * entropy_aux_loss.mean().detach(),
                    self.entrop_loss_weight * per_sample_entropy.mean().detach(),
                    self.entrop_loss_weight * codebook_entropy.mean().detach(),
                ),
            )
        else:
            return (
                z_q,
                loss.unsqueeze(1).unsqueeze(1).unsqueeze(1),
                self.commitment_loss_weight * commit_loss.mean().detach(),
            )


class InvQuantizerJit(nn.Module):
    """Use for decoder_jit to trace quantizer in discrete tokenizer"""

    def __init__(self, quantizer):
        super().__init__()
        self.quantizer = quantizer

    def forward(self, indices: torch.Tensor):
        codes = self.quantizer.indices_to_codes(indices)
        return codes.to(self.quantizer.dtype)
