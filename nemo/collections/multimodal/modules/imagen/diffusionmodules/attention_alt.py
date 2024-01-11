# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
Adapted from:
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
"""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd

USE_ALT = False


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


# Stable attention
class StableAttentionOp(torch.autograd.Function):
    # This function defines the attention weight computation in a stable way
    # The idea is to scale the gradients of weight matrix by the maximum absolute value.
    # In case of overflow, this will prevent weight gradients from exploding.
    # In case of underflow, since we clipped the scale to 1e-4, this will prevent underflow.

    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q, k / math.sqrt(k.shape[1])).softmax(dim=2)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors

        s = dw.detach().norm(float('inf'), dim=[1, 2], keepdim=True).clip(min=1e-4)
        dw = dw / s

        # Due to softmax, w is fp32, making db fp32.
        # Type casting is required for amp to work.
        db = torch._softmax_backward_data(grad_output=dw, output=w, dim=2, input_dtype=dw.dtype).to(q.dtype)
        s = s / math.sqrt(k.shape[1])

        dq = torch.einsum('nck,nqk->ncq', k, db) * s
        dk = torch.einsum('ncq,nqk->nck', q, db) * s

        return dq, dk


class QKVStableAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshaping q and k
        # try:
        #     q = q.view(bs * self.n_heads, ch, length)
        #     k = k.view(bs * self.n_heads, ch, length)
        # except Exception:
        q = q.reshape(bs * self.n_heads, ch, length)
        k = k.reshape(bs * self.n_heads, ch, length)

        weight = StableAttentionOp.apply(q, k)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length), weight

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length), weight

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class StableMaskedAttentionOp(torch.autograd.Function):
    # Robust attention operation in case of masked attention
    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, mask):
        max_neg_value = -float('inf')
        w = torch.einsum('ncq,nck->nqk', q, k / math.sqrt(k.shape[1]))
        w = w.masked_fill(mask, max_neg_value)
        w = w.softmax(dim=2)

        # When we use an arbitrary mask, there is a possibility that we get nans in softmax.
        # In this case, use nan_to_num to make it a stable number.
        # w = w.nan_to_num_()
        ctx.save_for_backward(q, k, w, mask)
        return w

    @staticmethod
    @custom_bwd
    def backward(ctx, dw):
        q, k, w, mask = ctx.saved_tensors
        max_neg_value = -torch.finfo(q.dtype).max
        s = dw.detach().norm(float('inf'), dim=[1, 2], keepdim=True).clip(min=1e-4)
        dw = dw / s
        db = torch._softmax_backward_data(grad_output=dw, output=w, dim=2, input_dtype=dw.dtype)

        # Masking db
        db_in = db.clone().masked_fill_(mask, 0)

        s = s / math.sqrt(k.shape[1])
        dq = torch.einsum('nck,nqk->ncq', k, db_in) * s
        dk = torch.einsum('ncq,nqk->nck', q, db_in) * s

        # These are dummy derivatives since mask is a constant
        dmask = (max_neg_value - w) * db.clone() * s

        return dq, dk, dmask


class QKVMaskedAttention(nn.Module):
    """
    A module which performs QKV attention.
    Attention mask is accepted as input.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v, mask):
        r"""
        Apply QKV attention with attention mask.

        Args:
            q: an [N x d x n_seq1] of queries.
            k: an [N x d x n_seq2] of keys.
            v: an [N x d x n_seq2] of values.
            mask: Attention mask of size N x n_seq1 x n_seq2

        Returns: an [N x d x n_seq1] tensor after attention.
        """

        bs, width, length_q = q.shape
        _, _, length_k = k.shape

        assert width % self.n_heads == 0
        ch = width // self.n_heads

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length_q),
            (k * scale).view(bs * self.n_heads, ch, length_k),
        )  # More stable with f16 than dividing afterwards

        # Duplicate mask n_heads times
        # mask = mask.repeat_interleave(self.n_heads, dim=0)
        mask = mask.unsqueeze(0).repeat(self.n_heads, 1, 1, 1).transpose(0, 1).flatten(0, 1)
        assert mask.shape == weight.shape
        max_neg_value = -float('inf')
        weight = weight.masked_fill(~mask, max_neg_value)

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # When we use an arbitrary mask, there is a possibility that we get nans in softmax.
        # In this case, use nan_to_num to make it a non-nan number.
        # weight = weight.nan_to_num_()
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length_k))
        # We also return weight here for attention visualization.
        return a.reshape(bs, -1, length_q), weight

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVStableMaskedAttention(nn.Module):
    """
    A module which performs QKV attention.
    Attention mask is accepted as input.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, q, k, v, mask):
        r"""
        Apply QKV attention with attention mask.

        Args:
            q: an [N x d x n_seq1] of queries.
            k: an [N x d x n_seq2] of keys.
            v: an [N x d x n_seq2] of values.
            mask: Attention mask of size N x n_seq1 x n_seq2

        Returns: an [N x d x n_seq1] tensor after attention.
        """

        bs, width, length_q = q.shape
        _, _, length_k = k.shape

        assert width % self.n_heads == 0
        ch = width // self.n_heads

        q = q.view(bs * self.n_heads, ch, length_q)
        k = k.view(bs * self.n_heads, ch, length_k)

        # Forming attention mask
        # mask = mask.repeat_interleave(self.n_heads, dim=0)
        mask = mask.unsqueeze(0).repeat(self.n_heads, 1, 1, 1).transpose(0, 1).flatten(0, 1)

        weight = StableMaskedAttentionOp.apply(q, k, ~mask)

        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length_k))
        # We also return weight here for attention visualization.
        return a.reshape(bs, -1, length_q), weight

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    Taken from: https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep
