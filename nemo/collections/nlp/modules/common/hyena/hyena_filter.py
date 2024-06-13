import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn

from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

# Code mostly taken from:
# https://github.com/HazyResearch/safari/blob/flashfftconv/src/models/sequence/hyena.py


@dataclass
class HyenaFilterSubmodules:
    positional_embedding: Union[ModuleSpec, type] = IdentityOp
    linear: Union[ModuleSpec, type] = IdentityOp
    activation: Union[ModuleSpec, type] = IdentityOp
    modulation: Union[ModuleSpec, type] = IdentityOp


def register(module: nn.Module, name: str, tensor: torch.Tensor, learnable: bool):
    if learnable:
        module.register_parameter(name, nn.Parameter(tensor))
    else:
        module.register_buffer(name, tensor)


class Sin(nn.Module):
    def __init__(self, dim: int, freq: float = 10, train_freq: bool = True):
        """
        Sinusoidal activation function with (optionally learned) per-channel frequency
        """
        super().__init__()
        self.freq = nn.Parameter(freq * torch.ones(1, dim)) if train_freq else freq * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        seq_len: int,
        learn_pos_emb_z: bool = True,
    ):
        """Complex exponential positional embeddings for Hyena filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filters is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        register(self, "z", z, learnable=learn_pos_emb_z)
        register(self, "t", t, learnable=False)

    def forward(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    def __init__(
        self,
        d_model: int,
        modulate: bool = True,
        learn_modulation: bool = False,
        fast_decay_pct: float = 0.3,
        slow_decay_pct: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.0,
    ):
        """
        Exponential decay modulation with (optionally learned) per-channel decay rate
        """
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        register(self, "deltas", deltas, learnable=learn_modulation)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_len: int = 1024,
        emb_dim: int = 3,
        learn_pos_emb_z: bool = True,
        mlp_width: int = 64,
        sine_freq: int = 1,
        num_inner_mlps: int = 2,
        normalized: bool = False,
        submodules: HyenaFilterSubmodules = None,
        **modulation_kwargs,
    ):
        """
        Implicit long filter with modulation.

        Args:
            d_model (int): number of channels in the input
            emb_dim (int): dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            mlp_width (int): Width of the MLP parametrizing the implicit filter. Defaults to 64
            seq_len (int): length of input sequence
            learn_pos_emb_z (bool): whether the positional embeddings are learned
            sine_freq (int): frequency of periodic activations
            num_inner_mlps (int): number of inner linear layers inside filter MLP
            normalized (bool): whether to apply normalization after modulation
        """
        super().__init__()

        if submodules is None:
            submodules = HyenaFilterSubmodules(
                positional_embedding=PositionalEmbedding,
                linear=nn.Linear,
                activation=Sin,
                modulation=ExponentialModulation,
            )

        self.d_model = d_model
        self.mlp_width = mlp_width

        act = build_module(submodules.activation, dim=mlp_width, freq=sine_freq)
        self.emb_dim = emb_dim
        if emb_dim % 2 == 0 or emb_dim < 3:
            raise ValueError("emb_dim must be odd and greater or equal to 3 (time, sine and cosine)")
        self.seq_len = seq_len

        self.pos_emb = build_module(submodules.positional_embedding, emb_dim, seq_len, learn_pos_emb_z)

        # uses a variable number of inner linear layers
        self.mlp = nn.Sequential(
            build_module(submodules.linear, emb_dim, mlp_width),
            act,
        )
        for i in range(num_inner_mlps):
            self.mlp.append(build_module(submodules.linear, mlp_width, mlp_width))
            self.mlp.append(act)
        # final linear layer
        self.mlp.append(build_module(submodules.linear, mlp_width, d_model, bias=False))

        self.modulation = build_module(submodules.modulation, d_model, **modulation_kwargs)

        self.normalized = normalized

    def forward(self, L):
        z, t = self.pos_emb(L)
        h = self.mlp(z)

        h = self.modulation(t, h)

        if self.normalized:
            h = h / torch.norm(h, dim=-1, p=1, keepdim=True)

        return h
