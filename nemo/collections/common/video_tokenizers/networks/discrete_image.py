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
"""The network definition for discrete image tokenization with VQ, LFQ, FSQ or ResidualFSQ."""
from collections import OrderedDict, namedtuple

import torch
from torch import nn

from nemo.collections.common.video_tokenizers.modules import DecoderType, DiscreteQuantizer, EncoderType
from nemo.collections.common.video_tokenizers.modules.quantizers import InvQuantizerJit

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "quant_info"])


class DiscreteImageTokenizer(nn.Module):
    def __init__(self, z_channels: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "DiscreteImageTokenizer")
        self.embedding_dim = embedding_dim

        encoder_name = kwargs.get("encoder", EncoderType.Default.name)
        self.encoder = EncoderType[encoder_name].value(z_channels=z_channels, **kwargs)

        decoder_name = kwargs.get("decoder", DecoderType.Default.name)
        self.decoder = DecoderType[decoder_name].value(z_channels=z_channels, **kwargs)
        self.quant_conv = nn.Conv2d(z_channels, embedding_dim, 1)
        self.post_quant_conv = nn.Conv2d(embedding_dim, z_channels, 1)

        quantizer_name = kwargs.get("quantizer", DiscreteQuantizer.RESFSQ.name)
        if quantizer_name == DiscreteQuantizer.VQ.name:
            assert "num_embeddings" in kwargs, f"`num_embeddings` must be provided for {quantizer_name}."
            kwargs.update(dict(embedding_dim=embedding_dim))
        elif quantizer_name == DiscreteQuantizer.LFQ.name:
            assert "codebook_size" in kwargs, f"`codebook_size` must be provided for {quantizer_name}."
            assert "codebook_dim" in kwargs, f"`codebook_dim` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.FSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.RESFSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}.name."
            assert "num_quantizers" in kwargs, f"`num_quantizers` must be provided for {quantizer_name}."
        self.quantizer = DiscreteQuantizer[quantizer_name].value(**kwargs)

    def to(self, *args, **kwargs):
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(DiscreteImageTokenizer, self).to(*args, **kwargs)

    def encoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder),
                    ("quant_conv", self.quant_conv),
                    ("quantizer", self.quantizer),
                ]
            )
        )

    def decoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("inv_quant", InvQuantizerJit(self.quantizer)),
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
                ]
            )
        )

    def last_decoder_layer(self):
        return self.decoder.conv_out

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantizer(h)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def decode_code(self, code_b):
        quant_b = self.quantizer.indices_to_codes(code_b)
        quant_b = self.post_quant_conv(quant_b)
        return self.decoder(quant_b)

    def forward(self, input):
        quant_info, quant_codes, quant_loss = self.encode(input)
        reconstructions = self.decode(quant_codes)
        if self.training:
            return dict(
                reconstructions=reconstructions,
                quant_loss=quant_loss,
                quant_info=quant_info,
            )
        return NetworkEval(
            reconstructions=reconstructions,
            quant_loss=quant_loss,
            quant_info=quant_info,
        )
