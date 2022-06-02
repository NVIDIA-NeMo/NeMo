# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""
import enum


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2
    retrieval_encoder = (
        3  # retrieval model encoder, it uses cross attention to be conditioned on the pre decoder output
    )
    retrieval_decoder = (
        4  # retrieval model decoder, it uses chunked cross attention to be conditioned on the retrieved information
    )
    decoder_pre_mlp = 5  # decoder that skips the computation after the self-attention
    retrieval_decoder_after_self_attn = 6  # retrieval decoder that skips the self-attention
