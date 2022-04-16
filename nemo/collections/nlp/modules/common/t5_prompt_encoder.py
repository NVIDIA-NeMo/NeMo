# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List, Optional

import torch
from torch import nn

from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import ChannelType, NeuralType

__all__ = ['PromptEncoder']


class PromptEncoder(NeuralModule, Exportable):
    """
    The Prompt Encoder network that is used to generate the virtual token embeddings
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "enc_taskname": NeuralType(('B', 'T', 'C'), ChannelType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"output_embeds": NeuralType(('B', 'T', 'C'), ChannelType())}

    def __init__(self, template: List[int], hidden_size: int, lstm_dropout: float, num_layers: int):
        """
        Initializes the PromptEncoder module.
        Args:
            template: the template sizes of the vitural tokens for different clozes
            hidden_size: hidden dimension
            lstm_dropout: the dropout used for the LSTM
            num_layers: number of layers used in the LSTM
        """
        super().__init__()
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [1] * sum(self.cloze_length)
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool()
        self.register_buffer('seq_indices', torch.LongTensor(list(range(len(self.cloze_mask)))))

        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask), self.hidden_size)
        # LSTM
        self.lstm_head = torch.nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // 2,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size)
        )

    @typecheck()
    def forward(self, enc_taskname) -> torch.Tensor:
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        if enc_taskname is not None:
            bz, task_seq, _ = enc_taskname.shape
            _, seq, emb = input_embeds.shape
            input_embeds = input_embeds.expand(bz, seq, emb).clone()
            length = min(task_seq, seq)
            input_embeds[:, 0:length, :] = enc_taskname[:, 0:length, :]
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        return output_embeds
