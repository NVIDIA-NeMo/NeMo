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

import os
import re
from typing import Dict, Optional

import torch

from nemo.core.classes import NeuralModule
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import ChannelType, FloatType, IntType, MaskType, NeuralType, StringType, VoidType
from nemo.utils import logging

__all__ = ['GPTModule']


class GPTModule(NeuralModule, Exportable):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "labels": NeuralType(('B', 'T'), ChannelType(), optional=True),
            'past_key_values': [[NeuralType(None, StringType(), optional=True)]],
            'use_cache': NeuralType(None, VoidType(), optional=True),
            'position_ids': NeuralType(('B', 'T'), ChannelType(), optional=True),
            "return_dict": NeuralType(None, StringType(), optional=True),
            "output_attentions": NeuralType(None, StringType(), optional=True),
            "output_hidden_states": NeuralType(None, StringType(), optional=True),
            "max_length": NeuralType(None, IntType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'loss': NeuralType(None, FloatType(), optional=True),
            'hidden_states': NeuralType(('B', 'T', 'D'), ChannelType()),
        }

    def restore_weights(self, restore_path: str):
        """Restores module/model's weights"""
        logging.info(f"Restoring weights from {restore_path}")

        if not os.path.exists(restore_path):
            logging.warning(f'Path {restore_path} not found')
            return

        pretrained_dict = torch.load(restore_path)

        # backward compatibility with NeMo0.11
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = pretrained_dict["state_dict"]

        # remove prefix from pretrained dict
        m = re.match(r"^gpt.*?\.", list(pretrained_dict.keys())[0])
        if m:
            prefix = m.group(0)
            pretrained_dict = {k[len(prefix) :]: v for k, v in pretrained_dict.items()}
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # starting with transformers 3.1.0, embeddings.position_ids is added to the model's state dict and could be
        # missing in checkpoints trained with older transformers version
        if 'embeddings.position_ids' in model_dict and 'embeddings.position_ids' not in pretrained_dict:
            pretrained_dict['embeddings.position_ids'] = model_dict['embeddings.position_ids']

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        logging.info(f"Weights for {type(self).__name__} restored from {restore_path}")

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        input_ids = torch.randint(low=0, high=2048, size=(2, 16), device=sample.device)
        token_type_ids = torch.randint(low=0, high=1, size=(2, 16), device=sample.device)
        attention_mask = torch.randint(low=0, high=1, size=(2, 16), device=sample.device)
        return tuple([input_ids, token_type_ids, attention_mask])
