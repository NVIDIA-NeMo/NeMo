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
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['BertModule']


class BertModule(NeuralModule, Exportable):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"last_hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

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
        m = re.match("^bert.*?\.", list(pretrained_dict.keys())[0])
        if m:
            prefix = m.group(0)
            pretrained_dict = {k[len(prefix) :]: v for k, v in pretrained_dict.items()}
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # starting with transformers 3.1.0, embeddings.position_ids is added to the model's state dict and could be
        # missing in checkpoints trained with older transformers version
        if 'embeddings.position_ids' in model_dict and 'embeddings.position_ids' not in pretrained_dict:
            pretrained_dict['embeddings.position_ids'] = model_dict['embeddings.position_ids']

        assert len(pretrained_dict) == len(model_dict)
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        logging.info(f"Weights for {type(self).__name__} restored from {restore_path}")

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        sz = (max_batch, max_dim)
        input_ids = torch.randint(low=0, high=max_dim - 1, size=sz, device=sample.device)
        token_type_ids = torch.randint(low=0, high=1, size=sz, device=sample.device)
        attention_mask = torch.randint(low=0, high=1, size=sz, device=sample.device)
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return tuple([input_dict])
