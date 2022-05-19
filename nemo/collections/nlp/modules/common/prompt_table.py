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

import enum
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from nemo.core.classes import Exportable, NeuralModule

try:
    from apex.transformer import tensor_parallel

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

__all__ = ['PromptTable', 'VirtualPromptSource', 'VirtualPromptStyle', 'VirtualPromptPlaceholderToken']


class VirtualPromptStyle(enum.Enum):
    P_TUNING = 'p-tuning'
    PROMPT_TUNING = 'prompt-tuning'
    INFERENCE = 'inference'


class VirtualPromptSource(enum.Enum):
    PROMPT_TABLE = 'prompt_table'
    PROMPT_ENCODER = 'prompt_encoder'


class VirtualPromptPlaceholderToken(enum.Enum):
    BASE = '<prompt_'
    END = '>'


class PromptTable(NeuralModule, Exportable):
    def __init__(self, existing_tasks, task_templates, task_id_num_to_name, hidden_size):
        super().__init__()

        self.task_templates = task_templates
        self.hidden_size = hidden_size
        self.prompt_table = torch.nn.ModuleDict()
        self.task_id_num_to_name = {}

        # Need to init prompt embeddings for each existing task before loading tuned weights
        if existing_tasks and existing_tasks[0] is not None:
            for taskname in existing_tasks:
                total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
                self.prompt_table[taskname] = PromptEmbedding(
                    init_from_prompt_text=False,
                    hidden_size=self.hidden_size,
                    total_virtual_tokens=total_virtual_tokens,
                )

        # Make sure tasknames and task id nums line up correctly in prompt table
        self.task_id_num_to_name = task_id_num_to_name

    def forward(self, task_id_num, input_ids=None):
        task_id_num = task_id_num.item()
        tasknames = self.task_id_num_to_name[task_id_num]
        return self.prompt_table[tasknames](input_ids)

    def remove_prompt(self, taskname):
        if taskname not in self.prompt_table:
            return

        # find the task_id_num assocaited with the tag to delete
        task_id_num = None
        for key, value in self.task_id_num_to_name.items():
            if value == taskname:
                task_id_num = key
                break

        del self.task_id_num_to_name[task_id_num]
        del self.prompt_table[taskname]

    def init_prompt_from_random(self, taskname, total_virtual_tokens):
        """Add new virtual prompt to be tuned.
           Intialize prompt weights using pytorch init method
        """
        # Initalize prompt embeddings from a pytorch random init method
        self.prompt_table[taskname] = PromptEmbedding(
            init_from_prompt_text=False, hidden_size=self.hidden_size, total_virtual_tokens=total_virtual_tokens,
        )

    def init_prompt_from_text(self, taskname, init_token_ids, word_embeddings, total_virtual_tokens):
        """Add new virtual prompt to be tuned.
           Intialize prompt weights from existing embeddings from specific vocab tokens.

        """
        # Trim or iterate until num_text_tokens matches total_virtual_tokens
        num_text_tokens = len(init_token_ids)

        if num_text_tokens > total_virtual_tokens:
            init_token_ids = init_token_ids[:total_virtual_tokens]
        elif num_text_tokens < total_virtual_tokens:
            num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
            init_token_ids = init_token_ids * num_reps

        # Set dictionary item keys and datatypes for broadcasting
        keys = ['text']
        datatype = torch.int64

        # Broadcast int ids across gpus for tensor parallel
        init_token_ids = init_token_ids[:total_virtual_tokens]
        init_token_ids = {'text': torch.tensor(init_token_ids, dtype=torch.int64)}
        init_token_ids_b = tensor_parallel.broadcast_data(keys, init_token_ids, datatype)
        init_token_ids = init_token_ids_b['text'].long()

        # Use a copy of token embedding weights to initalize the prompt embeddings
        word_embedding_weights = word_embeddings(init_token_ids).detach().clone()

        self.prompt_table[taskname] = PromptEmbedding(
            init_from_prompt_text=True,
            hidden_size=self.hidden_size,
            total_virtual_tokens=total_virtual_tokens,
            word_embedding_weights=word_embedding_weights,
        )

    def add_prompt_from_p_tuning_encoder(self, taskname, virtual_prompt_embeddings, total_virtual_tokens):
        """
        Add virtual prompts that have already been tuned using p-tuning. 
        """
        self.prompt_table[taskname] = PromptEmbedding(
            init_from_prompt_text=True,
            hidden_size=self.hidden_size,
            total_virtual_tokens=total_virtual_tokens,
            word_embedding_weights=virtual_prompt_embeddings,
        )


class PromptEmbedding(NeuralModule, Exportable):
    """Prompt embeddings

    Arugments:
        init_from_prompt_text: Whether to intialize prompt embeddings
                               from from certain lm embeddings
                               corresponding to a prompt string
        hidden_size: hidden size should match lm embedding size
        total_virtual_tokens: length of prompt initalized from torch init method
        word_embedding_weights: token embedding vectors for text init option
        init_method: pytorch init method
        prompt_embedding_dropout_prob: dropout probablity
    """

    def __init__(
        self,
        init_from_prompt_text,
        hidden_size,
        total_virtual_tokens,
        word_embedding_weights=None,
        init_method=init.xavier_normal_,
        prompt_embedding_dropout_prob=0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.total_virtual_tokens = total_virtual_tokens

        # Randomly init token and position embeddings
        self.prompt_embeddings = torch.nn.Embedding(self.total_virtual_tokens, self.hidden_size)
        init_method(self.prompt_embeddings.weight)

        # Set embedding weights to be embeddings from prompt tokens
        if init_from_prompt_text:
            self.prompt_embeddings.weight = nn.Parameter(word_embedding_weights)

        # Set fixed indicies for forward pass
        self.register_buffer('indices', torch.LongTensor(list(range(self.total_virtual_tokens))))
        self.embedding_dropout = torch.nn.Dropout(prompt_embedding_dropout_prob)

    def forward(self, input_ids=None):
        # Just get embeddings and dropout
        if input_ids is None:
            prompt_embeddings = self.prompt_embeddings(self.indices)
        else:
            prompt_embeddings = self.prompt_embeddings(input_ids)

        prompt_embeddings = self.embedding_dropout(prompt_embeddings)

        return prompt_embeddings
