# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


from typing import Dict, List

import torch
from torch.utils.data.dataset import Dataset


class GPTRequestDataset(Dataset):
    """
    Args:
        requests: List of prompts
        tokenizer: model tokenizer
        tokens_to_generate: int value denoting amount of tokens model should generate
        compute_logprobs: bool value denoting if model should generate tokens or compute logprobs
    Returns:
        data: class object
            {'data': tokens, 'tokens_to_generate': tokens_to_generate, 'compute_logprobs': compute_logprobs}
            * data: List of token's ids in respect to prompts
            * tokens_to_generate: int value denoting amount of tokens model should generate
            * compute_logprobs: bool value denoting if model should generate tokens or compute logprobs
    """

    def __init__(self, requests: List, tokenizer, tokens_to_generate: int, compute_logprobs: bool) -> None:
        super().__init__()
        self.requests = requests
        self.tokenizer = tokenizer
        self.tokens_to_generate = tokens_to_generate
        self.compute_logprobs = compute_logprobs
        self.tokens = []
        self.prompt_tags = []

        # tokenize prompt
        for request in self.requests:
            if type(request) == dict:
                prompt_tag = request['prompt_tag']
                self.prompt_tags.append(prompt_tag)
                text = request['text']
            else:
                text = request

            self.tokens.append(torch.tensor(self.tokenizer.text_to_ids(text)))

        if self.prompt_tags:
            self.data = {
                'prompt_tags': self.prompt_tags,
                'data': self.tokens,
                'tokens_to_generate': self.tokens_to_generate,
                'compute_logprobs': self.compute_logprobs,
            }

        else:
            self.data = {
                'data': self.tokens,
                'tokens_to_generate': self.tokens_to_generate,
                'compute_logprobs': self.compute_logprobs,
            }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data


class T5RequestDataset(Dataset):
    def __init__(self, request: Dict, tokenizer) -> None:
        super().__init__()
        self.request = request
        self.tokenizer = tokenizer
        self.add_eos_to_encoder_input = self.request['add_eos_to_encoder_input']

        # tokenize prompt
        self.request['tokenized_prompt'] = ' '.join(self.tokenizer.text_to_tokens(request['prompt']))
        tokens = self.tokenizer.text_to_ids(request['prompt'])
        self.request['tokens'] = torch.tensor(tokens)
        self.mask_prompt(self.request['prompt'])

    def mask_prompt(self, sample):
        sample = sample.split()
        sentinel_idx = 0
        for i, word in enumerate(sample):
            if word == '<mask>':
                sample[i] = f'<extra_id_{sentinel_idx}>'
                sentinel_idx += 1
        sample = ' '.join(sample)
        sample = self.tokenizer.text_to_ids(sample)
        if self.add_eos_to_encoder_input:
            sample = sample + [self.tokenizer.eos_id]
        sample = torch.LongTensor(sample)
        self.request['masked_sample'] = sample

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.request
