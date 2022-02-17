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

import json
import os

import pytest

from nemo.collections.nlp.data.language_modeling.megatron.gpt_prompt_tuning_dataset import GPTPromptTuningDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.core import Dataset


def get_prompt_tuning_dataset(tokenizer, dataset_path, num_prompt_tokens):
    dataset = GPTPromptTuningDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        prompt_table=[('A', 1)],
        num_prompt_tokens=num_prompt_tokens,
        micro_batch_size=4,
        max_seq_length=512,
        min_seq_length=1,
    )

    return dataset


def create_temp_dataset():
    example_dataset = [{'prompt_tag': 'A', 'text': 'Test sentence one, Answer: ', 'answer': 'test'} for i in range(24)]
    temp_file_name = 'temp_dataset_file.jsonl'

    with open(temp_file_name, 'w') as temp:
        for example in example_dataset:
            temp.write(json.dumps(example) + '\n')

    return temp_file_name


class TestMegatronGPTPromptTuningDataset:
    @pytest.mark.skip('disabled until prompt tuning is fixed for pipeline')
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_init_prompt_tuning_dataset(self):
        tokenizer = get_nmt_tokenizer(library='huggingface', model_name='gpt2')
        dataset_path = create_temp_dataset()
        num_prompt_tokens = 10

        dataset = get_prompt_tuning_dataset(tokenizer, dataset_path, num_prompt_tokens)

        print(type(dataset))

        assert isinstance(dataset, Dataset)

        os.remove(dataset_path)

    @pytest.mark.skip('disabled until prompt tuning is fixed for pipeline')
    @pytest.mark.run_only_on('GPU')
    @pytest.mark.unit
    def test_prompt_tuning_dataset_collate_fn(self):
        tokenizer = get_nmt_tokenizer(library='megatron', model_name='GPT2BPETokenizer')
        dataset_path = create_temp_dataset()
        num_prompt_tokens = 10

        dataset = get_prompt_tuning_dataset(tokenizer, dataset_path, num_prompt_tokens)
        batch = [dataset[i] for i in range(8)]
        batch = dataset.collate_fn(batch)

        assert len(batch) == 6

        tokens, labels, loss_mask, attention_mask, text_position_ids, prompt_tags = batch

        assert len(tokens) == len(loss_mask) == len(attention_mask) == len(text_position_ids)
        assert len(tokens) == len(prompt_tags)
        assert len(labels) == len(tokens)
        assert len(labels[0]) == len(loss_mask[0])
        assert len(tokens[0]) + (num_prompt_tokens) == attention_mask[0].size()[-1]

        os.remove(dataset_path)


if __name__ == "__main__":
    t = TestMegatronGPTPromptTuningDataset()
    t.test_init_prompt_tuning_dataset()
    t.test_prompt_tuning_dataset_collate_fn()
    print('-' * 50 + '\nALL PROMPT TUNING UNIT TESTS PASS!\n' + '-' * 50)
