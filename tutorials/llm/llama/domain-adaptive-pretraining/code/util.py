# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch


def check_directory_exists(directory):
    if os.path.isdir(directory):
        print(f"Directory '{directory}' exists")
    else:
        raise FileNotFoundError(f"The directory '{directory}' does not exist. Please create it.")


def load_weights(load_path, save_path):
    """
    This function loads llama2 weights (hugging face) and converts it to a Dict format suitable for NeMo

    Args:
            load_path (str): Path to llama2 weights downlaoded from hugging face
            save_path (str): Path to save modified dictionary containing the weights.

    Returns:
            None

    """
    model_type = "llama2"
    for i in range(8):
        state_dict = torch.load(f"{load_path}/consolidated.0{i}.pth")
        batch_dict = {}
        if model_type == "llama2":
            batch_dict['word_embeddings'] = state_dict['tok_embeddings.weight']
            batch_dict['output_layer'] = state_dict['output.weight']
        else:
            batch_dict['word_embeddings'] = state_dict['model']['embedding.word_embeddings.weight']  # embedding layer
            batch_dict['output_layer'] = state_dict['model']['output_layer.weight']  # output layer
        torch.save(batch_dict, f'{save_path}/embedding_{i}.pt')


def merge_embed(old_embd_path, new_embd_path, save_path):
    "Function to merge embeddings and convert back to hugging face format"
    model_type = "llama2"
    for i in range(8):
        state_dict = torch.load(f"{old_embd_path}/consolidated.0{i}.pth")
        batch_dict = torch.load(f'{new_embd_path}/embedding_{i}.pt')
        if model_type == "llama2":
            state_dict['output.weight'] = batch_dict['output_layer']
            state_dict['tok_embeddings.weight'] = batch_dict['word_embeddings']
        else:
            state_dict['tok_embeddings.weight'] = batch_dict['word_embeddings']
            state_dict['output.weight'] = batch_dict['output_layer']
        check_directory_exists(save_path)
        torch.save(state_dict, f"{save_path}/consolidated.0{i}.pth")
