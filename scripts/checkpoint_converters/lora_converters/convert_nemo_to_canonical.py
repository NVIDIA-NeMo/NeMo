#!/usr/bin/env
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""
Convert nemo style (fused) lora checkpoint to canonical (unfused) lora checkpoint.
Currently supports TP=PP=1 only.

Example usage:
python scripts/checkpoint_converters/lora_converters/convert_nemo_to_canonical.py \
    --lora_path nemo_style_lora_model.nemo \
    --output_path ./canonical_style_lora_model.nemo 

"""
import tempfile
from argparse import ArgumentParser
from typing import Dict

import torch
from omegaconf import OmegaConf, open_dict
from scripts.nlp_language_modeling.merge_lora_weights.merge import replace_number_add_offset

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector


def rename_keys(key):
    new_keys = []
    if "lora_kqv_adapter" in key:
        new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.q_adapter."))
        new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.k_adapter."))
        new_keys.append(key.replace(".lora_kqv_adapter.", ".lora_unfused_kqv_adapter.v_adapter."))
    elif "lora_hto4h_adapter" in key:
        new_keys.append(key.replace(".lora_hto4h_adapter.", ".lora_unfused_hto4h_adapter.gate_adapter."))
        new_keys.append(key.replace(".lora_hto4h_adapter.", ".lora_unfused_hto4h_adapter.up_adapter."))
    return new_keys


def reformat_module_names_to_hf(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_tensors = dict()
    for module_name, module_weight in tensors.items():
        # map linear_in and linear_out to lora_a/lora_b counterparts
        new_module_name = "base_model." + module_name.replace("linear_in", "lora_A").replace("linear_out", "lora_B")

        # map target modules to their vLLM/HF counterparts
        new_module_name = new_module_name.replace("q_adapter", "q_proj")
        new_module_name = new_module_name.replace("k_adapter", "k_proj")
        new_module_name = new_module_name.replace("v_adapter", "v_proj")
        new_module_name = new_module_name.replace("lora_dense_attention_adapter", "o_proj")
        new_module_name = new_module_name.replace("lora_4htoh_adapter", "down_proj")
        new_module_name = new_module_name.replace("gate_adapter", "gate_proj")
        new_module_name = new_module_name.replace("up_adapter", "up_proj")

        # map other parts of the module names to fit vLLM/huggingface
        new_module_name = new_module_name.replace(".adapter_layer", "")
        new_module_name = new_module_name.replace(".lora_unfused_kqv_proj", "")
        new_module_name = new_module_name.replace(".lora_unfused_hto4h_adapter", "")
        new_module_name = new_module_name.replace("self_attention", "self_attn")
        new_module_name = new_module_name.replace("decoder", "model")

        new_tensors[new_module_name] = module_weight
    return new_tensors


def convert_hto4h(lora_weights, lora_config):
    assert len(lora_weights) == 1, "Only single TP supported for now"
    keys_to_update = []
    for key in lora_weights[0].keys():
        if "lora_hto4h_adapter" in key:
            keys_to_update.append(key)

    for key in keys_to_update:
        if "linear_in" in key:
            for new_key in rename_keys(key):
                lora_weights[0][new_key] = lora_weights[0][key]
                print(new_key, lora_weights[0][new_key].shape)
        elif "linear_out" in key:
            for idx, new_key in enumerate(rename_keys(key)):
                orginal_shape = lora_weights[0][key].shape[0]
                lora_weights[0][new_key] = lora_weights[0][key][
                    idx * (orginal_shape // 2) : (idx + 1) * (orginal_shape // 2)
                ]
                print(new_key, lora_weights[0][new_key].shape)

        lora_weights[0].pop(key)
    return lora_weights


def convert_qkv(lora_weights, lora_model_cfg):
    assert len(lora_weights) == 1, "Only single TP supported for now"
    if (
        lora_model_cfg.get("num_query_groups", lora_model_cfg.num_attention_heads)
        != lora_model_cfg.num_attention_heads
    ):
        kv_channels = int(lora_model_cfg.hidden_size / lora_model_cfg.num_attention_heads)
        kv_size = int(lora_model_cfg.num_query_groups * kv_channels)
    else:
        kv_size = int(lora_model_cfg.hidden_size)
    q_size = lora_model_cfg.hidden_size
    k_size, v_size = kv_size, kv_size

    keys_to_update = []
    for key in lora_weights[0].keys():
        if "lora_kqv_adapter" in key:
            keys_to_update.append(key)

    for key in keys_to_update:
        if "linear_in" in key:
            for new_key in rename_keys(key):
                lora_weights[0][new_key] = lora_weights[0][key]
                print(new_key, lora_weights[0][new_key].shape)
        elif "linear_out" in key:
            srt = 0
            for new_key, size in zip(rename_keys(key), [q_size, k_size, v_size]):
                lora_weights[0][new_key] = lora_weights[0][key][srt : srt + size]
                print(new_key, lora_weights[0][new_key].shape)
                srt = srt + size

        lora_weights[0].pop(key)
    return lora_weights


def convert_lora(lora_nemo, save_path, hf_format=False):
    with tempfile.TemporaryDirectory() as tmpdir:
        NLPSaveRestoreConnector._unpack_nemo_file(lora_nemo, tmpdir)
        config_file = f"{tmpdir}/model_config.yaml"
        lora_config = OmegaConf.load(config_file)
        tp_size = lora_config.tensor_model_parallel_size
        pp_size = lora_config.pipeline_model_parallel_size

        lora_state_dict = [{}] * tp_size

        for pp in range(pp_size):
            for tp in range(tp_size):
                if tp_size == 1:
                    ckpt_file = f"{tmpdir}/model_weights.ckpt"
                elif pp_size == 1:
                    ckpt_file = f"{tmpdir}/mp_rank_{tp:02d}/model_weights.ckpt"
                else:
                    ckpt_file = f"{tmpdir}/tp_rank_{tp:02d}_pp_rank_{pp:03d}/model_weights.ckpt"

                l = torch.load(ckpt_file, map_location=torch.device('cpu'))
                if pp == 0:
                    lora_state_dict[tp] = l
                else:
                    # calculate layer offset
                    layer_offset = lora_config.num_layers // pp_size * pp
                    for key, value in l.items():
                        new_key = replace_number_add_offset(key, layer_offset)
                        lora_state_dict[tp][new_key] = value

        with open_dict(lora_config):
            lora_config.peft.lora_tuning.variant = "canonical"
        with open(f"{tmpdir}/model_config.yaml", "w") as f:
            OmegaConf.save(lora_config, f)
        lora_state_dict = convert_qkv(lora_state_dict, lora_config)
        lora_state_dict = convert_hto4h(lora_state_dict, lora_config)
        # TODO: currently suport tp=1
        lora_state_dict = lora_state_dict[0]
        if hf_format:
            lora_state_dict = reformat_module_names_to_hf(lora_state_dict)
            torch.save(lora_state_dict, f"{save_path}/model_weights_hf_formatted.pt")
        else:
            torch.save(lora_state_dict, f"{tmpdir}/model_weights.ckpt")
            NLPSaveRestoreConnector._make_nemo_file_from_folder(save_path, tmpdir)

    return lora_state_dict, lora_config


def fix_for_O2(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "model.module." not in k:
            new_state_dict[k.replace('model.', 'model.module.')] = v
    return new_state_dict


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        required=True,
        help="Path to NeMo style (fused) lora checkpoint in .nemo file format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the canonical (unfused) lora .nemo file.",
    )
    parser.add_argument("--hf_format", action='store_true', help="saves tensors in huggingface naming format.")
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    convert_lora(args.lora_path, args.output_path, args.hf_format)
