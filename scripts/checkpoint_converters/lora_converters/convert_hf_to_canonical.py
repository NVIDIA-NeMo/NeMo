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
Example usage of this script:
/checkpoints/bin/ is a folder containing the HF lora checkpoint (usually named adapter_model.bin)
and a HF lora config file (usually named adapter_config.json)
python scripts/checkpoint_converters/lora_converters/convert_hf_to_canonical.py \
    --hf_lora_path /checkpoints/bin/ \
    --output_path output_dir/converted_lora.nemo \
    --nemo_config model_config.yaml
"""

import json
import tempfile
from argparse import ArgumentParser
from typing import Dict

import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector

target_map = {
    "all": ["gate_proj", "o_proj", "up_proj", "down_proj", "k_proj", "q_proj", "v_proj"],
    "attention_qkv": ["k_proj", "q_proj", "v_proj"],
    "attention_dense": ["gate_proj", "o_proj", "up_proj"],
}


def map_target_modules_to_canonical(target_modules):
    tm = set(target_modules)
    for k, v in target_map.items():
        if tm == set(v):
            return [k]
    raise ValueError(f"Unknown target modules: {target_modules}")


def reformat_module_names_to_canonical(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_tensors = dict()
    for module_name, module_weight in tensors.items():
        # map linear_in and linear_out to lora_a/lora_b counterparts
        new_module_name = (
            module_name.replace("lora_A", "linear_in").replace("lora_B", "linear_out").replace("base_model.", "")
        )

        new_module_name = new_module_name.replace(".q_proj", ".adapter_layer.lora_unfused_kqv_adapter.q_adapter")
        new_module_name = new_module_name.replace(".k_proj", ".adapter_layer.lora_unfused_kqv_adapter.k_adapter")
        new_module_name = new_module_name.replace(".v_proj", ".adapter_layer.lora_unfused_kqv_adapter.v_adapter")
        new_module_name = new_module_name.replace(".o_proj", ".adapter_layer.lora_dense_attention_adapter")
        new_module_name = new_module_name.replace(".down_proj", ".adapter_layer.lora_4htoh_adapter")
        new_module_name = new_module_name.replace(
            ".gate_proj", ".adapter_layer.lora_unfused_hto4h_adapter.gate_adapter"
        )
        new_module_name = new_module_name.replace(".up_proj", ".adapter_layer.lora_unfused_hto4h_adapter.up_adapter")
        new_module_name = new_module_name.replace("self_attn", "self_attention")
        new_module_name = new_module_name.replace("model.model", "model.decoder")

        new_tensors[new_module_name] = module_weight
    return new_tensors


def convert_lora(lora_hf_path, save_path, lora_yaml):
    config_file = f"{lora_hf_path}/adapter_config.json"
    model_file = f"{lora_hf_path}/adapter_model.bin"
    hf_lora_config = json.loads(open(config_file).read())
    model = torch.load(model_file)
    # TODO: currently suport tp=1
    lora_state_dict = reformat_module_names_to_canonical(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        nemo_lora_config = OmegaConf.load(lora_yaml)
        with open_dict(nemo_lora_config):
            nemo_lora_config.peft.lora_tuning.variant = "canonical"
            nemo_lora_config.peft.lora_tuning.adapter_dim = hf_lora_config["r"]
            nemo_lora_config.peft.lora_tuning.alpha = hf_lora_config["lora_alpha"]
            nemo_lora_config.peft.lora_tuning.target_modules = map_target_modules_to_canonical(
                hf_lora_config["target_modules"]
            )

        with open(f"{tmpdir}/model_config.yaml", "w") as f:
            OmegaConf.save(nemo_lora_config, f)
        torch.save(lora_state_dict, f"{tmpdir}/model_weights.ckpt")
        NLPSaveRestoreConnector._make_nemo_file_from_folder(save_path, tmpdir)

    return True


def fix_for_O2(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "model.module." not in k:
            new_state_dict[k.replace('model.', 'model.module.')] = v
    return new_state_dict


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--hf_lora_path",
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
    parser.add_argument("--nemo_config", type=str, help="a model_config.yaml file which this script will update.")
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    convert_lora(args.hf_lora_path, args.output_path, args.nemo_config)
