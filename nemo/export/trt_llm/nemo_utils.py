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


import argparse
import copy
import csv
import datetime
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorrt_llm
from tensorrt_llm import str_dtype_to_trt
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import MoeConfig
from tensorrt_llm.models.modeling_utils import PretrainedConfig
from transformers import AutoTokenizer, LlamaConfig, PreTrainedTokenizer

from nemo.export.tarutils import TarPath
from nemo.export.trt_llm.decoder import DECODER_MODEL_TYPE
from nemo.export.trt_llm.model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    LINEAR_COLUMN,
    DecoderLayerConfig,
    EmbeddingConfig,
    LayernormConfig,
    LinearConfig,
    ModelConfig,
)
from nemo.export.trt_llm.nemo.nemo import UnpackedNemoCheckpointDir
from nemo.export.trt_llm.nemo.nemo_ckpt_convert import build_tokenizer, convert_dist_checkpoint, convert_nemo_model
from nemo.export.trt_llm.tensor_utils import get_tensor_from_dict, get_tensor_parallel_group, split

LOGGER = logging.getLogger("NeMo")


def _nemo_llm_decode(
    in_file: str,
    out_dir: str,
    tensor_parallelism: int = 1,
    processes: int = 1,
    storage_type: str = "bfloat16",
    load_checkpoints_on_gpu: bool = False,
    decoder_type: str = "gptnext",
    use_parallel_embedding: bool = False,
    save_nemo_model_config: bool = False,
) -> Tuple[Dict[str, np.ndarray], PretrainedConfig, PreTrainedTokenizer]:
    """Decodes the NEMO file and returns the weights dict, llm config and tokenizer."""
    args = argparse.Namespace()
    args.out_dir = out_dir
    args.tensor_parallelism = tensor_parallelism
    args.processes = processes
    args.storage_type = storage_type
    args.load_checkpoints_on_gpu = load_checkpoints_on_gpu
    args.verbose = False
    args.decoder_type = decoder_type
    args.use_parallel_embedding = use_parallel_embedding

    if not os.path.exists(in_file):
        LOGGER.error("%s does not exist", in_file)
        sys.exit(1)

    if os.path.isdir(in_file):
        nemo_dir = Path(in_file)
    else:
        nemo_dir = TarPath(in_file)

    try:
        unpacked_checkpoint_dir = UnpackedNemoCheckpointDir(
            nemo_dir, load_checkpoints_to_cpu=not args.load_checkpoints_on_gpu
        )

        start_time = datetime.datetime.now()
        dist_ckpt_folder = nemo_dir / "model_weights"

        if dist_ckpt_folder.exists():
            weights_dict, llm_config, tokenizer = convert_dist_checkpoint(unpacked_checkpoint_dir, args)
        else:
            raise Exception(
                "Not a supported nemo file format. " "Only distributed mcore nemo checkpoints are support."
            )

        LOGGER.info("Spent %s (h:m:s) to convert the model", datetime.datetime.now() - start_time)

        if save_nemo_model_config:
            # Copy the config file without using shutil.copy(...) because input may be a TarPath
            with (unpacked_checkpoint_dir._checkpoints_dir / "model_config.yaml").open("rb") as infile:
                with open(os.path.join(args.out_dir, "model_config.yaml"), "wb") as outfile:
                    outfile.write(infile.read())
    finally:
        if isinstance(nemo_dir, TarPath):
            nemo_dir.tarobject.close()

    return weights_dict, llm_config, tokenizer


def get_tokenzier(tokenizer_dir_or_path: Path) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NEMO weights dir."""
    if os.path.isdir(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer")):
        return AutoTokenizer.from_pretrained(os.path.join(tokenizer_dir_or_path, "huggingface_tokenizer"))

    model_path = tokenizer_dir_or_path / "tokenizer.model" if tokenizer_dir_or_path.is_dir() else tokenizer_dir_or_path
    tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
    return build_tokenizer(tokenizer_config)


def nemo_llm_to_model_config(
    in_file: str,
    decoder_type: str,
    nemo_export_dir: Union[str, Path],
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    save_nemo_model_config: bool = False,
) -> Tuple[List[ModelConfig], PreTrainedTokenizer]:
    """Converts the NEMO file and construct the `ModelConfig` before tensorrt_llm deployment."""
    dtype_str = dtype

    weights_dict, llm_model_config, tokenizer = _nemo_llm_decode(
        in_file=in_file,
        out_dir=nemo_export_dir,
        tensor_parallelism=tensor_parallel_size,
        processes=1,
        storage_type=dtype_str,
        load_checkpoints_on_gpu=False,
        decoder_type=decoder_type,
        save_nemo_model_config=save_nemo_model_config,
    )

    world_size = tensor_parallel_size * pipeline_parallel_size
    model_config_template = ModelConfig()
    model_config_template.dtype = dtype_str

    str_dtype_to_trt(dtype_str)

    model_configs = []
    for i in range(world_size):

        model_configs.append(copy.deepcopy(model_config_template))

        model_configs[i].vocab_embedding = EmbeddingConfig(weight=get_tensor_from_dict(weights_dict, "wte"))

        model_configs[i].positional_embedding = EmbeddingConfig(weight=get_tensor_from_dict(weights_dict, "wpe"))

        model_configs[i].final_layernorm = LayernormConfig(
            weight=get_tensor_from_dict(weights_dict, "final_layernorm.weight"),
            bias=get_tensor_from_dict(weights_dict, "final_layernorm.bias"),
        )
        model_configs[i].final_layernorm.layernorm_type = (
            LAYERNORM_RMS if isinstance(llm_model_config, LlamaConfig) else LAYERNORM_DEFAULT
        )
        model_configs[i].mapping = tensorrt_llm.Mapping(
            world_size=world_size, rank=i, tp_size=tensor_parallel_size, pp_size=pipeline_parallel_size
        )

    for i in range(llm_model_config.n_layer):
        for j in range(world_size):
            model_configs[j].layers.append(
                DecoderLayerConfig.from_nemo(
                    weights_dict=weights_dict,
                    llm_config=llm_model_config,
                    decoder_type=decoder_type,
                    layer_id=i,
                    rank=model_configs[j].mapping.tp_rank,
                    is_mcore=llm_model_config.is_mcore,
                )
            )

    lm_head_weight = get_tensor_from_dict(weights_dict, "lm_head.weight")

    if model_configs[0].vocab_size_padded != model_configs[0].vocab_size:
        pad_width = model_configs[0].vocab_size_padded - model_configs[0].vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0)

    for i in range(world_size):
        model_configs[i].lm_head = LinearConfig(linear_type=LINEAR_COLUMN)
        model_configs[i].lm_head.weight = np.ascontiguousarray(
            split(lm_head_weight, model_configs[i].mapping.tp_size, model_configs[i].mapping.tp_rank)
        )

    return model_configs, tokenizer


def to_word_list_format(
    word_dict: List[List[str]],
    tokenizer=None,
    ref_str="<extra_id_1>",
):
    '''
    format of word_dict
        len(word_dict) should be same to batch_size
        word_dict[i] means the words for batch i
        len(word_dict[i]) must be 1, which means it only contains 1 string
        This string can contains several sentences and split by ",".
        For example, if word_dict[2] = " I am happy, I am sad", then this function will return
        the ids for two short sentences " I am happy" and " I am sad".
    '''
    assert tokenizer is not None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    # The encoding of a single word can't always be trusted. See
    #   https://github.com/NVIDIA/NeMo/blob/bb575b72fd0be51ae10cc77d9f89ddb9e9d3b96d/nemo/collections/nlp/modules/common/text_generation_strategy.py#L229
    ids_ref = tokenizer.encode(ref_str)
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(f"{ref_str}{word}")
            if ids[0 : len(ids_ref)] == ids_ref:
                # It worked! We can obtain the token(s) associated to `word` by stripping the prefix tokens.
                ids = ids[len(ids_ref) :]
            else:
                # Unfortunately the prefix was merged with `word`. We could try with a different prefix, but
                # for now we just use the basic encoding since this should be a very rare edge case.
                ids = tokenizer.encode(word)
                logging.warning(f"The encoding of word '{word}' into tokens {ids} might be incorrect")

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


def nemo_llm_model_to_model_config(
    nemo_model: str,
    decoder_type: str,
    nemo_model_config: str,
    dtype_str: str = "float32",
) -> Tuple[List[ModelConfig], PreTrainedTokenizer]:
    """Converts the NEMO model object and construct the `ModelConfig` before tensorrt_llm deployment."""
    from megatron.core import parallel_state

    assert nemo_model_config is not None, "gpt_model_config must be provided when in is a nemo model"

    weights_dict, llm_model_config = convert_nemo_model(nemo_model, nemo_model_config, dtype_str, decoder_type)
    is_mcore = nemo_model_config.get("mcore_gpt", False)
    llm_model_config.is_mcore = is_mcore

    model_config = ModelConfig()
    model_config.use_prompt_tuning = False
    model_config.dtype = dtype_str
    model_config.use_parallel_embedding = True
    str_dtype_to_trt(dtype_str)

    model_config.vocab_embedding = EmbeddingConfig(weight=get_tensor_from_dict(weights_dict, "wte"), is_local=True)

    model_config.positional_embedding = EmbeddingConfig(
        weight=get_tensor_from_dict(weights_dict, "wpe"), is_local=True
    )

    model_config.final_layernorm = LayernormConfig(
        weight=get_tensor_from_dict(weights_dict, "final_layernorm.weight"),
        bias=get_tensor_from_dict(weights_dict, "final_layernorm.bias"),
    )
    model_config.final_layernorm.layernorm_type = (
        LAYERNORM_RMS if isinstance(llm_model_config, LlamaConfig) else LAYERNORM_DEFAULT
    )

    tensor_parallel_size = nemo_model_config.tensor_model_parallel_size
    pipeline_parallel_size = 1
    world_size = tensor_parallel_size * pipeline_parallel_size

    # hack since tensorrt_llm doesnt support DP natively so init all ranks with DP=1
    model_config.mapping = tensorrt_llm.Mapping(
        world_size=tensor_parallel_size * pipeline_parallel_size,
        rank=tensorrt_llm.mpi_rank() % world_size,
        tp_size=tensor_parallel_size,
        pp_size=pipeline_parallel_size,
    )
    model_config.mapping.rank = tensorrt_llm.mpi_rank()
    model_config.mapping.tp_group = get_tensor_parallel_group(tensor_parallel_size)

    LOGGER.info(
        f'''Resharing: Rank {tensorrt_llm.mpi_rank()} mapping:
        tp_rank  {parallel_state.get_tensor_model_parallel_rank()} -> {model_config.mapping.tp_rank},
        pp_rank  {parallel_state.get_pipeline_model_parallel_rank()} -> {model_config.mapping.pp_rank},
        tp_group {model_config.mapping.tp_group}'''
    )

    for i in range(llm_model_config.n_layer):
        model_config.layers.append(
            DecoderLayerConfig.from_nemo(
                weights_dict=weights_dict,
                llm_config=llm_model_config,
                decoder_type=decoder_type,
                layer_id=i,
                rank=model_config.mapping.tp_rank,
                is_mcore=llm_model_config.is_mcore,
            )
        )
    lm_head_weight = get_tensor_from_dict(weights_dict, "lm_head.weight")

    assert model_config.vocab_size_padded == model_config.vocab_size

    model_config.lm_head = LinearConfig(linear_type=LINEAR_COLUMN)
    model_config.lm_head.weight = lm_head_weight

    return [model_config]


def nemo_to_trtllm_config(
    in_file: str,
    decoder_type: str,
    nemo_export_dir: Union[str, Path],
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    use_parallel_embedding: bool = False,
    save_nemo_model_config: bool = False,
) -> Tuple[List[Dict], List[PretrainedConfig], PreTrainedTokenizer]:
    """Converts the NEMO file and construct the `PretrainedConfig` before tensorrt_llm deployment."""
    dtype_str = dtype

    weights_dict, nemo_model_config, tokenizer = _nemo_llm_decode(
        in_file=in_file,
        out_dir=nemo_export_dir,
        tensor_parallelism=tensor_parallel_size,
        processes=1,
        storage_type=dtype_str,
        use_parallel_embedding=use_parallel_embedding,
        load_checkpoints_on_gpu=False,
        decoder_type=decoder_type,
        save_nemo_model_config=save_nemo_model_config,
    )

    world_size = tensor_parallel_size * pipeline_parallel_size

    lm_head_weight = weights_dict["lm_head.weight"]

    vocab_size = weights_dict["transformer.vocab_embedding.weight"].shape[0]
    vocab_size_padded = pad_vocab_size(vocab_size, tensor_parallel_size)

    if vocab_size_padded != vocab_size:
        pad_width = vocab_size_padded - vocab_size
        lm_head_weight = np.pad(lm_head_weight, ((0, pad_width), (0, 0)), "constant", constant_values=0)

    hidden_act = nemo_model_config.get('activation')
    hidden_act = (
        hidden_act.split("-")[-1] if nemo_model_config.get('num_moe_experts', 0) else non_gated_version(hidden_act)
    )

    config = {
        'architecture': DECODER_MODEL_TYPE[decoder_type],
        'dtype': dtype_str,
        'num_hidden_layers': nemo_model_config.get('num_layers'),
        'num_attention_heads': nemo_model_config.get('num_attention_heads'),
        'num_key_value_heads': nemo_model_config.get('num_query_groups', nemo_model_config['num_attention_heads']),
        'head_size': nemo_model_config.get('kv_channels'),
        'hidden_size': nemo_model_config.get('hidden_size'),
        'intermediate_size': nemo_model_config.get('ffn_hidden_size'),
        'norm_epsilon': nemo_model_config.get('layernorm_epsilon'),
        'vocab_size': vocab_size_padded,
        'position_embedding_type': (
            "rope_gpt_neox" if nemo_model_config.get('position_embedding_type') == "rope" else "learned_absolute"
        ),
        'max_position_embeddings': nemo_model_config.get('max_position_embeddings'),
        'hidden_act': hidden_act,
        'use_parallel_embedding': use_parallel_embedding,
        'embedding_sharding_dim': 0,
        'share_embedding_table': False,
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
        },
        'bias': nemo_model_config.get('bias'),
        'apply_query_key_layer_scaling': False,
        'rotary_pct': nemo_model_config.get('rotary_percentage', 1.0),
        'rotary_base': nemo_model_config.get('rotary_base', 10000),
        'moe_num_experts': nemo_model_config.get('num_moe_experts', 0),
        'moe_top_k': nemo_model_config.get('moe_router_topk'),
        'moe_normalization_mode': nemo_model_config.get(
            'moe_renorm_mode', MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
        ),
        'moe_tp_mode': nemo_model_config.get('moe_tp_mode', MoeConfig.ParallelismMode.TENSOR_PARALLEL),
        'logits_dtype': 'float32',
        'world_size': world_size,
        'tp_size': tensor_parallel_size,
        'pp_size': pipeline_parallel_size,
    }

    model_configs = []
    weights_dicts = []
    num_layers = nemo_model_config.get('num_layers')
    rotary_scaling = nemo_model_config.get("seq_len_interpolation_factor")

    if decoder_type == "falcon":
        config["new_decoder_architecture"] = False if num_layers == 32 else True
        config["parallel_attention"] = True
    if rotary_scaling is not None:
        config["rotary_scaling"] = {"type": "linear", "factor": float(rotary_scaling)}

    pp_key = {
        "transformer.vocab_embedding.weight",
        "transformer.position_embedding.weight",
        "lm_head.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
    }

    for i in range(world_size):
        mapping = tensorrt_llm.Mapping(
            world_size=world_size, rank=i, tp_size=tensor_parallel_size, pp_size=pipeline_parallel_size
        )
        layers_range = mapping.pp_layers(num_layers)

        weights_dict_local = {}
        for k, v in weights_dict.items():
            if k in pp_key:
                continue
            new_key = k
            if new_key.endswith(".bin"):  # TP split
                if new_key.endswith(f"{mapping.tp_rank}.bin"):
                    new_key = new_key.replace(f".{mapping.tp_rank}.bin", "")
            if "layers" in new_key:  # PP
                layer_num = int(new_key.split(".")[2])
                if layer_num in layers_range:
                    new_key = new_key.replace(f"layers.{layer_num}", f"layers.{layer_num-layers_range[0]}")
            if config.get("new_decoder_architecture", False) and "post_layernorm" in new_key:
                new_key = new_key.replace("post_layernorm", "mlp_layernorm")
            weights_dict_local[new_key] = v

        if mapping.is_first_pp_rank():
            embedding_weight = (
                np.ascontiguousarray(
                    split(weights_dict["transformer.vocab_embedding.weight"], mapping.tp_size, mapping.tp_rank)
                )
                if use_parallel_embedding
                else weights_dict["transformer.vocab_embedding.weight"]
            )

            weights_dict_local["transformer.vocab_embedding.weight"] = embedding_weight

            pos_embedding_weight = weights_dict.get("transformer.position_embedding.weight")
            if pos_embedding_weight is not None:
                if use_parallel_embedding:
                    pos_embedding_weight = np.ascontiguousarray(
                        split(pos_embedding_weight, mapping.tp_size, mapping.tp_rank)
                    )
                weights_dict_local["transformer.position_embedding.weight"] = pos_embedding_weight

        if mapping.is_last_pp_rank():
            weights_dict_local["lm_head.weight"] = np.ascontiguousarray(
                split(lm_head_weight, mapping.tp_size, mapping.tp_rank)
            )
            weights_dict_local["transformer.ln_f.weight"] = weights_dict["transformer.ln_f.weight"]

            ln_f_bias = weights_dict.get("transformer.ln_f.bias")
            if ln_f_bias is not None:
                weights_dict_local["transformer.ln_f.bias"] = ln_f_bias

        model_config = PretrainedConfig(**config)
        model_config.mapping = mapping
        model_configs.append(model_config)
        weights_dicts.append(weights_dict_local)

    return weights_dicts, model_configs, tokenizer
