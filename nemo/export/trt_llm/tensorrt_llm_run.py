# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Defines the tensorrt_llm inference API that can support both single and multiple GPU LLM inferences.

Referrence impl in tensorrt_llm: examples/llama/summarize.py.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging

import tensorrt_llm
import torch
from mpi4py.futures import MPIPoolExecutor
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from transformers import PreTrainedTokenizer
from tensorrt_llm.logger import logger

from .tensorrt_llm_build import get_engine_name, MODEL_NAME  # isort:skip

LOGGER = logging.getLogger("NeMo")


@dataclass
class TensorrtLLMHostContext:
    """The host side context for TRT LLM inference."""

    executor: MPIPoolExecutor = None
    world_size: int = 1
    tokenizer: PreTrainedTokenizer = None
    max_batch_size: int = 0
    max_input_len: int = 0


@dataclass
class TensorrtLLMWorkerContext:
    """The MPI worker side context for TRT LLM inference."""

    decoder: tensorrt_llm.runtime.GenerationSession = None
    sampling_config: SamplingConfig = None
    max_batch_size: int = 0
    max_input_len: int = 0


# This is a global context that will be initialized during the model loading process as MPI worker.
tensorrt_llm_worker_context = TensorrtLLMWorkerContext()


def _read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)
    use_gpt_attention_plugin = config["plugin_config"]["gpt_attention_plugin"]
    remove_input_padding = config["plugin_config"]["remove_input_padding"]
    tensor_parallel_size = config["builder_config"]["tensor_parallel"]
    pipeline_parallel_size = config["builder_config"]["pipeline_parallel"]
    world_size = tensor_parallel_size * pipeline_parallel_size

    assert world_size <= torch.cuda.device_count(), f"Not enough GPUs, requesting {world_size}"

    num_heads = config["builder_config"]["num_heads"]
    num_kv_heads = config["builder_config"].get("num_kv_heads", num_heads)
    hidden_size = config["builder_config"]["hidden_size"] // tensor_parallel_size
    vocab_size = config["builder_config"]["vocab_size"]
    num_layers = config["builder_config"]["num_layers"]
    paged_kv_cache = config["plugin_config"]["paged_kv_cache"]
    tokens_per_block = config["builder_config"]["tokens_per_block"]
    max_prompt_embedding_table_size = config["builder_config"]["max_prompt_embedding_table_size"]

    num_heads = num_heads // tensor_parallel_size
    num_kv_heads = (num_kv_heads + tensor_parallel_size - 1) // tensor_parallel_size

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        dtype="bfloat16" if paged_kv_cache else ""
    )

    dtype = config["builder_config"]["precision"]
    max_input_len = config["builder_config"]["max_input_len"]
    max_batch_size = config["builder_config"]["max_batch_size"]

    return model_config, world_size, tensor_parallel_size, pipeline_parallel_size, dtype, max_input_len, max_batch_size


def _load(tokenizer: PreTrainedTokenizer, engine_dir, num_beams=1):
    """The impl of `load` API for on a single GPU worker."""
    try:
        tensorrt_llm.logger.set_level("info")

        engine_dir = Path(engine_dir)
        config_path = engine_dir / "config.json"
        model_config, world_size, tp_size, pp_size, \
            dtype, max_input_len, max_batch_size = _read_config(config_path)

        runtime_rank = tensorrt_llm.mpi_rank()

        assert runtime_rank < torch.cuda.device_count(), f"Rank {runtime_rank} out of bound"
        runtime_mapping = tensorrt_llm.Mapping(
            world_size, runtime_rank, tp_size=tp_size, pp_size=pp_size)

        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_name = get_engine_name(
            MODEL_NAME, dtype, tp_size, pp_size, runtime_rank)
        serialize_path = os.path.join(engine_dir, engine_name)
        logger.info(f"Reading from serialize path {serialize_path}")

        with open(serialize_path, "rb") as f:
            engine_buffer = f.read()
        decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, runtime_mapping, debug_mode=True
        )

        sampling_config = SamplingConfig(
            end_id=tokenizer.eos_token_id, pad_id=tokenizer.eos_token_id, num_beams=num_beams
        )

        # Initialize the global context so it can be used during `run` API.
        global tensorrt_llm_worker_context
        tensorrt_llm_worker_context.decoder = decoder
        tensorrt_llm_worker_context.sampling_config = sampling_config
        tensorrt_llm_worker_context.max_batch_size = max_batch_size
        tensorrt_llm_worker_context.max_input_len = max_input_len

    except Exception as e:
        print(e)
        raise e


def _forward(
        input_tensors: List[torch.IntTensor],
        max_output_len: int,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        prompt_table=None,
        task_vocab_size=None,
        task_ids: List[int]=None,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        streaming: bool = False,
        **sampling_kwargs,
) -> Optional[torch.IntTensor]:
    """The impl of `forward` API for on a single GPU worker with tensor as IO.

    Returns:
        the output tokens tensor with shape [batch_size, num_beams, output_len].
    """
    try:
        # Loading the global context initialized from the `load` API.
        global tensorrt_llm_worker_context
        decoder = tensorrt_llm_worker_context.decoder
        assert decoder is not None, "Invalid worker context, decoder is not loaded."
        sampling_config = tensorrt_llm_worker_context.sampling_config
        max_batch_size = tensorrt_llm_worker_context.max_batch_size
        max_input_len = tensorrt_llm_worker_context.max_input_len

        batch_size = len(input_tensors)
        assert (
            batch_size <= max_batch_size
        ), f"batch size {batch_size} exceedng max batch size {max_batch_size}"
        input_lengths = [t.shape[0] for t in input_tensors]
        max_length = max(input_lengths)
        assert (
            max_length <= max_input_len
        ), f"input length {max_length} exceedng max input length {max_input_len}"
        pad_id = sampling_config.pad_id

        if decoder.remove_input_padding:
            line_encoded = [torch.tensor(t, dtype=torch.int32).cuda() for t in input_tensors]
        else:
            line_encoded = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tensors, dtype=torch.int32), pad_id
            ).cuda()
            input_lengths = torch.tensor(input_lengths, dtype=torch.int32).cuda()

        if prompt_table is None:
            ptuning_args = []
        else:
            if task_vocab_size is None:
                raise Exception("task_vocab_size cannot be None")

            task_vocab_size = torch.tensor([task_vocab_size], dtype=torch.int32, device="cuda")
            task_ids = torch.tensor(task_ids, dtype=torch.int32, device="cuda")
            prompt_table = prompt_table.cuda()
            ptuning_args = [prompt_table, task_ids, task_vocab_size]

        with torch.no_grad():
            sampling_config.top_k = top_k
            sampling_config.top_p = top_p
            sampling_config.temperature = temperature
            for key, param in sampling_kwargs.items():
                # set any additional SamplingConfig kwargs
                setattr(sampling_config, key, param)

            decoder.setup(batch_size, max_context_length=max_length, max_new_tokens=max_output_len)

            if decoder.remove_input_padding:
                if stop_words_list is not None:
                    LOGGER.warning("stop_words_list should be set to None with remove_input_padding=True "
                                  "and it will be ignored.")

                if bad_words_list is not None:
                    LOGGER.warning("bad_words_list should be set to None with remove_input_padding=True "
                                  "and it will be ignored.")

                if no_repeat_ngram_size is not None:
                    LOGGER.warning("no_repeat_ngram_size should be set to None with remove_input_padding=True "
                                  "and it will be ignored.")

                output_ids = decoder.decode_batch(line_encoded, sampling_config)
            else:
                output_ids = decoder.decode(
                    line_encoded,
                    input_lengths,
                    sampling_config,
                    *ptuning_args,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    streaming=streaming,
                )

            torch.cuda.synchronize()

            runtime_rank = tensorrt_llm.mpi_rank()
            if runtime_rank == 0:
                return output_ids
            else:
                return None

    except Exception as e:
        print(e)
        raise e


def load(
    tokenizer: PreTrainedTokenizer, engine_dir: str, num_beams: int = 1
) -> TensorrtLLMHostContext:
    """Loaded the compiled LLM model and run it.

    It also supports running the TRT LLM model on multi-GPU.
    """
    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    world_size = config["builder_config"]["world_size"]
    if world_size == 1:
        _load(tokenizer, engine_dir, num_beams)
        executor = None
    else:
        executor = MPIPoolExecutor(max_workers=world_size)
        futures = []
        for _ in range(world_size):
            future = executor.submit(_load, tokenizer, engine_dir, num_beams)
            futures.append(future)
        for future in futures:
            future.result()

    max_batch_size = config["builder_config"]["max_batch_size"]
    max_input_len = config["builder_config"]["max_input_len"]

    return TensorrtLLMHostContext(
        executor=executor,
        world_size=world_size,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
    )


def forward(
        input_tensors: List[torch.IntTensor],
        max_output_len: int,
        host_context: TensorrtLLMHostContext,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        prompt_table=None,
        task_vocab_size=None,
        task_ids: List[int]=None,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        streaming: bool = False,
        **sampling_kwargs,
) -> Optional[torch.IntTensor]:
    """Run the loaded model with the host_context provided from the `load` API."""
    batch_size = len(input_tensors)
    max_batch_size = host_context.max_batch_size
    assert (
        batch_size <= max_batch_size
    ), f"batch size {batch_size} exceedng max batch size {max_batch_size}"
    max_length = max([t.shape[0] for t in input_tensors])
    max_input_len = host_context.max_input_len
    assert (
        max_length <= max_input_len
    ), f"input length {max_length} exceedng max input length {max_input_len}"

    world_size = host_context.world_size
    if world_size == 1:
        return _forward(
            input_tensors=input_tensors,
            max_output_len=max_output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            prompt_table=prompt_table,
            task_vocab_size=task_vocab_size,
            task_ids=task_ids,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            no_repeat_ngram_size=no_repeat_ngram_size,
            streaming=streaming,
            **sampling_kwargs
        )
    else:
        executor = host_context.executor
        futures = []
        for _ in range(world_size):
            future = executor.submit(
                _forward,
                input_tensors=input_tensors,
                max_output_len=max_output_len,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                prompt_table=prompt_table,
                task_vocab_size=task_vocab_size,
                task_ids=task_ids,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streaming=streaming,
                **sampling_kwargs
            )
            futures.append(future)
        for future in futures:
            result = future.result()
            if result is not None:
                return result

        raise RuntimeError("Internal error")


def generate(
        input_texts: List[str],
        max_output_len: int,
        host_context: TensorrtLLMHostContext,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        prompt_table=None,
        task_vocab_size=None,
        task_ids: List[int]=None,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        **sampling_kwargs,
) -> Optional[List[List[str]]]:
    """Generate the output sequence from the input sequence.

    Returns a 2D string list with shape [batch_size, num_beams].
    """
    tokenizer = host_context.tokenizer
    input_tensors = [
        torch.IntTensor(tokenizer.encode(t)) for t in input_texts
    ]

    batch_size = len(input_texts)

    stop_words_list_tensors = None
    if stop_words_list is not None:
        stop_words_list_tensors = [
            tokenizer.encode(t) for t in stop_words_list
        ]
        stop_words_list_tensors = torch.IntTensor(stop_words_list_tensors)
        stop_words_list_tensors = stop_words_list_tensors.unsqueeze(0).repeat(batch_size, 1, 1).to(
            torch.cuda.current_device())

    bad_words_list_tensors = None
    if bad_words_list is not None:
        bad_words_list_tensors = [
            tokenizer.encode(t) for t in bad_words_list
        ]
        bad_words_list_tensors = torch.IntTensor(bad_words_list_tensors)
        bad_words_list_tensors = bad_words_list_tensors.unsqueeze(0).repeat(batch_size, 1, 1).to(
            torch.cuda.current_device())

    if no_repeat_ngram_size is not None:
        no_repeat_ngram_size = torch.IntTensor(no_repeat_ngram_size).to(
            torch.cuda.current_device())

    outputs = forward(
        input_tensors=input_tensors,
        max_output_len=max_output_len,
        host_context=host_context,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        prompt_table=prompt_table,
        task_vocab_size=task_vocab_size,
        task_ids=task_ids,
        stop_words_list=stop_words_list_tensors,
        bad_words_list=bad_words_list_tensors,
        no_repeat_ngram_size=no_repeat_ngram_size,
        streaming=False,
        **sampling_kwargs
    )
    assert outputs is not None

    input_lengths = [t.shape[0] for t in input_tensors]

    output_lines_list = [
        tokenizer.batch_decode(outputs[b, :, input_lengths[b] :])
        for b in range(outputs.shape[0])
    ]
    return output_lines_list


def generate_streaming(
        input_texts: List[str],
        max_output_len: int,
        host_context: TensorrtLLMHostContext,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        prompt_table=None,
        task_vocab_size=None,
        task_ids: List[int]=None,
        stop_words_list=None,
        bad_words_list=None,
        no_repeat_ngram_size=None,
        **sampling_kwargs,
) -> Optional[List[List[str]]]:
    """Generate the output sequence from the input sequence.

    Returns a 2D string list with shape [batch_size, num_beams].
    """
    tokenizer = host_context.tokenizer
    input_tensors = [
        torch.IntTensor(tokenizer.encode(t)) for t in input_texts
    ]

    batch_size = len(input_texts)

    stop_words_list_tensors = None
    if stop_words_list is not None:
        stop_words_list_tensors = [
            tokenizer.encode(t) for t in stop_words_list
        ]
        stop_words_list_tensors = torch.IntTensor(stop_words_list_tensors)
        stop_words_list_tensors = stop_words_list_tensors.unsqueeze(0).repeat(batch_size, 1, 1).to(
            torch.cuda.current_device())

    bad_words_list_tensors = None
    if bad_words_list is not None:
        bad_words_list_tensors = [
            tokenizer.encode(t) for t in bad_words_list
        ]
        bad_words_list_tensors = torch.IntTensor(bad_words_list_tensors)
        bad_words_list_tensors = bad_words_list_tensors.unsqueeze(0).repeat(batch_size, 1, 1).to(
            torch.cuda.current_device())

    if no_repeat_ngram_size is not None:
        no_repeat_ngram_size = torch.IntTensor(no_repeat_ngram_size).to(
            torch.cuda.current_device())

    outputs = forward(
        input_tensors=input_tensors,
        max_output_len=max_output_len,
        host_context=host_context,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        prompt_table=prompt_table,
        task_vocab_size=task_vocab_size,
        task_ids=task_ids,
        stop_words_list=stop_words_list_tensors,
        bad_words_list=bad_words_list_tensors,
        no_repeat_ngram_size=no_repeat_ngram_size,
        streaming=True,
        **sampling_kwargs
    )
    assert outputs is not None

    input_lengths = [t.shape[0] for t in input_tensors]

    for cur_outputs in outputs:
        output_lines_list = [
            tokenizer.batch_decode(cur_outputs[b, :, input_lengths[b] :])
            for b in range(len(cur_outputs))
        ]
        yield output_lines_list

def unload(host_context: TensorrtLLMHostContext):
    """Frees the GPU resource from the TensorrtLLMHostContext and reset the host_context."""
    if host_context.executor is not None:
        host_context.executor.shutdown(wait=True)
        host_context.executor = None
        return

    global tensorrt_llm_worker_context
    tensorrt_llm_worker_context.decoder = None
    tensorrt_llm_worker_context = TensorrtLLMWorkerContext()
