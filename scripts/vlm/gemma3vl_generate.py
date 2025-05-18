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
Example:
  python scripts/vlm/gemma3vl_generate.py --local_model_path="path/to/converted_nemo_checkpoint"
"""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

import nemo.lightning as nl
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.vlm import Gemma3VLModel
from nemo.collections.vlm.inference.base import _setup_trainer_and_restore_model
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.utils.get_rank import get_last_rank


class SingleBatchIterator:
    def __init__(self, pixel_values, input_ids, position_ids):
        self.batch = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            position_ids=position_ids,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def gemma3_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
        "pixel_values": batch.get("pixel_values", None),
        "loss_mask": batch.get("loss_mask", None),
        "labels": batch.get("labels", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def main(args) -> None:
    # pylint: disable=C0115,C0116,C0301
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        sequence_parallel=args.tp > 1,
        ckpt_include_optimizer=False,
        ckpt_load_strictness="log_all",
        pipeline_dtype=torch.bfloat16,
    )
    trainer = nl.Trainer(
        devices=min(args.tp * args.pp, 8),
        num_nodes=max(args.tp * args.pp // 8, 1),
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        enable_checkpointing=False,
    )

    if args.local_model_path:
        path = Path(args.local_model_path)
        model: io.TrainerContext = io.load_context(path=ckpt_to_context_subdir(path), subpath="model")
        _setup_trainer_and_restore_model(path=path, trainer=trainer, model=model)
    else:
        fabric = trainer.to_fabric()
        model = fabric.import_model("hf://google/gemma-3-4b-it", Gemma3VLModel)

    model = model.module.cuda()
    model.eval()

    from transformers import AutoProcessor

    model_id = 'google/gemma-3-4b-it'
    processor = AutoProcessor.from_pretrained(model_id)
    gemma_tokenizer = AutoTokenizer(model_id)
    hf_tokenizer = gemma_tokenizer.tokenizer

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": "What animal is on the candy?"},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].cuda()
    # add additional dim to (B, N, C, H, W)
    pixel_values = inputs["pixel_values"].cuda().unsqueeze(0).to(dtype=torch.bfloat16)
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
    )
    generated_ids = input_ids.clone()

    stop_tokens = [1, 126]
    # Greedy generation loop
    for step in range(20):
        with torch.no_grad():
            if torch.distributed.get_rank() == 0:
                print(step)
            fwd_bwd_function = get_forward_backward_func()

            iterator = SingleBatchIterator(pixel_values, input_ids, position_ids)

            output = fwd_bwd_function(
                forward_step_func=gemma3_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )
            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]
                # All-gather operation
                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                # Concatenate along last dimension (dim=2)
                output = torch.cat(gathered_tensors, dim=2)

                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)
            else:
                next_token_ids = torch.ones((1, 1), device=generated_ids.device, dtype=generated_ids.dtype)

            torch.distributed.broadcast(next_token_ids, get_last_rank())
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            input_ids = generated_ids
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand_as(input_ids)
            )

            # If the generated token is the end of sequence token, stop generating
            if next_token_ids.item() in stop_tokens:
                break

    generated_texts = hf_tokenizer.decode(list(generated_ids[0]))
    if torch.distributed.get_rank() == 0:
        print("======== GENERATED TEXT OUTPUT ========")
        print(f"{generated_texts}")
        print("=======================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma3 Multimodal Inference")
    parser.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local path to the model if not loading from Hugging Face.",
    )
    parser.add_argument('--tp', default=1)
    parser.add_argument('--pp', default=1)
    args = parser.parse_args()

    main(args)
