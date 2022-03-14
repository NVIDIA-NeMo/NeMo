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

import json
import os
from argparse import ArgumentParser

import torch
from pytorch_lightning.trainer.trainer import Trainer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.language_modeling.megatron.request_dataset import GPTRequestDataset
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank

"""
Usage:
    a. If you need to run model on a few prompts from the file:
        python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --tokens_to_generate=32 \
            --batch_size=16 \

    b. If you need to run model on a prompt from the CLI:
        python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --tokens_to_generate=32 \
            --prompt=YOUR_PROMPT

    c. If you don't need to generate tokens and need model to compute logprobs:
         python megatron_gpt_eval.py \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --batch_size=16 \
            --compute_logprobs=True \
            --prompt .

    d. If you need to run a prompt-tuned model on a few prompts from a file:
        python megatron_gpt_eval.py \
            --use_soft_prompts \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --tokens_to_generate=32 \
            --batch_size=16 \

        The path_to_file containing the model prompts should be a json with prompts in the format:
            {'prompt_tag': tag1, 'text': prompt1}
            {'prompt_tag': tag1, 'text': prompt2}
            {'prompt_tag': tag3, 'text': prompt3}

    e. If you need to run a prompt-tuned model on a prompt from the CLI:
        python megatron_gpt_eval.py \
            --use_soft_prompts \
            --model_file=PATH_TO_MODEL \
            --tokens_to_generate=32 \
            --prompt_tag=PROMPT_TAG_STRING \
            --prompt=YOUR_PROMPT
"""

assert torch.cuda.is_available()


def main():
    parser = ArgumentParser()

    # args for loading the model, either from .nemo file or from PTL checkpoint
    parser.add_argument("--model_file", type=str, default="", required=False, help="Pass path to model's .nemo file")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        required=False,
        help="If not using a .nemo file. Path to PTL checkpoints saved during training. Ex: /raid/nemo_experiments/megatron_gpt/checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        required=False,
        help="If not using a .nemo file. Name of checkpoint to be used. Ex: megatron_gpt--val_loss=6.34-step=649-last.ckpt",
    )

    parser.add_argument(
        "--hparams_file",
        type=str,
        default=None,
        required=False,
        help="If not using a .nemo file. Path to config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    parser.add_argument(
        "--tensor_model_parallel_size", type=int, default=1, required=False, help="Needed if not using a .nemo file"
    )
    parser.add_argument(
        "--pipeline_model_parallel_size", type=int, default=1, required=False, help="Needed if not using a .nemo file",
    )

    # PTL Trainer args
    parser.add_argument("--devices", default=1, type=int, help="PyTorch Lightning Trainer devices flag")
    parser.add_argument("--num_nodes", default=1, type=int, help="PyTorch Lightning Trainer num_nodes flag")
    parser.add_argument("--precision", default=16, help="PyTorch Lightning Trainer precision flag")

    # evaluation args
    parser.add_argument(
        "--path_to_file", type=str, default="", required=False, help="Path to file with prompts (a text to complete)"
    )
    parser.add_argument(
        "--prompt", type=str, default="", required=False, help="Prompt for the model (a text to complete)"
    )
    parser.add_argument("--use_soft_prompts", action="store_true", help="Use model's existing soft prompts")
    parser.add_argument(
        "--prompt_tag", type=str, default="", required=False, help="Prompt tag string for task specific soft prompt"
    )
    parser.add_argument(
        "--tokens_to_generate", type=int, default="1", required=False, help="How many tokens to add to prompt"
    )
    parser.add_argument(
        "--stop_after_sentence",
        type=bool,
        default="True",
        required=False,
        help="True/False: whether to stop after full sentence has been generated.",
    )
    parser.add_argument("--batch_size", default=1, type=int, required=False, help="Evaluation batch_size")
    parser.add_argument(
        "--compute_logprobs", type=bool, default=False, required=False, help="Method for logprobs computation"
    )

    args = parser.parse_args()

    assert (
        args.devices * args.num_nodes == args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if args.model_file and args.checkpoint_dir:
        raise ValueError("Only one of model_file or checkpoint_dir should be used")

    # cast precision to int if 32 or 16
    if args.precision in ["32", "16"]:
        args.precision = int(float(args.precision))

    # trainer required for restoring model parallel models
    trainer = Trainer(
        plugins=[NLPDDPPlugin()],
        devices=args.devices,
        num_nodes=args.num_nodes,
        accelerator='gpu',
        precision=args.precision,
    )

    if args.model_file:
        model = MegatronGPTModel.restore_from(restore_path=args.model_file, trainer=trainer)
    elif args.checkpoint_dir:
        app_state = AppState()
        if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
            app_state.pipeline_model_parallel_size = args.pipeline_model_parallel_size
            app_state.tensor_model_parallel_size = args.tensor_model_parallel_size
            app_state.model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.model_parallel_size,
                _,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=app_state.tensor_model_parallel_size,
                pipeline_model_parallel_size_=app_state.pipeline_model_parallel_size,
            )
        # inject model parallel rank
        checkpoint_path = inject_model_parallel_rank(os.path.join(args.checkpoint_dir, args.checkpoint_name))

        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=args.hparams_file, trainer=trainer)

    model.freeze()

    def pad_collate(batch):
        tokens, tokens_to_generate = batch[0]['data'], batch[0]['tokens_to_generate']
        compute_logprobs = batch[0]['compute_logprobs']
        lens = [len(token) for token in tokens]

        tokens_pad = pad_sequence(tokens, batch_first=False, padding_value=50256)
        data = []

        if 'prompt_tags' in batch[0]:
            # Keep track of soft prompt tags
            prompt_tags = batch[0]['prompt_tags']

            for token, lenn, prompt_tag in zip(tokens_pad.T, lens, prompt_tags):
                data.append((token, lenn, tokens_to_generate, compute_logprobs, prompt_tag))
        else:
            for token, lenn in zip(tokens_pad.T, lens):
                data.append((token, lenn, tokens_to_generate, compute_logprobs))

        return data

    # defining type of request
    if args.path_to_file != "":
        request = []
        prompts = open(args.path_to_file, 'r', encoding='utf-8')

        for prompt in prompts.readlines():
            prompt = prompt.split('\n')[0]

            if args.use_soft_prompts and model.use_soft_prompts:
                prompt = json.loads(prompt)

            request.append(prompt)

        dataset = GPTRequestDataset(request, model.tokenizer, args.tokens_to_generate, args.compute_logprobs)
        request_dl = DataLoader(dataset=pad_collate(dataset), batch_size=int(args.batch_size))

    else:
        if args.use_soft_prompts and model.use_soft_prompts:
            request = [{'prompt_tag': args.prompt_tag, 'text': args.prompt}]
        else:
            request = [args.prompt]

        dataset = GPTRequestDataset(request, model.tokenizer, args.tokens_to_generate, args.compute_logprobs)
        request_dl = DataLoader(dataset=pad_collate(dataset), batch_size=1)

    # For GPT models that have had soft prompt tuning but you don't want to use any soft prompts
    if not args.use_soft_prompts and model.use_soft_prompts:
        model.use_soft_prompts = False

    response = trainer.predict(model, request_dl)

    print("***************************")
    print(response)
    print("***************************")
    if args.prompt and not args.compute_logprobs:
        print(f'Prompt: {args.prompt}\n\nResponse: {response[0][0][0]}')


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
