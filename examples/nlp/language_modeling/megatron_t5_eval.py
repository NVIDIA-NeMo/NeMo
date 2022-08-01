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


from argparse import ArgumentParser

import torch
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.language_modeling.megatron.request_dataset import T5RequestDataset
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import MegatronT5Model
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin, NLPSaveRestoreConnector
from nemo.utils.app_state import AppState

assert torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default="", required=True, help="Pass path to model's .nemo file")
    parser.add_argument(
        "--prompt", type=str, default="", required=True, help="Prompt for the model (a text to complete)"
    )
    parser.add_argument(
        "--tokens_to_generate", type=int, default="16", required=False, help="How many tokens to add to prompt"
    )
    parser.add_argument(
        "--tensor_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument(
        "--pipeline_model_parallel_size", type=int, default=1, required=False,
    )
    parser.add_argument(
        "--pipeline_model_parallel_split_rank", type=int, default=0, required=False,
    )
    parser.add_argument("--precision", default="16", type=str, help="PyTorch Lightning Trainer precision flag")
    args = parser.parse_args()

    # cast precision to int if 32 or 16
    if args.precision in ["32", "16"]:
        args.precision = int(float(args.precision))

    # trainer required for restoring model parallel models
    trainer = Trainer(
        plugins=NLPDDPPlugin(),
        devices=args.tensor_model_parallel_size * args.pipeline_model_parallel_size,
        accelerator='gpu',
        precision=args.precision,
    )

    app_state = AppState()
    if args.tensor_model_parallel_size > 1 or args.pipeline_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
        (
            app_state.tensor_model_parallel_rank,
            app_state.pipeline_model_parallel_rank,
            app_state.model_parallel_size,
            app_state.data_parallel_size,
            app_state.pipeline_model_parallel_split_rank,
        ) = fake_initialize_model_parallel(
            world_size=app_state.model_parallel_size,
            rank=trainer.global_rank,
            tensor_model_parallel_size_=args.tensor_model_parallel_size,
            pipeline_model_parallel_size_=args.pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank_=args.pipeline_model_parallel_split_rank,
        )

    model = MegatronT5Model.restore_from(
        restore_path=args.model_file, trainer=trainer, save_restore_connector=NLPSaveRestoreConnector(),
    )
    model.freeze()

    request = {
        "prompt": args.prompt,
        "tokens_to_generate": args.tokens_to_generate,
    }

    dataset = T5RequestDataset(request, model.tokenizer)

    request_dl = DataLoader(dataset)

    response = trainer.predict(model, request_dl)

    print("***************************")
    print(response)
    print(response[0]['completion']['text'])
    print("***************************")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
