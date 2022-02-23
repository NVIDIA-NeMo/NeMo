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
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
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
        "--tensor_model_parallel_size", type=int, default=1, required=True,
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=NLPDDPPlugin(), devices=args.tensor_model_parallel_size, precision=16, accelerator='gpu')

    app_state = AppState()
    if args.tensor_model_parallel_size > 1:
        app_state.model_parallel_size = args.tensor_model_parallel_size
        app_state.model_parallel_rank = compute_model_parallel_rank(trainer.local_rank, app_state.model_parallel_size)

    model = MegatronT5Model.restore_from(restore_path=args.model_file, trainer=trainer)
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
    print("***************************")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
