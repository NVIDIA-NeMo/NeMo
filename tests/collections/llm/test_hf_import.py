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

import argparse
from pathlib import Path

from nemo import lightning as nl
from nemo.collections import llm


def get_args():
    parser = argparse.ArgumentParser(description='Test Llama2 7B model model conversion from HF')
    parser.add_argument('--hf_model', type=str, help="Original HF model")
    parser.add_argument('--output_path', type=str, help="NeMo 2.0 export path")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    model = llm.LlamaModel(config=llm.Llama2Config7B)
    nemo2_path = llm.import_ckpt(model, "hf://" + args.hf_model, output_path=Path(args.output_path))

    trainer = nl.Trainer(
        devices=1,
        strategy=nl.MegatronStrategy(tensor_model_parallel_size=1),
        plugins=nl.MegatronMixedPrecision(precision='fp16'),
    )
    fabric = trainer.to_fabric()
    trainer.strategy.setup_environment()
    fabric.load_model(nemo2_path)
