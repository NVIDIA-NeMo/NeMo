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
import os
import shutil

import torch

from nemo.collections.common.video_tokenizers.cosmos_tokenizer import CausalVideoTokenizer
from nemo.export.tensorrt_lazy_compiler import trt_compile

parser = argparse.ArgumentParser(description="Export and run tokenizer in TensorRT")
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default="Cosmos-Tokenizer-CV4x8x8",
    help="Tokenizer name or path",
)
parser.add_argument(
    "--engine_path",
    type=str,
    default="outputs",
    help="Path to TensorRT engine",
)
parser.add_argument("--min_shape", type=int, nargs='+', help="min input shape for inference")
parser.add_argument("--opt_shape", type=int, nargs='+', help="opt input shape for inference")
parser.add_argument(
    "--max_shape", type=int, nargs='+', default=[1, 3, 9, 512, 512], help="max input shape for inference"
)
parser.add_argument("--clean", action="store_true", help="Clean all files in engine_path before export")

args = parser.parse_args()


def main():
    model = CausalVideoTokenizer.from_pretrained(args.tokenizer_name, use_pytorch=True, dtype="float")

    class VaeWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, input_tensor):
            output_tensor = self.vae.autoencode(input_tensor)
            return output_tensor

    model_wrapper = VaeWrapper(model)

    if args.clean and os.path.exists(args.engine_path):
        print(f"Remove existing {args.engine_path}")
        shutil.rmtree(args.engine_path)

    os.makedirs(args.engine_path, exist_ok=True)

    min_shape = args.min_shape
    opt_shape = args.opt_shape
    max_shape = args.max_shape

    if opt_shape is None:
        opt_shape = max_shape
    if min_shape is None:
        min_shape = opt_shape

    output_path = os.path.join(args.engine_path, "auto_encoder")
    trt_compile(
        model_wrapper,
        output_path,
        args={
            "precision": "bf16",
            "input_profiles": [
                {"input_tensor": [min_shape, opt_shape, max_shape]},
            ],
        },
    )

    input_tensor = torch.randn(max_shape).to('cuda').to(torch.float)
    output = model_wrapper(input_tensor)


if __name__ == '__main__':
    main()
