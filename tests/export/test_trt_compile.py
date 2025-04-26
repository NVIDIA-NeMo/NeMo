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

from __future__ import annotations

import tempfile
import unittest
from typing import List

import torch

TEST_CASE_1 = ["fp32"]
TEST_CASE_2 = ["fp16"]


class ListAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: List[torch.Tensor], y: torch.Tensor, z: torch.Tensor, bs: float = 0.1):
        y1 = y.clone()
        x1 = x.copy()
        z1 = z + y
        for xi in x:
            y1 = y1 + xi + bs
        return x1, [y1, z1], y1 + z1


@unittest.skip
class TestTRTCompile(unittest.TestCase):

    def setUp(self):
        self.gpu_device = torch.cuda.current_device()

    def tearDown(self):
        current_device = torch.cuda.current_device()
        if current_device != self.gpu_device:
            torch.cuda.set_device(self.gpu_device)

    def test_torch_trt(self):

        model = torch.nn.Sequential(*[torch.nn.PReLU(), torch.nn.PReLU()])
        data1 = model.state_dict()
        data1["0.weight"] = torch.tensor([0.1])
        data1["1.weight"] = torch.tensor([0.2])
        model.load_state_dict(data1)
        model.cuda()
        x = torch.randn(1, 16).to("cuda")

        with tempfile.TemporaryDirectory() as tempdir:
            args = {
                "method": "torch_trt",
                "dynamic_batchsize": [1, 4, 8],
            }
            input_example = (x,)
            output_example = model(*input_example)
            trt_compile(
                model,
                f"{tempdir}/test_lists",
                args=args,
            )
            self.assertIsNone(model._trt_compiler.engine)
            trt_output = model(*input_example)
            # Check that lazy TRT build succeeded
            self.assertIsNotNone(model._trt_compiler.engine)
            torch.testing.assert_close(trt_output, output_example, rtol=0.01, atol=0.01)

    def test_profiles(self):
        model = ListAdd().cuda()

        with torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
            args = {
                "export_args": {
                    "dynamo": False,
                },
                "input_profiles": [
                    {
                        "x_0": [[1, 8], [2, 16], [2, 32]],
                        "x_1": [[1, 8], [2, 16], [2, 32]],
                        "x_2": [[1, 8], [2, 16], [2, 32]],
                        "y": [[1, 8], [2, 16], [2, 32]],
                        "z": [[1, 8], [1, 16], [1, 32]],
                    }
                ],
                "output_lists": [[-1], [2], []],
            }
            x = torch.randn(1, 16).to("cuda")
            y = torch.randn(1, 16).to("cuda")
            z = torch.randn(1, 16).to("cuda")
            input_example = ([x, y, z], y.clone(), z.clone())
            output_example = model(*input_example)
            trt_compile(
                model,
                f"{tmpdir}/test_dynamo_trt",
                args=args,
            )
            self.assertIsNone(model._trt_compiler.engine)
            trt_output = model(*input_example)
            # Check that lazy TRT build succeeded
            self.assertIsNotNone(model._trt_compiler.engine)
            torch.testing.assert_close(trt_output, output_example, rtol=0.01, atol=0.01)

    def test_lists(self):
        model = ListAdd().cuda()

        with torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
            args = {
                "export_args": {
                    "dynamo": True,
                },
                "output_lists": [[-1], [2], []],
            }
            x = torch.randn(1, 16).to("cuda")
            y = torch.randn(1, 16).to("cuda")
            z = torch.randn(1, 16).to("cuda")
            input_example = ([x, y, z], y.clone(), z.clone())
            output_example = model(*input_example)
            trt_compile(
                model,
                f"{tmpdir}/test_lists",
                args=args,
            )
            self.assertIsNone(model._trt_compiler.engine)
            trt_output = model(*input_example)
            # Check that lazy TRT build succeeded
            self.assertIsNotNone(model._trt_compiler.engine)
            torch.testing.assert_close(trt_output, output_example, rtol=0.01, atol=0.01)


if __name__ == "__main__":
    unittest.main()
