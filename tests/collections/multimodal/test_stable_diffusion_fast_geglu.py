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

import random

import pytest
import torch

from nemo.collections.multimodal.modules.stable_diffusion import fast_geglu


def geglu(x_and_gate):
    x, gate = x_and_gate.chunk(2, dim=-1)
    return x * torch.nn.functional.gelu(gate)


geglu_compile = torch.compile(geglu)


class TestStableDiffusionFastGeGLU:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="fast_geglu can run only on CUDA")
    @pytest.mark.unit
    @pytest.mark.parametrize("dim_last", [1280, 2560, 5120])
    def test_correctness(self, dim_last):
        dtype = torch.float16
        dtype_ref = torch.float64

        random_seed = 42
        rng = random.Random()
        rng.seed(random_seed)
        torch.random.manual_seed(random_seed)

        for _ in range(10):
            batch_size = rng.randint(1, 16)
            shape1 = rng.randint(1, 8192)

            x_and_gate_fast = torch.randn(
                [batch_size, shape1, 2 * dim_last], device='cuda', dtype=dtype
            ).requires_grad_(True)
            x_and_gate_compile = x_and_gate_fast.detach().clone().requires_grad_(True)
            x_and_gate_ref = x_and_gate_fast.detach().clone().to(dtype_ref).requires_grad_(True)
            grad_output = torch.randn([batch_size, shape1, dim_last], device='cuda', dtype=dtype)

            output_fast = fast_geglu.geglu(x_and_gate_fast)
            output_compile = geglu_compile(x_and_gate_compile)
            output_ref = geglu(x_and_gate_ref)

            diff_max_fast = (output_fast - output_ref).abs().max().item()
            diff_mean_fast = (output_fast - output_ref).abs().mean().item()
            diff_max_compile = (output_compile - output_ref).abs().max().item()
            diff_mean_compile = (output_compile - output_ref).abs().mean().item()
            assert diff_max_fast <= 1.01 * diff_max_compile
            assert diff_mean_fast <= 1.01 * diff_mean_compile

            output_fast.backward(grad_output)
            output_compile.backward(grad_output)
            output_ref.backward(grad_output)

            diff_max_fast = (x_and_gate_fast.grad - x_and_gate_ref.grad).abs().max().item()
            diff_mean_fast = (x_and_gate_fast.grad - x_and_gate_ref.grad).abs().mean().item()
            diff_max_compile = (x_and_gate_compile.grad - x_and_gate_ref.grad).abs().max().item()
            diff_mean_compile = (x_and_gate_compile.grad - x_and_gate_ref.grad).abs().mean().item()
            assert diff_max_fast <= 1.01 * diff_max_compile
            assert diff_mean_fast <= 1.01 * diff_mean_compile
