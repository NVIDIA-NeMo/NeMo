# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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


import torch

try:
    from apex._autocast_utils import _cast_if_autocast_enabled

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


def dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, None, Tensor, float, bool) -> Tensor
    if bias is not None:
        raise ValueError(f"bias is expected to be None when using the bias_dropout function.")
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_fused_train_(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


def bias_dropout_add_fused_train(x, bias, residual, prob):
    # re-enable torch grad to enable fused optimization.
    with torch.enable_grad():
        args = _cast_if_autocast_enabled(x, bias, residual, prob)
        with torch.cuda.amp.autocast(enabled=False):
            return bias_dropout_add_fused_train_(*args)


@torch.jit.script
def bias_dropout_add_fused_inference_(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


def bias_dropout_add_fused_inference(x, bias, residual, prob):
    args = _cast_if_autocast_enabled(x, bias, residual, prob)
    with torch.cuda.amp.autocast(enabled=False):
        return bias_dropout_add_fused_inference_(*args)
