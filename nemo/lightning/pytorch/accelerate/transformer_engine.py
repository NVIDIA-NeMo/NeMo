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

from types import MethodType

import torch
from transformer_engine import pytorch as te


def accelerate(model, fp8_autocast=False):
    _accelerate(model)
    if fp8_autocast:
        apply_fp8_autocast(model)


def _accelerate(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            has_bias = module.bias is not None
            if any(p % 16 != 0 for p in module.weight.shape):
                continue
            te_module = te.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            with torch.no_grad():
                te_module.weight.copy_(module.weight)
                if has_bias:
                    te_module.bias.copy_(module.bias)

            setattr(module, name.split(".")[-1], te_module)
        elif isinstance(module, torch.nn.LayerNorm):
            te_module = te.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.bias.copy_(module.bias)
            setattr(module, name.split(".")[-1], te_module)
        elif isinstance(module, torch.nn.RMSNorm):
            te_module = te.RMSNorm(module.normalized_shape[0], eps=module.eps, dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.bias.copy_(module.bias)
            setattr(module, name.split(".")[-1], te_module)


def te_accelerated(model):
    for name, module in model.named_modules():
        if isinstance(module, (te.LayerNorm, te.Linear, te.TransformerLayer)):
            return True

    return False


def apply_fp8_autocast(model, fp8_recipe_handler=None):
    import transformer_engine.common.recipe as te_recipe

    kwargs = fp8_recipe_handler.to_kwargs() if fp8_recipe_handler is not None else {}
    if "fp8_format" in kwargs:
        kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
    use_during_eval = kwargs.pop("use_autocast_during_eval", False)
    fp8_recipe = te_recipe.DelayedScaling(**kwargs)
    new_forward = _contextual_fp8_autocast(model.forward, fp8_recipe, use_during_eval)

    if hasattr(model.forward, "__func__"):
        model.forward = MethodType(new_forward, model)
    else:
        model.forward = new_forward

    return model


def _contextual_fp8_autocast(model_forward, fp8_recipe, use_during_eval=False):
    from transformer_engine.pytorch import fp8_autocast

    def forward(self, *args, **kwargs):
        enabled = use_during_eval or self.training
        with fp8_autocast(enabled=enabled, fp8_recipe=fp8_recipe):
            return model_forward(*args, **kwargs)

    forward.__wrapped__ = model_forward

    return forward
