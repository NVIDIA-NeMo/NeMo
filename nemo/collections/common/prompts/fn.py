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

from typing import Callable, Sequence

import torch
from lhotse import CutSet

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

PROMPT_FORMAT_FNS = {}


def registered_prompt_format_fn(
    prompt_fn: Callable[[CutSet, TokenizerSpec], tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]
):
    """
    Decorator for registering prompt functions under a name.

    Example::

        >>> @registered_prompt_format_fn
        ... def my_prompt(cuts, tokenizer):
        ...     pass
        ...
        ... prompt_fn = get_prompt_format_fn("my_prompt")
    """
    global PROMPT_FORMAT_FNS

    PROMPT_FORMAT_FNS[prompt_fn.__name__] = prompt_fn
    return prompt_fn


def get_prompt_format_fn(
    name: str,
) -> Callable[[CutSet, TokenizerSpec], tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]]:
    if name not in PROMPT_FORMAT_FNS:
        raise ValueError(
            f"Unknown prompt format function name: {name} " f"(must be one of: {list(PROMPT_FORMAT_FNS.keys())}"
        )
    return PROMPT_FORMAT_FNS[name]
