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
