# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024 Arc Institute. All rights reserved.
# Copyright (c) 2024 Michael Poli. All rights reserved.
# Copyright (c) 2024 Stanford University. All rights reserved
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

import logging


if __name__ != "__main__":
    from megatron.core.utils import log_single_rank
else:
    from typing import Any

    def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
        """Log a message from the current rank."""
        print(*args[1:], **kwargs)


logger = logging.getLogger(__name__)


class Symbols:
    """Symbols for the hybrid layer allocation."""

    HYENA_SHORT = 'S'
    HYENA_MEDIUM = 'D'
    HYENA = 'H'
    ATTENTION = '*'
    VALID = {HYENA_SHORT, HYENA_MEDIUM, HYENA, ATTENTION}


def _allocate_override(total_layers_count: int, override_pattern: str) -> list:
    layer_type_list = list(override_pattern)
    override_pattern_length = len(layer_type_list)
    if override_pattern_length != total_layers_count:
        raise ValueError(
            "The hybrid override pattern is the wrong "
            f"length: got {override_pattern_length}, expected "
            f"{total_layers_count}"
        )
    for layer_type in layer_type_list:
        if layer_type not in Symbols.VALID:
            raise ValueError(f"In hybrid override pattern, '{layer_type}' is not " f"one of {Symbols.VALID}")

    return layer_type_list


def allocate_layers(
    total_layers_count: int,
    override_pattern: str,
) -> list:
    """Allocate the layers for the hybrid model."""
    layer_type_list = _allocate_override(total_layers_count, override_pattern)
    log_single_rank(logger, logging.INFO, "Using hybrid override pattern")
    actual_hyena_short_layers_count = layer_type_list.count(Symbols.HYENA_SHORT)
    actual_hyena_medium_layers_count = layer_type_list.count(Symbols.HYENA_MEDIUM)
    actual_hyena_layers_count = layer_type_list.count(Symbols.HYENA)
    actual_attention_layers_count = layer_type_list.count(Symbols.ATTENTION)
    allocation_string = ''.join(layer_type_list)
    log_single_rank(
        logger,
        logging.INFO,
        f"Hybrid allocation ({Symbols.HYENA_SHORT} is hyena_short_conv, "
        f"{Symbols.HYENA_MEDIUM} is hyena_medium_conv, "
        f"{Symbols.HYENA} is hyena, "
        f"{Symbols.ATTENTION} is attention, ",
    )
    log_single_rank(logger, logging.INFO, allocation_string)
    log_single_rank(
        logger,
        logging.INFO,
        f"{actual_hyena_short_layers_count} heyna_short_conv layers in " f"{total_layers_count} total layers.",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"{actual_hyena_medium_layers_count} heyna_medium_conv layers in " f"{total_layers_count} total layers.",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"{actual_hyena_layers_count} heyna layers in " f"{total_layers_count} total layers.",
    )
    log_single_rank(
        logger,
        logging.INFO,
        f"{actual_attention_layers_count} attention layers in " f"{total_layers_count} total layers.",
    )

    return layer_type_list


if __name__ == "__main__":
    test_cases = [
        (4, "SDH*"),
        (8, "SSDDH*H*"),
    ]
    for t in test_cases:
        print("")
        allocate_layers(*t)
