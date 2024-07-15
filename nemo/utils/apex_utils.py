import warnings
from typing import List, Optional

import torch


def _reconfigure_microbatch_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
) -> None:

    import megatron.core.num_microbatches_calculator as mb_calculator

    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = mb_calculator.build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size
    )


def get_micro_batch_size():
    from megatron.core.num_microbatches_calculator import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.micro_batch_size
