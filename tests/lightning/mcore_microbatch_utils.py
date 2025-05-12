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

import contextlib


# @akoumparouli: use a context manager that saves/restores gbs/mbs when using
# reconfigure_num_microbatches_calculator to avoid interference between tests.
@contextlib.contextmanager
def reconfigure_num_microbatches_calculator_manager(*args, **kwargs):
    import megatron.core.num_microbatches_calculator as mb_calc

    # Store current mbs, gbs values
    if not mb_calc._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        _mbs = mb_calc.get_micro_batch_size()
        _gbs = mb_calc.get_current_global_batch_size()

        # use user's settings
        mb_calc.reconfigure_num_microbatches_calculator(*args, **kwargs)
    else:
        _mbs, _gbs = 1, 1

    try:
        # run user's code
        yield
        # @akoumparouli: no catch
    finally:
        # restore old mbs, gbs
        if not mb_calc._GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
            mb_calc.reconfigure_num_microbatches_calculator(0, None, _gbs, _mbs, data_parallel_size=1)
