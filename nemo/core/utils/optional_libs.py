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

import importlib.util


def is_lib_available(name: str) -> bool:
    """
    Checks if the library/package with `name` is available in the system
    NB: try/catch with importlib.import_module(name) requires importing the library, which can be slow.
    So, `find_spec` should be preferred
    """
    return importlib.util.find_spec(name) is not None


TRITON_AVAILABLE = is_lib_available("triton")

try:
    from nemo.core.utils.k2_guard import k2 as _

    K2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    K2_AVAILABLE = False
