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

import pytest

from nemo.export.utils._mock_import import _mock_import


def test_mock_import_existing_module():
    """Test mocking an existing module."""
    import math as math_org

    with _mock_import("math"):
        import math

        assert math is math_org


def test_mock_import_non_existing_module():
    """Test mocking a non-existing module."""
    with _mock_import("non.existing.module"):
        import non.existing.module

    with pytest.raises(ModuleNotFoundError):
        import non.existing.module
