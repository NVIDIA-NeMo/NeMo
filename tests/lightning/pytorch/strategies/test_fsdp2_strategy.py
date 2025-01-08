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

from importlib.metadata import version
from unittest.mock import patch

from packaging.version import Version as PkgVersion

from nemo.lightning.pytorch.strategies import FSDP2Strategy


def get_torch_version_str():
    import torch

    if hasattr(torch, '__version__'):
        return str(torch.__version__)
    else:
        return version("torch")


if PkgVersion(get_torch_version_str()) >= PkgVersion("2.4"):

    class TestFSDP2Strategy:
        @patch('nemo.lightning.pytorch.strategies.fsdp2_strategy.create_checkpoint_io')
        def test_checkpoint_io(self, mock_create_checkpoint_io):
            class Dummy: ...

            mock_create_checkpoint_io.side_effect = lambda *args, **kwargs: Dummy()
            strategy = FSDP2Strategy()

            first_io = strategy.checkpoint_io
            mock_create_checkpoint_io.assert_called_once()

            assert first_io == strategy.checkpoint_io

            new_io = object()
            strategy.checkpoint_io = new_io
            assert new_io == strategy.checkpoint_io

            strategy2 = FSDP2Strategy()
            second_io = strategy2.checkpoint_io
            mock_create_checkpoint_io.assert_called()

            assert first_io != second_io
            assert second_io == strategy2.checkpoint_io
