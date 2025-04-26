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

import os
import tarfile
import tempfile
import typing

import numpy as np
import pytest
import torch
from PIL import Image
from pytriton.model_config import Tensor

from nemo.deploy.utils import (
    NEMO1,
    NEMO2,
    broadcast_list,
    cast_output,
    ndarray2img,
    nemo_checkpoint_version,
    str_list2numpy,
    str_ndarray2list,
    typedict2tensor,
)


class TestTypedict2Tensor:
    class SampleTypedict:
        int_field: int
        float_field: float
        bool_field: bool
        str_field: str
        int_list: typing.List[int]
        float_list: typing.List[float]
        bool_list: typing.List[bool]
        str_list: typing.List[str]

    def test_typedict2tensor_basic(self):
        tensors = typedict2tensor(self.SampleTypedict)
        assert len(tensors) == 8
        assert all(isinstance(t, Tensor) for t in tensors)

        # Check int field
        int_tensor = next(t for t in tensors if t.name == "int_field")
        assert int_tensor.dtype == np.int32
        assert int_tensor.shape == (1,)

        # Check float field
        float_tensor = next(t for t in tensors if t.name == "float_field")
        assert float_tensor.dtype == np.float32
        assert float_tensor.shape == (1,)

        # Check bool field
        bool_tensor = next(t for t in tensors if t.name == "bool_field")
        assert bool_tensor.dtype == np.bool_
        assert bool_tensor.shape == (1,)

        # Check str field
        str_tensor = next(t for t in tensors if t.name == "str_field")
        assert str_tensor.dtype == bytes
        assert str_tensor.shape == (1,)

    def test_typedict2tensor_with_overwrite(self):
        overwrite_kwargs = {"optional": True}
        tensors = typedict2tensor(self.SampleTypedict, overwrite_kwargs=overwrite_kwargs)
        assert all(t.optional for t in tensors)

    def test_typedict2tensor_list_types(self):
        tensors = typedict2tensor(self.SampleTypedict)

        # Check int list
        int_list_tensor = next(t for t in tensors if t.name == "int_list")
        assert int_list_tensor.dtype == np.int32
        assert int_list_tensor.shape == (1,)

        # Check float list
        float_list_tensor = next(t for t in tensors if t.name == "float_list")
        assert float_list_tensor.dtype == np.float32
        assert float_list_tensor.shape == (1,)

        # Check bool list
        bool_list_tensor = next(t for t in tensors if t.name == "bool_list")
        assert bool_list_tensor.dtype == np.bool_
        assert bool_list_tensor.shape == (1,)

        # Check str list
        str_list_tensor = next(t for t in tensors if t.name == "str_list")
        assert str_list_tensor.dtype == bytes
        assert str_list_tensor.shape == (1,)


class TestNemoCheckpointVersion:
    def test_nemo2_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NEMO 2.0 structure
            os.makedirs(os.path.join(tmpdir, "context"))
            os.makedirs(os.path.join(tmpdir, "weights"))
            assert nemo_checkpoint_version(tmpdir) == NEMO2

    def test_nemo1_checkpoint_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NEMO 1.0 structure (no context/weights dirs)
            assert nemo_checkpoint_version(tmpdir) == NEMO1

    def test_nemo2_checkpoint_tar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "checkpoint.tar")
            with tarfile.open(tar_path, "w") as tar:
                # Create NEMO 2.0 structure in tar
                context_info = tarfile.TarInfo("context")
                context_info.type = tarfile.DIRTYPE
                tar.addfile(context_info)

                weights_info = tarfile.TarInfo("weights")
                weights_info.type = tarfile.DIRTYPE
                tar.addfile(weights_info)

            assert nemo_checkpoint_version(tar_path) == NEMO2

    def test_nemo1_checkpoint_tar(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "checkpoint.tar")
            with tarfile.open(tar_path, "w") as tar:
                # Create empty tar (NEMO 1.0)
                pass

            assert nemo_checkpoint_version(tar_path) == NEMO1


class TestStringConversions:
    def test_str_list2numpy(self):
        input_list = ["hello", "world", "test"]
        result = str_list2numpy(input_list)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)
        assert all(isinstance(x, bytes) for x in result.flatten())

    def test_str_ndarray2list(self):
        input_array = np.array([b"hello", b"world", b"test"]).reshape(3, 1)
        result = str_ndarray2list(input_array)
        assert isinstance(result, list)
        assert result == ["hello", "world", "test"]

    def test_str_conversion_roundtrip(self):
        input_list = ["hello", "world", "test"]
        numpy_array = str_list2numpy(input_list)
        output_list = str_ndarray2list(numpy_array)
        assert input_list == output_list


class TestImageConversions:
    def test_ndarray2img(self):
        # Create a test image array
        img_array = np.random.randint(0, 255, size=(2, 100, 100, 3), dtype=np.uint8)
        result = ndarray2img(img_array)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(img, Image.Image) for img in result)
        assert all(img.size == (100, 100) for img in result)


class TestCastOutput:
    def test_cast_tensor(self):
        input_tensor = torch.tensor([1, 2, 3])
        result = cast_output(input_tensor, np.int32)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        assert result.shape == (3, 1)

    def test_cast_numpy(self):
        input_array = np.array([1, 2, 3])
        result = cast_output(input_array, np.float32)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3, 1)

    def test_cast_string(self):
        input_list = ["hello", "world"]
        result = cast_output(input_list, bytes)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 1)

    def test_cast_1d_to_2d(self):
        input_array = np.array([1, 2, 3])
        result = cast_output(input_array, np.int32)
        assert result.ndim == 2
        assert result.shape == (3, 1)


class TestBroadcastList:
    def test_broadcast_list_no_distributed(self):
        with pytest.raises(RuntimeError, match="Distributed environment is not initialized"):
            broadcast_list(["test"])

    def test_broadcast_list_distributed(self, monkeypatch):
        # Mock distributed environment
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

        # Mock broadcast_object_list
        def mock_broadcast_object_list(object_list, src, group=None):
            if src == 0:
                object_list[0] = ["test"]

        monkeypatch.setattr(torch.distributed, "broadcast_object_list", mock_broadcast_object_list)

        result = broadcast_list(["test"])
        assert result == ["test"]
