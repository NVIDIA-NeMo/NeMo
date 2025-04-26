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


import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo.export.utils.model_loader import (
    TarFileSystemReader,
    load_model_weights,
    load_sharded_metadata_zarr,
    nemo_to_path,
    nemo_weights_directory,
)


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    # Create a temporary directory structure mimicking a NeMo checkpoint
    weights_dir = tmp_path / "model_weights"
    weights_dir.mkdir()

    # Create metadata.json
    metadata = {"sharded_backend": "torch_dist"}
    with open(weights_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return tmp_path


def test_nemo_to_path():
    # Test directory path
    dir_path = "/path/to/checkpoint"
    with patch("os.path.isdir", return_value=True):
        result = nemo_to_path(dir_path)
        assert isinstance(result, Path)
        assert str(result) == dir_path


def test_tar_file_system_reader():
    path = Path("/some/path")
    reader = TarFileSystemReader(path)
    assert reader.path == path


@patch("zarr.open")
def test_load_sharded_metadata_zarr(mock_zarr_open):
    checkpoint_dir = MagicMock()

    # Mock directory structure
    subdir = MagicMock()
    subdir.name = "test_tensor"
    subdir.is_dir.return_value = True
    subdir.__truediv__.return_value.exists.return_value = True
    checkpoint_dir.iterdir.return_value = [subdir]

    # Mock zarr array
    mock_array = MagicMock()
    mock_array.dtype.name = "float32"
    mock_array.__getitem__.return_value = np.array([1.0])
    mock_zarr_open.return_value = mock_array

    state_dict = load_sharded_metadata_zarr(checkpoint_dir)
    assert "test_tensor" in state_dict
    assert isinstance(state_dict["test_tensor"], torch.Tensor)


def test_nemo_weights_directory(mock_checkpoint_dir):
    # Test model_weights directory
    result = nemo_weights_directory(mock_checkpoint_dir)
    assert result == mock_checkpoint_dir / "model_weights"

    # Test weights directory
    shutil.rmtree(mock_checkpoint_dir / "model_weights")
    weights_dir = mock_checkpoint_dir / "weights"
    weights_dir.mkdir()
    result = nemo_weights_directory(mock_checkpoint_dir)
    assert result == weights_dir

    # Test fallback to checkpoint directory
    weights_dir.rmdir()
    result = nemo_weights_directory(mock_checkpoint_dir)
    assert result == mock_checkpoint_dir


@patch("nemo.export.utils.model_loader.load_sharded_metadata_zarr")
@patch("nemo.export.utils.model_loader.load_sharded_metadata_torch_dist")
def test_load_model_weights(mock_torch_dist, mock_zarr, mock_checkpoint_dir):
    # Test torch_dist backend
    load_model_weights(mock_checkpoint_dir)
    mock_torch_dist.assert_called_once()
    mock_zarr.assert_not_called()

    # Test zarr backend
    mock_torch_dist.reset_mock()
    metadata = {"sharded_backend": "zarr"}
    with open(mock_checkpoint_dir / "model_weights" / "metadata.json", "w") as f:
        json.dump(metadata, f)

    load_model_weights(mock_checkpoint_dir)
    mock_zarr.assert_called_once()
    mock_torch_dist.assert_not_called()

    # Test unsupported backend
    metadata = {"sharded_backend": "unsupported"}
    with open(mock_checkpoint_dir / "model_weights" / "metadata.json", "w") as f:
        json.dump(metadata, f)

    with pytest.raises(NotImplementedError):
        load_model_weights(mock_checkpoint_dir)
