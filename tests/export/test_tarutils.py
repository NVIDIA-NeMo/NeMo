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

import tarfile
import tempfile
from pathlib import Path

import pytest

from nemo.export.tarutils import TarPath


@pytest.fixture
def sample_tar():
    # Create a temporary directory and tar file with sample content
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        test_dir = Path(temp_dir) / "test_dir"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "file3.txt").write_text("content3")

        # Create tar file
        tar_path = Path(temp_dir) / "test.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(test_dir, arcname=".")

        yield str(tar_path)


def test_tar_path_initialization(sample_tar):
    # Test initialization with string path
    with TarPath(sample_tar) as path:
        assert isinstance(path, TarPath)
        assert path.exists()

    # Test initialization with tarfile object
    with tarfile.open(sample_tar, "r") as tar:
        path = TarPath(tar)
        assert isinstance(path, TarPath)
        assert path.exists()


def test_path_operations(sample_tar):
    with TarPath(sample_tar) as root:
        # Test path division
        file_path = root / "file1.txt"
        assert str(file_path) == f"{sample_tar}/file1.txt"

        # Test nested path division
        subdir_path = root / "subdir" / "file3.txt"
        assert str(subdir_path) == f"{sample_tar}/subdir/file3.txt"

        # Test name property
        assert file_path.name == "file1.txt"
        assert subdir_path.name == "file3.txt"

        # Test suffix property
        assert file_path.suffix == ".txt"
        assert (root / "subdir").suffix == ""


def test_file_operations(sample_tar):
    with TarPath(sample_tar) as root:
        # Test file existence
        assert (root / "file1.txt").exists()
        assert (root / "file1.txt").is_file()

        # Test directory existence
        assert (root / "subdir").exists()
        assert (root / "subdir").is_dir()

        # Test non-existent path
        assert not (root / "nonexistent.txt").exists()

        # Test file reading
        with (root / "file1.txt").open("r") as f:
            content = f.read()
            assert content == b"content1"


def test_directory_operations(sample_tar):
    with TarPath(sample_tar) as root:
        # Test iterdir
        entries = list(root.iterdir())
        assert len(entries) == 5  # file1.txt, file2.txt, subdir, ., file3.txt

        # Test glob
        txt_files = list(root.glob("*.txt"))
        assert len(txt_files) == 3
        assert all(f.suffix == ".txt" for f in txt_files)

        # Test rglob
        all_txt_files = list(root.rglob("*.txt"))
        assert len(all_txt_files) == 3
        assert all(f.suffix == ".txt" for f in all_txt_files)


def test_error_handling(sample_tar):
    with TarPath(sample_tar) as root:
        # Test opening non-existent file
        with pytest.raises(FileNotFoundError):
            (root / "nonexistent.txt").open("r")

        # Test invalid mode
        with pytest.raises(NotImplementedError):
            (root / "file1.txt").open("w")

        # Test invalid initialization
        with pytest.raises(ValueError):
            TarPath(123)  # Invalid type
