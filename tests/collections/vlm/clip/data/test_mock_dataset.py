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


from unittest.mock import MagicMock

import pytest
import torch

from nemo.collections.vlm.clip.data.mock import MockDataModule, _MockClipDataset


@pytest.fixture
def mock_tokenizer():
    """
    Provides a mock tokenizer that simulates huggingface's tokenizer behavior.
    """
    tokenizer = MagicMock()
    tokenizer.tokenizer = tokenizer
    tokenizer.vocab_size = 1024
    return tokenizer


@pytest.fixture
def mock_image_processor():
    """
    Provides a mock image processor that simulates CLIP's image processor behavior.
    """
    processor = MagicMock()
    processor.crop_size = {"height": 224, "width": 224}
    return processor


@pytest.fixture
def mock_data_module(mock_tokenizer, mock_image_processor):
    """Fixture to create and set up a mock data module."""
    dm = MockDataModule(
        seq_length=16,
        tokenizer=mock_tokenizer,
        image_processor=mock_image_processor,
        micro_batch_size=2,
        num_train_samples=20,
        num_val_samples=5,
        num_test_samples=5,
    )
    dm.setup()
    return dm


def test_mock_clip_dataset_length(mock_tokenizer, mock_image_processor):
    """Ensure the dataset's length matches the configured number of samples."""
    ds = _MockClipDataset(
        tokenizer=mock_tokenizer,
        image_processor=mock_image_processor,
        name="train",
        num_samples=100,
        seq_length=16,
    )
    assert len(ds) == 100


def test_mock_clip_dataset_item_shapes(mock_tokenizer, mock_image_processor):
    """Check that a sample has the expected keys and shapes."""
    seq_length = 16
    ds = _MockClipDataset(
        tokenizer=mock_tokenizer,
        image_processor=mock_image_processor,
        name="train",
        num_samples=1,
        seq_length=seq_length,
    )
    sample = ds[0]
    assert "images" in sample
    assert "captions" in sample
    assert sample["images"].shape == (3, 224, 224)
    assert len(sample["captions"]) == seq_length


def test_data_module_train_dataloader(mock_data_module):
    """Check the train dataloader returns batches of the expected shape."""
    train_dl = mock_data_module.train_dataloader()
    batch = next(iter(train_dl))
    assert isinstance(batch, dict)
    assert "images" in batch
    assert "captions" in batch
    assert batch["images"].shape == torch.Size([2, 3, 224, 224])
    assert batch["captions"].shape == torch.Size([2, 16])


def test_data_module_val_dataloader(mock_data_module):
    """Check the val dataloader returns a non-empty dataset."""
    val_dl = mock_data_module.val_dataloader()
    val_batch = next(iter(val_dl))
    assert val_batch["images"].shape == torch.Size([2, 3, 224, 224])
    assert val_batch["captions"].shape == torch.Size([2, 16])


def test_data_module_test_dataloader(mock_data_module):
    """Check the test dataloader returns a non-empty dataset."""
    test_dl = mock_data_module.test_dataloader()
    test_batch = next(iter(test_dl))
    assert test_batch["images"].shape == torch.Size([2, 3, 224, 224])
    assert test_batch["captions"].shape == torch.Size([2, 16])


def test_data_module_state_dict(mock_data_module):
    """Test state dict saving and loading."""
    state = mock_data_module.state_dict()
    assert isinstance(state, dict)
    # Since we don't have a trainer connected, it should return empty dict
    assert len(state) == 0


def test_data_module_with_task_encoder(mock_tokenizer, mock_image_processor):
    """Test data module with task encoder integration."""
    task_encoder = MagicMock()
    # Mock return value for encode_sample
    task_encoder.encode_sample.return_value = {
        "input_ids": torch.ones(16, dtype=torch.long),
        "pixel_values": torch.ones(3, 224, 224),
    }

    dm = MockDataModule(
        seq_length=16,
        tokenizer=mock_tokenizer,
        image_processor=mock_image_processor,
        micro_batch_size=2,
        num_train_samples=2,  # Reduced sample size for testing
        num_val_samples=2,
        num_test_samples=2,
        task_encoder=task_encoder,
    )
    dm.setup()

    # Get a single item from the dataset to verify task encoder is called
    sample = dm._train_ds[0]
    task_encoder.encode_sample.assert_called_once()

    # Verify the returned sample has the expected format
    assert "input_ids" in sample
    assert "pixel_values" in sample
    assert isinstance(sample["input_ids"], torch.Tensor)
    assert isinstance(sample["pixel_values"], torch.Tensor)
    assert sample["input_ids"].shape == (16,)
    assert sample["pixel_values"].shape == (3, 224, 224)


def test_data_module_without_processors():
    """Test data module initialization without processors."""
    dm = MockDataModule(
        seq_length=16,
        micro_batch_size=2,
    )
    dm.setup()

    # Should use default processors
    assert dm.tokenizer is not None
    assert dm.image_processor is not None


def test_mock_clip_dataset_seed_consistency(mock_tokenizer, mock_image_processor):
    """Test that the same seed produces consistent results."""
    ds1 = _MockClipDataset(
        tokenizer=mock_tokenizer,
        image_processor=mock_image_processor,
        name="train",
        num_samples=10,
        seq_length=16,
        seed=42,
    )

    ds2 = _MockClipDataset(
        tokenizer=mock_tokenizer,
        image_processor=mock_image_processor,
        name="train",
        num_samples=10,
        seq_length=16,
        seed=42,
    )

    # Check if same seed produces same results
    sample1 = ds1[0]
    sample2 = ds2[0]
    assert torch.equal(sample1["images"], sample2["images"])
    assert torch.equal(sample1["captions"], sample2["captions"])


def test_data_module_batch_collation(mock_data_module):
    """Test the collation function of the data module."""
    train_dl = mock_data_module.train_dataloader()
    batch = next(iter(train_dl))

    # Test if attention mask is None as specified in collate_fn
    assert "attention_mask" in batch
    assert batch["attention_mask"] is None
