from dataclasses import dataclass

import pytest
import torch

from nemo.collections.asr.parts.mixins.transcription import move_to_device


@dataclass
class _Batch:
    data: torch.Tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires GPUs.")
@pytest.mark.parametrize(
    "batch",
    [
        torch.tensor([0]),
        (torch.tensor([0]),),
        [torch.tensor([0])],
        {"data": torch.tensor([0])},
        _Batch(torch.tensor([0])),
        "not a tensor",
    ],
)
def test_transcription_move_to_device(batch):
    cuda_batch = move_to_device(batch, device="cuda")
    assert type(batch) == type(cuda_batch)
    if isinstance(batch, _Batch):
        assert cuda_batch.data.is_cuda
    elif isinstance(batch, dict):
        assert cuda_batch["data"].is_cuda
    elif isinstance(batch, (list, tuple)):
        assert cuda_batch[0].is_cuda
    elif isinstance(batch, torch.Tensor):
        assert cuda_batch.is_cuda
    else:
        assert cuda_batch == batch
