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
import torch

from nemo.collections.common.data.dataset import ConcatDataset


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_datasets():
    return [ListDataset([f"a{i}" for i in range(10)]), ListDataset([f"b{i}" for i in range(7)])]


@pytest.mark.unit
def test_iter_is_repeatable_across_epochs():
    a, b = make_datasets()
    concat = ConcatDataset(
        datasets=[a, b], sampling_technique='round-robin', shuffle=False, global_rank=0, world_size=2
    )
    first = list(iter(concat))
    second = list(iter(concat))
    assert first == second
    assert first == ["a0", "b0", "a1", "b1", "a2", "b2", "a3", "a4"]
    assert concat.datasets[0] is a
    assert concat.datasets[1] is b


@pytest.mark.unit
def test_rank_shards_match_expected_slices():
    a, b = make_datasets()
    rank0 = ConcatDataset(
        datasets=[a, b], sampling_technique='round-robin', shuffle=False, global_rank=0, world_size=2
    )
    rank1 = ConcatDataset(
        datasets=[a, b], sampling_technique='round-robin', shuffle=False, global_rank=1, world_size=2
    )
    assert list(iter(rank0)) == ["a0", "b0", "a1", "b1", "a2", "b2", "a3", "a4"]
    assert list(iter(rank1)) == ["a5", "b3", "a6", "b4", "a7", "b5", "a8", "b6"]


@pytest.mark.unit
def test_ranks_get_disjoint_shards():
    a = ListDataset([f"a{i}" for i in range(8)])
    b = ListDataset([f"b{i}" for i in range(8)])
    rank0 = ConcatDataset(
        datasets=[a, b], sampling_technique='round-robin', shuffle=False, global_rank=0, world_size=2
    )
    rank1 = ConcatDataset(
        datasets=[a, b], sampling_technique='round-robin', shuffle=False, global_rank=1, world_size=2
    )

    r0_first, r0_second = list(iter(rank0)), list(iter(rank0))
    r1_first, r1_second = list(iter(rank1)), list(iter(rank1))

    assert r0_first == r0_second == ["a0", "b0", "a1", "b1", "a2", "b2", "a3", "b3"]
    assert r1_first == r1_second == ["a4", "b4", "a5", "b5", "a6", "b6", "a7", "b7"]
    assert set(r0_first).isdisjoint(r1_first)
    assert sorted(set(r0_first) | set(r1_first)) == sorted([f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)])


@pytest.mark.unit
def test_world_size_one_is_stable_and_complete():
    a, b = make_datasets()
    concat = ConcatDataset(
        datasets=[a, b], sampling_technique='round-robin', shuffle=False, global_rank=0, world_size=1
    )
    expected = [
        "a0",
        "b0",
        "a1",
        "b1",
        "a2",
        "b2",
        "a3",
        "b3",
        "a4",
        "b4",
        "a5",
        "b5",
        "a6",
        "b6",
        "a7",
        "a8",
        "b0",
    ]
    first = list(iter(concat))
    second = list(iter(concat))
    assert first == second == expected
