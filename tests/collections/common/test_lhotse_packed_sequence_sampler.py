# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from copy import deepcopy

import pytest
from lhotse import CutSet
from lhotse.dataset import DynamicCutSampler
from lhotse.indexing import create_jsonl_index
from lhotse.lazy import LazyIndexedManifestIterator
from lhotse.testing.dummies import dummy_cut
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.dataloader import (
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
)
from nemo.collections.common.data.lhotse.packed_sequence_sampler import (
    PackedSequenceDynamicCutSampler,
)
from nemo.collections.common.data.lhotse.sampling import MultimodalSamplingConstraint


def _make_cuts():
    return CutSet.from_cuts(
        dummy_cut(index, duration=duration)
        for index, duration in enumerate([7.0, 4.0, 6.0, 3.0])
    )


def _make_sampler(cuts):
    return PackedSequenceDynamicCutSampler(
        cuts,
        constraint=MultimodalSamplingConstraint(
            token_equivalent_duration=1.0,
            batch_tokens=10,
            measure_total_length=False,
            use_packed_sequence_sampling=True,
        ),
        shuffle=False,
        seed=0,
    )


def _drain(iterator):
    batches = []
    while True:
        try:
            batches.append(next(iterator))
        except StopIteration:
            return batches


def _batch_ids(batches):
    return [[cut.id for cut in batch] for batch in batches]


@pytest.mark.parametrize(
    ("packed", "sampler_type"),
    [(False, DynamicCutSampler), (True, PackedSequenceDynamicCutSampler)],
)
def test_packed_sequence_config_flag_selects_exact_sampler(
    tmp_path, packed, sampler_type
):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    config = make_structured_with_schema_warnings(
        OmegaConf.create(
            {
                "cuts_path": str(cuts_path),
                "force_finite": True,
                "shuffle": False,
                "num_workers": 0,
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_tokens": 10,
                "token_equivalent_duration": 1.0,
                "use_packed_sequence_sampling": packed,
                "measure_total_length": False,
                "seed": 0,
                "shard_seed": 0,
            }
        )
    )

    sampler, _ = get_lhotse_sampler_from_config(
        config,
        global_rank=0,
        world_size=1,
        tokenizer=object(),
    )

    assert type(sampler) is sampler_type


def test_packed_sequence_sampler_enforces_exact_token_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path)))

    assert _batch_ids(batches) == [
        ["dummy-mono-cut-0000"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0002"],
        ["dummy-mono-cut-0003"],
    ]
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [7, 10, 3]
    assert all(sum(cut.num_tokens for cut in batch) <= 10 for batch in batches)


@pytest.mark.parametrize("indexed", [False, True])
def test_packed_sequence_sampler_resume_preserves_deferred_example(tmp_path, indexed):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    if indexed:
        create_jsonl_index(cuts_path)

        def make_source():
            return CutSet(LazyIndexedManifestIterator(cuts_path))

    else:

        def make_source():
            return CutSet.from_jsonl_lazy(cuts_path)

    sampler = _make_sampler(make_source())
    iterator = iter(sampler)
    assert _batch_ids([next(iterator)]) == [["dummy-mono-cut-0000"]]
    state = sampler.state_dict()
    assert len(state["deferred_examples"]) == 1
    expected_remaining = _batch_ids(_drain(iterator))

    resumed = _make_sampler(make_source())
    resumed.load_state_dict(deepcopy(state))
    actual_remaining = _batch_ids(_drain(iter(resumed)))

    assert (
        actual_remaining
        == expected_remaining
        == [
            ["dummy-mono-cut-0001", "dummy-mono-cut-0002"],
            ["dummy-mono-cut-0003"],
        ]
    )


def test_packed_sequence_sampler_rejects_single_example_over_budget():
    cuts = CutSet.from_cuts([dummy_cut(0, duration=11.0)])
    with pytest.warns(UserWarning, match="eagerly read CutSet"):
        sampler = _make_sampler(cuts)
    iterator = iter(sampler)
    with pytest.raises(ValueError, match="individual example.*exceeds batch_tokens=10"):
        next(iterator)
