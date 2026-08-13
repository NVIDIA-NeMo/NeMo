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
from lhotse.dataset.iterable_dataset import IdentityDataset, IterableDatasetWrapper
from lhotse.indexing import create_jsonl_index
from lhotse.lazy import LazyIndexedManifestIterator
from lhotse.testing.dummies import dummy_cut
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.audio_token_estimator import (
    AudioTokenEstimator,
)
from nemo.collections.common.data.lhotse.dataloader import (
    get_lhotse_sampler_from_config,
    make_structured_with_schema_warnings,
)
from nemo.collections.common.data.lhotse.packed_sequence_sampler import (
    PackedSequenceDynamicCutSampler,
    _select_best_fit_indices,
)
from nemo.collections.common.data.lhotse.sampling import (
    MultimodalSamplingConstraint,
)


def _make_cuts(durations=(7.0, 4.0, 6.0, 3.0)):
    return CutSet.from_cuts(
        dummy_cut(index, duration=duration) for index, duration in enumerate(durations)
    )


def _make_sampler(
    *cuts,
    batch_tokens=10,
    batch_size=None,
    shuffle_buffer_size=4,
):
    return PackedSequenceDynamicCutSampler(
        *cuts,
        constraint=MultimodalSamplingConstraint(
            token_equivalent_duration=1.0,
            audio_token_estimator=AudioTokenEstimator.from_config(
                {
                    "preprocessor": {
                        "n_fft": 16000,
                        "hop_length": 16000,
                        "stft_pad_amount": 8000,
                    },
                    "subsampling": [],
                },
                sample_rate=16000,
            ),
            batch_size=batch_size,
            batch_tokens=batch_tokens,
            measure_total_length=False,
            use_packed_sequence_sampling=True,
        ),
        # PackedSequenceDynamicCutSampler consumes this value but deliberately
        # passes shuffle=False to its DynamicCutSampler parent.
        shuffle=True,
        shuffle_buffer_size=shuffle_buffer_size,
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
                "shuffle": True,
                "shuffle_buffer_size": 4,
                "num_workers": 0,
                "use_multimodal_sampling": True,
                "pretokenize": False,
                "batch_tokens": 10,
                "token_equivalent_duration": 1.0,
                "audio_token_estimator": {
                    "preprocessor": {
                        "n_fft": 16000,
                        "hop_length": 16000,
                        "stft_pad_amount": 8000,
                    },
                    "subsampling": [],
                },
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
    assert sampler.shuffle_buffer_size == 4
    assert sampler.shuffle is not packed


def test_best_fit_subset_is_exact_and_prefers_earlier_candidates():
    assert _select_best_fit_indices([6, 6, 3], capacity=6) == [0]
    assert _select_best_fit_indices([5, 3, 2], capacity=5) == [0]
    assert _select_best_fit_indices([4, 3, 2], capacity=5) == [1, 2]
    assert _select_best_fit_indices([4, 3, 2], capacity=7, max_items=1) == [0]
    assert _select_best_fit_indices([4, 3, 2], capacity=7, max_items=2) == [0, 1]


def test_packed_sequence_sampler_backfills_and_enforces_exact_token_cap(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)

    batches = list(_make_sampler(CutSet.from_jsonl_lazy(cuts_path)))

    assert _batch_ids(batches) == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0003"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0002"],
    ]
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [10, 10]
    assert all(sum(cut.num_tokens for cut in batch) <= 10 for batch in batches)


def test_packed_sequence_sampler_squeezes_around_large_next_candidate(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((9.0, 12.0, 4.0, 3.0)).to_file(cuts_path)

    batches = list(
        _make_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            batch_tokens=16,
            shuffle_buffer_size=4,
        )
    )

    assert _batch_ids(batches) == [
        [
            "dummy-mono-cut-0000",
            "dummy-mono-cut-0002",
            "dummy-mono-cut-0003",
        ],
        ["dummy-mono-cut-0001"],
    ]
    assert [sum(cut.num_tokens for cut in batch) for batch in batches] == [16, 12]


def test_packed_sequence_sampler_anchor_prevents_starvation(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((6.0, 6.0, 4.0, 4.0, 4.0, 4.0)).to_file(cuts_path)

    batches = list(
        _make_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            batch_tokens=10,
            shuffle_buffer_size=4,
        )
    )

    # Candidate 1 cannot fit beside candidate 0, but becomes the mandatory
    # anchor in the very next batch rather than being perpetually bypassed.
    assert _batch_ids(batches)[:2] == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0002"],
        ["dummy-mono-cut-0001", "dummy-mono-cut-0003"],
    ]
    assert sorted(cut_id for batch in _batch_ids(batches) for cut_id in batch) == [
        f"dummy-mono-cut-{index:04d}" for index in range(6)
    ]


def test_packed_sequence_sampler_honors_max_examples(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts((4.0, 3.0, 3.0, 2.0)).to_file(cuts_path)

    batches = list(
        _make_sampler(
            CutSet.from_jsonl_lazy(cuts_path),
            batch_tokens=10,
            batch_size=2,
            shuffle_buffer_size=4,
        )
    )

    assert all(len(batch) <= 2 for batch in batches)
    assert _batch_ids(batches)[0] == [
        "dummy-mono-cut-0000",
        "dummy-mono-cut-0001",
    ]


@pytest.mark.parametrize("indexed", [False, True])
def test_packed_sequence_sampler_resume_preserves_packing_buffer(tmp_path, indexed):
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
    assert _batch_ids([next(iterator)]) == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0003"]
    ]
    state = sampler.state_dict()
    if indexed:
        assert state["packing_buffer_tokens"] == [(1,), (2,)]
    else:
        assert state["packing_buffer_tokens"] is None
    expected_remaining = _batch_ids(_drain(iterator))

    resumed = _make_sampler(make_source())
    resumed.load_state_dict(deepcopy(state))
    actual_remaining = _batch_ids(_drain(iter(resumed)))

    assert (
        actual_remaining
        == expected_remaining
        == [["dummy-mono-cut-0001", "dummy-mono-cut-0002"]]
    )


def test_packed_sequence_sampler_indexed_state_contains_only_origin_tokens(tmp_path):
    cuts_path = tmp_path / "cuts.jsonl"
    _make_cuts().to_file(cuts_path)
    create_jsonl_index(cuts_path)
    sampler = _make_sampler(CutSet(LazyIndexedManifestIterator(cuts_path)))
    iterator = iter(sampler)
    next(iterator)

    live_buffer = list(sampler._batcher.reuse_cuts_buffer)
    state = sampler.state_dict()

    assert state["packing_buffer_tokens"] == [(1,), (2,)]
    assert all(
        isinstance(token, int)
        for candidate_tokens in state["packing_buffer_tokens"]
        for token in candidate_tokens
    )
    live_buffer[0][0].custom["mutated_after_snapshot"] = True
    assert state["packing_buffer_tokens"] == [(1,), (2,)]


def test_packed_sequence_sampler_indexed_tuple_source_resume(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _make_cuts().to_file(first_path)
    _make_cuts((1.0, 1.0, 1.0, 1.0)).to_file(second_path)
    create_jsonl_index(first_path)
    create_jsonl_index(second_path)

    def make_sources():
        return (
            CutSet(LazyIndexedManifestIterator(first_path)),
            CutSet(LazyIndexedManifestIterator(second_path)),
        )

    sampler = _make_sampler(*make_sources())
    iterator = iter(sampler)
    first_batch = next(iterator)
    assert isinstance(first_batch, tuple)
    assert _batch_ids([first_batch[0]]) == [
        ["dummy-mono-cut-0000", "dummy-mono-cut-0003"]
    ]
    state = sampler.state_dict()
    assert state["packing_buffer_tokens"] == [(1, 1), (2, 2)]
    expected = [
        (_batch_ids([left]), _batch_ids([right])) for left, right in _drain(iterator)
    ]

    resumed = _make_sampler(*make_sources())
    resumed.load_state_dict(deepcopy(state))
    actual = [
        (_batch_ids([left]), _batch_ids([right]))
        for left, right in _drain(iter(resumed))
    ]

    assert actual == expected


def test_packed_sequence_sampler_multiworker_stateful_resume(tmp_path):
    stateful_dataloader = pytest.importorskip("torchdata.stateful_dataloader")
    StatefulDataLoader = stateful_dataloader.StatefulDataLoader
    cuts_path = tmp_path / "cuts.jsonl"
    CutSet.from_cuts(
        dummy_cut(index, duration=[7.0, 4.0, 6.0, 3.0][index % 4])
        for index in range(160)
    ).to_file(cuts_path)
    create_jsonl_index(cuts_path)

    def make_loader():
        source = CutSet(LazyIndexedManifestIterator(cuts_path))
        wrapper = IterableDatasetWrapper(
            IdentityDataset(),
            _make_sampler(source, shuffle_buffer_size=16),
        )
        return StatefulDataLoader(
            wrapper,
            batch_size=None,
            num_workers=2,
            persistent_workers=True,
            snapshot_every_n_steps=1,
            timeout=10,
        )

    def consume(loader):
        return _batch_ids(list(loader))

    full = make_loader()
    expected = consume(full)
    full._iterator._shutdown_workers()

    partial = make_loader()
    iterator = iter(partial)
    prefix = _batch_ids([next(iterator) for _ in range(24)])
    state = partial.state_dict()
    partial._iterator._shutdown_workers()

    resumed = make_loader()
    resumed.load_state_dict(deepcopy(state))
    suffix = consume(resumed)
    resumed._iterator._shutdown_workers()

    assert prefix + suffix == expected


def test_packed_sequence_sampler_rejects_single_example_over_budget():
    cuts = CutSet.from_cuts([dummy_cut(0, duration=11.0)])
    with pytest.warns(UserWarning, match="eagerly read CutSet"):
        sampler = _make_sampler(cuts)
    iterator = iter(sampler)
    with pytest.raises(ValueError, match="individual example.*exceeds batch_tokens=10"):
        next(iterator)


def test_packed_sequence_sampler_rejects_invalid_buffer_size():
    with pytest.raises(ValueError, match="positive packing-buffer size"):
        _make_sampler(_make_cuts(), shuffle_buffer_size=0)
