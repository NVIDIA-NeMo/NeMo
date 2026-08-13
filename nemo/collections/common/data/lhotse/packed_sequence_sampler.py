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

from collections.abc import Sequence
from typing import Any

from lhotse.dataset import DynamicCutSampler
from lhotse.dataset.sampling.dynamic import DurationBatcher, Filter
from lhotse.lazy import get_graph_origin, resolve_iterator_source

from lhotse import CutSet


def _select_best_fit_indices(
    lengths: Sequence[int], capacity: int, max_items: int | None = None
) -> list[int]:
    """Select an exact best-fit subset, preferring earlier items on ties."""
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative (got {capacity})")
    if any(length < 0 for length in lengths):
        raise ValueError("lengths must be non-negative")
    if not lengths or capacity == 0 or max_items == 0:
        return []

    mask = (1 << (capacity + 1)) - 1

    if max_items is None or max_items >= len(lengths):
        # Prefix reachability makes reconstruction deterministic: when the
        # target was already reachable without the current (later) item, skip
        # it. Consequently equal-quality solutions favor earlier candidates.
        prefixes = [1]
        for length in lengths:
            reachable = prefixes[-1]
            prefixes.append(reachable | ((reachable << length) & mask))

        target = prefixes[-1].bit_length() - 1
        selected = []
        for index in range(len(lengths) - 1, -1, -1):
            if (prefixes[index] >> target) & 1:
                continue
            selected.append(index)
            target -= lengths[index]
        selected.reverse()
        return selected

    max_items = min(max_items, len(lengths))
    if max_items < 0:
        raise ValueError(f"max_items must be non-negative (got {max_items})")

    # Count-aware reachability is only needed when batch_size is configured.
    # Packed training normally leaves it null and takes the faster path above.
    layers = [1] + [0] * max_items
    prefixes = [tuple(layers)]
    for item_index, length in enumerate(lengths):
        for count in range(min(max_items, item_index + 1), 0, -1):
            layers[count] |= (layers[count - 1] << length) & mask
        prefixes.append(tuple(layers))

    reachable = 0
    for layer in layers:
        reachable |= layer
    target = reachable.bit_length() - 1
    # Prefer fewer examples when token utilization ties, then earlier examples.
    count = next(count for count, layer in enumerate(layers) if (layer >> target) & 1)
    selected = []
    for index in range(len(lengths) - 1, -1, -1):
        if (prefixes[index][count] >> target) & 1:
            continue
        selected.append(index)
        target -= lengths[index]
        count -= 1
    selected.reverse()
    return selected


class ExactTokenBatcher(DurationBatcher):
    """Bounded-lookahead best-fit batching for padding-free sequences."""

    def __init__(self, *args, packing_buffer_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        if packing_buffer_size <= 0:
            raise ValueError(
                "shuffle_buffer_size must be a positive packing-buffer size "
                f"(got {packing_buffer_size})"
            )
        self.packing_buffer_size = packing_buffer_size
        self._source_exhausted = False

    @staticmethod
    def _detuplify(examples):
        if isinstance(examples[0], tuple):
            if len(examples[0]) == 1:
                return CutSet.from_cuts(example[0] for example in examples)
            tuple_of_example_lists = list(zip(*examples))
            return tuple(CutSet.from_cuts(items) for items in tuple_of_example_lists)
        return CutSet.from_cuts(examples)

    @staticmethod
    def _measured_example(example_or_tuple):
        return (
            example_or_tuple[0]
            if isinstance(example_or_tuple, tuple)
            else example_or_tuple
        )

    def _fill_packing_buffer(self) -> None:
        while (
            len(self.reuse_cuts_buffer) < self.packing_buffer_size
            and not self._source_exhausted
        ):
            try:
                self.reuse_cuts_buffer.append(next(self.cuts_iter))
            except StopIteration:
                self._source_exhausted = True

    def _measure_integer_length(self, example_or_tuple) -> int:
        length = self.constraint.measure_length(
            self._measured_example(example_or_tuple)
        )
        integer_length = int(length)
        if integer_length != length:
            raise ValueError(
                "Packed sequence sampling requires integer token lengths, "
                f"but measured {length!r}."
            )
        return integer_length

    def _limits(self) -> tuple[int, int | None]:
        max_tokens = getattr(self.constraint, "batch_tokens", None)
        max_examples = getattr(self.constraint, "batch_size", None)
        if max_tokens is None:
            internal = getattr(self.constraint, "_internal", None)
            max_tokens = getattr(internal, "max_tokens", None)
            max_examples = getattr(internal, "max_examples", max_examples)
        if max_tokens is None:
            raise ValueError(
                "Packed sequence sampling requires batch_tokens to define the exact token cap."
            )
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            raise ValueError(f"batch_tokens must be positive (got {max_tokens})")
        if max_examples is not None:
            max_examples = int(max_examples)
            if max_examples <= 0:
                raise ValueError(
                    f"batch_size must be positive or null (got {max_examples})"
                )
        return max_tokens, max_examples

    def _discard(self, examples) -> None:
        try:
            self.diagnostics.discard(examples)
        except AttributeError:
            self.diagnostics.discard(examples[0])

    def _collect_batch(self):
        self._fill_packing_buffer()
        if not self.reuse_cuts_buffer:
            raise StopIteration()

        pool = list(self.reuse_cuts_buffer)
        lengths = [self._measure_integer_length(example) for example in pool]
        max_tokens, max_examples = self._limits()

        anchor_length = lengths[0]
        if anchor_length > max_tokens:
            raise ValueError(
                f"An individual example ({anchor_length} tokens) exceeds "
                f"batch_tokens={max_tokens}. Set max_tokens less than or equal "
                "to batch_tokens so it is filtered before batching."
            )

        remaining_items = None if max_examples is None else max_examples - 1
        tail_indices = _select_best_fit_indices(
            lengths[1:], max_tokens - anchor_length, max_items=remaining_items
        )
        selected_indices = {0, *(index + 1 for index in tail_indices)}
        examples = [
            example for index, example in enumerate(pool) if index in selected_indices
        ]
        deferred = [
            example
            for index, example in enumerate(pool)
            if index not in selected_indices
        ]
        self.reuse_cuts_buffer.clear()
        self.reuse_cuts_buffer.extend(deferred)

        self.constraint.reset()
        for example in examples:
            self.constraint.add(self._measured_example(example))
        if self.constraint.exceeded():
            raise AssertionError(
                "Best-fit packed batch exceeded its configured constraint."
            )

        is_final_batch = self._source_exhausted and not self.reuse_cuts_buffer
        if (
            is_final_batch
            and self.drop_last
            and not self.constraint.close_to_exceeding()
        ):
            self._discard(examples)
            raise StopIteration()

        return self._detuplify(examples)


class PackedSequenceDynamicCutSampler(DynamicCutSampler):
    """Dynamic sampler with one bounded best-fit pool and an exact token cap.

    ``shuffle_buffer_size`` controls the post-filter packing pool. The parent
    reservoir shuffler is intentionally disabled because indexed data sources
    already provide an O(1)-memory Feistel permutation.
    """

    def __init__(
        self,
        *args,
        shuffle: bool = False,
        shuffle_buffer_size: int = 20000,
        **kwargs,
    ):
        if shuffle_buffer_size is None or shuffle_buffer_size <= 0:
            raise ValueError(
                "shuffle_buffer_size must be a positive packing-buffer size "
                f"(got {shuffle_buffer_size})"
            )
        # Consume the public `shuffle` argument for config compatibility, but
        # do not allocate DynamicCutSampler's second, reservoir-style buffer.
        del shuffle
        super().__init__(
            *args,
            shuffle=False,
            shuffle_buffer_size=shuffle_buffer_size,
            **kwargs,
        )
        self._batcher = None
        self._restored_packing_buffer_tokens = []
        self._restored_legacy_examples = []
        self._inject_restored_packing_buffer = False

    def _uses_indexed_restore(self) -> bool:
        return bool(self.cuts) and all(
            getattr(source, "has_constant_time_access", False) for source in self.cuts
        )

    @staticmethod
    def _capture_packing_buffer_tokens(buffer) -> list[tuple[Any, ...]]:
        saved = []
        for example_or_tuple in buffer:
            examples = (
                example_or_tuple
                if isinstance(example_or_tuple, tuple)
                else (example_or_tuple,)
            )
            tokens = tuple(get_graph_origin(example) for example in examples)
            if any(token is None for token in tokens):
                raise RuntimeError(
                    "PackedSequenceDynamicCutSampler could not checkpoint its packing buffer: "
                    "an indexed candidate is missing graph-origin metadata."
                )
            saved.append(tokens)
        return saved

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        if self._uses_indexed_restore():
            if self._batcher is not None:
                state["packing_buffer_tokens"] = self._capture_packing_buffer_tokens(
                    self._batcher.reuse_cuts_buffer
                )
            else:
                state["packing_buffer_tokens"] = list(
                    self._restored_packing_buffer_tokens
                )
        else:
            # Replay restoration deterministically rebuilds the post-filter pool.
            state["packing_buffer_tokens"] = None
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        tokens = state_dict.pop("packing_buffer_tokens", None)
        self._restored_packing_buffer_tokens = [] if tokens is None else list(tokens)
        # Read checkpoints created by the previous one-candidate implementation.
        self._restored_legacy_examples = list(state_dict.pop("deferred_examples", []))
        self._batcher = None
        super().load_state_dict(state_dict)
        if self.shuffle_buffer_size is None or self.shuffle_buffer_size <= 0:
            raise ValueError(
                "Restored shuffle_buffer_size must be a positive packing-buffer size "
                f"(got {self.shuffle_buffer_size})"
            )

    def allow_iter_to_reset_state(self):
        super().allow_iter_to_reset_state()
        self._restored_packing_buffer_tokens = []
        self._restored_legacy_examples = []

    def _fast_forward(self):
        # Indexed restoration resumes each source after all buffered candidates;
        # reconstruct them by immutable graph token. O(N) replay rebuilds the pool.
        self._inject_restored_packing_buffer = self._uses_indexed_restore()
        try:
            super()._fast_forward()
        finally:
            self._inject_restored_packing_buffer = False
        self._restored_packing_buffer_tokens = []
        self._restored_legacy_examples = []

    def _restore_packing_buffer(self) -> list[tuple[Any, ...]]:
        active_sources = self._active_cuts or []
        restored = []
        for tokens in self._restored_packing_buffer_tokens:
            if len(tokens) != len(active_sources):
                raise RuntimeError(
                    "Packed sampler checkpoint source count does not match the active source graph: "
                    f"{len(tokens)} != {len(active_sources)}."
                )
            restored.append(
                tuple(
                    resolve_iterator_source(source)[token]
                    for source, token in zip(active_sources, tokens)
                )
            )
        restored.extend(self._restored_legacy_examples)
        return restored

    def _initialize_epoch_iterator(self, *, rebuild_sources: bool) -> None:
        if rebuild_sources or self._active_cuts is None:
            self._active_cuts = self._make_epoch_sources()
        source_iterators = [
            iter(resolve_iterator_source(source)) for source in self._active_cuts
        ]
        filtered_examples = Filter(
            iterator=zip(*source_iterators),
            predicate=lambda examples: all(
                self._filter_fn(example) for example in examples
            ),
            diagnostics=self.diagnostics,
        )
        self._batcher = ExactTokenBatcher(
            filtered_examples,
            max_duration=self.max_duration,
            max_cuts=self.max_cuts,
            constraint=self.constraint,
            drop_last=self.drop_last,
            quadratic_duration=self.quadratic_duration,
            diagnostics=self.diagnostics,
            packing_buffer_size=self.shuffle_buffer_size,
        )
        if self._inject_restored_packing_buffer:
            self._batcher.reuse_cuts_buffer.extend(self._restore_packing_buffer())
        self.cuts_iter = iter(self._batcher)
