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

from typing import Any

from lhotse import CutSet
from lhotse.dataset import DynamicCutSampler
from lhotse.dataset.sampling.dynamic import DurationBatcher, Filter
from lhotse.lazy import resolve_iterator_source


class ExactTokenBatcher(DurationBatcher):
    """Candidate-aware next-fit batching for padding-free packed sequences."""

    @staticmethod
    def _detuplify(examples):
        if isinstance(examples[0], tuple):
            if len(examples[0]) == 1:
                return CutSet.from_cuts(example[0] for example in examples)
            tuple_of_example_lists = list(zip(*examples))
            return tuple(CutSet.from_cuts(items) for items in tuple_of_example_lists)
        return CutSet.from_cuts(examples)

    def _collect_batch(self):
        self.constraint.reset()
        examples = []
        while True:
            try:
                if self.reuse_cuts_buffer:
                    next_example_or_tuple = self.reuse_cuts_buffer.popleft()
                else:
                    next_example_or_tuple = next(self.cuts_iter)
            except StopIteration:
                if examples and (
                    not self.drop_last or self.constraint.close_to_exceeding()
                ):
                    return self._detuplify(examples)
                try:
                    self.diagnostics.discard(examples)
                except AttributeError:
                    self.diagnostics.discard(examples[0])
                raise StopIteration()

            measured_example = (
                next_example_or_tuple[0]
                if isinstance(next_example_or_tuple, tuple)
                else next_example_or_tuple
            )
            if self.constraint.would_exceed(measured_example):
                if not examples:
                    measured_tokens = self.constraint.measure_length(measured_example)
                    raise ValueError(
                        f"An individual example ({measured_tokens} tokens) exceeds "
                        f"batch_tokens={self.constraint.batch_tokens}. Set max_tokens "
                        "less than or equal to batch_tokens so it is filtered before batching."
                    )
                self.reuse_cuts_buffer.appendleft(next_example_or_tuple)
                break

            examples.append(next_example_or_tuple)
            self.constraint.add(measured_example)
            if self.constraint.reached_limit():
                break

        return self._detuplify(examples)


class PackedSequenceDynamicCutSampler(DynamicCutSampler):
    """DynamicCutSampler whose packed token budget is a hard upper bound.

    The first candidate that does not fit is deferred to the next batch. The
    deferred payload is included in sampler state because indexed source state
    has already advanced past that candidate when a checkpoint is taken.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batcher = None
        self._restored_deferred_examples = []
        self._inject_restored_deferred_examples = False

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        if self._batcher is not None:
            deferred_examples = list(self._batcher.reuse_cuts_buffer)
        else:
            deferred_examples = list(self._restored_deferred_examples)
        state["deferred_examples"] = deferred_examples
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._restored_deferred_examples = list(state_dict.pop("deferred_examples", []))
        self._batcher = None
        super().load_state_dict(state_dict)

    def allow_iter_to_reset_state(self):
        super().allow_iter_to_reset_state()
        self._restored_deferred_examples = []

    def _fast_forward(self):
        # Indexed restoration resumes the source after the lookahead candidate,
        # so inject its saved payload. Replay restoration reconstructs that
        # lookahead naturally while replaying completed batches.
        self._inject_restored_deferred_examples = bool(self.cuts) and all(
            getattr(source, "has_constant_time_access", False) for source in self.cuts
        )
        try:
            super()._fast_forward()
        finally:
            self._inject_restored_deferred_examples = False
        self._restored_deferred_examples = []

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
        )
        if self._inject_restored_deferred_examples:
            self._batcher.reuse_cuts_buffer.extend(self._restored_deferred_examples)
        self.cuts_iter = iter(self._batcher)
