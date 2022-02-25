# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import abc

import torch

__all__ = [
    "MegatronPretrainingBatchSampler",
    "MegatronPretrainingRandomBatchSampler",
]


class BaseMegatronBatchSampler:
    """Base class for Megatron style BatchSampler."""

    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __iter__(self):
        ...

    @property
    @abc.abstractmethod
    def num_micro_batch_times_micro_batch_size(self) -> int:
        """The size of global batch on each data parallel rank."""
        ...

    @num_micro_batch_times_micro_batch_size.setter
    @abc.abstractclassmethod
    def num_micro_batch_times_micro_batch_size(self) -> None:
        """The size of global batch on each data parallel rank."""
        ...


class MegatronPretrainingBatchSampler(BaseMegatronBatchSampler):
    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        num_micro_batch_times_micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
    ) -> None:
        # Sanity checks.
        if total_samples <= 0:
            raise RuntimeError('no sample to consume: {}'.format(self.total_samples))
        if consumed_samples >= total_samples:
            raise RuntimeError('no samples left to consume: {}, {}'.format(self.consumed_samples, self.total_samples))
        if num_micro_batch_times_micro_batch_size <= 0:
            raise RuntimeError(
                f"num_micro_batch_times_micro_batch_size size must be greater than 0: {num_micro_batch_times_micro_batch_size}"
            )
        if data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0: {data_parallel_size}")
        if data_parallel_rank >= data_parallel_size:
            raise RuntimeError(
                'data_parallel_rank should be smaller than data size: {}, {}'.format(
                    self.data_parallel_rank, data_parallel_size
                )
            )
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self._num_micro_batch_times_micro_batch_size = num_micro_batch_times_micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.num_micro_batch_times_micro_batch_size_times_data_parallel_size = (
            self._num_micro_batch_times_micro_batch_size * data_parallel_size
        )
        self.drop_last = drop_last

    def __len__(self):
        return (
            self.total_samples - self.consumed_samples - 1
        ) // self.num_micro_batch_times_micro_batch_size_times_data_parallel_size + 1

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self._num_micro_batch_times_micro_batch_size
        end_idx = start_idx + self._num_micro_batch_times_micro_batch_size
        return start_idx, end_idx

    @property
    def num_micro_batch_times_micro_batch_size(self) -> int:
        return self._num_micro_batch_times_micro_batch_size

    @num_micro_batch_times_micro_batch_size.setter
    def num_micro_batch_times_micro_batch_size(self, new_num_micro_batch_times_micro_batch_size) -> None:
        self._num_micro_batch_times_micro_batch_size = new_num_micro_batch_times_micro_batch_size
        self.num_micro_batch_times_micro_batch_size_times_data_parallel_size = (
            self._num_micro_batch_times_micro_batch_size * self.data_parallel_size
        )

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self._num_micro_batch_times_micro_batch_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronPretrainingRandomBatchSampler(BaseMegatronBatchSampler):
    """Megatron style Random Batch Sampler.
    Major difference is that `__iter__` yields a global_batch, not a microbatch.
    A global_batch consists of :math:`num_micro_batch \\times micro_batch_size`
    Args:
        total_samples: The number of data samples, i.e. ``len(dataset)``.
        consumed_samples: The number of samples already consumed in pretraining.
        num_micro_batch_times_micro_batch_size: The number of data in each batch returned from `__iter__`.
        data_parallel_rank:
        data_parallel_size:
    """

    def __init__(
        self,
        total_samples: int,
        consumed_samples: int,
        num_micro_batch_times_micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
    ) -> None:
        if total_samples <= 0:
            raise ValueError(f"no sample to consume: total_samples of {total_samples}")
        if num_micro_batch_times_micro_batch_size <= 0:
            raise ValueError(
                f"Invalid num_micro_batch_times_micro_batch_size: {num_micro_batch_times_micro_batch_size}"
            )
        if data_parallel_size <= 0:
            raise ValueError(f"Invalid data_parallel_size: {data_parallel_size}")
        if data_parallel_rank >= data_parallel_size:
            raise ValueError(
                f"data_parallel_rank should be smaller than data parallel size: {data_parallel_rank} < {data_parallel_size}"
            )
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self._num_micro_batch_times_micro_batch_size = num_micro_batch_times_micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.num_micro_batch_times_micro_batch_size_times_data_parallel_size = (
            self._num_micro_batch_times_micro_batch_size * self.data_parallel_size
        )
        self.last_batch_size = (
            self.total_samples % self.num_micro_batch_times_micro_batch_size_times_data_parallel_size
        )

    def __len__(self):
        return (
            self.total_samples - self.consumed_samples - 1
        ) // self.num_micro_batch_times_micro_batch_size_times_data_parallel_size + 1

    @property
    def num_micro_batch_times_micro_batch_size(self) -> int:
        return self._num_micro_batch_times_micro_batch_size

    @num_micro_batch_times_micro_batch_size.setter
    def num_micro_batch_times_micro_batch_size(self, new_num_micro_batch_times_micro_batch_size) -> None:
        self._num_micro_batch_times_micro_batch_size = new_num_micro_batch_times_micro_batch_size
        self.num_micro_batch_times_micro_batch_size_times_data_parallel_size = (
            self._num_micro_batch_times_micro_batch_size * self.data_parallel_size
        )

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        # note(mkozuki): might be better to uncomment
        # assert current_epoch_samples % (self.data_parallel_size * apex.transformer.pipeline_parallel.utils.get_micro_batch_size()) == 0

        # data sharding and random sampling
        bucket_size = (
            self.total_samples // self.num_micro_batch_times_micro_batch_size_times_data_parallel_size
        ) * self._num_micro_batch_times_micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self._num_micro_batch_times_micro_batch_size:
                self.consumed_samples += self.num_micro_batch_times_micro_batch_size_times_data_parallel_size
                yield batch
                batch = []
