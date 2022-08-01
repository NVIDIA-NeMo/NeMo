# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, List

import numpy as np
import torch.utils.data as pt_data
from torch.utils.data import Dataset, IterableDataset

__all__ = ['ConcatDataset', 'ConcatMapDataset']


class ConcatDataset(IterableDataset):
    """
    A dataset that accepts as argument multiple datasets and then samples from them based on the specified 
    sampling technique.
    Args:
        datasets (list): A list of datasets to sample from.
        shuffle (bool): Whether to shuffle individual datasets. Only works with non-iterable datasets. 
            Defaults to True.
        sampling_technique (str): Sampling technique to choose which dataset to draw a sample from.
            Defaults to 'temperature'. Currently supports 'temperature', 'random' and 'round-robin'.
        sampling_temperature (int): Temperature value for sampling. Only used when sampling_technique = 'temperature'.
            Defaults to 5.
        sampling_probabilities (list): Probability values for sampling. Only used when sampling_technique = 'random'.
        global_rank (int): Worker rank, used for partitioning map style datasets. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning map style datasets. Defaults to 1.
    """

    def __init__(
        self,
        datasets: List[Any],
        shuffle: bool = True,
        sampling_technique: str = 'temperature',
        sampling_temperature: int = 5,
        sampling_probabilities: List[float] = None,
        global_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        supported_sampling_techniques = ['temperature', 'random', 'round-robin']
        self.datasets = datasets
        self.iterables = [None] * len(datasets)
        self.shuffle = shuffle
        self.global_rank = global_rank
        self.world_size = world_size
        self.sampling_kwargs = {}
        if sampling_technique == 'temperature':
            self.index_generator = ConcatDataset.temperature_generator
            self.sampling_kwargs['temperature'] = sampling_temperature
        elif sampling_technique == 'random':
            self.index_generator = ConcatDataset.random_generator
            self.sampling_kwargs['p'] = sampling_probabilities
        elif sampling_technique == 'round-robin':
            self.index_generator = ConcatDataset.round_robin_generator
        else:
            raise ValueError(f"Currently we only support sampling techniques in {supported_sampling_techniques}.")
        self.length = 0

        if isinstance(datasets[0], IterableDataset):
            self.kind = 'iterable'
        else:
            self.kind = 'map'

        for idx, dataset in enumerate(datasets):
            isiterable = isinstance(dataset, IterableDataset)
            if (isiterable and not self.kind == 'iterable') or (not isiterable and self.kind == 'iterable'):
                raise ValueError("All datasets in ConcatDataset must be of the same kind (Iterable or Map).")

            if self.kind == 'map':
                self.length += len(dataset) // world_size
            else:
                self.length += len(dataset)

    def get_iterable(self, dataset):
        if isinstance(dataset, IterableDataset):
            return dataset.__iter__()
        else:
            indices = np.arange(len(dataset))
            if self.shuffle:
                np.random.shuffle(indices)
            return iter(indices)

    def __iter__(self):
        worker_info = pt_data.get_worker_info()
        if worker_info is None:
            max_elements = self.length
            wid = 0
            wnum = 1
        else:
            wid = worker_info.id
            wnum = worker_info.num_workers
            max_elements = len(range(wid, self.length, wnum))

        if self.kind == 'map':
            for idx in range(len(self.datasets)):
                start_idx = (len(self.datasets[idx]) // self.world_size) * self.global_rank
                end_idx = start_idx + (len(self.datasets[idx]) // self.world_size)
                if self.global_rank == self.world_size - 1:
                    end_idx = len(self.datasets[idx])
                indices = range(start_idx + wid, end_idx, wnum)
                self.datasets[idx] = pt_data.Subset(self.datasets[idx], indices)

        for idx, dataset in enumerate(self.datasets):
            iterable = self.get_iterable(dataset)
            self.iterables[idx] = iterable

        n = 0
        ind_gen = self.index_generator(self.datasets, **self.sampling_kwargs)
        while n < max_elements:
            n += 1
            try:
                ind = next(ind_gen)
            except StopIteration:
                return
            try:
                val = next(self.iterables[ind])
                if self.kind == 'map':
                    val = self.datasets[ind][val]
                yield val
            except StopIteration:
                self.iterables[ind] = self.get_iterable(self.datasets[ind])
                n -= 1

    def __len__(self):
        return self.length

    @staticmethod
    def temperature_generator(datasets, **kwargs):
        temp = kwargs.get('temperature')
        if not temp:
            raise ValueError("Temperature generator expects a 'temperature' keyword argument.")

        lengths = []
        num = len(datasets)
        for dataset in datasets:
            lengths.append(len(dataset))

        p = np.array(lengths) / np.sum(lengths)
        p = np.power(p, 1 / temp)
        p = p / np.sum(p)

        while True:
            ind = np.random.choice(np.arange(num), p=p)
            yield ind

    @staticmethod
    def round_robin_generator(datasets, **kwargs):
        num = len(datasets)
        while True:
            for i in range(num):
                yield i

    @staticmethod
    def random_generator(datasets, **kwargs):
        p = kwargs.get('p')
        if not p:
            raise ValueError("Random generator expects a 'p' keyowrd argument for sampling probabilities.")

        num = len(datasets)
        if len(p) != num:
            raise ValueError("Length of probabilities list must be equal to the number of datasets.")

        while True:
            ind = np.random.choice(np.arange(num), p=p)
            yield ind


class ConcatMapDataset(Dataset):
    """
    A dataset that accepts as argument multiple datasets and then samples from them based on the specified 
    sampling technique.
    Args:
        datasets (list): A list of datasets to sample from.
        sampling_technique (str): Sampling technique to choose which dataset to draw a sample from.
            Defaults to 'temperature'. Currently supports 'temperature', 'random' and 'round-robin'.
        sampling_temperature (int): Temperature value for sampling. Only used when sampling_technique = 'temperature'.
            Defaults to 5.
        sampling_probabilities (list): Probability values for sampling. Only used when sampling_technique = 'random'.
    """

    def __init__(
        self,
        datasets: List[Any],
        sampling_technique: str = 'temperature',
        sampling_temperature: int = 5,
        sampling_probabilities: List[float] = None,
        consumed_samples: int = 0,
    ):
        super().__init__()
        self.datasets = datasets
        self.sampling_kwargs = {}
        self.size = 0
        self.sampling_technique = sampling_technique
        self.sampling_temperature = sampling_temperature
        self.sampling_probabilities = sampling_probabilities
        self.consumed_samples = consumed_samples
        self.np_rng = np.random.RandomState(consumed_samples)

        for dataset in datasets:
            self.size += len(dataset)

        # Pointer into the next index to fetch from each dataset
        self.dataset_index = np.zeros(len(self.datasets), dtype=np.uint8)
        self.permuted_dataset_indices = []
        for dataset in self.datasets:
            permuted_indices = np.arange(len(dataset))
            self.np_rng.shuffle(permuted_indices)
            self.permuted_dataset_indices.append(permuted_indices)

        if self.sampling_technique == 'temperature':
            lengths = []
            for dataset in datasets:
                lengths.append(len(dataset))

            p = np.array(lengths) / np.sum(lengths)
            p = np.power(p, 1 / self.sampling_temperature)
            p = p / np.sum(p)
            self.p = p

        elif self.sampling_technique == 'random':
            if not self.sampling_probabilities:
                raise ValueError(
                    "Random generator expects a 'sampling_probabilities' - a list of probability values corresponding to each dataset."
                )

            if len(self.sampling_probabilities) != len(self.datasets):
                raise ValueError(
                    f"Length of probabilities list must be equal to the number of datasets. Found {len(sampling_probabilities)} probs and {len(self.datasets)} datasets."
                )

            p = np.array(self.sampling_probabilities)
            self.p = p / np.sum(p)  # Ensure probabilities sum to 1

    def __len__(self):
        return self.size

    def _get_dataset_index(self, idx):
        if self.sampling_technique == 'temperature' or self.sampling_technique == 'random':
            return self.np_rng.choice(np.arange(len(self.datasets)), p=self.p)
        elif self.sampling_technique == 'round-robin':
            return idx % len(self.datasets)

    def __getitem__(self, idx):
        # Get the dataset we want to sample from
        dataset_index = self._get_dataset_index(idx)

        # Get the index of the sample we want to fetch from the dataset
        sample_idx = self.dataset_index[dataset_index]

        # If the sample idx > dataset size, reset to 0.
        if sample_idx > len(self.datasets[dataset_index]):
            sample_idx = 0
            self.dataset_index[dataset_index] = 0

        # Sample index -> shuffled sample index
        shuffled_sample_idx = self.permuted_dataset_indices[dataset_index][sample_idx]

        sample = self.datasets[dataset_index][shuffled_sample_idx]
        self.dataset_index[dataset_index] += 1

        return sample
