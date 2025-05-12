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

import lhotse
from torch.utils.data import DataLoader, IterableDataset
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging


class SpeechLMDataSampler(MegatronDataSampler):
    """
    Overwrite the default MegatronDataSampler to not add batch sampler when iteralbe dataset is used.
    """

    def transform_dataloader(self, dataloader: DataLoader, consumed_samples: int = 0) -> DataLoader:
        """
        Overwrites the MegatronDataSampler.transform_dataloader() function
        """
        sampler = getattr(dataloader, 'sampler', None)

        if isinstance(sampler, lhotse.dataset.sampling.base.CutSampler):
            logging.info(f"Using Lhotse sampler for dataloader {dataloader}. Skipping Megatron data sampler.")
            return dataloader

        try:
            _ = len(dataloader.dataset)
            has_len = True
        except TypeError:
            has_len = False

        if not has_len:
            logging.info(f"Dataset {dataloader.dataset} does not have __len__ method. Skipping Megatron data sampler.")
            return dataloader

        if isinstance(dataloader.dataset, IterableDataset):
            logging.info(f"Dataset {dataloader.dataset} is an IterableDataset. Skipping Megatron data sampler.")
            return dataloader

        return super().transform_dataloader(dataloader, consumed_samples)
