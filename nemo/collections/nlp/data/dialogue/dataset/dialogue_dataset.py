# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2019 The Google Research Authors.
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

from nemo.core.classes import Dataset

__all__ = ['DialogueDataset']


class DialogueDataset(Dataset):
    '''
    Base class for Dialogue Datasets
        1. Performs Model-dependent (but Data-independent) operations (tokenization etc)
        2. This can allow the same model preprocessing for multiple datasources
        3. Users can configurate which labels to use for modelling 
            (e.g. intent classification, slot filling or sequence generation etc)
    '''

    def __init__(self, dataset_split: str, dialogues_processor: object, **kwargs):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
