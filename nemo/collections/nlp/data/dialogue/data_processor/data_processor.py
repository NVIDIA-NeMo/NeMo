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


import random

from nemo.collections.nlp.data.data_utils.data_preprocessing import DataProcessor

__all__ = ['DialogueDataProcessor']


class DialogueDataProcessor(DataProcessor):
    """
    Base class for Data Processing for all data sources

    Data Processor is designed to be Model-independent (but Data-dependent) so that
        - Encourages experimentation with a variety of models \
            (BERT-style; GPT-style; T5-style), \
            which have different tokenization/preprocessing requirements
        - Facilitates experiments with a variety of data sources, 
           as data is processed into a common format
        
    Roles 
        1. Processes raw files into Dialogue Input Examples. 
        2. Keeps all possibly relevant information from the raw files, which 
            the Dataset class can then determine which labels to use
    
    """

    def __init__(self):
        raise NotImplementedError()

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    @staticmethod
    def get_relevant_idxs(dataset_split, n_samples, dev_proportion):
        """
        Obtain indexes for each dataset_split, when train and dev sets are not in separate files
        
        Args: 
            dataset_split: train, dev or test
            n_samples: total number of samples
            dev_proportion: value from 1 to 99 that represent proportion of data in dev set
        Returns:
            idxs: indices for relevant samples
        """

        if dataset_split in ["train", "dev"]:
            n_dev = int(n_samples * (dev_proportion / 100))
            dev_idxs = random.sample(list(range(n_samples)), n_dev)
            if dataset_split == "dev":
                idxs = dev_idxs
            else:
                dev_idxs_set = set(dev_idxs)
                train_idxs = [idx for idx in list(range(n_samples)) if idx not in dev_idxs_set]
                idxs = train_idxs

        elif dataset_split == "test":
            idxs = list(range(n_samples))

        else:
            raise ValueError("please select dataset split from train, dev and test")

        return idxs
