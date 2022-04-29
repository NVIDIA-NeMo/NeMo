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
