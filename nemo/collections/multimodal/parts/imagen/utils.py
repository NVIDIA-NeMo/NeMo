# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import torch


def random_dropout(embeddings, drop_rate):
    r"""
    Function to perform random dropout for embeddings.
    When we drop embeddings, we zero them out.
    Args:
        embeddings (tensor): Input embeddings
        drop_rate (float): Rate of dropping the embedding.
    """
    nsamples = embeddings.shape[0]
    zero_flag = torch.ones(nsamples, 1, 1).to(embeddings.dtype) * (1 - drop_rate)
    zero_flag = torch.bernoulli(zero_flag).cuda()
    embeddings = embeddings * zero_flag
    return embeddings
