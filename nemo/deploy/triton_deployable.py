# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


from abc import ABC, abstractmethod
import numpy as np


class ITritonDeployable(ABC):
    @abstractmethod
    def get_triton_input(self):
        pass

    @abstractmethod
    def get_triton_output(self):
        pass

    @abstractmethod
    def triton_infer_fn(self, **inputs: np.ndarray):
        pass
