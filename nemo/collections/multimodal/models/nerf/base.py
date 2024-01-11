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

from nemo.core.classes.common import Serialization
from nemo.core.classes.modelPT import ModelPT


class NerfModelBase(ModelPT, Serialization):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.save_hyperparameters()
        self._cfg = cfg

    @staticmethod
    def is_module_updatable(module):
        return hasattr(module, 'update_step') and callable(module.update_step)

    def list_available_models(self):
        pass

    def setup_training_data(self, config):
        pass

    def setup_validation_data(self, config):
        pass
