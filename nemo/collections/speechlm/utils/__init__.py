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


from nemo.collections.speechlm.utils.hydra_utils import get_object_list_from_config, to_dict_config
from nemo.collections.speechlm.utils.io import get_nested_attr, import_ckpt, load_distributed_ckpt
from nemo.collections.speechlm.utils.model_transform import SpeechToTextLLMPEFT
