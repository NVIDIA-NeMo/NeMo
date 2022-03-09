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

from nemo.collections.nlp.modules.common.huggingface.albert import AlbertEncoder
from nemo.collections.nlp.modules.common.huggingface.bert import BertEncoder
from nemo.collections.nlp.modules.common.huggingface.camembert import CamembertEncoder
from nemo.collections.nlp.modules.common.huggingface.distilbert import DistilBertEncoder
from nemo.collections.nlp.modules.common.huggingface.huggingface_utils import (
    get_huggingface_lm_model,
    get_huggingface_pretrained_lm_models_list,
)
from nemo.collections.nlp.modules.common.huggingface.roberta import RobertaEncoder
