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

"""
Implemented following the guide at https://www.promptingguide.ai/models/gemma#gemma-7b-prompt-format
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

GEMMA_BOS = "<start_of_turn>"
GEMMA_END_OF_TURN = "<end_of_turn>"
GEMMA_NL = "\n\n"


class GemmaPromptFormatter(PromptFormatter):
    NAME = "gemma"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": f"{GEMMA_BOS}user\n|message|{GEMMA_END_OF_TURN}\n{GEMMA_BOS}model\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            # Note: that trailing NL is bothering me.
            "template": f"|message|{GEMMA_END_OF_TURN}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
