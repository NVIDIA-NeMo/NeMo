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

from typing import Any, Mapping

from nemo.collections.speechlm.utils.model_transform import SPEECHLM_PEFT_RESUME
from nemo.lightning.pytorch.strategies import MegatronStrategy
from nemo.utils import logging


class SpeechLMMegatronStrategy(MegatronStrategy):

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        """
        Overwrite the load_model_state_dict method to skip loading PTL checkpoint in first stage of PEFT.
        This is needed to avoid
        """
        cached_ckpt_load_strictness = self.ckpt_load_strictness
        if SPEECHLM_PEFT_RESUME in checkpoint:
            logging.info("Resuming from PEFT jobs, skip PTL checkpoint loading in first stage.")
            strict = False
            self.ckpt_load_strictness = None
        super().load_model_state_dict(checkpoint, strict=strict)
        self.ckpt_load_strictness = cached_ckpt_load_strictness
