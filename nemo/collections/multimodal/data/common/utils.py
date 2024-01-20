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

import open_clip
import torch


def get_collate_fn(first_stage_key="images_moments", cond_stage_key="captions"):
    def collate_fn_with_tokenize(batch):
        images_moments = [s[first_stage_key] for s in batch]
        cond_inputs = [s[cond_stage_key] for s in batch]
        if cond_stage_key == "captions":
            tokens = open_clip.tokenize(cond_inputs)
        else:
            tokens = torch.stack(cond_inputs)
        batch = {
            first_stage_key: torch.cat(images_moments),
            cond_stage_key: tokens,
        }
        return batch

    return collate_fn_with_tokenize
