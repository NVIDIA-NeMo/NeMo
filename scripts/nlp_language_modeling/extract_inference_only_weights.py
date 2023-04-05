# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""A script to extract the final p-tuning representations used for inference.
Here is an example usage command:

```python
python scripts/nlp_language_modeling/extract_inference_only_weights.py p_tuning.nemo
```

"""
import argparse
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("nemo", help="path to nemo file", type=str)
parser.add_argument("taskname", help="taskname for the nemo model", type=str, default="taskname", required=False)
args = parser.parse_args()

os.system(f"tar xvf {args.nemo}")

for p in '', 'mp_rank_00/', 'tp_rank_00_pp_rank_000/':
    try:
        a = torch.load(f'{p}model_weights.ckpt')
        break
    except FileNotFoundError:
        pass
inf_weights = a['prompt_table'][f'prompt_table.{args.taskname}.prompt_embeddings.weight']
torch.save(inf_weights, "p_tuned.inf_only.ckpt")
