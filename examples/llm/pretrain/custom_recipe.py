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

import nemo_run as run

from nemo.collections import llm
from nemo.collections.llm.recipes import llama3_8b, llama3_70b


def custom_llama3_8b():
    pretrain = llama3_8b.pretrain_recipe(num_nodes=1, num_gpus_per_node=8)

    pretrain.trainer.val_check_interval = 400
    pretrain.log.ckpt.save_top_k = -1
    pretrain.log.ckpt.every_n_train_steps = 400

    pretrain.trainer.max_steps = 1000

    return pretrain


def custom_llama3_70b():
    pretrain = llama3_70b.pretrain_recipe(num_nodes=1, num_gpus_per_node=8)

    pretrain.trainer.val_check_interval = 400
    pretrain.log.ckpt.save_top_k = -1
    pretrain.log.ckpt.every_n_train_steps = 400

    pretrain.trainer.max_steps = 1000

    return pretrain


if __name__ == "__main__":
    # When running this file, it will run the `custom_llama3_8b` recipe

    # To select the `custom_llama3_70b` recipe, use the following command:
    #   python custom_recipe.py --factory custom_llama3_70b
    #   This will automatically call the custom_llama3_70b that's defined above

    # Note that any parameter can be overwritten by using the following syntax:
    # python custom_recipe.py trainer.max_steps=2000

    # You can even apply transformations when triggering the CLI as if it's python code
    # python custom_recipe.py "trainer.max_steps*=2"

    run.cli.main(llm.pretrain, default_factory=custom_llama3_8b)
