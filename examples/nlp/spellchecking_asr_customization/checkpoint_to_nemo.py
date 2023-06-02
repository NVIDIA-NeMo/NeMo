# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script converts checkpoint .ckpt to .nemo file.

This script uses the `examples/nlp/spellchecking_asr_customization/conf/spellchecking_asr_customization_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'.
"""

from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models import SpellcheckingAsrCustomizationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="spellchecking_asr_customization_config")
def main(cfg: DictConfig) -> None:
    logging.debug(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    SpellcheckingAsrCustomizationModel.load_from_checkpoint(cfg.checkpoint_path).save_to(cfg.target_nemo_path)


if __name__ == "__main__":
    main()
