# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from omegaconf import OmegaConf

from nemo.collections.tts.models import EasyMagpieTTSModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf/magpietts", config_name="easy_magpietts_lhotse")
def main(cfg):
    logging.info("Initializing EasyMagpieTTSModel from config only.")
    # ModelPT eagerly builds a dataloader whenever train_ds is configured. This
    # checker is intentionally construction-only and should also work with the
    # placeholder dataset path in the example config.
    cfg.model.train_ds = None
    logging.info("Model config:\n%s", OmegaConf.to_yaml(cfg.model, resolve=False))

    model = EasyMagpieTTSModel(cfg=cfg.model, trainer=None)

    logging.info("Initialized EasyMagpieTTSModel successfully.")
    logging.info("decoder_type=%s", model.decoder_type)
    logging.info("decoder=%s", type(model.decoder).__name__)
    logging.info("lm_text_head=%s", None if model.lm_text_head is None else type(model.lm_text_head).__name__)
    logging.info("num_parameters=%s", sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
