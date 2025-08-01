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
import os
import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from pathlib import Path

from nemo.collections.speechlm2 import DataModule, DuplexS2SDataset, DuplexS2SSpeechDecoderModel
from nemo.core.config import hydra_runner
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra_runner(config_path="conf", config_name="s2s_duplex_predict")
def main(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = True
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))
    log_dir = Path(cfg.output_path)
    OmegaConf.save(cfg, log_dir / "exp_config.yaml")

    if os.path.isdir(cfg.ckpt_path):
        # Hugging Face format
        model = DuplexS2SSpeechDecoderModel.from_pretrained(cfg.ckpt_path)
    else:
        # PyTorch Lightning format
        model = DuplexS2SSpeechDecoderModel.load_from_checkpoint(cfg.ckpt_path, map_location="cpu")

    model.cfg.prediction = cfg.prediction

    dataset = DuplexS2SDataset(
        tokenizer=model.tokenizer,
        frame_length=cfg.data.frame_length,
        source_sample_rate=cfg.data.source_sample_rate,
        target_sample_rate=cfg.data.target_sample_rate,
        input_roles=cfg.data.input_roles,
        output_roles=cfg.data.output_roles,
    )
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)

    # Predictions are saved by the prediction writer, which is a callback provided in the config
    trainer.predict(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
