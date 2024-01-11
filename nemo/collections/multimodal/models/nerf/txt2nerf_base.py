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

from nemo.collections.multimodal.models.nerf.base import NerfModelBase


class Txt2NerfBase(NerfModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.prompt = cfg.prompt
        self.negative_prompt = cfg.negative_prompt
        self.front_prompt = cfg.front_prompt
        self.side_prompt = cfg.side_prompt
        self.back_prompt = cfg.back_prompt

        self.nerf_cfg = cfg.nerf
        self.renderer_cfg = cfg.renderer
        self.guidance_cfg = cfg.guidance

        nerf = self.from_config_dict(cfg.nerf)
        material = self.from_config_dict(cfg.material)
        background = self.from_config_dict(cfg.background)
        self.renderer = self.build_renderer(cfg.renderer, nerf, material, background)
        self.guidance = None

    def build_renderer(self, cfg, nerf, material, background):
        renderer = self.from_config_dict(cfg)
        renderer.nerf = nerf
        renderer.material = material
        renderer.background = background
        return renderer

    def build_guidance(self, cfg):
        self.guidance = self.from_config_dict(cfg)
        self.guidance.eval()
        for p in self.guidance.parameters():
            p.requires_grad = False

    def prepare_embeddings(self):
        # TODO(ahmadki): add top view ?
        self.text_z = {
            "default": self.guidance.get_text_embeds([self.prompt]),
            "uncond": self.guidance.get_text_embeds([self.negative_prompt]),
            "front": self.guidance.get_text_embeds([f"{self.prompt}{self.front_prompt}"]),
            "side": self.guidance.get_text_embeds([f"{self.prompt}{self.side_prompt}"]),
            "back": self.guidance.get_text_embeds([f"{self.prompt}{self.back_prompt}"]),
        }

    def on_fit_start(self) -> None:
        self.build_guidance(self.guidance_cfg)
        self.prepare_embeddings()

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        if self.is_module_updatable(self.guidance):
            self.guidance.update_step(epoch=self.current_epoch, global_step=self.global_step)

        if self.is_module_updatable(self.renderer.nerf):
            self.renderer.nerf.update_step(epoch=self.current_epoch, global_step=self.global_step)

        if self.is_module_updatable(self.renderer.material):
            self.renderer.material.update_step(epoch=self.current_epoch, global_step=self.global_step)

        if self.is_module_updatable(self.renderer.background):
            self.renderer.background.update_step(epoch=self.current_epoch, global_step=self.global_step)

        if self.is_module_updatable(self.renderer):
            self.renderer.update_step(epoch=self.current_epoch, global_step=self.global_step)

        dataset = self.trainer.train_dataloader.dataset
        if self.is_module_updatable(dataset):
            dataset.update_step(epoch=self.current_epoch, global_step=self.global_step)

    def mesh(self, resolution, batch_size=128, density_thresh=None):
        return self.nerf.mesh(resolution=resolution, batch_size=batch_size, density_thresh=density_thresh)

    def on_save_checkpoint(self, checkpoint):
        # remove guidance from checkpoint.
        # We can still laod the model without guidance checkpoints because the module is not initalized
        # at __init__ time.
        keys_to_remove = [key for key in checkpoint['state_dict'].keys() if key.startswith('guidance.')]
        for key in keys_to_remove:
            del checkpoint['state_dict'][key]
