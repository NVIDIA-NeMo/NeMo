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
import os
import random

import cv2
import imageio
import numpy as np
import torch

from nemo.collections.multimodal.models.nerf.txt2nerf_base import Txt2NerfBase
from nemo.collections.multimodal.modules.nerf.loss.laplacian_smooth_loss import LaplacianSmoothLoss
from nemo.collections.multimodal.modules.nerf.loss.normal_consistency_loss import NormalConsistencyLoss
from nemo.collections.multimodal.modules.nerf.materials.materials_base import ShadingEnum
from nemo.core import optim


# TODO(ahmadki): split dmtet from dreamfusion
class DreamFusion(Txt2NerfBase):
    def __init__(self, cfg):
        super(DreamFusion, self).__init__(cfg)

        self.guidance_scale = cfg.guidance_scale

        self.iters = cfg.iters
        self.latent_iter_ratio = cfg.latent_iter_ratio
        self.albedo_iter_ratio = cfg.albedo_iter_ratio
        self.min_ambient_ratio = cfg.min_ambient_ratio
        self.textureless_ratio = cfg.textureless_ratio

        # Lambdas
        self.lambda_sds = cfg.loss.lambda_sds
        self.lambda_opacity = cfg.loss.lambda_opacity
        self.lambda_entropy = cfg.loss.lambda_entropy
        self.lambda_orientation = cfg.loss.lambda_orientation
        self.lambda_2d_normal_smooth = cfg.loss.lambda_2d_normal_smooth
        self.lambda_3d_normal_smooth = cfg.loss.lambda_3d_normal_smooth
        self.lambda_mesh_normal = cfg.loss.lambda_mesh_normal
        self.lambda_mesh_laplacian = cfg.loss.lambda_mesh_laplacian

        if self.lambda_mesh_normal > 0:
            self.normal_consistency_loss_fn = NormalConsistencyLoss()
        if self.lambda_mesh_laplacian > 0:
            self.laplacian_smooth_loss_fn = LaplacianSmoothLoss()

        # Video
        self.test_images = []
        self.test_depths = []

    def training_step(self, batch, batch_idx):
        # experiment iterations ratio
        # i.e. what proportion of this experiment have we completed (in terms of iterations) so far?
        exp_iter_ratio = self.global_step / self.iters

        # TODO(ahmadki): move to database
        if exp_iter_ratio < self.latent_iter_ratio:
            ambient_ratio = 1.0
            shading_type = ShadingEnum.NORMAL
            as_latent = True
        else:
            if exp_iter_ratio <= self.albedo_iter_ratio:
                ambient_ratio = 1.0
                shading_type = None
            else:
                # random shading
                ambient_ratio = self.min_ambient_ratio + (1.0 - self.min_ambient_ratio) * random.random()
                rand = random.random()
                if rand >= (1.0 - self.textureless_ratio):
                    shading_type = ShadingEnum.TEXTURELESS
                else:
                    shading_type = ShadingEnum.LAMBERTIAN

            as_latent = False

        return_normal_image = bool(self.lambda_2d_normal_smooth)
        return_normal_perturb = bool(self.lambda_3d_normal_smooth)
        return_vertices = bool(self.lambda_mesh_laplacian)
        return_faces = bool(self.lambda_mesh_normal) or bool(self.lambda_mesh_laplacian)
        return_faces_normals = bool(self.lambda_mesh_normal)
        outputs = self(
            rays_o=batch['rays_o'],  # [B, H, W, 3]
            rays_d=batch['rays_d'],  # [B, H, W, 3]
            mvp=batch['mvp'],  # [B, 4, 4]
            perturb=True,
            ambient_ratio=ambient_ratio,
            shading_type=shading_type,
            binarize=False,
            return_normal_image=return_normal_image,
            return_normal_perturb=return_normal_perturb,
            return_vertices=return_vertices,
            return_faces=return_faces,
            return_faces_normals=return_faces_normals,
        )

        if as_latent:
            pred_rgb = (
                torch.cat([outputs['image'], outputs['opacity']], dim=-1).permute(0, 3, 1, 2).contiguous()
            )  # [B, 4, H, W]
        else:
            pred_rgb = outputs['image'].permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]

        # TODO(ahmadki): move into guidance
        azimuth = batch['azimuth']
        text_z = [self.text_z['uncond']] * azimuth.shape[0]
        for b in range(azimuth.shape[0]):
            if azimuth[b] >= -90 and azimuth[b] < 90:
                if azimuth[b] >= 0:
                    r = 1 - azimuth[b] / 90
                else:
                    r = 1 + azimuth[b] / 90
                start_z = self.text_z['front']
                end_z = self.text_z['side']
            else:
                if azimuth[b] >= 0:
                    r = 1 - (azimuth[b] - 90) / 90
                else:
                    r = 1 + (azimuth[b] + 90) / 90
                start_z = self.text_z['side']
                end_z = self.text_z['back']
            pos_z = r * start_z + (1 - r) * end_z
            text_z.append(pos_z)
        text_z = torch.cat(text_z, dim=0)

        loss_dict = {}

        # SDS loss
        guidance_loss = self.guidance.train_step(
            text_z, pred_rgb, as_latent=as_latent, guidance_scale=self.guidance_scale
        )
        loss_dict['lambda_sds'] = guidance_loss * self.lambda_sds

        # opacity loss
        if self.lambda_opacity > 0 and 'opacity' in outputs:
            loss_opacity = (outputs['opacity'] ** 2).mean()
            loss_dict['loss_opacity'] = self.lambda_opacity * loss_opacity

        # entropy loss
        if self.lambda_entropy > 0 and 'weights' in outputs:
            alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
            loss_entropy = (-alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            lambda_entropy = self.lambda_entropy * min(1, 2 * self.global_step / self.iters)
            loss_dict['loss_entropy'] = lambda_entropy * loss_entropy

        if self.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:
            pred_normal = outputs['normal_image']
            loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + (
                pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]
            ).square().mean()
            loss_dict['loss_smooth'] = self.lambda_2d_normal_smooth * loss_smooth

        # orientation loss
        if self.lambda_orientation > 0 and all(key in outputs for key in ['weights', 'normals', 'dirs']):
            loss_orientation = (
                outputs['weights'].detach() * (outputs['normals'] * outputs['dirs']).sum(-1).clamp(min=0) ** 2
            )
            loss_orientation = loss_orientation.mean()
            loss_dict['loss_orientation'] = self.lambda_orientation * loss_orientation

        if self.lambda_3d_normal_smooth > 0 and all(key in outputs for key in ['normals', 'normal_perturb']):
            loss_normal_perturb = (outputs['normal_perturb'] - outputs['normals']).abs().mean()
            loss_dict['loss_normal_smooth'] = self.lambda_3d_normal_smooth * loss_normal_perturb

        if self.lambda_mesh_normal > 0 and all(key in outputs for key in ['face_normals', 'faces']):
            normal_consistency_loss = self.normal_consistency_loss_fn(
                face_normals=outputs['face_normals'], t_pos_idx=outputs['faces']
            )
            loss_dict['normal_consistency_loss'] = self.lambda_mesh_normal * normal_consistency_loss

        if self.lambda_mesh_laplacian > 0 and all(key in outputs for key in ['verts', 'faces']):
            laplacian_loss = self.laplacian_smooth_loss_fn(verts=outputs['verts'], faces=outputs['faces'])
            loss_dict['laplacian_loss'] = self.lambda_mesh_laplacian * laplacian_loss

        loss = sum(loss_dict.values())

        self.log_dict(loss_dict, prog_bar=False, rank_zero_only=True)
        self.log('loss', loss, prog_bar=True, rank_zero_only=True)

        # TODO(ahmadki): LearningRateMonitor
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, rank_zero_only=True)

        self.log('global_step', self.global_step + 1, prog_bar=True, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # save image
        images, depths = self._shared_predict(batch)

        save_path = os.path.join(self.trainer.log_dir, 'validation')
        os.makedirs(save_path, exist_ok=True)
        for i, (image, depth) in enumerate(zip(images, depths)):
            # Save image
            cv2.imwrite(
                os.path.join(
                    save_path,
                    f'{self.current_epoch:04d}_{self.global_step:04d}_{self.global_rank:04d}_{batch_idx:04d}_{i:04d}_rgb.png',
                ),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
            # Save depth
            cv2.imwrite(
                os.path.join(
                    save_path,
                    f'{self.current_epoch:04d}_{self.global_step:04d}_{self.global_rank:04d}_{batch_idx:04d}_{i:04d}_depth.png',
                ),
                depth,
            )

    def test_step(self, batch, batch_idx):
        # save image
        images, depths = self._shared_predict(batch)
        self.test_images.append(images)
        self.test_depths.append(depths)

    def on_test_epoch_end(self):
        save_path = os.path.join(self.trainer.log_dir, 'test')
        os.makedirs(save_path, exist_ok=True)

        images = np.concatenate(self.test_images, axis=0)
        imageio.mimwrite(
            os.path.join(os.path.join(save_path, f'{self.current_epoch:04d}_{self.global_step:04d}_rgb.mp4')),
            images,
            fps=25,
            quality=8,
            macro_block_size=1,
        )

        depths = np.concatenate(self.test_depths, axis=0)
        imageio.mimwrite(
            os.path.join(os.path.join(save_path, f'{self.current_epoch:04d}_{self.global_step:04d}_depth.mp4')),
            depths,
            fps=25,
            quality=8,
            macro_block_size=1,
        )

        self.test_images.clear()
        self.test_depths.clear()

    def predict_step(self, batch, batch_idx):
        return self._shared_predict(self, batch)

    def forward(
        self,
        rays_o,
        rays_d,
        mvp,
        perturb,
        ambient_ratio,
        shading_type,
        binarize,
        return_normal_image,
        return_normal_perturb,
        return_vertices,
        return_faces,
        return_faces_normals,
    ):
        outputs = self.renderer(
            rays_o=rays_o,
            rays_d=rays_d,
            mvp=mvp,
            perturb=perturb,
            ambient_ratio=ambient_ratio,
            shading_type=shading_type,
            binarize=binarize,
            return_normal_image=return_normal_image,
            return_normal_perturb=return_normal_perturb,
            return_vertices=return_vertices,
            return_faces=return_faces,
            return_faces_normals=return_faces_normals,
        )
        return outputs

    def _shared_predict(self, data):
        outputs = self(
            rays_o=data['rays_o'],  # [B, H, W, 3]
            rays_d=data['rays_d'],  # [B, H, W, 3]
            mvp=data['mvp'],
            perturb=False,
            ambient_ratio=data['ambient_ratio'] if 'ambient_ratio' in data else 1.0,  # TODO(ahmadki): move to dataset
            shading_type=data['shading_type'] if 'shading_type' in data else None,  # TODO(ahmadki): move to dataset
            binarize=False,
            return_normal_image=False,
            return_normal_perturb=False,
            return_vertices=False,
            return_faces=False,
            return_faces_normals=False,
        )

        images_np = outputs['image'].detach().cpu().numpy()
        images_np = (images_np * 255).astype(np.uint8)

        depths_np = outputs['depth'].detach().cpu().numpy()
        depths_np = (depths_np - depths_np.min()) / (np.ptp(depths_np) + 1e-6)
        depths_np = (depths_np * 255).astype(np.uint8)

        return images_np, depths_np

    # TODO(ahmadki): rework
    def setup_optimization(self):
        cfg = self._cfg.optim
        optimizer_args = dict(cfg)
        optimizer_args.pop('name', None)

        optimizer = optim.get_optimizer(cfg.name)

        optimizer = optimizer(params=self.parameters(), **optimizer_args)

        self._optimizer = optimizer

    def configure_optimizers(self):
        self.setup_optimization()
        return self._optimizer
