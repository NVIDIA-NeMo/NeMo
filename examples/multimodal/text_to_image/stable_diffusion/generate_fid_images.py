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

import torch
from omegaconf.omegaconf import open_dict

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.multimodal.parts.stable_diffusion.pipeline import pipeline
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner


@hydra_runner(config_path='conf', config_name='sd_fid_images')
def main(cfg):
    # Read configuration parameters
    nnodes_per_cfg = cfg.fid.nnodes_per_cfg
    ntasks_per_node = cfg.fid.ntasks_per_node
    local_task_id = cfg.fid.local_task_id
    num_images_to_eval = cfg.fid.num_images_to_eval
    path = cfg.fid.coco_captions_path

    node_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    node_id_per_cfg = node_id % nnodes_per_cfg

    current_node_cfg = cfg.fid.classifier_free_guidance[node_id // nnodes_per_cfg]
    with open_dict(cfg):
        cfg.infer.unconditional_guidance_scale = current_node_cfg
    save_path = os.path.join(cfg.fid.save_path, str(current_node_cfg))

    # Read and store captions
    captions = []
    caption_files = sorted(os.listdir(path))
    assert len(caption_files) >= num_images_to_eval
    for file in caption_files[:num_images_to_eval]:
        with open(os.path.join(path, file), 'r') as f:
            captions += f.readlines()

    # Calculate partition sizes and select the partition for the current node
    partition_size_per_node = num_images_to_eval // nnodes_per_cfg
    start_idx = node_id_per_cfg * partition_size_per_node
    end_idx = (node_id_per_cfg + 1) * partition_size_per_node if node_id_per_cfg != nnodes_per_cfg - 1 else None
    captions = captions[start_idx:end_idx]

    local_task_id = int(local_task_id) if local_task_id is not None else int(os.environ.get("SLURM_LOCALID", 0))
    partition_size_per_task = int(len(captions) // ntasks_per_node)

    # Select the partition for the current task
    start_idx = local_task_id * partition_size_per_task
    end_idx = (local_task_id + 1) * partition_size_per_task if local_task_id != ntasks_per_node - 1 else None
    input = captions[start_idx:end_idx]

    print(f"Current worker {node_id}:{local_task_id} will generate {len(input)} images")

    os.makedirs(save_path, exist_ok=True)

    # Modify the model configuration
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.use_flash_attention = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None
        model_cfg.global_batch_size = cfg.infer.batch_size * ntasks_per_node

    # Set up the trainer and model for inference
    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronLatentDiffusion, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )
    model = megatron_diffusion_model.model
    model.cuda().eval()

    # Generate images using the model and save them
    cfg.infer.prompts = input
    rng = torch.Generator().manual_seed(cfg.infer.seed + local_task_id * 10 + node_id_per_cfg * 100)
    output = pipeline(model, cfg, rng=rng)
    for i, image in enumerate(img for batch in output for img in batch):
        image_num = i + partition_size_per_node * node_id_per_cfg + partition_size_per_task * local_task_id
        image.save(os.path.join(save_path, f'image{image_num:06d}.png'))


if __name__ == "__main__":
    main()
