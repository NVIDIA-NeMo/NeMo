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

"""
python clip_script.py --captions_path /path/to/coco2014_val/captions \
                      --fid_images_path /path/to/synthetic_images \
                      --output_path /path/to/output/clip_scores.csv

1. `--captions_path`: The path to the real images captions directory. In this example,
   it is set to `/path/to/coco2014_val/captions`. This path should point to the
   directory containing the COCO 2014 validation dataset captions.

2. `--fid_images_path`: The path to the directory containing subfolders with synthetic
   images. In this example, it is set to `/path/to/synthetic_images`. Each subfolder
   should contain a set of synthetic images for which you want to compute CLIP scores
   against the captions from `--captions_path`.

3. `--output_path`: The path to the output CSV file where the CLIP scores will be saved.
   In this example, it is set to `/path/to/output/clip_scores.csv`. This file will
   contain a table with two columns: `cfg` and `clip_score`. The `cfg`
   column lists the names of the subfolders in `--fid_images_path`, and the
   `clip_score` column lists the corresponding average CLIP scores between the synthetic
   images in each subfolder and the captions from `--captions_path`.
"""

import argparse
import csv
import os
from glob import glob

import open_clip
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


class CLIPEncoder(nn.Module):
    def __init__(self, clip_version='ViT-B/32', pretrained='', cache_dir=None, device='cuda'):
        super().__init__()

        self.clip_version = clip_version
        if not pretrained:
            if self.clip_version == 'ViT-H-14':
                self.pretrained = 'laion2b_s32b_b79k'
            elif self.clip_version == 'ViT-g-14':
                self.pretrained = 'laion2b_s12b_b42k'
            else:
                self.pretrained = 'openai'

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_version, pretrained=self.pretrained, cache_dir=cache_dir
        )

        self.model.eval()
        self.model.to(device)

        self.device = device

    @torch.no_grad()
    def get_clip_score(self, text, image):
        if isinstance(image, str):  # filenmae
            image = Image.open(image)
        if isinstance(image, Image.Image):  # PIL Image
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if not isinstance(text, (list, tuple)):
            text = [text]
        text = open_clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        return similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--captions_path', default='/coco2014/coco2014_val_sampled_30k/captions/', type=str)
    parser.add_argument('--fid_images_path', default=None, type=str)
    parser.add_argument('--output_path', default='./clip_scores.csv', type=str)
    parser.add_argument('--clip_version', default='ViT-L-14', type=str)
    args = parser.parse_args()

    # Initialize distributed training
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    captions_path = args.captions_path
    print('Init CLIP Encoder..')
    encoder = CLIPEncoder(clip_version=args.clip_version)

    # Create output CSV file
    with open(args.output_path, 'w', newline='') as csvfile:
        fieldnames = ['cfg', 'clip_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through subfolders in fid_images_path
        for subfolder in os.listdir(args.fid_images_path):
            subfolder_path = os.path.join(args.fid_images_path, subfolder)
            if os.path.isdir(subfolder_path):
                images = sorted(
                    glob(f'{subfolder_path}/*.png'), key=lambda x: (int(x.split('/')[-1].strip('.png').strip('image')))
                )
                texts = sorted(glob(f'{captions_path}/*.txt'))
                print(images[:5], texts[:5])
                # this enables computing clip on the smaller images set
                texts = texts[: len(images)]
                assert len(images) == len(texts)
                print(f'Number of images text pairs: {len(images)}')

                imgs = torch.utils.data.DataLoader(
                    images, sampler=torch.utils.data.distributed.DistributedSampler(images)
                )
                txts = torch.utils.data.DataLoader(
                    texts, sampler=torch.utils.data.distributed.DistributedSampler(texts)
                )

                ave_sim = torch.tensor(0.0).cuda()
                count = 0
                for text, img in zip(tqdm(txts), imgs):
                    with open(text[0], 'r') as f:
                        text = f.read().strip()
                    sim = encoder.get_clip_score(text, img[0])
                    ave_sim += sim[0, 0]
                    count += 1
                    if count % 2000 == 0:
                        print(ave_sim / count)

                torch.distributed.all_reduce(ave_sim)
                ave_sim /= len(images)
                print(f'The CLIP similarity for CFG {subfolder}: {ave_sim}')

                # Write CLIP score to output CSV file
                writer.writerow({'cfg': subfolder, 'clip_score': float(ave_sim)})
