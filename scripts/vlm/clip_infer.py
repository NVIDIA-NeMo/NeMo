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

"""
Usage example:
    python scripts/vlm/clip_infer.py \
    --image_url https://upload.wikimedia.org/wikipedia/commons/0/0f/1665_Girl_with_a_Pearl_Earring.jpg \
    --hf_path hf://openai/clip-vit-large-patch14 \
    --classes "a dog" "a boy" "a girl"



It should generate a high probability for "a girl" tag, e.g.
Nemo: CLIP text probability:  [('a dog', 0.0048940657), ('a boy', 0.002311793), ('a girl', 0.9927942)]
HF: CLIP text probability:  [('a dog', 0.0048940657), ('a boy', 0.002311793), ('a girl', 0.9927942)]
"""
import argparse
import os

import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import CLIPModel as HFCLIPModel

import nemo.lightning as nl
from nemo.collections.vlm import CLIPModel


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from a URL or local file path.

    Args:
        image_path (str): The URL or local path to the image.

    Returns:
        Image.Image: The loaded PIL image object, or None if loading fails.
    """
    try:
        if os.path.exists(image_path):  # Check if it's a local file path
            image = Image.open(image_path)
        else:  # Assume it's a remote URL
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        return image
    except (requests.exceptions.RequestException, FileNotFoundError, IOError) as e:
        print(f"Error loading image from {image_path}: {e}")
        return None


def main(args) -> None:
    # pylint: disable=C0115,C0116
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        ckpt_include_optimizer=False,
        ckpt_save_optimizer=False,
    )

    trainer = nl.Trainer(
        devices=1,
        max_steps=1000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        val_check_interval=1000,
        limit_val_batches=50,
    )

    hf_repo = args.hf_path.split("//")[1]
    processor = AutoProcessor.from_pretrained(hf_repo)
    max_length = processor.tokenizer.model_max_length

    # Load the image
    raw_image = load_image(args.image_url)
    if raw_image is None:
        return  # Exit if the image can't be loaded

    fabric = trainer.to_fabric()
    model = fabric.import_model(args.hf_path, CLIPModel)
    model = model.module.cuda()

    # Freeze the models, We have a few nesting in the model
    vision_model = model.module.module.vision_model.eval()
    text_model = model.module.module.text_model.eval()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        # %% Zero-shot classification
        classes = args.classes

        inputs = processor(
            text=classes,
            images=[raw_image],
            return_tensors="pt",
            truncation=True,  # Truncate if the sentence is longer than max_seq_length
            padding='max_length',  # Pad to max_seq_length
            max_length=max_length,
        )

        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        model_hf = HFCLIPModel.from_pretrained(hf_repo)
        model_hf = model_hf.to("cuda")
        output_hf = model_hf(**inputs)

        image_embeds_nemo = vision_model(inputs["pixel_values"].cuda().to(torch.bfloat16))
        image_embeds_hf = output_hf["image_embeds"]

        text_embeds_nemo = text_model(inputs["input_ids"].cuda())
        text_embeds_hf = output_hf["text_embeds"]

        image_embeds_nemo /= image_embeds_nemo.norm(dim=-1, keepdim=True)
        text_embeds_nemo /= text_embeds_nemo.norm(dim=-1, keepdim=True)

        nemo_probs = (100.0 * image_embeds_nemo @ text_embeds_nemo.T).softmax(dim=-1)
        hf_probs = (100.0 * image_embeds_hf @ text_embeds_hf.T).softmax(dim=-1)

        print(f"Nemo: CLIP text probability: ", list(zip(classes, nemo_probs[0].cpu().numpy())))
        print(f"HF: CLIP text probability: ", list(zip(classes, hf_probs[0].cpu().numpy())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip Verification Script")
    parser.add_argument(
        "--image_url",
        type=str,
        default="1665_Girl_with_a_Pearl_Earring.jpg",
        help="URL of the image to use for inference.",
    )

    parser.add_argument(
        "--hf_path",
        type=str,
        default="hf://openai/clip-vit-large-patch14",
        help="Path to the Huggingface model.",
    )

    parser.add_argument(
        '--classes', nargs='+', type=str, help="Classes for texts", default=["a dog", "a boy", "a girl"]
    )
    args = parser.parse_args()

    main(args)
