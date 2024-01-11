# Imagen
## A. Overview

Imagen is a multi-stage text-to-image diffusion model with an unprecedented degree of photorealism and a deep level of language understanding. Given a text prompt, Imagen first generates an image at a 64x64 resolution and then upsamples the generated image to 256x256 and 1024x1024 resolutions, all using diffusion models.

**Table of Contents:**
- [Imagen](#imagen)
  - [A. Overview](#a-overview)
  - [B. Imagen Pipeline](#b-imagen-pipeline)
  - [C. Files in this folder](#c-files-in-this-folder)
  - [D. Imagen Training](#d-imagen-training)
    - [D.1 Training Dataset](#d1-training-dataset)
    - [D.2 Training configs](#d2-training-configs)
  - [E. Imagen Inference](#e-imagen-inference)
    - [E.1 Inference Settings](#e1-inference-settings)
    - [E.2 Running the sample inference code](#e2-running-the-sample-inference-code)
    - [E.3 Inference GPU Memory Usage](#e3-inference-gpu-memory-usage)
      - [E.3.1 FP16 Inference](#e31-fp16-inference)
      - [E.3.2 FP32 Inference](#e32-fp32-inference)
      - [E.3.3 AMP Inference (Autocast Enabled)](#e33-amp-inference-autocast-enabled)
  - [F. UNet Architecture](#f-unet-architecture)
    - [F.1 U-Net (used for base model)](#f1-u-net-used-for-base-model)
    - [F.2 Efficient U-Net (used for SR models)](#f2-efficient-u-net-used-for-sr-models)

## B. Imagen Pipeline

Imagen comprises a frozen text encoder (e.g. T5-XXL) to map input text into a sequence of embeddings, and a 64x64 image diffusion model, followed by two super-resolution diffusion models for generating 256x256 and 1024x1024 images. All diffusion models are conditioned on the text embedding sequence and use classifier-free guidance.

## C. Files in this folder

- [imagen_training.py](imagen_training.py): Script for running inference
- [imagen_generate_images.py](imagen_generate_images.py): Script for generating images for FID-CLIP analysis
- [imagen_infer.py](imagen_infer.py): Script for running inference

## D. Imagen Training

All three diffusion models (64x64, 256x256, 1024x1024) can be trained independently.

### D.1 Training Dataset

### D.2 Training configs
| configs  | Description  |
|---|---|
| base64-2b.yaml  | 2b-parameter base 64x64 model as described in Imagen paper  |
| base64-500m.yaml | 500m-parameter base 64x64 model with decreased number of embedding channels|
|sr256-400m.yaml| 400m-parameter sr 256x256 model as described in Imagen paper |
|sr1024-400m.yaml| 400m-parameter sr 1024x1024 model as described in Imagen paper |

## E. Imagen Inference

### E.1 Inference Settings

[inference_pipeline.yaml](conf/inference_pipeline.yaml) specifies every config for running the sample inference code. Specifically:
- num_images_per_promt: The number of images you want to generate for each text prompt
- model_name: Different pre-defined configs (not used for now)
- run_ema_model: Either run reg/ema model for pretrained models
- customized_model: Instead of loading pre-defined models, load specified checkpoint. .ckpt checkpoint (generated during in-the-middle of training) and .nemo checkpoint (generated once training completed) are both acceptable 
- target_resolution: should be one of [64, 256, 1024]
- inference_precision: Running inference in one of [16, 32, AMP] mode
- dynamic_thresholding: Whether to use dynamic thresholding when generating images
- texts: List of text prompts that are used to generate images
- output_path: The path to save generate images
- encoder_path: If not set (null), it will download text encoder first time running the inference code (and will be saved to HF_HOME), you can also load it offline by setting it to the prepared folder
- samplers: List of sampler settings that are used for each model. `step` (the number of iterations to denoise the image, ideally the larger the better, but also consume more time) and `cfg` for classifier free guidance value. You can tweak these values for better visual quality.

### E.2 Running the sample inference code
```
(inside NeMo root folder)
python examples/multimodal/generative/imagen/imagen_infer.py
```

### E.3 Inference GPU Memory Usage

#### E.3.1 FP16 Inference
| Output\Batch size | 1     | 8     |
|-------------------|-------|-------|
| 64x64             | 11.7G | 11.9G |
| 256x256           | 12.5G | 13.0G |
| 1024x1024         | 14.1G | 21.6G |

#### E.3.2 FP32 Inference
| Output\Batch size | 1     | 8     |
|-------------------|-------|-------|
| 64x64             | 21.7G | 22.6G |
| 256x256           | 23.4G | 24.5G |
| 1024x1024         | 26.6G | 40.6G |

#### E.3.3 AMP Inference (Autocast Enabled)
| Output\Batch size | 1     | 8     |
|-------------------|-------|-------|
| 64x64             | 22.4G | 23.4G |
| 256x256           | 24.0G | 25.1G |
| 1024x1024         | 26.4G | 33.7G |

## F. UNet Architecture

We have prepared two types of UNet for Imagen according to the paper. Base model (64x64) and SR models (256x256, 1024x1024) are using different UNet models.

### F.1 U-Net (used for base model)



### F.2 Efficient U-Net (used for SR models)

