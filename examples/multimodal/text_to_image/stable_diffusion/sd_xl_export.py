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
import torch.nn as nn

from nemo.collections.multimodal.models.text_to_image.stable_diffusion.diffusion_engine import MegatronDiffusionEngine
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner
from nemo.utils.trt_utils import build_engine


@hydra_runner(config_path='conf', config_name='sd_xl_export')
def main(cfg):
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None
        model_cfg.first_stage_config._target_ = 'nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.autoencoder.AutoencoderKLInferenceWrapper'
        model_cfg.fsdp = False

    torch.backends.cuda.matmul.allow_tf32 = True
    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronDiffusionEngine, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )

    model = megatron_diffusion_model.model
    model.cuda().eval()

    output_dir = cfg.infer.out_path
    os.makedirs(f"{output_dir}/onnx/unet_xl/", exist_ok=True)
    os.makedirs(f"{output_dir}/onnx/clip1/", exist_ok=True)
    os.makedirs(f"{output_dir}/onnx/clip2/", exist_ok=True)
    os.makedirs(f"{output_dir}/onnx/vae/", exist_ok=True)
    os.makedirs(f"{output_dir}/plan/", exist_ok=True)

    in_channels = model.model.diffusion_model.in_channels
    seq_length = model.conditioner.embedders[0].max_length
    adm_in_channels = model.model.diffusion_model.adm_in_channels

    def get_dummy_inputs(model_name):
        dummy_input = {}
        if 'unet' in model_name:
            dummy_input["x"] = torch.ones(2, in_channels, cfg.infer.height // 8, cfg.infer.width // 8, device="cuda")
            dummy_input["y"] = torch.ones(2, adm_in_channels, device="cuda")
            dummy_input["timesteps"] = torch.ones(2, device="cuda")
            dummy_input["context"] = torch.ones(2, 80, 2048, device="cuda")
        elif 'vae' in model_name:
            dummy_input["z"] = torch.ones(2, in_channels, cfg.infer.height // 8, cfg.infer.width // 8, device="cuda")
        elif 'clip' in model_name:
            dummy_input["input_ids"] = torch.randint(high=10, size=(2, seq_length), device='cuda')
        return dummy_input

    def get_input_profile(model_name, batch_size, static_batch=False, min_batch_size=1, max_batch_size=8):
        assert batch_size >= min_batch_size and batch_size <= max_batch_size
        if static_batch:
            min_batch_size = batch_size if static_batch else min_batch_size
            max_batch_size = batch_size if static_batch else max_batch_size
        input_profile = {}
        dummy_input = get_dummy_inputs(model_name)
        for key, value in dummy_input.items():
            input_profile[key] = [
                (min_batch_size, *(value.shape[1:])),
                (batch_size, *(value.shape[1:])),
                (max_batch_size, *(value.shape[1:])),
            ]
        return input_profile

    def get_input_output_names(model_name):
        if model_name == 'unet_xl':
            input_names = [
                "x",
                "timesteps",
                "context",
                "y",
            ]
            output_names = ["out"]
        elif model_name == 'vae':
            input_names = ["z"]
            output_names = ["dec"]
        elif model_name == 'clip1':
            input_names = ["input_ids"]
            output_names = ["z"]
        elif model_name == 'clip2':
            input_names = ["input_ids"]
            output_names = ["z", "z_pooled"]
        else:
            raise NotImplementedError(f"{model_name} is not supported")
        return input_names, output_names

    def get_dynamic_axis(model_name):
        if 'unet' in model_name:
            dynamic_axes = {
                "x": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
                "timesteps": {0: "steps"},
                "y": {0: "batch_size", 1: "adm_in"},
                "context": {0: "batch_size", 1: "sequence_length"},
            }
        elif 'vae' in model_name:
            dynamic_axes = {"z": {0: 'batch_size'}, "dec": {0: 'batch_size'}}
        elif 'clip' in model_name:
            dynamic_axes = {"input_ids": {0: 'batch_size'}, "z": {0: 'batch_size'}, "z_pooled": {0: 'batch_size'}}
        else:
            raise NotImplementedError(f"{model_name} is not supported")
        return dynamic_axes

    # Unet Export
    unet_xl = model.model.diffusion_model
    model_name = 'unet_xl'
    dummy_input = get_dummy_inputs(model_name)
    input_names, output_names = get_input_output_names(model_name)
    dynamic_axes = get_dynamic_axis(model_name)

    if not os.path.exists(f"{output_dir}/onnx/unet_xl/unet_xl.onnx"):
        torch.onnx.export(
            unet_xl,
            (dummy_input,),
            f"{output_dir}/onnx/unet_xl/unet_xl.onnx",
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=14,
        )

    ###VAE Export
    class VAEWrapper(nn.Module):
        def __init__(self, model):
            super(VAEWrapper, self).__init__()
            self.model = model

        def forward(self, z):
            z = self.model.post_quant_conv(z)
            dec = self.model.decoder(z)
            return dec

    VAE = VAEWrapper(model.first_stage_model)
    model_name = 'vae'
    dummy_input = get_dummy_inputs(model_name)
    input_names, output_names = get_input_output_names(model_name)
    dynamic_axes = get_dynamic_axis(model_name)

    if not os.path.exists(f"{output_dir}/onnx/vae/vae.onnx"):
        torch.onnx.export(
            VAE,
            (dummy_input,),
            f"{output_dir}/onnx/vae/vae.onnx",
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )

    #### CLIP Export
    class FrozenCLIPWrapper(nn.Module):
        def __init__(self, model):
            super(FrozenCLIPWrapper, self).__init__()
            self.model = model

        def forward(self, input_ids):
            outputs = self.model.transformer(input_ids=input_ids, output_hidden_states=True)
            z = outputs.hidden_states[11]
            seq_len = (z.shape[1] + 8 - 1) // 8 * 8
            z = torch.nn.functional.pad(z, (0, 0, 0, seq_len - z.shape[1]), value=0.0)
            return z

    clip1 = FrozenCLIPWrapper(model.conditioner.embedders[0])
    model_name = 'clip1'
    dummy_input = get_dummy_inputs(model_name)
    input_names, output_names = get_input_output_names(model_name)
    dynamic_axes = get_dynamic_axis(model_name)

    if not os.path.exists(f"{output_dir}/onnx/clip1/clip1.onnx"):
        torch.onnx.export(
            clip1,
            (dummy_input,),
            f"{output_dir}/onnx/clip1/clip1.onnx",
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )

    class FrozenOpenCLIPWrapper(nn.Module):
        def __init__(self, model):
            super(FrozenOpenCLIPWrapper, self).__init__()
            self.model = model

        def forward(self, input_ids):
            z = self.model.encode_with_transformer(input_ids)
            z_layer = z['penultimate']
            seq_len = (z_layer.shape[1] + 8 - 1) // 8 * 8
            z_layer = torch.nn.functional.pad(z_layer, (0, 0, 0, seq_len - z_layer.shape[1]), value=0.0)
            return z_layer, z["pooled"]

    clip2 = FrozenOpenCLIPWrapper(model.conditioner.embedders[1])
    model_name = 'clip2'
    dummy_input = get_dummy_inputs(model_name)
    input_names, output_names = get_input_output_names(model_name)
    dynamic_axes = get_dynamic_axis(model_name)

    if not os.path.exists(f"{output_dir}/onnx/clip1/clip2.onnx"):
        torch.onnx.export(
            clip2,
            (dummy_input,),
            f"{output_dir}/onnx/clip2/clip2.onnx",
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )

    del model, trainer, megatron_diffusion_model
    torch.cuda.empty_cache()
    batch_size = cfg.infer.get('batch_size', 1)
    min_batch_size = cfg.trt.min_batch_size
    max_batch_size = cfg.trt.max_batch_size
    static_batch = cfg.trt.static_batch
    fp16 = cfg.trainer.precision in ['16', '16-mixed', 16]
    for model_name in ['unet_xl', 'vae', 'clip1', 'clip2']:
        if not os.path.exists(f"{output_dir}/plan/{model_name}.plan"):
            build_engine(
                f"{output_dir}/onnx/{model_name}/{model_name}.onnx",
                f"{output_dir}/plan/{model_name}.plan",
                fp16=fp16 if model_name != 'vae' else False,
                input_profile=get_input_profile(
                    model_name,
                    batch_size,
                    static_batch=static_batch,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                ),
                timing_cache=None,
            )


if __name__ == "__main__":
    main()
