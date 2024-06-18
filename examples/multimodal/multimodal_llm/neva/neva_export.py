# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import shutil
import tempfile
from time import time

import tensorrt as trt
import torch
import yaml
from tensorrt_llm.builder import Builder
from transformers import AutoModel

from nemo.core.config import hydra_runner
from nemo.export import TensorRTLLM
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_nemo_model

logger = trt.Logger(trt.Logger.INFO)


def export_visual_wrapper_onnx(
    visual_wrapper, input, output_dir, input_names=['input'], dynamic_axes={'input': {0: 'batch'}}
):
    logger.log(trt.Logger.INFO, "Exporting onnx")
    os.makedirs(f'{output_dir}/onnx', exist_ok=True)
    torch.onnx.export(
        visual_wrapper,
        input,
        f'{output_dir}/onnx/visual_encoder.onnx',
        opset_version=17,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )


def build_trt_engine(
    model_type, input_sizes, output_dir, max_batch_size, dtype=torch.bfloat16, image_size=None, num_frames=None
):
    part_name = 'visual_encoder'
    onnx_file = '%s/onnx/%s.onnx' % (output_dir, part_name)
    engine_file = '%s/%s.engine' % (output_dir, part_name)
    config_file = '%s/%s' % (output_dir, "config.json")
    logger.log(trt.Logger.INFO, "Building TRT engine for %s" % part_name)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config_args = {"precision": str(dtype).split('.')[-1], "model_type": model_type}
    if image_size is not None:
        config_args["image_size"] = image_size
    if num_frames is not None:
        config_args["num_frames"] = num_frames

    config_wrapper = Builder().create_builder_config(**config_args)
    config = config_wrapper.trt_builder_config

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    # Delete onnx files since we don't need them now
    shutil.rmtree(f'{output_dir}/onnx')

    nBS = -1
    nMinBS = 1
    nOptBS = max(nMinBS, int(max_batch_size / 2))
    nMaxBS = max_batch_size

    inputT = network.get_input(0)

    # input sizes can be a list of ints (e.g., [3, H, W]) when inputs are images,
    # or a list of three int lists (e.g., [[1, 1, 2700], [1, 500, 2700], [1, 4096, 2700]]).
    assert isinstance(input_sizes, list), "input_sizes must be a list"
    if isinstance(input_sizes[0], int):
        logger.log(trt.Logger.INFO, f"Processed input sizes {input_sizes}")
        inputT.shape = [nBS, *input_sizes]
        min_size = opt_size = max_size = input_sizes
    elif len(input_sizes) == 3 and isinstance(input_sizes[0], list):
        min_size, opt_size, max_size = input_sizes
        logger.log(trt.Logger.INFO, f"Processed min/opt/max input sizes {min_size}/{opt_size}/{max_size}")
    else:
        raise ValueError(f"invalid input sizes: {input_sizes}")

    profile.set_shape(inputT.name, [nMinBS, *min_size], [nOptBS, *opt_size], [nMaxBS, *max_size])
    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO, "Succeeded building %s in %d s" % (engine_file, t1 - t0))
        with open(engine_file, 'wb') as f:
            f.write(engine_string)

    Builder.save_config(config_wrapper, config_file)


def build_neva_engine(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # extract NeMo checkpoint
    with tempfile.TemporaryDirectory() as temp:
        mp0_weights, nemo_config, _ = load_nemo_model(cfg.model.visual_model_path, temp)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            vision_x = self.encoder(pixel_values=images, output_hidden_states=True)
            vision_x = vision_x.hidden_states[-2]
            vision_x = vision_x[:, 1:]
            vision_x = self.connector(vision_x)
            return vision_x

    encoder = AutoModel.from_pretrained(
        vision_config["from_pretrained"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    vision_encoder = encoder.vision_model
    hf_config = encoder.config
    dtype = hf_config.torch_dtype

    # connector
    assert nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "mlp2x_gelu"
    vision_connector = torch.nn.Sequential(
        torch.nn.Linear(vision_config["hidden_size"], nemo_config["hidden_size"], bias=True),
        torch.nn.GELU(),
        torch.nn.Linear(nemo_config["hidden_size"], nemo_config["hidden_size"], bias=True),
    ).to(dtype=dtype)

    key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
    for layer in range(0, 3, 2):
        vision_connector[layer].load_state_dict(
            {
                'weight': mp0_weights[f"{key_prefix}.{layer}.weight"].to(dtype),
                'bias': mp0_weights[f"{key_prefix}.{layer}.bias"].to(dtype),
            }
        )

    # export the whole wrapper
    wrapper = VisionEncoderWrapper(vision_encoder, vision_connector).to(device, dtype)
    image_size = hf_config.vision_config.image_size
    dummy_image = torch.empty(
        1, 3, image_size, image_size, dtype=dtype, device=device
    )  # dummy image shape [B, C, H, W]

    engine_dir = os.path.join(cfg.infer.output_dir, "visual_engine")

    export_visual_wrapper_onnx(wrapper, dummy_image, engine_dir)
    build_trt_engine(
        cfg.model.type,
        [3, image_size, image_size],
        engine_dir,
        cfg.infer.visual.max_batch_size,
        dtype,
        image_size=image_size,
    )


def build_video_neva_engine(cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # extract NeMo checkpoint
    with tempfile.TemporaryDirectory() as temp:
        connector = NLPSaveRestoreConnector()
        connector._unpack_nemo_file(path2file=cfg.model.visual_model_path, out_folder=temp)
        config_yaml = os.path.join(temp, connector.model_config_yaml)
        with open(config_yaml) as f:
            nemo_config = yaml.safe_load(f)
        if nemo_config["tensor_model_parallel_size"] > 1:
            path = os.path.join(temp, 'mp_rank_00', connector.model_weights_ckpt)
        else:
            path = os.path.join(temp, connector.model_weights_ckpt)
        mp0_weights = connector._load_state_dict_from_disk(path)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            b, num_frames, c, h, w = images.shape
            images = images.view(b * num_frames, c, h, w)
            vision_x = self.encoder(pixel_values=images, output_hidden_states=True)  # [(B num_frames), C, H, W]
            vision_x = vision_x.hidden_states[-2]
            vision_x = vision_x[:, 1:]

            # reshape back to [B, num_frames, img_size, hidden_size]
            vision_x = vision_x.view(b, num_frames, -1, vision_x.shape[-1])

            vision_x = self.connector(vision_x)
            return vision_x

    encoder = AutoModel.from_pretrained(
        vision_config["from_pretrained"], torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    vision_encoder = encoder.vision_model
    hf_config = encoder.config
    dtype = hf_config.torch_dtype

    # connector
    assert nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "linear"
    vision_connector = torch.nn.Linear(vision_config["hidden_size"], nemo_config["hidden_size"], bias=True)

    key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
    vision_connector.load_state_dict(
        {
            'weight': mp0_weights[f"{key_prefix}.weight"].to(dtype),
            'bias': mp0_weights[f"{key_prefix}.bias"].to(dtype),
        }
    )

    engine_dir = os.path.join(cfg.infer.output_dir, "visual_engine")

    # export the whole wrapper
    wrapper = VisionEncoderWrapper(vision_encoder, vision_connector).to(device, dtype)
    image_size = hf_config.vision_config.image_size
    num_frames = nemo_config['data']['num_frames']
    dummy_video = torch.empty(1, num_frames, 3, image_size, image_size, dtype=dtype, device=device)  # dummy image
    export_visual_wrapper_onnx(wrapper, dummy_video, engine_dir)
    build_trt_engine(
        cfg.model.type,
        [num_frames, 3, image_size, image_size],  # [num_frames, 3, H, W]
        engine_dir,
        cfg.infer.visual.max_batch_size,
        dtype,
        image_size=image_size,
        num_frames=num_frames,
    )


def build_trtllm_engines(cfg):
    engine_dir = os.path.join(cfg.infer.output_dir, "llm_engine")
    trt_llm_exporter = TensorRTLLM(model_dir=engine_dir, load_model=False)
    trt_llm_exporter.export(
        nemo_checkpoint_path=cfg.model.visual_model_path if cfg.model.type == "neva" else cfg.model.llm_model_path,
        model_type=cfg.model.llm_model_type,
        tensor_parallel_size=cfg.infer.llm.tensor_parallelism,
        max_input_len=cfg.infer.llm.max_input_len,
        max_output_len=cfg.infer.llm.max_output_len,
        max_batch_size=cfg.infer.llm.max_batch_size,
        max_prompt_embedding_table_size=cfg.infer.llm.max_multimodal_len,
        dtype=cfg.model.precision,
        load_model=False,
    )


@hydra_runner(config_path='conf', config_name='neva_export')
def main(cfg):
    build_trtllm_engines(cfg)

    if cfg.model.type == "neva":
        build_neva_engine(cfg)
    elif cfg.model.type == "video-neva":
        build_video_neva_engine(cfg)
    else:
        raise RuntimeError(f"Invalid model type {cfg.model.type}")


if __name__ == '__main__':
    main()
