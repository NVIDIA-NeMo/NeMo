# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin, NLPSaveRestoreConnector
from nemo.utils import logging, model_utils
from nemo.utils.app_state import AppState


"""
Usage:
python megatron_change_num_partitions.py \
    --model_file=PATH_TO_SRC_FILE \
    --target_file=PATH_TO_TGT_FILE \
    --tensor_model_parallel_size=2 \
    --target_tensor_model_parallel_size=1
"""


def merge_partition(model, partitions, write_path=None):
    idx = 0
    for name, param in model.named_parameters():
        if param.shape == partitions[0][idx].shape:
            concated = partitions[0][idx].data
        elif param.shape[0] == partitions[0][idx].shape[0]:
            concated = torch.cat([partitions[i][idx].data for i in range(len(partitions))], dim=-1)
        else:
            concated = torch.cat([partitions[i][idx].data for i in range(len(partitions))], dim=0)
        if concated.shape != param.shape:
            logging.info(
                f"Warning: Shape mismatch for parameter {name} required shape: {param.shape}, merged shape: {concated.shape}. Narrowing to match required size."
            )
            if concated.shape[1:] == param.shape[1:]:
                concated = torch.narrow(concated, 0, 0, param.shape[0])
            elif concated.shape[:-1] == param.shape[:-1]:
                concated = torch.narrow(concated, -1, 0, param.shape[-1])
            else:
                raise RuntimeError(
                    f"Can not handle parameter {name}, required shape: {param.shape}, merged shape: {concated.shape}."
                )
        param.data = concated
        idx += 1

    if write_path is not None:
        model.save_to(write_path)


def split_partition(model, partitions, tp_size, write_path=None):
    if len(partitions) != 1:
        raise ValueError(
            "Can only split partitions of model with TP=1. For partitions of models with TP>1, merge first."
        )

    if tp_size < 1:
        raise ValueError("TP size must to be >= 1.")

    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = 1  # not supported yet in this script
    app_state.tensor_model_parallel_size = tp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    app_state.tensor_model_parallel_rank = tp_size - 1

    idx = 0
    splits = []
    for _, param in model.named_parameters():
        if param.shape == partitions[0][idx].shape:
            split = [partitions[0][idx].data] * tp_size
        elif param.shape[0] == partitions[0][idx].shape[0]:
            split = torch.split(partitions[0][idx].data, param.shape[-1], dim=-1)
        else:
            split = torch.split(partitions[0][idx].data, param.shape[0], dim=0)
        splits.append(split)
        idx += 1

    for i in range(tp_size - 1, -1, -1):
        app_state.tensor_model_parallel_rank = i

        idx = 0
        for name, param in model.named_parameters():
            split_val = splits[idx][i]

            if param.shape != split_val.shape:
                logging.info(
                    f"Warning: Shape mismatch for parameter {name} required shape: {param.shape}, split shape: {split_val.shape}. Padding to match required size."
                )

                if split_val.shape[1:] == param.shape[1:]:
                    pad = [0, 0] * len(split_val.shape)
                    pad[-1] = param.shape[0] - split_val.shape[0]
                    split_val = torch.nn.functional.pad(split_val, pad, 'constant')
                elif split_val.shape[:-1] == param.shape[:-1]:
                    pad = [0, param.shape[-1] - split_val.shape[-1]]
                    split_val = torch.nn.functional.pad(split_val, pad, 'constant')
                else:
                    raise RuntimeError(
                        f"Can not handle parameter {name}, required shape: {param.shape}, split shape: {split_val.shape}."
                    )

            param.data = split_val
            idx += 1

        if write_path is not None:
            model.save_to(write_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True, help="Path to source .nemo file")
    parser.add_argument("--target_file", type=str, required=True, help="Path to write target .nemo file")
    parser.add_argument("--tensor_model_parallel_size", type=int, required=True, help="TP size of source model")
    parser.add_argument("--target_tensor_model_parallel_size", type=int, required=True, help="TP size of target model")
    parser.add_argument(
        "--model_class",
        type=str,
        default="nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel",
        help="NeMo model class. This script should support all NeMo megatron models that use Tensor Parallel",
    )
    parser.add_argument("--precision", default=16, help="PyTorch Lightning Trainer precision flag")

    args = parser.parse_args()

    precision = args.precision
    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))
    tp_size = args.tensor_model_parallel_size
    tgt_tp_size = args.target_tensor_model_parallel_size
    cls = model_utils.import_class_by_path(args.model_class)

    trainer = Trainer(devices=1, plugins=NLPDDPPlugin(), accelerator="cpu", precision=precision)
    app_state = AppState()
    app_state.data_parallel_rank = 0
    app_state.pipeline_model_parallel_size = 1  # not supported yet in this script
    app_state.tensor_model_parallel_size = tp_size
    app_state.model_parallel_size = app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size

    if tp_size > 1:
        partitions = []
        for i in range(tp_size):
            app_state.tensor_model_parallel_rank = i
            model = cls.restore_from(restore_path=args.model_file, trainer=trainer, map_location=torch.device("cpu"))
            params = [p for _, p in model.named_parameters()]
            partitions.append(params)
            # app_state is being updated incorrectly during restore
            app_state.data_parallel_rank = 0
            app_state.pipeline_model_parallel_size = 1  # not supported yet in this script
            app_state.tensor_model_parallel_size = tp_size
            app_state.model_parallel_size = (
                app_state.pipeline_model_parallel_size * app_state.tensor_model_parallel_size
            )

        model.cfg.tensor_model_parallel_size = 1
        app_state.model_parallel_size = 1
        trainer = Trainer(devices=1, plugins=NLPDDPPlugin(), accelerator="cpu", precision=precision)
        model = cls(model.cfg, trainer).to('cpu')
        model._save_restore_connector = NLPSaveRestoreConnector()

        if tgt_tp_size > 1:
            merge_partition(model, partitions)
        else:
            merge_partition(model, partitions, args.target_file)
    else:
        app_state.model_parallel_size = 1
        model = cls.restore_from(restore_path=args.model_file, trainer=trainer)

    if tgt_tp_size > 1:
        partitions = []
        params = [p for _, p in model.named_parameters()]
        partitions.append(params)

        model.cfg.tensor_model_parallel_size = tgt_tp_size
        app_state.model_parallel_size = tgt_tp_size
        trainer = Trainer(devices=1, plugins=NLPDDPPlugin(), accelerator="cpu", precision=precision)
        model = cls(model.cfg, trainer).to('cpu')
        model._save_restore_connector = NLPSaveRestoreConnector()

        split_partition(model, partitions, tgt_tp_size, args.target_file)

    logging.info("Successfully finished changing partitions!")


if __name__ == '__main__':
    main()
