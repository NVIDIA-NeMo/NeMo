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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
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

import argparse
import os
import sys

import torch
from pytorch_lightning import Trainer

from nemo.core import ModelPT
from nemo.core.classes import Exportable
from nemo.core.config.pytorch_lightning import TrainerConfig
from nemo.utils import logging

try:
    from contextlib import nullcontext
except ImportError:
    # handle python < 3.7
    from contextlib import suppress as nullcontext


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"Export NeMo models to ONNX/Torchscript",
    )
    parser.add_argument("source", help="Source .nemo file")
    parser.add_argument("out", help="Location to write result to")
    parser.add_argument("--autocast", action="store_true", help="Use autocast when exporting")
    parser.add_argument("--runtime-check", action="store_true", help="Runtime check of exported net result")
    parser.add_argument("--verbose", default=None, help="Verbose level for logging, numeric")
    parser.add_argument("--max-batch", type=int, default=None, help="Max batch size for model export")
    parser.add_argument("--max-dim", type=int, default=None, help="Max dimension(s) for model export")
    parser.add_argument("--onnx-opset", type=int, default=None, help="ONNX opset for model export")
    parser.add_argument("--device", default="cuda", help="Device to export for")
    args = parser.parse_args(argv)
    return args


def nemo_export(argv):
    args = get_args(argv)
    loglevel = logging.INFO
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    if args.verbose is not None:
        numeric_level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % numeric_level)
        loglevel = numeric_level
    logging.setLevel(loglevel)
    logging.info("Logging level set to {}".format(loglevel))

    """Convert a .nemo saved model into .riva Riva input format."""
    nemo_in = args.source
    out = args.out

    # Create a PL trainer object which is required for restoring Megatron models
    cfg_trainer = TrainerConfig(
        gpus=1,
        accelerator="ddp",
        num_nodes=1,
        # Need to set the following two to False as ExpManager will take care of them differently.
        logger=False,
        checkpoint_callback=False,
    )
    trainer = Trainer(cfg_trainer)

    logging.info("Restoring NeMo model from '{}'".format(nemo_in))
    try:
        with torch.inference_mode():
            # Restore instance from .nemo file using generic model restore_from
            model = ModelPT.restore_from(restore_path=nemo_in, trainer=trainer)
    except Exception as e:
        logging.error(
            "Failed to restore model from NeMo file : {}. Please make sure you have the latest NeMo package installed with [all] dependencies.".format(
                nemo_in
            )
        )
        raise e

    logging.info("Model {} restored from '{}'".format(model.__class__.__name__, nemo_in))

    if not isinstance(model, Exportable):
        logging.error("Your NeMo model class ({}) is not Exportable.".format(model.__class__.__name__))
        sys.exit(1)

    #
    #  Add custom export parameters here
    #
    in_args = {}
    if args.max_batch is not None:
        in_args["max_batch"] = args.max_batch
    if args.max_dim is not None:
        in_args["max_dim"] = args.max_dim

    autocast = nullcontext
    model.to(device=args.device).freeze()
    if args.autocast:
        autocast = torch.cuda.amp.autocast
    try:
        with autocast(), torch.inference_mode():
            _, descriptions = model.export(
                out, check_trace=args.runtime_check, onnx_opset_version=args.onnx_opset, verbose=args.verbose,
            )

    except Exception as e:
        logging.error(
            "Export failed. Please make sure your NeMo model class ({}) has working export() and that you have the latest NeMo package installed with [all] dependencies.".format(
                model.__class__
            )
        )
        raise e


if __name__ == '__main__':
    nemo_export(sys.argv[1:])
