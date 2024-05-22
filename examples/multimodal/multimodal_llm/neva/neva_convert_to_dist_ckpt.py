# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from argparse import ArgumentParser
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.models.multimodal_llm.neva.neva_model import MegatronNevaModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        required=True,
        help="Path to NeMo legacy checkpoints",
    )
    parser.add_argument("--output_path", type=str, default=None, required=True, help="Path to output .nemo file.")
    parser.add_argument("--gpus_per_node", type=int, required=False, default=8)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument(
        "--precision",
        type=str,
        required=False,
        default='bf16-mixed',
        choices=['32-true', '16-mixed', 'bf16-mixed'],
        help="Precision value for the trainer that matches with precision of the ckpt",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    cfg = {
        'trainer': {
            'devices': args.gpus_per_node,
            'num_nodes': args.num_nodes,
            'accelerator': 'gpu',
            'precision': args.precision,
        },
        'model': {
            'native_amp_init_scale': 2**32,
            'native_amp_growth_interval': 1000,
            'hysteresis': 2,
            'gradient_as_bucket_view': True,
        },
        'cluster_type': 'BCP',
    }
    cfg = OmegaConf.create(cfg)

    # Set precision None after precision plugins are created as PTL >= 2.1 does not allow both
    # precision plugins and precision to exist
    cfg.trainer.precision = None
    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(args.input_path):
        save_restore_connector.model_extracted_dir = args.input_path

    model = MegatronNevaModel.restore_from(
        restore_path=args.input_path,
        trainer=trainer,
        save_restore_connector=save_restore_connector,
        strict=False,
    )

    model.save_to(args.output_path)
    logging.info(f'NeMo model saved to: {args.output_path}')


if __name__ == '__main__':
    main()
