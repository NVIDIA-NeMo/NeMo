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

import pickle

import pandas as pd
from omegaconf import OmegaConf

from nemo.collections.common.tokenizers.column_coder import ColumnCodes
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="tabular_data_tokenizer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(OmegaConf.to_yaml(cfg))
    table = pd.read_csv(cfg.table_csv_file)
    example_arrays = {}
    for col in cfg.table_structure:
        col_name = col['name']
        example_arrays[col_name] = table[col_name].dropna().unique()
    cc = ColumnCodes.get_column_codes(cfg.table_structure, example_arrays)
    with open(cfg.tokenizer_file, 'wb') as handle:
        pickle.dump(cc, handle)


if __name__ == '__main__':
    main()
