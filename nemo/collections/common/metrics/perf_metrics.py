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

from typing import Any, Dict, List, Optional

import numpy as np
from lightning.pytorch.callbacks import Callback

from nemo.collections.common.parts.perf_metrics_utils import read_tb_log
from nemo.utils import flops_formulas, logging

__all__ = ["FLOPsMeasurementCallback"]


class FLOPsMeasurementCallback(Callback):
    """
    Calculate FLOPs per second after last train step for a given job run.

    Args:
        model_config (Dict[str, Any]): params for running the experiment/job.
        Expects a nested dictionary with parent keys
            1. run- for assessing model name (Eg. 'gpt3', 'llama2', etc.) from sub-key 'name'.
                'name' usually has value like- train_gpt3_5b_*, which is matched to model name 'gpt3'.
            2. exp_manager- for accessing 'explicit_log_dir'. tensorboard log file is stored here,
                used for accessing step time needed for calculating TFLOPs per sec per GPU
            3. trainer- for accessing 'num_nodes' and 'devices' needed for calculating
                TFLOPs per sec per GPU
            4. model- Hyperparams for the model. Specifically- global batch size, sequence length,
                hidden size,  ffn hidden size, num_layers, num_attention_heads, num_query_groups,
                moe_router_topk. (list might increase with new models as required)
        log_dir (Optional[str]): Directory with tenbsorboard log file. If present, will overrride
            'explicit_log_dir' in model_config. Defaults to None.
        model_name (Optional[str]): If present, will override 'name' under 'run' in model_config.
            Defaults to None.
    """

    higher_is_better = True

    def __init__(
        self,
        model_config: Dict[str, Any],
        log_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.cfg = model_config

        self.run_cfg = self.cfg.get('run', {})
        # exp_manager = None is valid and indicates no exp_manager should be initialized
        self.exp_cfg = self.cfg.get('exp_manager', {}) or {}
        self.train_cfg = self.cfg.get('trainer', {})
        self.model_cfg = self.cfg.get('model', {})

        # use config params only when NOT provided explicitly
        self.model = self.run_cfg.get('name', "") if model_name is None else model_name
        self.log_dir = self.exp_cfg.get('explicit_log_dir', None) if log_dir is None else log_dir

        self.num_nodes = self.train_cfg.get('num_nodes', None)
        self.num_gpus_per_node = self.train_cfg.get('devices', None)

        gbs = self.model_cfg.get('global_batch_size', None)
        enc_seq_len = self.model_cfg.get('encoder_seq_length', None)
        hs = self.model_cfg.get('hidden_size', None)
        layers = self.model_cfg.get('num_layers', None)
        ffn_hs = self.model_cfg.get('ffn_hidden_size', None)
        attention_heads = self.model_cfg.get('num_attention_heads', None)
        moe_router_topk = self.model_cfg.get('moe_router_topk', None)

        # this handles both- 1. key is present, value is None; 2. key is absent
        query_groups = self.model_cfg.get('num_query_groups', None)
        if query_groups is None:
            query_groups = attention_heads

        self.flops_config = flops_formulas.FLOPSConfig(
            gbs=gbs,
            enc_seq_len=enc_seq_len,
            hs=hs,
            layers=layers,
            ffn_hs=ffn_hs,
            attention_heads=attention_heads,
            moe_router_topk=moe_router_topk,
            query_groups=query_groups,
        )

        self.model = self.model.lower() if self.model is not None else self.model

    def on_train_end(self, trainer, pl_module):
        """
        PyTorch Lightning callback hook to calculate TFLOPs per sec per GPU after training
        """
        tflops_per_sec_per_gpu = -1

        try:
            if "peft" in self.cfg["model"]:
                raise NotImplementedError("FLOPs measurement not supported for finetuning jobs")

            step_time_list = read_tb_log(self.log_dir, "train_step_timing in s")
            tflops_per_sec_per_gpu = self.eval_tflops_per_sec_per_gpu(step_time_list)
        except Exception as exc:
            logging.error(f"Failed to calculate TFLOPs per sec per GPU.\n{exc}")

        logging.info(f"TFLOPs per sec per GPU={tflops_per_sec_per_gpu:.2f}")
        if pl_module.logger:
            pl_module.logger.experiment.add_scalar("tflops_per_sec_per_gpu", tflops_per_sec_per_gpu)

    def eval_tflops_per_sec_per_gpu(self, train_step_time: List | float | int) -> float:
        """
        Args:
            train_step_time (Any[List, float, int]): Train step time (in seconds).
            Step time will be less stable for initial steps (~10 steps)- less
            accurate measurement
            Use average step time over several steps for higher accuracy
        Returns:
            (float): Model TFLOPs per sec per gpu
        """
        total_flops, flops_per_gpu = self.eval_model_flops()

        if not isinstance(train_step_time, list):
            train_step_time = [train_step_time]
        # efficient mean computation if num train steps is very large
        step_time_arr = np.array(train_step_time)
        train_step_time = np.mean(step_time_arr[len(step_time_arr) // 2 :])

        return flops_per_gpu / (1e12 * train_step_time)

    def eval_model_flops(self):
        """
        Calculate model FLOPs for a given model
        """

        model_flops_map = {
            "gpt3": flops_formulas.gpt3,
            "llama2": flops_formulas.llama2,
            "llama3": flops_formulas.llama3,
            "nemotron": flops_formulas.nemotron,
            "mixtral": flops_formulas.mixtral,
            "bert": flops_formulas.bert,
        }

        if self.model is not None:
            model_matches = [model for model in model_flops_map if model in self.model]
            self.model = model_matches[0] if len(model_matches) > 0 else self.model
        if self.model not in model_flops_map:
            logging.info(f"FLOPs measurement supported for {list(model_flops_map.keys())}")
            raise KeyError(f"Failed to extract valid model name from or missing FLOPs calculations for {self.model}")

        total_flops = model_flops_map[self.model](self.flops_config)
        flops_per_gpu = total_flops / (self.num_nodes * self.num_gpus_per_node)

        return total_flops, flops_per_gpu
