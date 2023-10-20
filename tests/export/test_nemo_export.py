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

from pathlib import Path
import pytest
from nemo.export import TensorRTLLM
from tests.infer_data_path import get_infer_test_data, download_nemo_checkpoint


class TestNemoExport:
    @pytest.mark.unit
    def test_trt_llm_export(self):
        """Here we test the trt-llm transfer and infer function"""

        test_data = get_infer_test_data()
        test_at_least_one = False

        for model_name, model_info in test_data.items():
            if model_info["location"] == "HF":
                download_nemo_checkpoint(
                    model_info["checkpoint_link"], model_info["checkpoint_dir"], model_info["checkpoint"]
                )

            print(
                "Path: {0} and model: {1} is next ...".format(
                    model_info["checkpoint"], model_name
                )
            )
            if Path(model_info["checkpoint"]).exists():
                for n_gpu in model_info["total_gpus"]:
                    print(
                        "Path: {0} and model: {1} with {2} gpus will be tested".format(
                            model_info["checkpoint"], model_name, n_gpu
                        )
                    )
                    Path(model_info["trt_llm_model_dir"]).mkdir(parents=True, exist_ok=True)

                    prompt_embeddings_checkpoint_path = None
                    if "p_tuning_checkpoint" in model_info.keys():
                        if Path(model_info["p_tuning_checkpoint"]).exists():
                            prompt_embeddings_checkpoint_path = model_info["p_tuning_checkpoint"]

                    trt_llm_exporter = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])

                    trt_llm_exporter.export(
                        nemo_checkpoint_path=model_info["checkpoint"],
                        model_type=model_info["model_type"],
                        n_gpus=n_gpu,
                        prompt_embeddings_checkpoint_path=prompt_embeddings_checkpoint_path,
                    )
                    output = trt_llm_exporter.forward(
                        input_texts=["Hi, how are you?", "I am good, thanks, how about you?"],
                        max_output_token=200,
                    )
                    print("output 1: ", output)

                    trt_llm_exporter2 = TensorRTLLM(model_dir=model_info["trt_llm_model_dir"])
                    output = trt_llm_exporter2.forward(
                        input_texts=["Let's see how this works", "Did you get the result yet?"],
                        max_output_token = 200,
                    )
                    print("output 2: ", output)

                test_at_least_one = True

        assert test_at_least_one, "At least one nemo checkpoint has to be tested."

