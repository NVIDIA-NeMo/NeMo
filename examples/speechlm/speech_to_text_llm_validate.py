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
This is an example script for validating multi-modal speech-to-text LLM using NeMo.
All SpeechLMs that has the three componnets (audio encoder, modality adapter and LLM) are supported.
Some example models are:
- SALM (https://arxiv.org/abs/2310.09424)
- VoiceTextBlender (https://arxiv.org/abs/2410.17485)

Example usage:

export WANDB_API_KEY=${WANDB} && \
export CUDA_VISIBLE_DEVICES="1" && \
export HF_TOKEN=${HFTOKEN} && \
export HF_HOME="/home/heh/.huggingface/" && \
export HF_HUB_CACHE="/media/data/cache" && \
export NEMO_MODELS_CACHE="/media/data/pretrained_models/" && \
python speech_to_text_llm_validate.py \
    --config-path="/home/heh/github/NeMo-main/examples/speechlm/conf/salm"  \
    --config-name "salm_llama3.2-1b_fc_fc_peft" \
    ~data.train_ds \
    data.validation_ds.manifest_filepath=$VAL_MANIFESTS \
    data.train_ds.num_workers=$NUM_WORKERS \
    data.validation_ds.num_workers=$NUM_WORKERS \
    ++data.validation_ds.name=$VAL_NAMES \
    data.common.global_batch_size=$GLOBAL_BATCH \
    data.common.micro_batch_size=$MICRO_BATCH \
    ++model.resume_from_path=<path to model checkpoint>
"""


from nemo.collections.speechlm.recipes import speech_to_text_llm_validate
from nemo.core.config import hydra_runner


@hydra_runner(config_path="./conf/salm", config_name="salm_llama3.2-1b_fc_fc_peft")
def main(cfg):
    """main function for running validation."""
    return speech_to_text_llm_validate(cfg)


if __name__ == "__main__":
    main()
