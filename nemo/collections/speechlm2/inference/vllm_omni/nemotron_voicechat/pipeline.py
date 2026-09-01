# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Single-stage ``nemotron_voicechat`` Omni pipeline.

The VoiceChat wrapper intentionally runs NemotronDuplexH and EarTTS as
independent one-stage engines.  This pipeline is the Nemotron half: it
consumes one acoustic encoder embedding per :class:`StreamingInput` update
and emits the text token plus whichever auxiliary channel (ASR or function)
the checkpoint contains.  NeMo forwards the text token to the separate
``eartts`` pipeline.

The pipeline is registered against ``model_type = "nemotron_voicechat"``,
which the component checkpoint does not report natively, so the converted
wrapper directory remains the model root::

    <wrapper>/
        config.json               # {"model_type": "nemotron_voicechat"}
        nemotron/                 # directory or symlink → Nemotron ckpt
        eartts/                   # directory or symlink → EarTTS ckpt

Only ``nemotron/`` is loaded by this pipeline.  ``eartts/`` is passed
directly to a second :class:`AsyncOmni` instance.  The bundled deploy YAML at
``nemo/collections/speechlm2/inference/vllm_omni/deploy/nemotron_voicechat.yaml``
points this stage at its component via ``model_subdir`` / ``tokenizer_subdir``.
"""

from __future__ import annotations

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_SCHED_ASYNC = (
    "nemo.collections.speechlm2.inference.vllm_omni." "nemotron_voicechat.scheduler.NemotronVoicechatARAsyncScheduler"
)


NEMOTRON_VOICECHAT_PIPELINE = PipelineConfig(
    model_type="nemotron_voicechat",
    model_arch="NemotronDuplexHForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="nemotron",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            model_arch="NemotronDuplexHForCausalLM",
            # Stock vLLM-Omni 0.26 only includes single-stage AR requests in
            # the client multimodal pooler payload for the "audio" engine
            # output path. The final output remains text, so sampled text
            # tokens stay on RequestOutput while OmniOutput.multimodal_outputs
            # carries the optional ASR/function token beside it.
            engine_output_type="audio",
            scheduler_cls=_SCHED_ASYNC,
            sampling_constraints={"detokenize": False},
        ),
    ),
)


__all__ = [
    "NEMOTRON_VOICECHAT_PIPELINE",
]
