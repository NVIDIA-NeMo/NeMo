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

"""Single-stage streaming EarTTS pipeline.

NeMo submits text tokens directly to this engine after NemotronDuplexH has
produced them.  CFG is represented by two explicit requests in the same
engine, giving the conditional and unconditional streams independent vLLM KV
caches while allowing :class:`EarTTSCFGScheduler` to keep them in lockstep.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_CFG_SCHEDULER = (
    "nemo.collections.speechlm2.inference.vllm_omni."
    "eartts.scheduler.EarTTSCFGScheduler"
)


EARTTS_PIPELINE = PipelineConfig(
    model_type="eartts",
    model_arch="EarTTSForCausalLM",
    hf_architectures=("EarTTSForCausalLM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="eartts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            # vLLM-Omni derives the entry-stage ``generate`` task from
            # ``owns_tokenizer`` (runtime name: ``is_comprehension``). EarTTS
            # consumes token-id placeholders and deploy keeps
            # ``skip_tokenizer_init: true``, but this flag must still be true
            # for a direct SamplingParams request to pass task validation.
            owns_tokenizer=True,
            model_arch="EarTTSForCausalLM",
            engine_output_type="audio",
            retains_state_across_chunks=True,
            scheduler_cls=_CFG_SCHEDULER,
            sampling_constraints={"detokenize": False},
        ),
    ),
)


__all__ = ["EARTTS_PIPELINE"]
