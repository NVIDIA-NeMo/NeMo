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

from omegaconf.dictconfig import DictConfig

from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
    NemotronVoicechatInferenceWrapper,
)
from nemo.collections.speechlm2.inference.model_wrappers.engine_selection import precision_matches_cfg
from nemo.collections.speechlm2.inference.pipelines.streaming_s2s_pipeline import StreamingS2SPipeline
from nemo.utils import logging as logger


class S2SPipelineBuilder:
    """Factory that builds a streaming S2S pipeline."""

    @classmethod
    def build_pipeline(cls, cfg: DictConfig) -> StreamingS2SPipeline:
        """
        Build the streaming S2S pipeline based on the config.

        Precision and determinism are torch process globals: they must be in
        effect before any weights load and stay in effect for the whole run.
        That cannot be owned by an object this call returns, so the caller
        scopes it and this call *requires* it, rather than warning afterwards::

            with inference_precision_from_cfg(cfg.s2s):
                pipeline = S2SPipelineBuilder.build_pipeline(cfg)
                try:
                    pipeline.run(audio_filepaths)
                finally:
                    pipeline.shutdown()

        :meth:`StreamingS2SPipeline.shutdown` releases any vLLM-Omni runtime;
        it is a no-op for native engines.

        Args:
            cfg: (DictConfig) Config
        Returns:
            Returns StreamingS2SPipeline object
        """
        if not precision_matches_cfg(cfg.s2s):
            raise RuntimeError(
                "This process is not configured the way cfg.s2s asks: allow_tf32, "
                "matmul_precision and deterministic are torch process globals that have to be "
                "applied before the checkpoint loads, and they are not in effect. Wrap the call:\n"
                "    from nemo.collections.speechlm2.inference.model_wrappers.engine_selection "
                "import inference_precision_from_cfg\n"
                "    with inference_precision_from_cfg(cfg.s2s):\n"
                "        pipeline = S2SPipelineBuilder.build_pipeline(cfg)\n"
                "        try:\n"
                "            ...\n"
                "        finally:\n"
                "            pipeline.shutdown()"
            )

        s2s_model = NemotronVoicechatInferenceWrapper(model_cfg=cfg.s2s)

        logger.info(f"S2S model `{cfg.s2s.model_path}` loaded")

        pipeline = StreamingS2SPipeline(cfg, s2s_model)
        logger.info(f"`{type(pipeline).__name__}` pipeline loaded")
        return pipeline
