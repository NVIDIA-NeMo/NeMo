# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""MTP speculative-decoding draft model for NeMo SpeechLM checkpoints.

``NeMoSpeechLMMTP`` wraps vLLM's ``NemotronHMTP`` and adapts the
weight-loading step for the NeMo checkpoint layout where all LLM
weights (including MTP layers) carry an ``llm.`` prefix:

    NeMo checkpoint            NemotronHMTP expects
    ──────────────────────     ────────────────────
    llm.mtp.layers.0.*    →    mtp.layers.0.*
    llm.lm_head.weight    →    lm_head.weight
    llm.model.embed_*     →    (embeddings shared from target model)

Only the ``mtp.*`` and ``lm_head.*`` weights are loaded here; the
embedding table is shared with the target model by vLLM's MTP framework.
"""

from collections.abc import Iterable

import torch
from vllm.model_executor.models.nemotron_h_mtp import NemotronHMTP
from vllm.sequence import IntermediateTensors

from nemo.collections.speechlm2.vllm.salm.audio import _pad_to_vocab_size


class NeMoSpeechLMMTP(NemotronHMTP):
    """NemotronH MTP draft model for NeMo SpeechLM checkpoints.

    Extends NemotronHMTP in two ways:

    * ``load_weights`` strips the ``llm.`` prefix from checkpoint names.
    * ``embed_input_ids`` fuses audio-feature embeddings into the token
      embeddings at placeholder positions, exactly like the target model.
      The MTP heads were trained on the same mixed text+audio embedding
      stream as the backbone, so the draft must see it too. vLLM probes
      ``draft_model.embed_input_ids(ids, multimodal_embeddings=None)`` at
      load time (``llm_base_proposer.load_model``); without this method
      the probe raises AttributeError and speculative decoding silently
      falls back to text-only draft inputs, which collapses acceptance
      rates on audio prompts.
    """

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed token IDs and merge audio embeddings at placeholder positions.

        Mirrors ``NeMoSpeechLMForConditionalGeneration.embed_input_ids``. The
        embedding table itself is shared from the target model by vLLM's MTP
        framework, so text-token rows are identical to the target's.
        """
        inputs_embeds = self.model.get_input_embeddings(input_ids)

        if multimodal_embeddings is None or is_multimodal is None or not is_multimodal.any():
            return inputs_embeds

        audio_embeds = torch.cat(list(multimodal_embeddings), dim=0)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[is_multimodal] = audio_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        lm_head_vocab = None
        for name, module in self.named_modules():
            if hasattr(module, "org_vocab_size") and "lm_head" in name:
                lm_head_vocab = module.org_vocab_size
                break

        def _strip_llm_prefix(items):
            for name, tensor in items:
                if name.startswith("llm."):
                    name = name[len("llm.") :]
                if name == "lm_head.weight" and lm_head_vocab is not None:
                    tensor = _pad_to_vocab_size(tensor, lm_head_vocab)
                yield name, tensor

        return super().load_weights(_strip_llm_prefix(weights))
