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

"""Reading vLLM-Omni stage outputs.

The shape of an ``OmniStageOutput`` is not part of any contract we control:
the multimodal payload may sit on the stage output or nested on its
completion, and the key it arrives under depends on whether the stage is
client-facing. These readers are the one place that tolerates that, so the
version-sensitivity is contained rather than spread through the session.
"""

from typing import Any, NamedTuple

import torch


class StepTokens(NamedTuple):
    """Per-frame tokens sampled by Nemotron for one acoustic frame.

    ``asr`` and ``function`` are *None* when the checkpoint has no such
    channel. ASR and function channels are independently optional.
    """

    text: int
    asr: int | None = None
    function: int | None = None


def _multimodal_output(stage_output: Any, req_out: Any) -> Any:
    """Return a stage's multimodal payload, whichever level carries it.

    vLLM-Omni attaches the payload to the ``MultimodalCompletionOutput`` in
    ``request_output.outputs[0]`` and also lifts it onto the stage output
    itself, so check both (mirroring
    ``vllm_omni.metrics.utils.first_multimodal_output``).

    Stock vLLM-Omni 0.26 surfaces EarTTS audio codes here. The registered
    Nemotron pipeline also uses the final multimodal engine-output route so
    optional ASR/function tensors accompany its text ``RequestOutput``.
    """
    mm = getattr(stage_output, "multimodal_output", None)
    if mm:
        return mm
    outputs = getattr(req_out, "outputs", None) or ()
    for completion in outputs:
        nested = getattr(completion, "multimodal_output", None)
        if nested:
            return nested
    return {}


def _audio_codes(mm: Any) -> Any:
    """Return this step's EarTTS acoustic codes from a multimodal payload.

    ``EarTTSForCausalLM.make_omni_output`` publishes them under
    ``model_outputs``, which vLLM-Omni's output processor remaps to the
    drainable ``audio`` modality key: in DELTA mode that key is emptied after
    every step, so each payload carries only the frames computed this step.
    Keys other than the modality's own are retained across steps and merged
    with :class:`TensorAccumulationStrategy` ``CONCAT_LAST`` for audio, which
    widens a ``T x num_quantizers`` frame instead of appending to it — so
    ``audio_codes`` is read last, only for a stage that is not client-facing.
    """
    if mm is None:
        return None
    for key in ("audio", "model_outputs", "audio_codes"):
        value = mm.get(key)
        if value is not None:
            return value
    return None


def _step_delta(value: Any, finished: bool, *, skip_finished: bool = True):
    """Mirror of ``_step_delta`` in the vllm-omni example: pull the
    new-this-step multimodal chunk from an :class:`OmniStageOutput`'s
    ``multimodal_output`` value (which may be a tensor, a list of tensors,
    ``None``, or absent).

    ``skip_finished`` drops a terminal duplicate. The split streaming
    requests use one-token segments, so callers pass ``False`` and separately
    skip each request's prefill output.
    """
    if finished and skip_finished:
        return None
    if isinstance(value, torch.Tensor):
        return value if value.numel() > 0 else None
    if isinstance(value, list) and value:
        last = value[-1]
        return last if isinstance(last, torch.Tensor) and last.numel() > 0 else None
    return None


def _step_tokens(stage_output: Any) -> StepTokens:
    """Extract text and optional auxiliary tokens from one Nemotron output.

    The text token remains on the stock vLLM ``RequestOutput`` even when the
    stage uses the multimodal engine-output path. Auxiliary tensors may be
    lifted onto ``OmniStageOutput`` or remain nested on its completion.
    """
    req_out = stage_output.request_output
    mm = _multimodal_output(stage_output, req_out)
    finished = bool(getattr(req_out, "finished", False))
    if req_out and req_out.outputs and req_out.outputs[0].token_ids:
        text_tok = int(req_out.outputs[0].token_ids[-1])
    else:
        text_tok = 0

    def last_token(key: str) -> int | None:
        delta = _step_delta(mm.get(key), finished, skip_finished=False)
        return int(delta[-1].item()) if delta is not None else None

    return StepTokens(
        text=text_tok,
        asr=last_token("asr_tokens"),
        function=last_token("function_tokens"),
    )
