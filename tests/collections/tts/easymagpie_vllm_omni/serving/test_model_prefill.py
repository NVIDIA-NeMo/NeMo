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
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.codec.encoder import EasyMagpieCodecEncoderOutput
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from torch import nn


def test_text_prefill_embeddings_add_phoneme_bos_at_position_three():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(text_prefill_num=4, phoneme_stacking_factor=1)
    model.embedding_dim = 3
    model.has_phoneme = True
    model.phonemes_delay = 3
    model.phoneme_bos_id = 7
    model.text_embedding = nn.Embedding(32, 3)
    model.phoneme_embeddings = nn.ModuleList([nn.Embedding(16, 3)])

    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.phoneme_embeddings[0].weight.zero_()
        for index, token_id in enumerate((10, 11, 12, 13), start=1):
            model.text_embedding.weight[token_id] = torch.tensor([index, 0, 0])
        model.phoneme_embeddings[0].weight[7] = torch.tensor([0, 0, 10])

    rows = model._build_text_prefill_embeds(
        torch.device("cpu"),
        torch.float32,
        {"text_prefill_num": 4, "prefill_text_tokens": [10, 11, 12, 13]},
    )

    torch.testing.assert_close(
        rows,
        torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 10]], dtype=torch.float32),
    )


class _FakeReferenceCodecEncoder(nn.Module):
    def forward(self, audio, audio_lens, *, audio_frame_embedder=None):
        del audio_frame_embedder
        self.audio = audio.detach().clone()
        self.audio_lens = audio_lens.detach().clone()
        count = int(audio.shape[0])
        return EasyMagpieCodecEncoderOutput(
            acoustic_codes=torch.zeros(audio.shape[0], 4, 2, dtype=torch.long),
            acoustic_lens=torch.full((audio.shape[0],), 2, dtype=torch.long),
            reference_speaker_embeddings=torch.full((count, 4, 2), 9.0),
            reference_speaker_embedding_lens=torch.full((count,), 4, dtype=torch.long),
        )


def test_embed_multimodal_returns_reference_embeddings_when_codec_encoder_is_present():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        ensure_reference_audio_available=lambda: None,
    )
    model.model_path = "unused"
    model.codec_encoder = _FakeReferenceCodecEncoder()
    model.code_predictor = SimpleNamespace(embed_audio_frame=lambda codes: codes.float())
    model._combined_embeddings = torch.zeros(1, 2)

    outputs = model.embed_multimodal(
        audio_values=[torch.tensor([1.0, 2.0, 3.0, 4.0])],
        audio_lens=torch.tensor([4]),
    )

    assert [output.shape for output in outputs] == [(4, 2)]
    torch.testing.assert_close(outputs[0], torch.full((4, 2), 9.0))
    torch.testing.assert_close(
        model.codec_encoder.audio[0],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    torch.testing.assert_close(model.codec_encoder.audio_lens, torch.tensor([4]))


def test_embed_multimodal_rejects_reference_audio_when_codec_encoder_is_absent():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        ensure_reference_audio_available=lambda: (_ for _ in ()).throw(RuntimeError("no reference-audio encoder"))
    )
    model.codec_encoder = None

    with pytest.raises(RuntimeError, match="no reference-audio encoder"):
        model.embed_multimodal(audio_values=[torch.ones(4)], audio_lens=torch.tensor([4]))


def _raw_reference_prefill_model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model._maybe_set_lt_sampling_params = lambda info: None
    model._build_reference_audio_prefill_chunk = lambda **kwargs: torch.full((kwargs["span_len"], 2), 7.0)
    model._build_prefill_embeds = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("known-speaker path must not run")
    )
    return model


def test_prefill_infers_raw_reference_when_speaker_id_is_absent():
    model = _raw_reference_prefill_model()

    _, rows, _ = model._preprocess_prefill(
        input_ids=torch.tensor([1, 1]),
        input_embeds=torch.ones(2, 2),
        span_len=2,
        device=torch.device("cpu"),
        info_dict={
            "_omni_num_computed_tokens": 0,
            "_omni_prompt_len": 2,
        },
    )

    torch.testing.assert_close(rows, torch.full((2, 2), 7.0))


def test_reference_prefill_rejects_non_default_task_mode():
    model = _reference_prefill_model()

    with pytest.raises(ValueError, match="Only task_mode_id=0"):
        model._build_reference_audio_prefill_chunk(
            input_embeds=torch.ones(2, 3),
            info_dict={"task_mode_id": 1},
            offset=0,
            span_len=2,
            prompt_len=2,
        )


def _reference_prefill_model():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(
        text_prefill_num=4,
    )
    model.model_path = "unused"
    model.embedding_dim = 3
    model.has_phoneme = False
    model.task_embedding = None
    model.num_task_embeddings = 0
    model.text_embedding = nn.Embedding(32, 3)
    model._encode_context_text = lambda text, device: torch.tensor([20], device=device)
    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.text_embedding.weight[20] = torch.tensor([0, 8, 0])
        for index, token_id in enumerate((10, 11, 12, 13), start=1):
            model.text_embedding.weight[token_id] = torch.tensor([index, 0, 0])
    return model


def test_reference_audio_prefill_preserves_speaker_rows_and_is_chunk_invariant():
    model = _reference_prefill_model()
    info = {
        "context_text": "[EN]",
        "text_prefill_num": 4,
        "prefill_text_tokens": [10, 11, 12, 13],
    }
    input_embeds = torch.ones(7, 3)

    whole = model._build_reference_audio_prefill_chunk(
        input_embeds=input_embeds,
        info_dict=info,
        offset=0,
        span_len=7,
        prompt_len=7,
    )
    chunks = torch.cat(
        [
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[:4],
                info_dict=info,
                offset=0,
                span_len=4,
                prompt_len=7,
            ),
            model._build_reference_audio_prefill_chunk(
                input_embeds=input_embeds[4:],
                info_dict=info,
                offset=4,
                span_len=3,
                prompt_len=7,
            ),
        ]
    )

    torch.testing.assert_close(chunks, whole)
    torch.testing.assert_close(whole[:2], torch.ones(2, 3))
    torch.testing.assert_close(whole[2], torch.tensor([0, 8, 0], dtype=torch.float32))
    torch.testing.assert_close(
        whole[3:],
        torch.tensor([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=torch.float32),
    )
