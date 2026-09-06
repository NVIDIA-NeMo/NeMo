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
import json
from types import SimpleNamespace

import pytest
import torch
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration
from safetensors.torch import save_file
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


def _speaker_model(model_path, embedding_dim=3):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_path = str(model_path)
    model.embedding_dim = embedding_dim
    model._combined_embeddings = torch.zeros(1, embedding_dim)
    model._speaker_embedding_buffers = {}
    model._speaker_context_registry = {}
    return model


def test_context_bundle_is_preloaded_as_nonpersistent_buffers(tmp_path):
    speaker_dir = tmp_path / "speaker_embeddings"
    speaker_dir.mkdir()
    registry = {
        "schema_version": 1,
        "contexts": {
            "eng": {"tensor_key": "speaker.eng", "context_text": "[EN]", "prompt_len": 67},
            "deu": {"tensor_key": "speaker.deu", "context_text": "[DE]", "prompt_len": 69},
        },
    }
    (speaker_dir / "contexts.json").write_text(json.dumps(registry))
    tensors = {"speaker.eng": torch.ones(2, 3), "speaker.deu": torch.full((4, 3), 2.0)}
    save_file(tensors, speaker_dir / "contexts.safetensors")
    model = _speaker_model(tmp_path)

    model._preload_known_speaker_embeddings()

    assert sorted(model._speaker_embedding_buffers) == ["deu", "eng"]
    assert model._speaker_context_registry == registry["contexts"]
    torch.testing.assert_close(
        model._load_known_speaker_embedding("eng", torch.device("cpu"), torch.float32), tensors["speaker.eng"]
    )
    assert not any(name.startswith("_speaker_context_") for name in model.state_dict())


def test_context_registry_drives_prompt_length_and_context_text(tmp_path):
    speaker_dir = tmp_path / "speaker_embeddings"
    speaker_dir.mkdir()
    registry = {
        "schema_version": 1,
        "contexts": {"eng": {"tensor_key": "speaker.eng", "context_text": "[EN]", "prompt_len": 67}},
    }
    (speaker_dir / "contexts.json").write_text(json.dumps(registry))

    assert EasyMagpieTTSForConditionalGeneration.get_prompt_len("eng", str(tmp_path), tokenize=list) == 67
    with pytest.raises(FileNotFoundError, match="speaker_id 'deu'"):
        EasyMagpieTTSForConditionalGeneration.get_prompt_len("deu", str(tmp_path), tokenize=list)

    model = _speaker_model(tmp_path)
    model._speaker_context_registry = registry["contexts"]
    with pytest.raises(ValueError, match=r"requires context_text '\[EN\]'"):
        model._build_prefill_embeds(torch.device("cpu"), {"speaker_id": "eng", "context_text": "[DE]"})


def test_load_weights_uses_vllm_026_auto_loader(monkeypatch):
    loaded_weights = []

    class FakeAutoWeightsLoader:
        def __init__(self, module):
            assert isinstance(module, nn.Module)

        def load_weights(self, weights):
            loaded_weights.extend(weights)
            return {"layer.weight"}

    monkeypatch.setattr("vllm.model_executor.models.utils.AutoWeightsLoader", FakeAutoWeightsLoader)
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.backbone = nn.Module()
    model.code_predictor = SimpleNamespace(init_forbidden_mask=lambda: None)

    assert model.load_weights([]) == {"backbone.layer.weight"}
    assert loaded_weights == []


def test_phoneme_eos_is_fed_once_then_masked():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model.arch = SimpleNamespace(phoneme_stacking_factor=1, audio_bos_id=1025)
    model.has_phoneme = True
    model.phonemes_delay = 0
    model.phoneme_bos_id = 10
    model.phoneme_eos_id = 11
    model.speech_delay = 99
    model.embedding_dim = 3
    model.num_codebooks = 2
    model._combined_embeddings = torch.zeros(2, 3)
    model._dec_text_tokens = torch.zeros(2, dtype=torch.long)
    model._dec_text_mask = torch.zeros(2, dtype=torch.long)
    model._dec_phoneme_tokens = torch.zeros(2, 1, dtype=torch.long)
    model._dec_phoneme_valid = torch.zeros(2, dtype=torch.long)
    model._dec_audio_codes = torch.zeros(2, 2, dtype=torch.long)
    model._dec_audio_valid = torch.zeros(2, dtype=torch.long)
    input_ids = torch.zeros(1, dtype=torch.long)

    _, _, update = model._preprocess_decode(
        input_ids,
        0,
        input_ids.device,
        {"decode_offset": 1, "last_phoneme_token": torch.tensor([[model.phoneme_eos_id]])},
    )

    assert model._dec_phoneme_valid[0].item() == 1
    assert update["phoneme_ended"].item() is True

    model._preprocess_decode(
        input_ids,
        1,
        input_ids.device,
        {
            "decode_offset": 2,
            "last_phoneme_token": torch.tensor([[3]]),
            "phoneme_ended": update["phoneme_ended"],
        },
    )

    assert model._dec_phoneme_valid[1].item() == 0
    assert "phoneme_ended" in model.gpu_resident_buffer_keys


def test_two_stage_output_copies_codes_once_and_uses_async_output():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    nn.Module.__init__(model)
    model._single_stage_audio = False
    model._out_codes = torch.tensor([[1, 2], [3, 4]])
    hidden = torch.zeros(2, 3)

    output = model.make_omni_output(hidden)

    assert set(output.multimodal_outputs) == {"codes"}
    torch.testing.assert_close(output.multimodal_outputs["codes"]["audio"], model._out_codes)
    assert model.use_async_omni_output
    assert model.eager_omni_postprocess_before_async_output
