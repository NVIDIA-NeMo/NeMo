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

"""CPU-only tests for vLLM-Omni EarTTS classifier-free guidance.

Guidance arithmetic and MaskGIT trajectory sharing are numerical properties
that an end-to-end run cannot localise, so they are checked directly against
the real model definition. Requires vllm-omni to be installed; skipped
otherwise rather than run against a stubbed vLLM, which would only test the
stubs.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.fixture(scope="module")
def eartts():
    pytest.importorskip("vllm_omni")
    from nemo.collections.speechlm2.inference.vllm_omni.eartts import eartts as eartts_module

    return eartts_module


def test_unconditional_embedding_replaces_only_text_branch(eartts):
    config = SimpleNamespace(
        hidden_size=2,
        emb_vocab_size=4,
        codebook_size=3,
        latent_size=2,
        num_quantizers=2,
        use_gated_fusion_for_text_audio=False,
        use_audio_prompt_frozen_projection=False,
    )
    embedding = eartts.EarTTSInputEmbedding(config)
    with torch.no_grad():
        for rvq in embedding.rvq_embs:
            rvq.weight.zero_()
        embedding.rvq_embs[0].weight[0] = torch.tensor([1.0, 0.0])
        embedding.rvq_embs[1].weight[0] = torch.tensor([0.0, 2.0])
        embedding.embed_code.weight.copy_(torch.eye(2))
        embedding.embed_subword.embed_subwords.weight.zero_()
        embedding.embed_subword.embed_subwords.weight[1] = torch.tensor([3.0, 4.0])
        embedding.bos_emb.zero_()
        embedding.null_emb.copy_(torch.tensor([9.0, 10.0]))

    kwargs = dict(
        acoustic_tokens=torch.zeros(2, 2, dtype=torch.long),
        text_tokens=torch.ones(2, dtype=torch.long),
        text_mask=torch.ones(2, dtype=torch.long),
        bos_mask=torch.zeros(2, dtype=torch.long),
        speaker_latent=torch.zeros(2, 2),
    )
    original = embedding(**kwargs)
    explicit_no_cfg = embedding(
        **kwargs,
        cfg_is_uncond=torch.zeros(2, dtype=torch.bool),
    )
    guided_input = embedding(
        **kwargs,
        cfg_is_uncond=torch.tensor([False, True]),
    )

    torch.testing.assert_close(original, explicit_no_cfg, rtol=0, atol=0)
    torch.testing.assert_close(guided_input[0], torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(guided_input[1], torch.tensor([10.0, 12.0]))
    torch.testing.assert_close(
        guided_input[1] - embedding.null_emb,
        torch.tensor([1.0, 2.0]),
    )
    assert dict(embedding.named_parameters())["null_emb"] is embedding.null_emb


def test_cfg_rows_are_ordered_and_guided_after_mlp(eartts):
    hidden = torch.tensor([[5.0], [10.0], [3.0], [20.0]])
    enabled = torch.ones(4, dtype=torch.bool)
    is_uncond = torch.tensor([True, False, True, False])
    pair_id = torch.tensor([20, 10, 10, 20])
    scale = torch.tensor([1.5, 2.0, 2.0, 1.5])
    valid = torch.ones(4, dtype=torch.bool)

    (
        ordered,
        ordered_is_uncond,
        ordered_scale,
        active,
        partner,
        conditional_rep,
        inverse,
    ) = eartts._prepare_cfg_sampling_batch(
        hidden,
        enabled,
        is_uncond,
        pair_id,
        scale,
        valid,
    )
    torch.testing.assert_close(
        ordered,
        torch.tensor([[10.0], [20.0], [3.0], [5.0]]),
    )
    assert ordered_is_uncond.tolist() == [False, False, True, True]
    assert active.tolist() == [True, True, True, True]
    assert partner.tolist() == [2, 3, 0, 1]
    assert conditional_rep.tolist() == [0, 1, 0, 1]

    guided = eartts._apply_cfg_after_mlp(
        ordered,
        ordered_is_uncond,
        ordered_scale,
        active,
        partner,
    )
    torch.testing.assert_close(
        guided,
        torch.tensor([[24.0], [42.5], [24.0], [42.5]]),
    )
    assert torch.equal(ordered[inverse], hidden)

    # An incomplete or disabled batch must fall straight through: no
    # reordering, nothing active, and guidance a no-op.
    plain = torch.randn(3, 4)
    ordered, roles, scales, active, partner, _, inverse = eartts._prepare_cfg_sampling_batch(
        plain,
        cfg_enabled=torch.tensor([True, True, False]),
        cfg_is_uncond=torch.tensor([False, True, False]),
        cfg_pair_id=torch.tensor([7, 7, -1]),
        cfg_scale=torch.ones(3),
        valid=torch.ones(3, dtype=torch.bool),
    )
    assert torch.equal(ordered, plain)
    assert not active.any()
    assert torch.equal(inverse, torch.arange(3))
    assert torch.equal(eartts._apply_cfg_after_mlp(ordered, roles, scales, active, partner), plain)


def test_maskgit_shares_one_code_trajectory_per_pair(eartts):
    config = SimpleNamespace(
        num_quantizers=2,
        codebook_size=8,
        noise_scale=0.7,
        num_iter=2,
        exponent=3.0,
        latent_size=4,
        hidden_size=4,
        intermediate_size=8,
        mog_num_layers=0,
        mog_num_predictions=4,
        mog_low_rank=None,
        top_p_or_k=None,
        mog_min_log_std=-4.0,
        mog_eps=1e-6,
    )
    sampler = eartts.MaskGITSampler(config)
    torch.manual_seed(3)
    with torch.no_grad():
        for parameter in sampler.parameters():
            parameter.normal_(mean=0.0, std=0.2)

    hidden = torch.randn(4, 4)
    enabled = torch.ones(4, dtype=torch.bool)
    roles = torch.tensor([True, False, True, False])
    pairs = torch.tensor([2, 1, 1, 2])
    scales = torch.full((4,), 1.25)
    valid = torch.ones(4, dtype=torch.bool)
    torch.manual_seed(11)
    codes = sampler(hidden, enabled, roles, pairs, scales, valid)

    assert torch.equal(codes[0], codes[3])
    assert torch.equal(codes[1], codes[2])

    no_cfg_hidden = torch.randn(3, 4)
    torch.manual_seed(17)
    implicit_no_cfg = sampler(no_cfg_hidden)
    torch.manual_seed(17)
    explicit_no_cfg = sampler(
        no_cfg_hidden,
        cfg_enabled=torch.zeros(3, dtype=torch.bool),
        cfg_is_uncond=torch.zeros(3, dtype=torch.bool),
        cfg_pair_id=torch.full((3,), -1, dtype=torch.long),
        cfg_scale=torch.zeros(3),
        valid=torch.ones(3, dtype=torch.bool),
    )
    assert torch.equal(implicit_no_cfg, explicit_no_cfg)


def test_client_facing_stage_emits_the_drainable_audio_key(eartts):
    """A final AR audio stage must publish codes under ``model_outputs``.

    vLLM-Omni remaps that key onto the drainable ``audio`` modality, so DELTA
    streaming empties it every step. Any other key is retained across steps and
    merged with ``CONCAT_LAST``, which widens a ``T x num_quantizers`` frame
    instead of appending frames to it.
    """
    hidden = torch.zeros(1, 4)
    codes = torch.tensor([[3, 5]], dtype=torch.long)

    for single_stage_audio, expected_key in ((True, "model_outputs"), (False, "audio_codes")):
        model = object.__new__(eartts.EarTTSForCausalLM)
        nn.Module.__init__(model)
        model._single_stage_audio = single_stage_audio
        model._out_codes = codes.clone()

        output = model.make_omni_output(hidden)

        assert list(output.multimodal_outputs) == [expected_key]
        torch.testing.assert_close(output.multimodal_outputs[expected_key], codes)

        stashed = model.postprocess(hidden, output.multimodal_outputs)
        torch.testing.assert_close(stashed["last_acoustic_codes"], codes)

    # Per-request CFG metadata lands in model-owned buffers whose addresses
    # must stay stable, because CUDA graphs capture them.
    model = object.__new__(eartts.EarTTSForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(guidance_scale=0.5)
    model._cfg_enabled = torch.zeros(8, dtype=torch.bool)
    model._cfg_is_uncond = torch.zeros(8, dtype=torch.bool)
    model._cfg_pair_id = torch.full((8,), -1, dtype=torch.long)
    model._cfg_scale = torch.zeros(8)
    addresses = tuple(
        value.data_ptr()
        for value in (
            model._cfg_enabled,
            model._cfg_is_uncond,
            model._cfg_pair_id,
            model._cfg_scale,
        )
    )

    model._write_cfg_state(
        start=1,
        span_len=2,
        info_dict={
            "cfg_enabled": True,
            "cfg_role": "cond",
            "cfg_pair_id": "request-7",
            "cfg_scale": 1.75,
        },
    )
    model._write_cfg_state(
        start=3,
        span_len=1,
        info_dict={
            "cfg_enabled": torch.tensor(True),
            "cfg_role": ["uncond"],
            "cfg_pair_id": "request-7",
            "cfg_scale": torch.tensor(1.75),
        },
    )

    assert model._cfg_enabled[1:4].all()
    assert model._cfg_is_uncond[1:4].tolist() == [False, False, True]
    assert model._cfg_pair_id[1] == model._cfg_pair_id[3]
    torch.testing.assert_close(model._cfg_scale[1:4], torch.full((3,), 1.75))
    assert addresses == tuple(
        value.data_ptr()
        for value in (
            model._cfg_enabled,
            model._cfg_is_uncond,
            model._cfg_pair_id,
            model._cfg_scale,
        )
    )

    with pytest.raises(AssertionError, match="cfg_role"):
        model._write_cfg_state(
            start=4,
            span_len=1,
            info_dict={
                "cfg_enabled": True,
                "cfg_role": "conditional",
                "cfg_pair_id": 4,
            },
        )
