# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This file implemented unit tests for loading all pretrained AlignerModel NGC checkpoints and generating Mel-spectrograms.
The test duration breakdowns are shown below. In general, each test for a single model is ~24 seconds on an NVIDIA RTX A6000.
"""
import pytest
import torch

from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.models import AlignerModel
from nemo.collections.tts.parts.utils.helpers import binarize_attention

available_models = [model.pretrained_model_name for model in AlignerModel.list_available_models()]


@pytest.fixture(params=available_models, ids=available_models)
def pretrained_model(request, get_language_id_from_pretrained_model_name):
    model_name = request.param
    language_id = get_language_id_from_pretrained_model_name(model_name)
    model = AlignerModel.from_pretrained(model_name=model_name)
    return model, language_id


@pytest.mark.nightly
def test_inference(pretrained_model, audio_text_pair_example_english):
    model, _ = pretrained_model
    audio, audio_len, text_raw = audio_text_pair_example_english

    # Generate mel-spectrogram
    spec, spec_len = model.preprocessor(input_signal=audio, length=audio_len)

    # Process text
    text_normalized = model.normalizer.normalize(text_raw, punct_post_process=True)
    text_tokens = model.tokenizer(text_normalized)
    text = torch.tensor(text_tokens, device=spec.device).unsqueeze(0).long()
    text_len = torch.tensor(len(text_tokens), device=spec.device).unsqueeze(0).long()

    # Run the Aligner
    _, _ = model(spec=spec, spec_len=spec_len, text=text, text_len=text_len)


@pytest.mark.unit
def test_metrics_applies_bin_loss_scale():
    """``on_train_epoch_start`` ramps ``self.bin_loss_scale`` from 0 to 1 across
    ``bin_loss_warmup_epochs`` (mirroring FastPitch's ``bin_loss_weight``, see
    ``fastpitch.py``'s ``bin_loss = self.bin_loss_fn(...) * bin_loss_weight``). ``_metrics``
    must multiply the bin loss by that scale before adding it to the total loss, otherwise the
    warmup config knob has no effect and bin_loss is applied at full weight starting the very
    epoch it switches on.
    """
    torch.manual_seed(0)
    batch, num_speakers, spec_len_val, text_len_val = 1, 1, 6, 4
    attn_soft = torch.softmax(torch.randn(batch, num_speakers, spec_len_val, text_len_val), dim=-1)
    attn_logprob = torch.log_softmax(torch.randn(batch, num_speakers, spec_len_val, text_len_val), dim=-1)
    spec_len = torch.tensor([spec_len_val])
    text_len = torch.tensor([text_len_val])

    class _Stub:
        pass

    def run(bin_loss_scale):
        self = _Stub()
        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.add_bin_loss = True
        self.bin_loss_scale = bin_loss_scale
        return AlignerModel._metrics(self, attn_soft, attn_logprob, spec_len, text_len)

    # Ground truth: the raw (unscaled) bin_loss magnitude, computed independently of _metrics.
    attn_hard = binarize_attention(attn_soft, text_len, spec_len)
    raw_bin_loss = BinLoss()(hard_attention=attn_hard, soft_attention=attn_soft)

    loss_start, fsl_start, bl_start, _ = run(bin_loss_scale=0.0)
    loss_full, fsl_full, bl_full, _ = run(bin_loss_scale=1.0)

    # At the very start of the warmup (scale == 0), bin_loss must not move the total loss...
    torch.testing.assert_close(loss_start, fsl_start)
    # ...because it was actually scaled down to (near) zero, not coincidentally cancelled out.
    torch.testing.assert_close(bl_start, torch.zeros_like(bl_start))
    # Fully warmed up (scale == 1), bin_loss must contribute at its full, true magnitude.
    torch.testing.assert_close(bl_full, raw_bin_loss)
    torch.testing.assert_close(loss_full, fsl_full + raw_bin_loss)
