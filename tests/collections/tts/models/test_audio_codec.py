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


import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.tts.models import AudioCodecModel


def create_codec_config():
    audio_encoder = {
        'cls': 'nemo.collections.tts.modules.audio_codec_modules.MultiResolutionSTFTEncoder',
        'params': {
            'out_dim': 40,
            'resolutions': [[960, 240, 960], [1920, 480, 1920]],
            'resolution_filter_list': [256, 512],
        },
    }
    audio_decoder = {
        'cls': 'nemo.collections.tts.modules.audio_codec_modules.ResNetDecoder',
        'params': {
            'input_dim': 40,
            'input_filters': 512,
            'n_hidden_layers': 6,
            'hidden_filters': 512,
            'pre_up_sample_rates': [],
            'pre_up_sample_filters': [],
            'resblock_up_sample_rates': [10, 8, 6],
            'resblock_up_sample_filters': [256, 128, 32],
        },
    }
    vector_quantizer = {
        'cls': 'nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer',
        'params': {
            'num_groups': 8,
            'num_levels_per_group': [4, 4, 4, 4, 4],
        },
    }
    generator_loss = {
        'cls': 'nemo.collections.tts.losses.audio_codec_loss.GeneratorSquaredLoss',
    }
    discriminator_loss = {
        'cls': 'nemo.collections.tts.losses.audio_codec_loss.DiscriminatorSquaredLoss',
    }

    model_cfg = DictConfig(
        {
            'sample_rate': 24000,
            'samples_per_frame': 480,
            'loss_resolutions': [[960, 240, 960], [1920, 480, 1920]],
            'mel_loss_dims': [160, 320],
            'commit_loss_scale': 0.0,
            'audio_encoder': DictConfig(audio_encoder),
            'audio_decoder': DictConfig(audio_decoder),
            'vector_quantizer': DictConfig(vector_quantizer),
            'generator_loss': DictConfig(generator_loss),
            'discriminator_loss': DictConfig(discriminator_loss),
        }
    )
    return model_cfg


@pytest.fixture()
def codec_model():
    model_cfg = create_codec_config()
    codec_model = AudioCodecModel(cfg=model_cfg)
    return codec_model


@pytest.fixture()
def acoustic_codec_model():
    semantic_model_cfg = create_codec_config()
    semantic_model_cfg.vector_quantizer.params.num_groups = 1
    semantic_model_cfg.audio_encoder.params.out_dim = 5
    semantic_model_cfg.audio_decoder.params.input_dim = 5

    acoustic_model_cfg = create_codec_config()
    acoustic_model_cfg.semantic_codec = semantic_model_cfg
    acoustic_model_cfg.audio_encoder.params.out_dim = 35
    acoustic_codec_model = AudioCodecModel(cfg=acoustic_model_cfg)

    return acoustic_codec_model


def create_hybrid_codec_model(vae_std=None, residual_dropout_rate=1.0, mean_loss_scale=0.0):
    semantic_model_cfg = create_codec_config()
    semantic_model_cfg.vector_quantizer.params.num_groups = 1
    semantic_model_cfg.audio_encoder.params.out_dim = 5
    semantic_model_cfg.audio_decoder.params.input_dim = 5

    hybrid_model_cfg = create_codec_config()
    hybrid_model_cfg.semantic_codec = semantic_model_cfg
    hybrid_model_cfg.audio_encoder.params.out_dim = 35
    hybrid_model_cfg.hybrid_codec = {
        'continuous_dim': 35,
        'residual_dropout_rate': residual_dropout_rate,
        'kl_loss_scale': 0.1,
        'mean_loss_scale': mean_loss_scale,
    }
    if vae_std is not None:
        hybrid_model_cfg.hybrid_codec.vae_std = vae_std
    hybrid_model_cfg.optim = {"_target_": "torch.optim.Adam", "lr": 1e-3}
    return AudioCodecModel(cfg=hybrid_model_cfg)


@pytest.fixture()
def hybrid_codec_model():
    return create_hybrid_codec_model()


class TestAudioCodecModel:
    @pytest.mark.unit
    def test_forward(self, codec_model):
        batch_size = 2
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)
        output_audio, output_audio_len = codec_model.forward(
            audio=audio, audio_len=audio_len, sample_rate=codec_model.sample_rate
        )
        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()

    @pytest.mark.unit
    def test_forward_with_acoustic_codec(self, acoustic_codec_model):
        batch_size = 3
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)
        output_audio, output_audio_len = acoustic_codec_model.forward(
            audio=audio, audio_len=audio_len, sample_rate=acoustic_codec_model.sample_rate
        )
        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()

    @pytest.mark.unit
    def test_encode_and_decode(self, codec_model):
        batch_size = 4
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)

        tokens, tokens_len = codec_model.encode(audio=audio, audio_len=audio_len, sample_rate=codec_model.sample_rate)
        assert tokens.shape[0] == batch_size
        assert tokens.shape[2] == tokens_len.max()

        output_audio, output_audio_len = codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()

    @pytest.mark.unit
    def test_encode_and_decode_with_acoustic_codec(self, acoustic_codec_model):
        batch_size = 5
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)

        tokens, tokens_len = acoustic_codec_model.encode(
            audio=audio, audio_len=audio_len, sample_rate=acoustic_codec_model.sample_rate
        )
        assert tokens.shape[0] == batch_size
        assert tokens.shape[2] == tokens_len.max()

        output_audio, output_audio_len = acoustic_codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()

    @pytest.mark.unit
    def test_hybrid_codec_forward(self, hybrid_codec_model):
        batch_size = 2
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)

        output_audio, output_audio_len = hybrid_codec_model.forward(
            audio=audio, audio_len=audio_len, sample_rate=hybrid_codec_model.sample_rate
        )

        assert output_audio.shape[0] == batch_size
        assert output_audio.shape[1] == output_audio_len.max()

    @pytest.mark.unit
    def test_hybrid_codec_residual_dropout(self, hybrid_codec_model):
        batch_size = 3
        audio = torch.randn(size=(batch_size, 20000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=20000)

        hybrid_codec_model.train()
        hybrid = hybrid_codec_model._encode_hybrid(
            audio=audio,
            audio_len=audio_len,
            sample_rate=hybrid_codec_model.sample_rate,
        )

        assert hybrid.decoder_inputs.shape[0] == batch_size
        assert hybrid.decoder_inputs.shape[1] == 40
        assert hybrid.residual_mu.shape[1] == 35
        assert not hybrid.residual_enabled.any()
        assert torch.allclose(hybrid.decoder_inputs, hybrid.semantic_embedding)
        assert hybrid.kl_loss.ndim == 0
        assert torch.isfinite(hybrid.kl_loss)
        assert torch.equal(hybrid.mean_loss, torch.zeros_like(hybrid.mean_loss))
        assert hybrid_codec_model.residual_logvar is not None
        assert hybrid_codec_model.vector_quantizer is None
        assert hybrid_codec_model.num_codebooks == 1
        assert not any(parameter.requires_grad for parameter in hybrid_codec_model.semantic_codec.parameters())

        optimizer = hybrid_codec_model.configure_optimizers()
        optimizer_params = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        hybrid_modules = (
            hybrid_codec_model.semantic_to_decoder,
            hybrid_codec_model.residual_mu,
            hybrid_codec_model.residual_logvar,
            hybrid_codec_model.residual_to_decoder,
        )
        assert all(id(parameter) in optimizer_params for module in hybrid_modules for parameter in module.parameters())
        assert all(
            id(parameter) not in optimizer_params for parameter in hybrid_codec_model.semantic_codec.parameters()
        )

    @pytest.mark.unit
    def test_hybrid_codec_fixed_vae_std(self, monkeypatch):
        vae_std = 0.625
        hybrid_codec_model = create_hybrid_codec_model(vae_std=vae_std, residual_dropout_rate=0.0, mean_loss_scale=0.1)
        audio = torch.randn(size=(2, 12000))
        audio_len = torch.tensor([12000, 10000])
        monkeypatch.setattr(torch, "randn_like", torch.ones_like)
        monkeypatch.setattr(
            torch,
            "randn",
            lambda *args, **kwargs: torch.ones(*args, device=kwargs.get("device"), dtype=kwargs.get("dtype")),
        )

        hybrid_codec_model.train()
        hybrid = hybrid_codec_model._encode_hybrid(
            audio=audio,
            audio_len=audio_len,
            sample_rate=hybrid_codec_model.sample_rate,
        )

        assert torch.allclose(hybrid.residual_logvar.exp(), torch.full_like(hybrid.residual_logvar, vae_std**2))
        assert torch.equal(hybrid.kl_loss, torch.zeros_like(hybrid.kl_loss))
        frame_index = torch.arange(hybrid.residual_mu.shape[-1])
        valid = (frame_index.unsqueeze(0) < hybrid.encoded_len.unsqueeze(1)).unsqueeze(1)
        expected_mean_loss = (hybrid.residual_mu.square() * valid).sum() / (valid.sum() * hybrid.residual_mu.shape[1])
        assert torch.allclose(hybrid.mean_loss, expected_mean_loss)
        assert hybrid_codec_model.residual_logvar is None
        assert not any(key.startswith("residual_logvar.") for key in hybrid_codec_model.state_dict())
        assert torch.allclose(hybrid.decoder_inputs[0, 5:], hybrid.residual_mu[0] + vae_std)
        assert torch.allclose(hybrid.decoder_inputs[1, 5:], hybrid.residual_mu[1] + vae_std)

    @pytest.mark.unit
    def test_hybrid_codec_mean_loss_is_opt_in_and_fixed_variance_only(self):
        assert create_hybrid_codec_model(vae_std=0.625).mean_loss_scale == 0.0
        with pytest.raises(ValueError, match="only supported when fixed"):
            create_hybrid_codec_model(mean_loss_scale=0.1)

    @pytest.mark.unit
    def test_hybrid_codec_encode_and_decode(self, hybrid_codec_model):
        batch_size = 2
        audio = torch.randn(size=(batch_size, 12000))
        audio_len = torch.randint(size=[batch_size], low=10000, high=12000)

        hybrid_codec_model.eval()
        tokens, residual_mu, residual_logvar, tokens_len = hybrid_codec_model.encode_hybrid(
            audio=audio,
            audio_len=audio_len,
            sample_rate=hybrid_codec_model.sample_rate,
        )

        assert tokens.shape[1] == 1
        assert residual_mu.shape == residual_logvar.shape
        assert residual_mu.shape[1] == 35
        assert residual_mu.shape[2] == tokens.shape[2]

        code_only_audio, code_only_len = hybrid_codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        hybrid_audio, hybrid_len = hybrid_codec_model.decode_hybrid(
            tokens=tokens,
            tokens_len=tokens_len,
            residual=residual_mu,
        )

        assert code_only_audio.shape[0] == batch_size
        assert hybrid_audio.shape[0] == batch_size
        assert torch.equal(code_only_len, hybrid_len)

    @pytest.mark.unit
    def test_lhotse_validation_dataloader(self, codec_model, monkeypatch):
        validation_loader = object()
        cfg = DictConfig({'dataloader_params': {'use_lhotse': True}})
        monkeypatch.setattr(codec_model, '_get_lhotse_dataloader', lambda _: validation_loader)

        codec_model.setup_validation_data(cfg)

        assert codec_model._validation_dl is validation_loader
