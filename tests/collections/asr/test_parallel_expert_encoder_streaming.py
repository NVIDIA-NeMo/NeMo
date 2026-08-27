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

"""Tests for the cache-aware streaming variant of the parallel expert encoder.

The load-bearing claim is that the subclass adds **no behaviour**: its streaming step is its ASR
branch's step plus fusion, bit for bit, and its offline forward is the base class's.
"""

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder, StreamingParallelExpertEncoder
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from tests.collections.asr.test_parallel_expert_encoder import (
    _MEL_FEATURES,
    _N_SPK,
    _SUBSAMPLING_FACTOR,
    toy_asr_encoder_cfg,
    toy_diarization_model_cfg,
)


@pytest.fixture(autouse=True)
def _cpu_default_device():
    """Pin this module's default device to CPU, and restore it afterwards.

    Several ``tests/collections/speechlm2`` modules call ``torch.set_default_device('cuda')`` at
    import time and never restore it, so in a full-suite run this file would otherwise inherit a
    CUDA default and mix devices.
    """
    previous = torch.get_default_device()
    torch.set_default_device('cpu')
    yield
    torch.set_default_device(previous)


def streaming_asr_encoder_cfg() -> DictConfig:
    """Tiny *cache-aware* ConformerEncoder config, so the streaming interface is exercisable."""
    cfg = toy_asr_encoder_cfg()
    cfg.att_context_size = [8, 1]
    cfg.att_context_style = 'chunked_limited'
    cfg.causal_downsampling = True
    cfg.conv_context_size = 'causal'
    return cfg


def build_toy_streaming_pe_encoder(**overrides):
    """A real StreamingParallelExpertEncoder over the tiny cache-aware ASR + diar configs."""
    kwargs = dict(
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type=None,
        online_inference_length=500,
    )
    kwargs.update(overrides)
    return StreamingParallelExpertEncoder(**kwargs)


@pytest.mark.unit
def test_only_the_streaming_subclass_advertises_the_streaming_interface():
    """The base class must NOT claim streaming capability -- that is the point of the split."""
    base = ParallelExpertEncoder(
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type=None,
    )
    assert not isinstance(base, StreamingEncoder)
    assert not hasattr(base, "cache_aware_stream_step")

    enc = build_toy_streaming_pe_encoder()
    assert isinstance(enc, StreamingEncoder)
    # ...while still satisfying the `isinstance(..., ParallelExpertEncoder)` gate in salm_automodel.
    assert isinstance(enc, ParallelExpertEncoder)
    enc.setup_streaming_params()
    # streaming_cfg is the ASR branch's, not a copy that can drift out of sync.
    assert enc.streaming_cfg is enc.asr_encoder.streaming_cfg


@pytest.mark.unit
def test_streaming_step_delegates_exactly_to_asr_branch():
    """PE's cache-aware step must be its ASR branch's step plus fusion -- nothing else. Any extra
    transform of the signal (e.g. a stray re-normalization) shows up here as a nonzero diff."""
    enc = build_toy_streaming_pe_encoder(speaker_activity_threshold=0.5).eval()
    enc._fuse_diar_and_asr = lambda asr_encoded, spk_targets: asr_encoded
    bare = enc.asr_encoder
    enc.setup_streaming_params()

    torch.manual_seed(0)
    mel = torch.randn(1, _MEL_FEATURES, 512)
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    shift = enc.streaming_cfg.shift_size
    shift = shift[1] if isinstance(shift, (list, tuple)) else shift

    pe_state = list(enc.get_initial_cache_state(batch_size=1, dtype=mel.dtype, device=mel.device))
    bare_state = list(bare.get_initial_cache_state(batch_size=1, dtype=mel.dtype, device=mel.device))
    assert all(torch.equal(a, b) for a, b in zip(pe_state, bare_state))

    n_steps = 0
    with torch.no_grad():
        for step in range(3):
            chunk = mel[:, :, step * shift : step * shift + chunk_size]
            if chunk.shape[-1] < chunk_size:
                break
            kwargs = dict(
                processed_signal=chunk,
                processed_signal_length=torch.tensor([chunk.shape[-1]]),
                keep_all_outputs=False,
                drop_extra_pre_encoded=0 if step == 0 else enc.streaming_cfg.drop_extra_pre_encoded,
            )
            pe_out = enc.cache_aware_stream_step(
                cache_last_channel=pe_state[0],
                cache_last_time=pe_state[1],
                cache_last_channel_len=pe_state[2],
                spk_targets=torch.zeros(1, chunk_size // _SUBSAMPLING_FACTOR, _N_SPK),
                **kwargs,
            )
            bare_out = bare.cache_aware_stream_step(
                cache_last_channel=bare_state[0],
                cache_last_time=bare_state[1],
                cache_last_channel_len=bare_state[2],
                **kwargs,
            )
            assert torch.equal(pe_out[0], bare_out[0]), f"encoder output diverged at step {step}"
            for pe_cache, bare_cache in zip(pe_out[2:4], bare_out[2:4]):
                assert torch.equal(pe_cache, bare_cache), f"cache diverged at step {step}"
            pe_state, bare_state = list(pe_out[2:]), list(bare_out[2:])
            n_steps += 1
    assert n_steps > 0, "test exercised no streaming steps"


@pytest.mark.unit
def test_offline_forward_is_inherited_unchanged():
    """Mounting the streaming subclass must not perturb the offline path."""
    torch.manual_seed(0)
    common = dict(
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        diarization_model_cfg=toy_diarization_model_cfg(),
        asr_normalize_type=None,
        online_inference_length=500,
    )
    torch.manual_seed(0)
    base = ParallelExpertEncoder(**common).eval()
    torch.manual_seed(0)
    streaming = StreamingParallelExpertEncoder(**common).eval()
    streaming.load_state_dict(base.state_dict())

    mel = torch.randn(1, _MEL_FEATURES, 128)
    length = torch.tensor([mel.shape[-1]])
    spk_targets = torch.rand(1, mel.shape[-1] // _SUBSAMPLING_FACTOR, _N_SPK)
    with torch.no_grad():
        base_out, base_len = base(audio_signal=mel, length=length, spk_targets=spk_targets.clone())
        strm_out, strm_len = streaming(audio_signal=mel, length=length, spk_targets=spk_targets.clone())
    assert torch.equal(base_out, strm_out)
    assert torch.equal(base_len, strm_len)


@pytest.mark.unit
def test_diarizer_drops_the_same_pre_encode_frames_as_the_asr_branch():
    """Both branches must consume the same `drop_extra_pre_encoded`, or the fusion goes stale.

    The ASR branch drops N cache frames from its output; if the diarizer does not, it emits N+2
    frames per N ASR frames and `_align_diar_frames` keeps the OLDEST ones -- so the speaker signal
    ends up one chunk behind the ASR frames it is added to. `perception.forward` does not pass
    `drop_extra_pre_encoded`, so PE must fall back to the ASR branch's streaming config, not to 0.
    """
    enc = build_toy_streaming_pe_encoder().eval()
    enc.setup_streaming_params()
    state = enc.get_initial_cache_state(batch_size=1, dtype=torch.float32, device=torch.device("cpu"))
    cache_last_channel, cache_last_time, cache_last_channel_len = state

    seen = []

    def spy(processed_signal, processed_signal_length, align_target, drop_extra_pre_encoded):
        # Record the plumbing and stand in for the diarizer: the toy Sortformer cannot pre-encode
        # a chunk this small, and what is under test is which `drop` value it is handed.
        seen.append(drop_extra_pre_encoded)
        return torch.zeros(1, align_target, enc.n_spk)

    enc._stream_diarizer = spy
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    mel = torch.randn(1, _MEL_FEATURES, chunk_size)
    with torch.no_grad():
        # No `drop_extra_pre_encoded` argument -- exactly how perception.forward calls it.
        enc.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=torch.tensor([chunk_size]),
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
        )
    assert seen == [enc.streaming_cfg.drop_extra_pre_encoded], f"diarizer got drop={seen}, not the ASR branch's"


@pytest.mark.unit
def test_stream_step_without_initial_cache_state_raises_actionable_error():
    """`nn.Module.__getattr__` would raise AttributeError and mask the real instruction."""
    enc = build_toy_streaming_pe_encoder().eval()
    enc.setup_streaming_params()
    chunk_size = enc.streaming_cfg.chunk_size
    chunk_size = chunk_size[1] if isinstance(chunk_size, (list, tuple)) else chunk_size
    with pytest.raises(RuntimeError, match="get_initial_cache_state"):
        enc.cache_aware_stream_step(
            processed_signal=torch.randn(1, _MEL_FEATURES, chunk_size),
            processed_signal_length=torch.tensor([chunk_size]),
        )
