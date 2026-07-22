# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Audio augmentation utilities for speech models."""

import glob
import os
import random
import subprocess
import tempfile
from typing import Optional, Tuple, List, Union
from torchaudio.functional import filtfilt
from nemo.collections.speechlm2.parts.add_background_noise import AddBackgroundNoise
try:
    from torch_audiomentations import (
        Compose,
        # AddBackgroundNoise,
        Gain,
        # BandPassFilter,
        LowPassFilter,
        PitchShift,
        # Shift,
        # AddColoredNoise,
        # PolarityInversion
    )
    _TORCH_AUDIOMENTATIONS_AVAILABLE = True
except ImportError:
    _TORCH_AUDIOMENTATIONS_AVAILABLE = False
    Compose = Gain = LowPassFilter = PitchShift = None

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, fftconvolve, lfilter
from nemo.collections.speechlm2.parts.precision import fp32_precision
from pathlib import Path

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False


class AudioAugmenter:
    """Audio augmentation with noise, impulse responses, and codec simulation."""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self._noise_files_cache = {}
        self._roomir_files_cache = {}
        self._micir_files_cache = {}
        self._lowpass_filter_cache = {}
        # build_audio_aug was constructing a new Compose + AddBackgroundNoise every step;
        # cache pipelines keyed by (sample_rate, noise file list) so AddBackgroundNoise LRU/cache persists.
        self._compose_aug_cache = {}

    def add_noise_to_batch(
        self,
        batch_audio: torch.Tensor,
        noise_folder: str,
        snr_db: float = 20.0,
        noise_prob_scale_user: float = 0.3,
        noise_prob_scale_user_min_snr: float = -3.0,
        noise_prob_scale_user_max_snr: float = 24.0,
        snr_measure_dur: float = 0.0,
        noise_resample: bool = True,
        noise_prob_low_pass: float = 0.1,
    ) -> torch.Tensor:
        """Add noise to batch audio with specified SNR."""
        batch_size = batch_audio.shape[0]
        audio_length = batch_audio.shape[1]

        if noise_folder not in self._noise_files_cache:
            noise_files = [f for f in glob.glob(os.path.join(noise_folder, "*.wav"))]
            if not noise_files:
                raise ValueError(f"No noise files found in {noise_folder}")
            self._noise_files_cache[noise_folder] = noise_files
        else:
            noise_files = self._noise_files_cache[noise_folder]

        for i in range(batch_size):
            def get_scale_factor(signal, noise, snr_db_local):
                if snr_measure_dur > 0:
                    signal = signal[: int(snr_measure_dur * self.sample_rate)]
                    noise = noise[: int(snr_measure_dur * self.sample_rate)]
                signal_power = torch.mean(signal ** 2) + 1e-8
                noise_power = torch.mean(noise ** 2) + 1e-8

                target_noise_power = signal_power / (10 ** (snr_db_local / 10))
                scaling_factor = torch.sqrt(target_noise_power / noise_power)
                return scaling_factor

            if random.random() < noise_prob_scale_user:
                scaling_factor = get_scale_factor(
                    batch_audio[i],
                    batch_audio[i],
                    random.randint(
                        int(noise_prob_scale_user_min_snr), int(noise_prob_scale_user_max_snr)
                    ),
                )
                batch_audio[i] = batch_audio[i] * scaling_factor

            def get_noise(noise_files):
                noise_path = random.choice(noise_files)
                noise, sr = sf.read(noise_path, dtype='float32')

                if noise_resample and sr != self.sample_rate:
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=self.sample_rate)

                if len(noise.shape) > 1:
                    noise = np.mean(noise, axis=1)

                noise_tensor = torch.tensor(noise, dtype=batch_audio.dtype, device=batch_audio.device)
                scaling_factor = get_scale_factor(batch_audio[i], noise_tensor, snr_db)
                noise_tensor = noise_tensor * scaling_factor
                return noise_tensor

            noise = get_noise(noise_files)
            noise2 = get_noise(noise_files)
            noise3 = get_noise(noise_files)
            noise = torch.cat([noise, noise2, noise3], axis=0)

            if noise.size(0) < audio_length:
                repeat_times = (audio_length // noise.size(0)) + 1
                noise = noise.repeat(repeat_times)[:audio_length]
            else:
                start_idx = torch.randint(0, noise.size(0) - audio_length + 1, (1,)).item()
                noise = noise[start_idx : start_idx + audio_length]

            if random.random() < noise_prob_low_pass:
                cutoff = 1000.0
                noise = self._apply_lowpass_filter(noise, cutoff)

            batch_audio[i] = batch_audio[i] + noise

        return batch_audio
    
    def _apply_lowpass_filter(self, audio: torch.Tensor, cutoff: float, order: int = 5) -> torch.Tensor:
        """Apply a low-pass Butterworth filter to audio."""
        cache_key = (cutoff, self.sample_rate, order)
        if cache_key not in self._lowpass_filter_cache:
            nyquist = 0.5 * self.sample_rate
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            self._lowpass_filter_cache[cache_key] = (b, a)
        
        b, a = self._lowpass_filter_cache[cache_key]
        y_cpu = lfilter(b, a, audio.cpu().numpy())
        y_gpu = torch.tensor(y_cpu, dtype=torch.float32, device=audio.device)
        return y_gpu
    
    def add_room_ir_to_batch(
        self,
        batch_audio: torch.Tensor,
        audio_lens: Optional[torch.Tensor],
        roomir_folder: str,
        use_loudness_norm: bool = True,
    ) -> torch.Tensor:
        """Apply room impulse response to batch audio."""
        batch_size = batch_audio.shape[0]

        if roomir_folder not in self._roomir_files_cache:
            roomir_files = [f for f in glob.glob(os.path.join(roomir_folder, "*.wav"))]
            if not roomir_files:
                raise ValueError(f"No room IR files found in {roomir_folder}")
            self._roomir_files_cache[roomir_folder] = roomir_files
        else:
            roomir_files = self._roomir_files_cache[roomir_folder]
        
        for i in range(batch_size):
            audio_length = audio_lens[i].item() if audio_lens is not None else batch_audio.shape[1]
            
            ir_path = random.choice(roomir_files)
            ir, sr = sf.read(ir_path, dtype='float32')
            
            if sr != self.sample_rate:
                ir = librosa.resample(ir, orig_sr=sr, target_sr=self.sample_rate)
            
            if len(ir.shape) > 1:
                ir = np.mean(ir, axis=1)
            
            audio_cpu = batch_audio[i, :audio_length].cpu().numpy()
            
            if use_loudness_norm and PYLOUDNORM_AVAILABLE:
                meter = pyln.Meter(self.sample_rate)
                try:
                    speech_loudness = meter.integrated_loudness(audio_cpu)
                    # Check for invalid loudness values (-inf for silent audio)
                    if speech_loudness == float('-inf') or not np.isfinite(speech_loudness):
                        use_loudness_norm = False
                except Exception:
                    use_loudness_norm = False
            
            convolved = fftconvolve(audio_cpu, ir, mode="full")[:audio_length]
            
            # Calculate RMS before convolution for gain compensation
            input_rms = np.sqrt(np.mean(audio_cpu ** 2)) + 1e-8
            convolved_rms = np.sqrt(np.mean(convolved ** 2)) + 1e-8
            
            if use_loudness_norm and PYLOUDNORM_AVAILABLE:
                try:
                    convolved_loudness = meter.integrated_loudness(convolved)
                    # Check for invalid loudness values
                    if convolved_loudness != float('-inf') and np.isfinite(convolved_loudness) and np.isfinite(speech_loudness):
                        convolved = pyln.normalize.loudness(convolved, convolved_loudness, speech_loudness)
                        # Validate output doesn't contain NaN or inf
                        if not np.isfinite(convolved).all():
                            convolved = fftconvolve(audio_cpu, ir, mode="full")[:audio_length]
                    else:
                        # Fallback: Use RMS-based gain compensation to restore signal level
                        gain_compensation = input_rms / convolved_rms
                        gain_compensation = min(gain_compensation, 10.0)  # Max 20dB boost
                        convolved = convolved * gain_compensation
                except Exception:
                    # Still apply gain compensation as fallback
                    gain_compensation = input_rms / convolved_rms
                    gain_compensation = min(gain_compensation, 10.0)
                    convolved = convolved * gain_compensation
            else:
                # If loudness normalization is disabled, apply simple RMS-based gain compensation
                gain_compensation = input_rms / convolved_rms
                gain_compensation = min(gain_compensation, 10.0)
                convolved = convolved * gain_compensation
            
            # Clip to prevent extreme values
            convolved = np.clip(convolved, -1.0, 1.0)

            batch_audio[i, :audio_length] = torch.tensor(convolved, dtype=batch_audio.dtype, device=batch_audio.device)
        
        return batch_audio
    
    def add_mic_ir_to_batch(
        self,
        batch_audio: torch.Tensor,
        audio_lens: Optional[torch.Tensor],
        micir_folder: str,
        use_loudness_norm: bool = True,
    ) -> torch.Tensor:
        """Apply microphone impulse response to batch audio."""
        batch_size = batch_audio.shape[0]
        
        if micir_folder not in self._micir_files_cache:
            micir_files = [f for f in glob.glob(os.path.join(micir_folder, "*.wav"))]
            if not micir_files:
                raise ValueError(f"No mic IR files found in {micir_folder}")
            self._micir_files_cache[micir_folder] = micir_files
        else:
            micir_files = self._micir_files_cache[micir_folder]
        
        for i in range(batch_size):
            audio_length = audio_lens[i].item() if audio_lens is not None else batch_audio.shape[1]
            
            ir_path = random.choice(micir_files)
            ir, sr = sf.read(ir_path, dtype='float32')
            
            if sr != self.sample_rate:
                ir = librosa.resample(ir, orig_sr=sr, target_sr=self.sample_rate)
            
            if len(ir.shape) > 1:
                ir = np.mean(ir, axis=1)
            
            audio_cpu = batch_audio[i, :audio_length].cpu().numpy()
            
            if use_loudness_norm and PYLOUDNORM_AVAILABLE:
                meter = pyln.Meter(self.sample_rate)
                try:
                    speech_loudness = meter.integrated_loudness(audio_cpu)
                    # Check for invalid loudness values (-inf for silent audio)
                    if speech_loudness == float('-inf') or not np.isfinite(speech_loudness):
                        use_loudness_norm = False
                except Exception:
                    use_loudness_norm = False
            
            convolved = fftconvolve(audio_cpu, ir, mode="full")[:audio_length]
            
            # Calculate RMS before convolution for gain compensation
            input_rms = np.sqrt(np.mean(audio_cpu ** 2)) + 1e-8
            convolved_rms = np.sqrt(np.mean(convolved ** 2)) + 1e-8
            
            if use_loudness_norm and PYLOUDNORM_AVAILABLE:
                try:
                    convolved_loudness = meter.integrated_loudness(convolved)
                    # Check for invalid loudness values
                    if convolved_loudness != float('-inf') and np.isfinite(convolved_loudness) and np.isfinite(speech_loudness):
                        convolved = pyln.normalize.loudness(convolved, convolved_loudness, speech_loudness)
                        # Validate output doesn't contain NaN or inf
                        if not np.isfinite(convolved).all():
                            convolved = fftconvolve(audio_cpu, ir, mode="full")[:audio_length]
                    else:
                        # Fallback: Use RMS-based gain compensation
                        gain_compensation = input_rms / convolved_rms
                        gain_compensation = min(gain_compensation, 10.0)
                        convolved = convolved * gain_compensation
                except Exception:
                    # Still apply gain compensation as fallback
                    gain_compensation = input_rms / convolved_rms
                    gain_compensation = min(gain_compensation, 10.0)
                    convolved = convolved * gain_compensation
            else:
                # If loudness normalization is disabled, apply simple RMS-based gain compensation
                gain_compensation = input_rms / convolved_rms
                gain_compensation = min(gain_compensation, 10.0)
                convolved = convolved * gain_compensation
            
            # Clip to prevent extreme values
            convolved = np.clip(convolved, -1.0, 1.0)
            
            batch_audio[i, :audio_length] = torch.tensor(convolved, dtype=batch_audio.dtype, device=batch_audio.device)
        
        return batch_audio
    
    def add_codec_to_batch(
        self,
        batch_audio: torch.Tensor,
        audio_lens: Optional[torch.Tensor],
        codec_settings: dict,
    ) -> torch.Tensor:
        """Apply codec degradation to batch audio using FFmpeg."""
        batch_size = batch_audio.shape[0]
        
        for i in range(batch_size):
            audio_length = audio_lens[i].item() if audio_lens is not None else batch_audio.shape[1]
            
            codec_name, codec_args = random.choice(list(codec_settings.items()))
            
            audio_cpu = batch_audio[i, :audio_length].cpu().numpy()
            
            try:
                degraded = self._apply_ffmpeg_codec(audio_cpu, codec_args)
                
                # Validate output doesn't contain NaN or inf
                if np.isfinite(degraded).all():
                    degraded = np.clip(degraded, -1.0, 1.0)
                    batch_audio[i, :audio_length] = torch.tensor(degraded, dtype=batch_audio.dtype, device=batch_audio.device)
            except Exception:
                # Codec failed, skip augmentation for this sample
                pass
        
        return batch_audio
    
    def _apply_ffmpeg_codec(self, audio: np.ndarray, codec_args: list) -> np.ndarray:
        """Apply audio compression/decompression using FFmpeg."""
        target_len = len(audio)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = os.path.join(tmpdir, "input.wav")
            fmt = codec_args[codec_args.index("-f") + 1] if "-f" in codec_args else "wav"
            mid_file = os.path.join(tmpdir, f"compressed.{fmt}")
            out_wav = os.path.join(tmpdir, "output.wav")
            
            sf.write(in_wav, audio, samplerate=self.sample_rate, subtype='PCM_16')
            
            subprocess.run(
                ["ffmpeg", "-y", "-i", in_wav] + codec_args + [mid_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            
            subprocess.run(
                ["ffmpeg", "-y", "-i", mid_file, "-ar", str(self.sample_rate), out_wav],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            
            decoded, _ = sf.read(out_wav, dtype='float32')
            
            if len(decoded) > target_len:
                decoded = decoded[:target_len]
            elif len(decoded) < target_len:
                pad = np.zeros(target_len - len(decoded), dtype=decoded.dtype)
                decoded = np.concatenate([decoded, pad])
            
            return decoded

    def build_audio_aug(
        self, 
        audio_samples: torch.Tensor, 
        noise_files: List[Union[str, Path]]
    ) -> torch.Tensor:
        """
        Construct and apply an audio augmentation pipeline to the input batch.

        Parameters
        audio_samples : torch.Tensor
            Input audio batch.
        noise_files : List[str] or List[Path]
            List of paths to WAV noise files that will be used by
            AddBackgroundNoise.

        Returns
        torch.Tensor
            Augmented audio batch, same shape as `audio_samples`.
        """
        # Ensure correct shape
        if audio_samples.dim() == 2:            # [B, T]
            audio_samples = audio_samples.unsqueeze(1)  # [B, 1, T]
        elif audio_samples.dim() == 1:          # [T]
            audio_samples = audio_samples.unsqueeze(0).unsqueeze(0)
            
        # Random EQ
        if np.random.rand() > 0.1:
            eq = self.get_random_eq(self.sample_rate)
            with fp32_precision():
                audio_samples = self.apply_eq(audio_samples, eq)

        # Reuse one Compose + AddBackgroundNoise per (sr, noise set); rebuilding every step
        # rescans the filesystem and drops in-memory noise LRU — major step-time regression.
        noise_key = tuple(sorted(str(f) for f in noise_files))
        compose_key = (self.sample_rate, noise_key)
        if compose_key not in self._compose_aug_cache:
            if not _TORCH_AUDIOMENTATIONS_AVAILABLE:
                raise ImportError(
                    "torch_audiomentations is required for compose augmentation but is not installed. "
                    "Install it with: pip install torch_audiomentations"
                )
            self._compose_aug_cache[compose_key] = Compose(
                transforms=[
                    AddBackgroundNoise(
                        sounds_path=list(noise_key),
                        min_snr_db=5.0,
                        max_snr_db=25.0,
                        p=0.2,
                        cache_audio=True,
                        max_cache_items=512,
                        # DNS-style continuous noise: simple RMS is much faster than silence-aware windows.
                        silence_aware_rms=False,
                    ),
                    Gain(
                        min_gain_in_db=-6,
                        max_gain_in_db=6,
                        p=0.8,
                    ),
                    PitchShift(
                        min_transpose_semitones=-1.0,
                        max_transpose_semitones=1.0,
                        sample_rate=self.sample_rate,
                        p=0.0,
                    ),
                    LowPassFilter(
                        min_cutoff_freq=5000.0,
                        max_cutoff_freq=7500.0,
                        p=0.4
                    ),
                ]
            )
        apply_augmentation = self._compose_aug_cache[compose_key]

        # Apply augmentation
        with fp32_precision():
            transformed_audio =  apply_augmentation(audio_samples, sample_rate=self.sample_rate)
        
        # If any sample's absolute peak > 1, rescale that sample so its peak becomes 0.85
        # reduce over all non-batch dims (works for [B,T], [B,1,T], [B,C,T], etc.)
        reduce_dims = tuple(range(1, transformed_audio.dim()))
        max_abs = transformed_audio.abs().amax(dim=reduce_dims)  # shape: (B,)
        eps = 1e-12
        scale = torch.where(max_abs > 1.0, 0.85 / (max_abs + eps), torch.ones_like(max_abs))
        # broadcast scale to audio shape
        view_shape = [transformed_audio.shape[0]] + [1] * (transformed_audio.dim() - 1)
        transformed_audio = transformed_audio * scale.view(*view_shape)

        # Remove channel dim [B, 1, T] -> [B, T]
        return transformed_audio.squeeze(1)

    def apply_eq(self, signal, eq, sampling_rate=16000, display=False):
        """
        Filter a signal with a low-pass/high-pass/band-pass/band-stop
        Butterworth filter.

        Parameters
        ----------
        signal: ndarray of floats (shape [n_channels, n_samples])
            signal to filter

        sampling_rate: int
            sampling rate of the signal

        eq: audio_dspy.eq.EQ
            audio_dspy EQ instance containing filter parameters in .filters

        display: bool (default False)
            display input signal vs filtered signal

        Returns
        -------
        ndarray of floats (shape [n_channels, n_samples])
            filtered signal
        """

        filt_signal = signal
        for filt in eq.filters:
            a = torch.as_tensor(filt.a_coefs, dtype=torch.float32, device=filt_signal.device)
            b = torch.as_tensor(filt.b_coefs, dtype=torch.float32, device=filt_signal.device)
            filt_signal = filtfilt(filt_signal, a, b)

        if torch.any(torch.isnan(filt_signal)):
            raise ValueError('NaN found in filtered signal during EQ')

        if filt_signal.dtype != signal.dtype:
            filt_signal = filt_signal.to(dtype=signal.dtype)

        return filt_signal

    def get_random_eq(self, sampling_rate, display=False):
        """Generate random EQ"""
        try:
            import audio_dspy as adsp
        except ImportError:
            raise ImportError(
                "audio_dspy is required for EQ augmentation but is not installed. "
                "Install it with: pip install audio_dspy"
            )

        eq = adsp.EQ(sampling_rate)

        # Choose a filter type among low_shelf, bell, high_shelf
        filter_type_choice = np.random.randint(3)

        if filter_type_choice == 0:  # low_shelf
            cutoff_freq = np.random.uniform(100, 400)
            gain = np.random.uniform(1.05, 1.5)
            q_factor = np.random.uniform(0.1, 0.8)
            eq.add_lowshelf(cutoff_freq, q_factor, gain)

        if filter_type_choice == 1:  # bell
            # How many bands? 1 to 3
            n_bands = np.random.randint(1, 4)
            for sb in range(n_bands):
                cutoff_freq = np.random.uniform(100, 6000)
                gain = np.random.uniform(1.05, 1.5)
                q_factor = np.random.uniform(0.1, 1)
                eq.add_bell(cutoff_freq, q_factor, gain)

        if filter_type_choice == 2:  # high_shelf
            cutoff_freq = np.random.uniform(5000, 6000)
            gain = np.random.uniform(1.05, 1.5)
            q_factor = np.random.uniform(0.1, 0.8)
            eq.add_highshelf(cutoff_freq, q_factor, gain)

        if display:
            import matplotlib.pyplot as plt

            plt.figure()
            eq.plot_eq_curve()
            plt.show(block=False)

        return eq

DEFAULT_CODEC_SETTINGS = {
    "high_libopus_8k_5k": ["-ar", "8000", "-c:a", "libopus", "-application", "voip", "-b:a", "5.5k", "-f", "ogg"],
    "high_g726_8k_16k": ["-ar", "8000", "-c:a", "adpcm_g726", "-b:a", "16k", "-f", "wav"],
    "med_libopus_8k_9k": ["-ar", "8000", "-c:a", "libopus", "-application", "voip", "-b:a", "9.5k", "-f", "ogg"],
    "med_libopus_16k_12k": ["-ar", "16000", "-c:a", "libopus", "-application", "voip", "-b:a", "12k", "-f", "ogg"],
    "med_libvorbis_16k_32k": ["-ar", "16000", "-c:a", "libvorbis", "-b:a", "32k", "-f", "ogg"],
    "med_mp3_16k_32k": ["-ar", "16000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3"],
    "low_mulaw_8k": ["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"],
    "low_alaw_8k": ["-ar", "8000", "-c:a", "pcm_alaw", "-f", "wav"],
    "low_g722_16k": ["-ar", "16000", "-c:a", "g722", "-f", "wav"],
    "low_g726_8k_32k": ["-ar", "8000", "-c:a", "adpcm_g726", "-b:a", "32k", "-f", "wav"],
    "low_libopus_16k_32k": ["-ar", "16000", "-c:a", "libopus", "-application", "audio", "-b:a", "32k", "-f", "ogg"],
    "low_libopus_24k_32k": ["-ar", "24000", "-c:a", "libopus", "-application", "audio", "-b:a", "32k", "-f", "ogg"],
    "low_libopus_24k_48k": ["-ar", "24000", "-c:a", "libopus", "-application", "audio", "-b:a", "48k", "-f", "ogg"],
    "low_libvorbis_24k_64k": ["-ar", "24000", "-c:a", "libvorbis", "-b:a", "64k", "-f", "ogg"],
    "low_mp3_24k_64k": ["-ar", "24000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"]
}

