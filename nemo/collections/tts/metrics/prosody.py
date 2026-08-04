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

"""Reference-based acoustic prosody distances for TTS evaluation."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

import librosa
import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is an optional speedup.
    njit = None


SpeechRateCharMode = Literal["nonspace", "all", "alnum"]
F0Method = Literal["pyin", "yin", "none"]
F0Normalization = Literal["gt_median", "utterance_median", "none"]
EnergyNormalization = Literal["zscore", "none"]

_ALNUM_CHAR_RE = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class ProsodyDistanceConfig:
    """Configuration for reference-based acoustic prosody distance metrics.

    The default values are tuned for corpus-level TTS evaluation: pYIN is used
    for more stable F0 contours, F0 is converted to semitones relative to the
    reference median, intensity uses log-RMS z-scores, and contours are reduced
    before DTW to keep evaluation bounded.
    """

    sample_rate: int = 16000
    res_type: str = "soxr_hq"
    frame_shift_ms: float = 20.0
    frame_length_ms: float = 64.0
    fmin: float = 55.0
    fmax: float = 450.0
    f0_method: F0Method = "pyin"
    yin_silence_db_below_peak: float = 35.0
    pyin_n_thresholds: int = 24
    pyin_beta_a: float = 2.0
    pyin_beta_b: float = 18.0
    pyin_boltzmann_parameter: float = 2.0
    pyin_resolution: float = 0.25
    pyin_max_transition_rate: float = 12.0
    pyin_switch_prob: float = 0.01
    pyin_no_trough_prob: float = 0.01
    pyin_center: bool = True
    pyin_pad_mode: str = "constant"
    max_dtw_frames: int = 1000
    dtw_band_ratio: float = 0.05
    f0_nan_penalty: float = 6.0
    f0_normalization: F0Normalization = "gt_median"
    intensity_normalization: EnergyNormalization = "zscore"
    speech_rate_char_mode: SpeechRateCharMode = "nonspace"
    min_voiced_frames: int = 5


@dataclass(frozen=True)
class ProsodyDistanceResult:
    """Per-pair acoustic prosody distance metrics."""

    pitch_distance: float
    intensity_distance: float
    speech_rate_distance: float
    gt_duration_sec: float
    pred_duration_sec: float
    gt_speech_rate_cps: float
    pred_speech_rate_cps: float
    gt_char_count: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable dictionary."""
        return {
            "pitch_distance": self.pitch_distance,
            "intensity_distance": self.intensity_distance,
            "speech_rate_distance": self.speech_rate_distance,
            "gt_duration_sec": self.gt_duration_sec,
            "pred_duration_sec": self.pred_duration_sec,
            "gt_speech_rate_cps": self.gt_speech_rate_cps,
            "pred_speech_rate_cps": self.pred_speech_rate_cps,
            "gt_char_count": self.gt_char_count,
        }


if njit is not None:

    @njit
    def _dtw_distance_1d_numba(x: np.ndarray, y: np.ndarray, nan_penalty: float, band_radius: int) -> float:
        n = x.shape[0]
        m = y.shape[0]
        if n == 0 or m == 0:
            return np.nan

        inf = 1.0e30
        prev = np.empty(m + 1, dtype=np.float64)
        curr = np.empty(m + 1, dtype=np.float64)
        for j in range(m + 1):
            prev[j] = inf
            curr[j] = inf
        prev[0] = 0.0

        for i in range(1, n + 1):
            for j in range(m + 1):
                curr[j] = inf

            if band_radius < 0:
                j_start = 1
                j_end = m + 1
            else:
                j_start = max(1, i - band_radius)
                j_end = min(m, i + band_radius) + 1

            for j in range(j_start, j_end):
                xv = x[i - 1]
                yv = y[j - 1]
                x_nan = np.isnan(xv)
                y_nan = np.isnan(yv)
                if x_nan and y_nan:
                    cost = 0.0
                elif x_nan or y_nan:
                    cost = nan_penalty
                else:
                    diff = xv - yv
                    cost = diff if diff >= 0.0 else -diff

                best_prev = prev[j - 1]
                if prev[j] < best_prev:
                    best_prev = prev[j]
                if curr[j - 1] < best_prev:
                    best_prev = curr[j - 1]
                curr[j] = cost + best_prev

            tmp = prev
            prev = curr
            curr = tmp

        total = prev[m]
        if total >= inf / 2.0:
            return np.nan
        return total / float(n + m)

else:
    _dtw_distance_1d_numba = None


def compute_prosody_distances(
    gt_audio_path: str,
    pred_audio_path: str,
    text: Any,
    config: Optional[ProsodyDistanceConfig] = None,
) -> ProsodyDistanceResult:
    """Compute acoustic prosody distances between reference and generated audio.

    Args:
        gt_audio_path: Ground-truth/reference audio path.
        pred_audio_path: Generated/predicted audio path.
        text: Reference text used for character-per-second speech rate.
        config: Optional prosody distance configuration.

    Returns:
        ProsodyDistanceResult with pitch, intensity, and speech-rate distances.
    """
    cfg = config or ProsodyDistanceConfig()
    gt_audio, sr, gt_duration = _load_audio(gt_audio_path, cfg)
    pred_audio, _, pred_duration = _load_audio(pred_audio_path, cfg)

    hop_length, frame_length = _frame_params(sr, cfg)
    gt_log_energy = _compute_log_energy(gt_audio, frame_length=frame_length, hop_length=hop_length)
    pred_log_energy = _compute_log_energy(pred_audio, frame_length=frame_length, hop_length=hop_length)

    pitch_distance = float("nan")
    if cfg.f0_method != "none":
        gt_f0 = _compute_f0(
            gt_audio, sr=sr, frame_length=frame_length, hop_length=hop_length, log_energy=gt_log_energy, cfg=cfg
        )
        pred_f0 = _compute_f0(
            pred_audio,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            log_energy=pred_log_energy,
            cfg=cfg,
        )
        if np.isfinite(gt_f0).sum() >= cfg.min_voiced_frames and np.isfinite(pred_f0).sum() >= cfg.min_voiced_frames:
            gt_pitch, pred_pitch = _prepare_f0_for_metric(gt_f0, pred_f0, cfg)
            pitch_distance = _dtw_distance_1d(
                _maybe_reduce_for_dtw(gt_pitch, cfg.max_dtw_frames),
                _maybe_reduce_for_dtw(pred_pitch, cfg.max_dtw_frames),
                nan_penalty=cfg.f0_nan_penalty,
                band_ratio=cfg.dtw_band_ratio,
            )

    gt_intensity, pred_intensity = _prepare_intensity_for_metric(gt_log_energy, pred_log_energy, cfg)
    intensity_distance = _dtw_distance_1d(
        _maybe_reduce_for_dtw(gt_intensity, cfg.max_dtw_frames),
        _maybe_reduce_for_dtw(pred_intensity, cfg.max_dtw_frames),
        nan_penalty=0.0,
        band_ratio=cfg.dtw_band_ratio,
    )

    gt_char_count = _char_count(text, cfg.speech_rate_char_mode)
    gt_speech_rate = gt_char_count / gt_duration if gt_duration > 0.0 else float("nan")
    pred_speech_rate = gt_char_count / pred_duration if pred_duration > 0.0 else float("nan")
    speech_rate_distance = abs(gt_speech_rate - pred_speech_rate)

    return ProsodyDistanceResult(
        pitch_distance=_safe_float(pitch_distance),
        intensity_distance=_safe_float(intensity_distance),
        speech_rate_distance=_safe_float(speech_rate_distance),
        gt_duration_sec=_safe_float(gt_duration),
        pred_duration_sec=_safe_float(pred_duration),
        gt_speech_rate_cps=_safe_float(gt_speech_rate),
        pred_speech_rate_cps=_safe_float(pred_speech_rate),
        gt_char_count=gt_char_count,
    )


def _load_audio(path: str, cfg: ProsodyDistanceConfig) -> tuple[np.ndarray, int, float]:
    if not path:
        raise FileNotFoundError("empty audio filepath")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    audio, sr = librosa.load(path, sr=cfg.sample_rate, mono=True, res_type=cfg.res_type)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"empty audio after loading: {path}")
    return audio, int(sr), float(audio.shape[0] / sr)


def _frame_params(sr: int, cfg: ProsodyDistanceConfig) -> tuple[int, int]:
    hop_length = max(1, int(round(sr * cfg.frame_shift_ms / 1000.0)))
    frame_length = max(hop_length * 2, int(round(sr * cfg.frame_length_ms / 1000.0)))
    return hop_length, frame_length


def _compute_log_energy(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    return np.log(np.maximum(np.asarray(rms, dtype=np.float64), 1.0e-10))


def _compute_f0(
    audio: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    log_energy: np.ndarray,
    cfg: ProsodyDistanceConfig,
) -> np.ndarray:
    if cfg.f0_method == "pyin":
        f0, voiced_flag, _ = librosa.pyin(
            y=np.asarray(audio, dtype=np.float64),
            sr=sr,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            frame_length=frame_length,
            hop_length=hop_length,
            center=cfg.pyin_center,
            pad_mode=cfg.pyin_pad_mode,
            n_thresholds=cfg.pyin_n_thresholds,
            beta_parameters=(cfg.pyin_beta_a, cfg.pyin_beta_b),
            boltzmann_parameter=cfg.pyin_boltzmann_parameter,
            resolution=cfg.pyin_resolution,
            max_transition_rate=cfg.pyin_max_transition_rate,
            switch_prob=cfg.pyin_switch_prob,
            no_trough_prob=cfg.pyin_no_trough_prob,
            fill_na=np.nan,
        )
        f0 = np.asarray(f0, dtype=np.float64)
        if voiced_flag is not None:
            voiced_flag = np.asarray(voiced_flag, dtype=bool)
            min_len = min(len(f0), len(voiced_flag))
            f0 = f0[:min_len]
            f0[~voiced_flag[:min_len]] = np.nan
        return f0

    if cfg.f0_method == "yin":
        f0 = librosa.yin(
            audio,
            sr=sr,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )
        f0 = np.asarray(f0, dtype=np.float64)
        f0[(f0 < cfg.fmin) | (f0 > cfg.fmax)] = np.nan
        return _mask_yin_silence(f0, log_energy, cfg)

    if cfg.f0_method == "none":
        return np.asarray([], dtype=np.float64)

    raise ValueError(f"Unsupported f0_method={cfg.f0_method!r}")


def _mask_yin_silence(f0: np.ndarray, log_energy: np.ndarray, cfg: ProsodyDistanceConfig) -> np.ndarray:
    if len(log_energy) == 0:
        return f0

    min_len = min(len(f0), len(log_energy))
    f0 = f0[:min_len].copy()
    energy = log_energy[:min_len]
    rms_db = 20.0 * energy / math.log(10.0)
    finite = np.isfinite(rms_db)
    if finite.any():
        peak_db = float(np.max(rms_db[finite]))
        f0[rms_db < peak_db - cfg.yin_silence_db_below_peak] = np.nan
    return f0


def _prepare_f0_for_metric(
    gt_f0_hz: np.ndarray,
    pred_f0_hz: np.ndarray,
    cfg: ProsodyDistanceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.f0_normalization == "none":
        return gt_f0_hz, pred_f0_hz

    gt_median = float(np.nanmedian(gt_f0_hz)) if np.isfinite(gt_f0_hz).any() else float("nan")
    pred_median = float(np.nanmedian(pred_f0_hz)) if np.isfinite(pred_f0_hz).any() else float("nan")

    if cfg.f0_normalization == "gt_median":
        return _hz_to_semitones(gt_f0_hz, gt_median), _hz_to_semitones(pred_f0_hz, gt_median)
    if cfg.f0_normalization == "utterance_median":
        return _hz_to_semitones(gt_f0_hz, gt_median), _hz_to_semitones(pred_f0_hz, pred_median)
    raise ValueError(f"Unsupported f0_normalization={cfg.f0_normalization!r}")


def _prepare_intensity_for_metric(
    gt_log_energy: np.ndarray,
    pred_log_energy: np.ndarray,
    cfg: ProsodyDistanceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.intensity_normalization == "none":
        return gt_log_energy, pred_log_energy
    if cfg.intensity_normalization == "zscore":
        return _zscore(gt_log_energy), _zscore(pred_log_energy)
    raise ValueError(f"Unsupported intensity_normalization={cfg.intensity_normalization!r}")


def _dtw_distance_1d(x: np.ndarray, y: np.ndarray, nan_penalty: float, band_ratio: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("DTW expects 1-D arrays")
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    if band_ratio is None or band_ratio < 0:
        band_radius = -1
    else:
        band_radius = max(abs(len(x) - len(y)), int(math.ceil(float(band_ratio) * max(len(x), len(y)))))

    if _dtw_distance_1d_numba is not None:
        try:
            return _safe_float(_dtw_distance_1d_numba(x, y, float(nan_penalty), int(band_radius)))
        except Exception:
            pass
    return _dtw_distance_1d_python(x, y, float(nan_penalty), int(band_radius))


def _dtw_distance_1d_python(x: np.ndarray, y: np.ndarray, nan_penalty: float, band_radius: int) -> float:
    n = len(x)
    m = len(y)
    if n == 0 or m == 0:
        return float("nan")

    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr.fill(np.inf)
        if band_radius < 0:
            j_start = 1
            j_end = m + 1
        else:
            j_start = max(1, i - band_radius)
            j_end = min(m, i + band_radius) + 1

        for j in range(j_start, j_end):
            cost = _frame_distance(x[i - 1], y[j - 1], nan_penalty)
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    total = prev[m]
    if not np.isfinite(total):
        return float("nan")
    return float(total / (n + m))


def _frame_distance(x: float, y: float, nan_penalty: float) -> float:
    x_nan = np.isnan(x)
    y_nan = np.isnan(y)
    if x_nan and y_nan:
        return 0.0
    if x_nan or y_nan:
        return nan_penalty
    return abs(float(x) - float(y))


def _hz_to_semitones(f0_hz: np.ndarray, ref_hz: float) -> np.ndarray:
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    out = np.full_like(f0_hz, np.nan, dtype=np.float64)
    if not np.isfinite(ref_hz) or ref_hz <= 0.0:
        return out
    valid = np.isfinite(f0_hz) & (f0_hz > 0.0)
    out[valid] = 12.0 * np.log2(f0_hz[valid] / ref_hz)
    return out


def _zscore(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return values
    mean = float(np.mean(values[finite]))
    std = float(np.std(values[finite]))
    out = values.copy()
    if std < eps:
        out[finite] = out[finite] - mean
    else:
        out[finite] = (out[finite] - mean) / std
    return out


def _maybe_reduce_for_dtw(values: np.ndarray, max_frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if max_frames is None or max_frames <= 0 or len(values) <= max_frames:
        return values
    return _resample_1d_preserve_nans(values, int(max_frames))


def _resample_1d_preserve_nans(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return values
    if len(values) == 1:
        return np.full(target_len, values[0], dtype=np.float64)

    old_t = np.linspace(0.0, 1.0, len(values))
    new_t = np.linspace(0.0, 1.0, target_len)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return np.full(target_len, np.nan, dtype=np.float64)

    out = np.interp(new_t, old_t[valid], values[valid])
    valid_interp = np.interp(new_t, old_t, valid.astype(np.float64))
    out[valid_interp < 0.5] = np.nan
    return out.astype(np.float64)


def _char_count(text: Any, mode: SpeechRateCharMode) -> int:
    if text is None:
        return 0
    text = str(text)
    if mode == "nonspace":
        return sum(1 for ch in text if not ch.isspace())
    if mode == "alnum":
        return len(_ALNUM_CHAR_RE.findall(text))
    if mode == "all":
        return len(text)
    raise ValueError(f"Unsupported speech_rate_char_mode={mode!r}")


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(value):
        return float("nan")
    return value
