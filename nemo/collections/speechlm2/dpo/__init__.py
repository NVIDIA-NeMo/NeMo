# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Native Direct Preference Optimization support for SpeechLM2.

The implementation intentionally lives next to :class:`SALMAutomodel`: data
loading, pair formatting, reference capture, objective computation, and the
Lightning update lifecycle are normal package code rather than launch-time
overlays or experiment-local adapters.
"""

from nemo.collections.speechlm2.dpo.data import FiniteLhotsePreferenceDataModule
from nemo.collections.speechlm2.dpo.model import DPOSALMAutomodel

__all__ = ["DPOSALMAutomodel", "FiniteLhotsePreferenceDataModule"]
