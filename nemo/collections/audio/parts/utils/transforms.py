# Thin shim: nemotron_voicechat_inference_wrapper imports resample from .transforms;
# the actual implementation lives in resampling.py.
from nemo.collections.audio.parts.utils.resampling import resample

__all__ = ["resample"]
