import os
from pathlib import Path

from nemo.export.tarutils import TarPath

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "rank{}.safetensors"


def is_qnemo_checkpoint(path: str) -> bool:
    """Detect if a given path is a TensorRT-LLM a.k.a. "qnemo" checkpoint based on config & tensor data presence."""
    if os.path.isdir(path):
        path = Path(path)
    else:
        path = TarPath(path)
    config_path = path / CONFIG_NAME
    tensor_path = path / WEIGHTS_NAME.format(0)
    return config_path.exists() and tensor_path.exists()
