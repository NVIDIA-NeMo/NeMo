from pathlib import Path
from typing import Union

# NeMo2 checkpoint structure is a checkpoint directory, with a WEIGHTS_PATH and CONTEXT_PATH subdirectory structure.
#  WEIGHTS_PATH stores the weights while CONTEXT_PATH stores the hyper-parameters.
WEIGHTS_PATH: str = "weights"
CONTEXT_PATH: str = "context"


def ckpt_to_weights_subdir(filepath: Union[str, Path]) -> Path:
    """Given an input checkpoint filepath, clean it using `ckpt_to_dir` and then return the weights subdirectory."""
    return ckpt_to_dir(filepath=filepath) / WEIGHTS_PATH


def ckpt_to_context_subdir(filepath: Union[str, Path]) -> Path:
    """Given an input checkpoint filepath, clean it using `ckpt_to_dir` and then return the context subdirectory."""
    return ckpt_to_dir(filepath=filepath) / CONTEXT_PATH


def ckpt_to_dir(filepath: Union[str, Path]) -> Path:
    """PTL considers checkpoints as .ckpt files.
    This method removes the extension and returns a path
    to be used as a directory for distributed checkpoints
    """

    filepath = Path(filepath)
    if not filepath.suffix == ".ckpt":
        filepath = filepath.with_suffix(filepath.suffix + ".ckpt")

    # adding this assert because we will later remove directories based on the return value of this method
    assert filepath.suffix == ".ckpt", f"filepath: {filepath} must have .ckpt extension"

    # create a new path whose name is the original filepath without the .ckpt extension
    checkpoint_dir = filepath.with_name(filepath.stem)

    return checkpoint_dir
