import shutil
from pathlib import Path
from typing import Union

from nemo.lightning.io.artifact.base import Artifact


class PathArtifact(Artifact[Path]):
    def dump(self, value: Path, absolute_dir: Path, relative_dir: Path) -> Path:
        new_value = copy_file(value, absolute_dir, relative_dir)
        return new_value

    def load(self, path: Path) -> Path:
        return path


class FileArtifact(Artifact[str]):
    def dump(self, value: str, absolute_dir: Path, relative_dir: Path) -> str:
        new_value = copy_file(value, absolute_dir, relative_dir)
        return str(new_value)

    def load(self, path: str) -> str:
        return path


def copy_file(src: Union[Path, str], path: Union[Path, str], relative_dst: Union[Path, str]):
    relative_path = Path(relative_dst) / Path(src).name
    output = Path(path) / relative_path
    shutil.copy2(src, output)
    return relative_path
