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
import json
import os
import random
import re
import struct
import tarfile
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

# Knuth's multiplicative hash constant (golden-ratio derived, 32-bit).
_KNUTH_HASH = 2654435761

# Tar block size + the all-zeros block that marks end-of-archive in tar.
_TAR_BLOCK_SIZE = 512
_TAR_ZERO_BLOCK = b'\0' * _TAR_BLOCK_SIZE

# Recognized URL schemes whose authority ("host" component) is part of the
# logical path (e.g. the bucket name). Stripping just the scheme keeps the
# bucket+key in the relative path used to mirror under indexes_root.
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def resolve_idx_path(data_path: str | Path, indexes_root: Optional[str | Path] = None) -> str:
    """
    Compute the ``.idx`` sidecar path for *data_path*.

    When ``indexes_root`` is ``None`` (the default), return ``data_path + ".idx"``
    so the sidecar lives next to the data file, matching the conventional
    layout.

    When ``indexes_root`` is set, return a path under that root that mirrors
    the data file's directory structure. URL schemes are stripped (so the
    bucket/key remains as the relative key); leading separators on local paths
    are dropped. Examples::

        /data/foo/bar.jsonl       + indexes_root=/cache/idx
            -> /cache/idx/data/foo/bar.jsonl.idx
        ais://bucket/key/m.jsonl  + indexes_root=/cache/idx
            -> /cache/idx/bucket/key/m.jsonl.idx
        s3://b/path/data.tar      + indexes_root=/cache/idx
            -> /cache/idx/b/path/data.tar.idx

    The indexes_root argument itself can be local or a URL — joining respects
    URL semantics so e.g. mirroring into ``ais://cache/idx`` works the same way.
    """
    data_str = str(data_path)
    if indexes_root is None:
        return data_str + ".idx"

    # Normalize the data path into a relative "key" by stripping a URL scheme,
    # any leading slashes, and Windows-style drive letters (best-effort).
    rel = _URL_RE.sub("", data_str).lstrip("/\\")
    # Strip "C:" or "C:/" style drive prefixes.
    if len(rel) >= 2 and rel[1] == ":":
        rel = rel[2:].lstrip("/\\")

    root_str = str(indexes_root).rstrip("/\\")
    if _URL_RE.match(root_str):
        return f"{root_str}/{rel}.idx"
    return str(Path(root_str) / (rel + ".idx"))


class LazyShuffledRange:
    """
    Generates a permutation of ``range(n)`` lazily using a Feistel cipher,
    without materializing the full index list. Each element is computed on
    the fly in O(1) time and the object itself uses O(1) memory regardless
    of ``n``.

    The technique is known as *cycle-walking* format-preserving encryption:
    a Feistel network is a bijection on ``[0, 2^k)``, and repeatedly applying
    it until the output falls within ``[0, n)`` restricts it to a bijection
    on the desired domain.

    Args:
        n: Size of the range to permute.
        rng: A ``random.Random`` instance used to derive round keys.
        num_rounds: Number of Feistel rounds (more rounds = better uniformity,
            6 is a good default for typical dataset sizes).
    """

    def __init__(self, n: int, rng: random.Random, num_rounds: int = 6):
        self.n = n
        if n <= 1:
            return
        bits = (n - 1).bit_length()
        if bits < 2:
            bits = 2
        if bits % 2:
            bits += 1
        self._half = bits // 2
        self._mask = (1 << self._half) - 1
        self._num_rounds = num_rounds
        self._keys = [rng.getrandbits(64) for _ in range(num_rounds)]

    def _permute_one(self, x: int) -> int:
        left = (x >> self._half) & self._mask
        right = x & self._mask
        for key in self._keys:
            left, right = right, left ^ (((right * _KNUTH_HASH) ^ key) >> 32 & self._mask)
        return (left << self._half) | right

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        n = self.n
        if n <= 0:
            return
        if n == 1:
            yield 0
            return
        for i in range(n):
            x = i
            while True:
                x = self._permute_one(x)
                if x < n:
                    yield x
                    break


def _load_index(data_path: str, idx_path: str | None = None):
    """
    Load a memmap'd offset index for *data_path*.

    Returns ``(offsets, num_samples)`` where ``offsets`` always has
    ``num_samples + 1`` entries — the last one being the data file size
    (appended if absent in the on-disk index).

    Validates that all sample offsets fall within the data file.
    """
    if idx_path is None:
        idx_path = data_path + '.idx'
    offsets = np.memmap(idx_path, dtype=np.dtype('<u8'), mode='r')
    data_size = os.path.getsize(data_path)
    if offsets[-1] == data_size:
        num_samples = offsets.shape[0] - 1
    else:
        num_samples = offsets.shape[0]
        offsets = np.append(offsets, np.uint64(data_size))
    if num_samples > 0:
        max_offset = int(offsets[:num_samples].max())
        if max_offset >= data_size:
            raise ValueError(
                f"Index for {data_path} contains offset {max_offset} "
                f"beyond file size {data_size}. "
                f"The .idx file may have been created by an incompatible tool "
                f"or for a different file."
            )
    return offsets, num_samples


def _resolve_idx(idx: int, length: int) -> int:
    if idx < 0:
        idx += length
    if idx < 0 or idx >= length:
        raise IndexError("Index out of bounds")
    return idx


class IndexedJSONLReader:
    def __init__(self, jsonl_path: Path | str, idx_path: Path | str | None = None):
        self.data_path = str(jsonl_path)
        self.offsets, self._len = _load_index(self.data_path, str(idx_path) if idx_path else None)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        idx = _resolve_idx(idx, self._len)
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        with open(self.data_path, 'rb') as f:
            f.seek(start)
            data = f.read(end - start)
        return json.loads(data.decode('utf-8'))


class TarSample(NamedTuple):
    """A single sample extracted from a WebDataset tar archive."""

    json_data: dict
    audio_bytes: bytes
    audio_name: str


def _split_json_audio_pair(name_a, bytes_a, name_b, bytes_b) -> TarSample:
    """Classify two tar members into a ``TarSample`` regardless of order."""
    if name_a.endswith('.json'):
        return TarSample(json.loads(bytes_a), bytes_b, name_b)
    if name_b.endswith('.json'):
        return TarSample(json.loads(bytes_b), bytes_a, name_a)
    raise ValueError(f"Expected one .json member in tar sample pair, got: {name_a}, {name_b}")


class IndexedTarSampleReader:
    """
    Random access to WebDataset tar samples (``N.json`` + ``N.<audio>``) via an index file.
    Index format is identical to ``IndexedJSONLReader``: little-endian uint64 offsets,
    optionally followed by a sentinel equal to the tar file size.
    """

    def __init__(self, tar_path: str | Path, idx_path: str | Path | None = None):
        self.data_path = str(tar_path)
        self.offsets, self._len = _load_index(self.data_path, str(idx_path) if idx_path else None)
        self._data_size = int(self.offsets[-1])
        self._validate_index()

    def _validate_index(self):
        """Tar-specific validation: check that indexed offsets point to valid tar headers."""
        if self._len == 0:
            return
        # Validate first offset is a valid tar header.
        self._check_offset_is_tar_header(int(self.offsets[0]), label="first")
        # Strip trailing sentinels: some tools store the offset of the
        # end-of-archive zero-block marker as a sentinel instead of the
        # file size (which _load_index already handles).
        while self._len > 0:
            last = int(self.offsets[self._len - 1])
            with open(self.data_path, 'rb') as f:
                f.seek(last)
                buf = f.read(_TAR_BLOCK_SIZE)
            if len(buf) < _TAR_BLOCK_SIZE or buf == _TAR_ZERO_BLOCK:
                self._len -= 1
            else:
                break

    def _check_offset_is_tar_header(self, offset: int, label: str = ""):
        with open(self.data_path, 'rb') as f:
            f.seek(offset)
            buf = f.read(_TAR_BLOCK_SIZE)
        if len(buf) < _TAR_BLOCK_SIZE:
            raise ValueError(
                f"Tar index for {self.data_path}: {label} offset {offset} "
                f"is too close to EOF (file size {self._data_size})."
            )
        if buf == _TAR_ZERO_BLOCK:
            raise ValueError(
                f"Tar index for {self.data_path}: {label} offset {offset} "
                f"points to a zero block (end-of-archive marker), not a tar header. "
                f"The .idx file may have been created by an incompatible tool "
                f"or for a different file."
            )
        try:
            tarfile.TarInfo.frombuf(buf, tarfile.ENCODING, "surrogateescape")
        except tarfile.TarError as e:
            raise ValueError(
                f"Tar index for {self.data_path}: {label} offset {offset} "
                f"does not point to a valid tar header: {e}. "
                f"The .idx file may have been created by an incompatible tool "
                f"(e.g. has a binary header or stores per-member offsets) "
                f"or for a different file."
            ) from e

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        idx = _resolve_idx(idx, self._len)
        offset = int(self.offsets[idx])
        with open(self.data_path, 'rb') as f:
            f.seek(offset)
            try:
                name_a, bytes_a = _read_tar_member(f)
            except (EOFError, tarfile.TarError) as e:
                raise type(e)(
                    f"{e} — reading first member of sample {idx}/{self._len} "
                    f"at offset {offset} in {self.data_path} "
                    f"(file size {self._data_size})"
                ) from e
            try:
                name_b, bytes_b = _read_tar_member(f)
            except (EOFError, tarfile.TarError) as e:
                raise type(e)(
                    f"{e} — reading second member of sample {idx}/{self._len} "
                    f"(first member was '{name_a}', {len(bytes_a)} bytes) "
                    f"at offset {offset} in {self.data_path} "
                    f"(file size {self._data_size})"
                ) from e
        return _split_json_audio_pair(name_a, bytes_a, name_b, bytes_b)


class IndexedTarMemberReader:
    """
    Random access to a NeMo-style tar archive that stores **one regular member
    per sample** (e.g. ``<cut_id>.flac`` per line of an external NeMo manifest).

    Uses the same ``.idx`` format as :class:`IndexedJSONLReader` and
    :class:`IndexedTarSampleReader`: little-endian uint64 byte offsets, with
    a sentinel equal to the tar file size at the end. Each entry points at
    one tar header, and the corresponding payload starts ``512`` bytes later.

    Two access patterns:

    * Positional: ``reader[idx]`` returns ``(member_name, payload_bytes)``.
    * Name-keyed: ``reader.get(name)`` returns just the payload bytes. The
      name → position map is built lazily on first use by walking the tar
      headers (no payload reads), then cached for subsequent calls.
    """

    def __init__(
        self,
        tar_path: str | Path,
        idx_path: str | Path | None = None,
        auto_create_index: bool = True,
    ):
        self.data_path = str(tar_path)
        resolved_idx = str(idx_path) if idx_path else self.data_path + ".idx"
        if auto_create_index and not os.path.exists(resolved_idx):
            create_tar_index(self.data_path, resolved_idx)
        self.offsets, self._len = _load_index(self.data_path, resolved_idx)
        self._fh = None
        self._name_to_idx: dict[str, int] | None = None

    def _ensure_open(self):
        if self._fh is None:
            self._fh = open(self.data_path, "rb")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_fh"] = None  # file handles are not picklable
        return s

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[str, bytes]:
        idx = _resolve_idx(idx, self._len)
        offset = int(self.offsets[idx])
        self._ensure_open()
        self._fh.seek(offset)
        try:
            name, data = _read_tar_member(self._fh)
        except (EOFError, tarfile.TarError) as e:
            raise type(e)(
                f"{e} — reading sample {idx}/{self._len} at offset {offset} "
                f"in {self.data_path}"
            ) from e
        return name, data

    def _build_name_index(self) -> dict[str, int]:
        """Walk the tar headers once to build a name → sample-index map.

        Reads only the 512-byte tar headers (no payloads), so this is
        relatively cheap even on remote storage. Done lazily on first
        :meth:`get` call.

        ``tar.add`` writes a PAX extended header (``@PaxHeader``) before any
        member with a long path or extended attributes. We skip those and
        record the *regular* file's name at each indexed offset.
        """
        name_to_idx: dict[str, int] = {}
        self._ensure_open()
        for i in range(self._len):
            self._fh.seek(int(self.offsets[i]))
            while True:
                header = self._fh.read(_TAR_BLOCK_SIZE)
                if len(header) < _TAR_BLOCK_SIZE or header == _TAR_ZERO_BLOCK:
                    break
                info = tarfile.TarInfo.frombuf(
                    header, tarfile.ENCODING, "surrogateescape"
                )
                if info.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    name_to_idx[info.name] = i
                    break
                # Skip non-regular member (PAX/GNU long-name) data + padding.
                size_blocks = -(-info.size // _TAR_BLOCK_SIZE) * _TAR_BLOCK_SIZE
                self._fh.seek(size_blocks, 1)
        return name_to_idx

    def get(self, name: str) -> bytes:
        """Return the payload bytes of the tar member named ``name``."""
        if self._name_to_idx is None:
            self._name_to_idx = self._build_name_index()
        try:
            idx = self._name_to_idx[name]
        except KeyError as e:
            raise KeyError(
                f"Tar {self.data_path} has no member named '{name}'. "
                f"The .idx may be stale or the manifest is referencing a "
                f"different tar."
            ) from e
        _, data = self[idx]
        return data

    def __contains__(self, name: str) -> bool:
        if self._name_to_idx is None:
            self._name_to_idx = self._build_name_index()
        return name in self._name_to_idx


def _read_tar_member(f):
    """Read the next regular-file tar member, skipping non-regular entries
    (PAX headers, GNU long-name headers, directory entries, etc.).

    We read tar headers manually instead of using ``tarfile.open()`` because
    the stdlib ``tarfile`` module does not support random-access seeks into the
    middle of an archive — it always reads sequentially from the start.
    By parsing individual headers via ``TarInfo.frombuf`` we can seek to an
    arbitrary byte offset and read just the members we need in O(1).
    """
    while True:
        header_buf = f.read(_TAR_BLOCK_SIZE)
        if len(header_buf) < _TAR_BLOCK_SIZE or header_buf == _TAR_ZERO_BLOCK:
            raise EOFError("End of tar archive or unexpected EOF")
        info = tarfile.TarInfo.frombuf(header_buf, tarfile.ENCODING, "surrogateescape")
        data = f.read(info.size)
        if len(data) < info.size:
            raise EOFError("Unexpected end of tar file while reading data")
        remainder = info.size % _TAR_BLOCK_SIZE
        if remainder:
            f.seek(_TAR_BLOCK_SIZE - remainder, 1)
        if info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            continue
        return info.name, data


def create_index(jsonl_path, idx_path):
    """
    Creates a raw binary index file compatible with Megatron-Energon (CrudeJsonlDataset).

    Format: sequence of little-endian uint64 values
    ``[Offset_0, Offset_1, ..., Offset_N, File_Size]``

    Written atomically (tmp + ``os.replace``) so concurrent writers can't
    observe a half-written ``.idx``.
    """
    # Flush the write buffer every 8 MiB to limit memory usage on large files.
    flush_threshold = 8 * 1024 * 1024
    tmp_path = f"{idx_path}.tmp.{os.getpid()}"
    with open(jsonl_path, 'rb') as f_in, open(tmp_path, 'wb') as f_out:
        current_offset = 0
        write_buffer = bytearray()
        write_buffer.extend(struct.pack('<Q', current_offset))
        for line in f_in:
            current_offset += len(line)
            write_buffer.extend(struct.pack('<Q', current_offset))
            if len(write_buffer) > flush_threshold:
                f_out.write(write_buffer)
                write_buffer.clear()
        if write_buffer:
            f_out.write(write_buffer)
    os.replace(tmp_path, idx_path)


def create_tar_index(tar_path, idx_path):
    """
    Creates a raw binary index file for a WebDataset tar archive.
    Stores the byte offset of the first member of each sample (grouped by basename),
    followed by a sentinel equal to the tar file size.
    Format is identical to :func:`create_index`.

    Written atomically: data is staged in a per-process temp file next to
    ``idx_path`` and then ``os.replace()``-d into place, so concurrent writers
    can't observe a half-written ``.idx``.
    """
    offsets = []
    prev_stem = None
    with tarfile.open(tar_path, 'r:') as tar:
        for member in tar:
            if not member.isreg():
                continue
            stem = Path(member.name).stem
            if stem != prev_stem:
                offsets.append(member.offset)
                prev_stem = stem
    tmp_path = f"{idx_path}.tmp.{os.getpid()}"
    with open(tmp_path, 'wb') as f:
        buf = bytearray()
        for off in offsets:
            buf.extend(struct.pack('<Q', off))
        buf.extend(struct.pack('<Q', os.path.getsize(tar_path)))
        f.write(buf)
    os.replace(tmp_path, idx_path)
