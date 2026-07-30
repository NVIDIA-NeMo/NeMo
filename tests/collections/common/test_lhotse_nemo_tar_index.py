import io
import struct
import tarfile

from nemo.collections.common.data.lhotse.indexed_adapters import create_tar_index


def test_nemo_tar_index_sentinel_includes_trailing_record_padding(tmp_path):
    tar_path = tmp_path / "data.tar"
    with tarfile.open(tar_path, "w") as archive:
        payload = b"sample"
        info = tarfile.TarInfo("sample.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with tar_path.open("ab") as stream:
        stream.write(b"\0" * 10240)

    idx_path = tmp_path / "data.tar.idx"
    create_tar_index(tar_path, idx_path)

    with idx_path.open("rb") as stream:
        stream.seek(-8, io.SEEK_END)
        (sentinel,) = struct.unpack("<Q", stream.read(8))
    assert sentinel == tar_path.stat().st_size
