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
"""
Build O(1)-restore index sidecars for an arbitrary NeMo Lhotse ``input_cfg``.

Walks a NeMo dataloading config (``input_cfg`` YAML, including nested ``group``
entries and per-entry YAML references), discovers every JSONL/tar file an
indexed dataloader will need, and creates the corresponding ``.idx`` sidecars
next to each data file.

Two tar layouts are dispatched correctly:

* NeMo tarred audio (one regular member per sample, name-keyed) — uses
  ``nemo.collections.common.data.lhotse.indexed_adapters.create_tar_index``
  which records one offset per *basename group*.
* WebDataset/Shar tars (json + payload pairs) — uses
  ``lhotse.indexing.create_tar_index`` which records one offset per *member
  pair*.

Local files and remote URIs are both supported via lhotse's ``open_best``
(which routes to ``smart_open`` / AIStore SDK when available). The ``.idx`` is
written next to its source path, so the storage backend must accept writes at
that location — for read-only object stores, materialize the data locally
first or pre-build indexes at upload time.

Examples::

    # Build indexes for everything referenced by an input_cfg.yaml.
    python scripts/dataloading/build_indexes.py path/to/input_cfg.yaml

    # Multiple configs at once.
    python scripts/dataloading/build_indexes.py train.yaml validation.yaml

    # Show what would be built without writing anything.
    python scripts/dataloading/build_indexes.py --dry-run path/to/input_cfg.yaml

    # Rebuild even when an .idx already exists; parallelize across 16 workers.
    python scripts/dataloading/build_indexes.py --force --workers 16 path/to/input_cfg.yaml
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import click
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.common.data.lhotse.indexed_adapters import (
    create_tar_index as create_nemo_tar_index,
)
from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths


# --------------------------------------------------------------------------- #
# Tar layout taxonomy.
# --------------------------------------------------------------------------- #
# NEMO_TAR  — one regular member per sample, indexed by basename. Used by
#             nemo / nemo_tarred / multimodal_conversation / share_gpt audio
#             tars (read via IndexedTarMemberReader).
# WDS_TAR   — WebDataset-style: each sample is a pair of consecutive members
#             (e.g. {N}.json + {N}.<audio>). Used by lhotse_shar tars and
#             share_gpt_webdataset tars (read via IndexedTarSampleReader).
NEMO_TAR = "nemo_tar"
WDS_TAR = "wds_tar"
JSONL = "jsonl"


@dataclass(frozen=True)
class IndexJob:
    path: str
    kind: str  # one of {JSONL, NEMO_TAR, WDS_TAR}

    def idx_path(self) -> str:
        return self.path + ".idx"


# --------------------------------------------------------------------------- #
# Path discovery.
# --------------------------------------------------------------------------- #


def _as_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, (list, tuple, ListConfig)):
        return list(val)
    return [val]


def _flatten_path_spec(spec) -> list[str]:
    """
    NeMo's manifest_filepath / tarred_audio_filepaths accept several layouts:
      str, list[str], list[list[str]], list[tuple[str, weight]], ...
    Flatten any of those into a list of plain string paths.
    """
    out: list[str] = []
    for item in _as_list(spec):
        if isinstance(item, (str, Path)):
            out.append(str(item))
        elif isinstance(item, (list, tuple, ListConfig)):
            # [path] or [path, weight] or [[path], [path], ...]
            head = item[0]
            if isinstance(head, (str, Path)):
                out.append(str(head))
            else:
                out.extend(_flatten_path_spec(item))
    return out


def _expand_jsonl(spec) -> list[str]:
    return [p for raw in _flatten_path_spec(spec) for p in expand_sharded_filepaths(raw)]


def _expand_tars(spec) -> list[str]:
    return [p for raw in _flatten_path_spec(spec) for p in expand_sharded_filepaths(raw)]


def _resolve_input_cfg(val) -> ListConfig | None:
    """``input_cfg`` may be inline or a path to a YAML file. Materialize it."""
    if isinstance(val, (list, ListConfig)):
        return val
    if isinstance(val, (str, Path)):
        return OmegaConf.load(str(val))
    return None


# Types that don't read any data themselves — they delegate to
# ``read_cutset_from_config(config)`` and accept *any* underlying source's keys
# (``cuts_path``, ``shar_path``, ``manifest_filepath`` [+ ``tarred_audio_filepaths``],
# nested ``input_cfg``, …). Treat them as transparent passthroughs.
_TRANSFORM_TYPES = frozenset({
    "lhotse_as_conversation",
    "sqa_as_conversation",
    "s2s_as_conversation",
    "s2s_duplex_overlap_as_s2s_duplex",
    "s2s_duplex_reverse_role",
    "lhotse_magpietts_data_as_continuation",
    "nemo_tarred_to_duplex",
})

# Types that index nothing on their own.
_NO_INDEX_TYPES = frozenset({"txt", "txt_pair", "parquet", "multi_speaker_simulator"})


def _discover_keys(entry, jobs: list[IndexJob]) -> None:
    """
    Key-based dispatch: emit IndexJobs based on which underlying-source keys
    are present, regardless of ``type``. Used for transform types that
    delegate to ``read_cutset_from_config``, and as the inner step for
    concrete types that name them directly.
    """
    if (cuts_path := entry.get("cuts_path")) is not None:
        for p in _expand_jsonl(cuts_path):
            jobs.append(IndexJob(p, JSONL))
    if (shar_path := entry.get("shar_path")) is not None:
        _discover_shar(shar_path, jobs)
    if (mfp := entry.get("manifest_filepath")) is not None:
        for p in _expand_jsonl(mfp):
            jobs.append(IndexJob(p, JSONL))
        for p in _expand_tars(entry.get("tarred_audio_filepaths")):
            jobs.append(IndexJob(p, NEMO_TAR))
    if (paths := entry.get("paths")) is not None:
        for p in _expand_jsonl(paths):
            jobs.append(IndexJob(p, JSONL))
    if (sub := _resolve_input_cfg(entry.get("input_cfg"))) is not None:
        discover(sub, jobs)


def discover(entry, jobs: list[IndexJob]) -> None:
    """Walk one entry of an ``input_cfg`` and append every required IndexJob."""
    if isinstance(entry, (list, ListConfig)):
        for sub in entry:
            discover(sub, jobs)
        return
    if not isinstance(entry, (dict, DictConfig)):
        return

    typ = entry.get("type")
    if typ is None:
        # Top-level wrapper (``input_cfg: [...]``) — recurse into every value.
        for v in entry.values():
            discover(v, jobs)
        return

    if typ in _NO_INDEX_TYPES:
        return

    if typ == "group" or typ in _TRANSFORM_TYPES:
        # Group and transform passthroughs: dispatch by keys.
        _discover_keys(entry, jobs)
        return

    if typ in ("nemo", "nemo_tarred", "multimodal_conversation", "share_gpt"):
        for p in _expand_jsonl(entry.get("manifest_filepath")):
            jobs.append(IndexJob(p, JSONL))
        for p in _expand_tars(entry.get("tarred_audio_filepaths")):
            jobs.append(IndexJob(p, NEMO_TAR))
        return

    if typ == "share_gpt_webdataset":
        # Layout: data_dir/shard-N.tar [+ optional shard-N.tar.idx, manifest jsonl].
        data_dir = entry.get("data_dir")
        if data_dir is None:
            return
        for ext, kind in ((".tar", WDS_TAR), (".jsonl", JSONL)):
            for p in sorted(Path(data_dir).glob(f"*{ext}")):
                jobs.append(IndexJob(str(p), kind))
        return

    if typ == "lhotse":
        if (cuts_path := entry.get("cuts_path")) is not None:
            for p in _expand_jsonl(cuts_path):
                jobs.append(IndexJob(p, JSONL))
        if (shar_path := entry.get("shar_path")) is not None:
            _discover_shar(shar_path, jobs)
        return

    if typ == "lhotse_shar":
        _discover_shar(entry.get("shar_path"), jobs)
        return

    if typ == "txt_jsonl":
        for p in _expand_jsonl(entry.get("paths")):
            jobs.append(IndexJob(p, JSONL))
        return

    # Unknown type — nothing to do.
    return


def _discover_shar(shar_path, jobs: list[IndexJob]) -> None:
    """Index every uncompressed JSONL/tar shard inside one or more Shar dirs."""
    if shar_path is None:
        return
    if isinstance(shar_path, (str, Path)):
        candidates = [shar_path]
    elif isinstance(shar_path, (list, ListConfig)):
        candidates = []
        for item in shar_path:
            if isinstance(item, (str, Path)):
                candidates.append(item)
            elif isinstance(item, (list, tuple, ListConfig)) and item:
                candidates.append(item[0])  # [path, weight] form
    elif isinstance(shar_path, (dict, DictConfig)):
        # {field: [shard, ...]} layout — index every shard in every field.
        for v in shar_path.values():
            for raw in _flatten_path_spec(v):
                for p in expand_sharded_filepaths(raw):
                    if p.endswith(".jsonl"):
                        jobs.append(IndexJob(p, JSONL))
                    elif p.endswith(".tar"):
                        jobs.append(IndexJob(p, WDS_TAR))
        return
    else:
        return

    for d in candidates:
        d = Path(str(d))
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix == ".jsonl":
                jobs.append(IndexJob(str(p), JSONL))
            elif p.suffix == ".tar":
                jobs.append(IndexJob(str(p), WDS_TAR))


# --------------------------------------------------------------------------- #
# Index builders.
# --------------------------------------------------------------------------- #


def _build_one(job: IndexJob) -> tuple[IndexJob, str]:
    """Run the right indexer for *job*. Returns (job, status)."""
    from lhotse.indexing import create_jsonl_index, create_tar_index as create_wds_tar_index

    builders: dict[str, Callable[[str], object]] = {
        JSONL: create_jsonl_index,
        WDS_TAR: create_wds_tar_index,
        NEMO_TAR: create_nemo_tar_index,
    }
    builder = builders[job.kind]
    if job.kind == NEMO_TAR:
        # NeMo's create_tar_index has a (tar_path, idx_path) signature.
        builder(job.path, job.idx_path())
    else:
        builder(job.path)
    return job, "built"


def _is_indexed(job: IndexJob) -> bool:
    """True if a non-empty .idx already exists locally."""
    p = Path(job.idx_path())
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #


@click.command(context_settings={"show_default": True})
@click.argument("input_cfgs", type=click.Path(exists=True, dir_okay=False), nargs=-1, required=True)
@click.option("--force", is_flag=True, help="Rebuild .idx files even if they already exist.")
@click.option("--workers", type=int, default=4, help="Number of parallel index builders.")
@click.option("--dry-run", is_flag=True, help="List the jobs without writing anything.")
def main(input_cfgs: tuple[str, ...], force: bool, workers: int, dry_run: bool):
    """
    Build .idx sidecars for every JSONL/tar referenced by INPUT_CFGS.

    INPUT_CFGS are NeMo Lhotse dataloading configs (``input_cfg`` YAML).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    jobs: list[IndexJob] = []
    for cfg_path in input_cfgs:
        cfg = OmegaConf.load(cfg_path)
        discover(cfg, jobs)

    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[IndexJob] = []
    for j in jobs:
        key = (j.path, j.kind)
        if key not in seen:
            seen.add(key)
            unique.append(j)

    todo = unique if force else [j for j in unique if not _is_indexed(j)]
    skipped = len(unique) - len(todo)

    logging.info("Discovered %d files (%d already indexed, %d to build).",
                 len(unique), skipped, len(todo))

    if dry_run or not todo:
        for j in todo:
            logging.info("  [%s] %s", j.kind, j.path)
        return

    failures: list[tuple[IndexJob, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(_build_one, j): j for j in todo}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                _, status = fut.result()
                logging.info("  [%s] %s -> %s", status, j.kind, j.path)
            except BaseException as e:  # noqa: BLE001 — surface any failure
                failures.append((j, e))
                logging.error("  [FAIL] %s %s: %s", j.kind, j.path, e)

    if failures:
        logging.error("\n%d index build(s) failed:", len(failures))
        for j, e in failures:
            logging.error("  %s (%s): %s", j.path, j.kind, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
