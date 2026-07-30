#!/usr/bin/env python
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
Convert an existing NeMo/Lhotse input_cfg and its ``.idx`` sidecars to one
dataset-level ``.idxpack``.

The input YAML may contain nested groups and transform wrappers. This initial
integration packs the formats consumed by the indexed runtime: native NeMo
manifests/tars, Nemotron text JSONL/tars, and ShareGPT JSONL manifests. No
source manifest or tar is rescanned: the command consumes existing ``.idx``
files, normally from ``--indexes-root``.

Example::

    python scripts/dataloading/convert_indexes_to_idxpack.py \
        --indexes-root /lustre/.../indexes_mirror \
        --output /lustre/.../packs/granary-v2.idxpack \
        /path/to/granary_v2_packed-lustre.yaml
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Optional

import click
from lhotse.index_pack import IndexPack, IndexPackCollectionSpec, write_index_pack
from omegaconf import DictConfig, ListConfig, OmegaConf
from scripts.dataloading.build_indexes import (
    _NO_INDEX_TYPES,
    _TRANSFORM_TYPES,
    JSONL,
    NEMO_TAR,
    _expand_jsonl,
    _expand_tars,
    _flatten_path_spec,
    _resolve_input_cfg,
)


def _add_collection(
    collections: list[IndexPackCollectionSpec],
    *,
    role: str,
    kind: str,
    source_spec,
    paths,
) -> None:
    paths = tuple(map(str, paths))
    if not paths:
        return
    candidate = IndexPackCollectionSpec(
        role=role,
        kind=kind,
        source_spec=source_spec,
        paths=paths,
    )
    for existing in collections:
        if existing.key != candidate.key:
            continue
        if existing.paths != candidate.paths:
            raise ValueError(
                f"Collection-key collision for role={role!r}, kind={kind!r}, " f"source_spec={source_spec!r}"
            )
        return
    collections.append(candidate)


def _discover_paths_collections(
    raw_paths,
    collections: list[IndexPackCollectionSpec],
) -> None:
    jsonls = []
    tars = []
    for raw in _flatten_path_spec(raw_paths):
        for expanded in _expand_jsonl(raw):
            path = Path(expanded)
            if path.is_dir():
                tars.extend(map(str, sorted(path.rglob("*.tar"))))
            elif path.suffix == ".tar":
                tars.append(str(path))
            else:
                jsonls.append(str(path))
    if jsonls and tars:
        raise ValueError(
            "Packed Nemotron text paths must be homogeneous. Split mixed "
            "JSONL/tar paths into separate dataset entries."
        )
    _add_collection(
        collections,
        role="paths",
        kind=NEMO_TAR if tars else JSONL,
        source_spec=raw_paths,
        paths=jsonls or tars,
    )


def _require_scalar_spec(value, field: str) -> None:
    if not isinstance(value, (str, Path)):
        raise ValueError(
            f"Packed {field} must be a string/Path (brace expansion is " "supported); list forms are not supported."
        )


def _shard_number(path: str) -> int | None:
    matches = re.findall(r"\d+", Path(path).stem)
    return int(matches[-1]) if matches else None


def _validate_native_pair(manifests: list[str], tars: list[str]) -> None:
    if len(manifests) != len(tars):
        raise ValueError(
            "Packed native NeMo data requires one manifest per tar shard: "
            f"manifests={len(manifests)}, tars={len(tars)}"
        )
    if len(manifests) < 2:
        return
    manifest_ids = [_shard_number(path) for path in manifests]
    tar_ids = [_shard_number(path) for path in tars]
    if None in manifest_ids or None in tar_ids:
        raise ValueError(
            "Cannot verify native NeMo manifest/tar shard identity from file " "names; use numbered shard names."
        )
    if manifest_ids != tar_ids:
        raise ValueError(
            "Native NeMo manifest/tar shards are not positionally aligned: "
            f"manifest ids={manifest_ids}, tar ids={tar_ids}"
        )


def discover_pack_collections(
    entry,
    collections: Optional[list[IndexPackCollectionSpec]] = None,
) -> list[IndexPackCollectionSpec]:
    """Discover ordered, runtime-addressable collections in one input_cfg."""
    if collections is None:
        collections = []
    if isinstance(entry, (list, ListConfig)):
        for item in entry:
            discover_pack_collections(item, collections)
        return collections
    if not isinstance(entry, (dict, DictConfig)):
        return collections

    typ = entry.get("type")
    if typ in _NO_INDEX_TYPES:
        return collections

    if typ is None:
        for value in entry.values():
            discover_pack_collections(value, collections)
        return collections

    if typ == "group":
        sub = _resolve_input_cfg(entry.get("input_cfg"))
        if sub is not None:
            discover_pack_collections(sub, collections)
        return collections

    if typ in _TRANSFORM_TYPES:
        sub = _resolve_input_cfg(entry.get("input_cfg"))
        if sub is not None:
            discover_pack_collections(sub, collections)
            return collections
        if entry.get("manifest_filepath") is None:
            return collections

    supported = {
        "nemo",
        "nemo_tarred",
        "nemotron_text_converation",
        "share_gpt",
        *_TRANSFORM_TYPES,
    }
    if typ not in supported:
        raise NotImplementedError(f"idxpack conversion does not support dataset type {typ!r}.")

    if typ in {"nemo", "nemo_tarred", "share_gpt", *_TRANSFORM_TYPES} and entry.get("manifest_filepath") is not None:
        raw = entry.get("manifest_filepath")
        _require_scalar_spec(raw, "manifest_filepath")
        manifests = _expand_jsonl(raw)
        _add_collection(
            collections,
            role="manifest",
            kind=JSONL,
            source_spec=raw,
            paths=manifests,
        )
        if entry.get("tarred_audio_filepaths") is not None:
            if typ == "share_gpt":
                raise NotImplementedError(
                    "Packed ShareGPT supports JSONL manifests with direct/remote "
                    "audio paths, not paired audio tar files."
                )
            raw_tars = entry.get("tarred_audio_filepaths")
            _require_scalar_spec(raw_tars, "tarred_audio_filepaths")
            tars = _expand_tars(raw_tars)
            _validate_native_pair(manifests, tars)
            _add_collection(
                collections,
                role="tar",
                kind=NEMO_TAR,
                source_spec=raw_tars,
                paths=tars,
            )

    if typ == "nemotron_text_converation":
        _discover_paths_collections(entry.get("paths"), collections)

    return collections


@click.command(context_settings={"show_default": True})
@click.argument("input_cfg", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--indexes-root",
    default=None,
    help="Root of the existing mirrored .idx sidecars. Omit for sidecars next to sources.",
)
@click.option("--overwrite", is_flag=True, help="Atomically replace an existing output pack.")
@click.option(
    "--native-tar-paths-only",
    is_flag=True,
    help=(
        "Store native NeMo tar shard names without copying tar-member offsets. "
        "Use for AIS URL-backed audio; manifest offsets remain fully packed."
    ),
)
@click.option("--dry-run", is_flag=True, help="Print discovered collections without writing.")
def main(
    input_cfg: str,
    output: str,
    indexes_root: Optional[str],
    overwrite: bool,
    native_tar_paths_only: bool,
    dry_run: bool,
) -> None:
    """Convert one INPUT_CFG dataset and its existing sidecars to one idxpack."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = OmegaConf.load(input_cfg)
    collections = discover_pack_collections(config)
    if native_tar_paths_only:
        collections = [
            (
                replace(collection, offsets_required=False)
                if collection.kind == NEMO_TAR and collection.role == "tar"
                else collection
            )
            for collection in collections
        ]
    num_paths = sum(len(collection.paths) for collection in collections)
    click.echo(f"Discovered {len(collections)} collections with {num_paths} ordered paths.")
    if dry_run:
        for collection in collections:
            click.echo(
                f"  role={collection.role} kind={collection.kind} "
                f"paths={len(collection.paths)} offsets={collection.offsets_required} key={collection.key.hex()}"
            )
        return
    write_index_pack(
        output,
        collections,
        indexes_root=indexes_root,
        overwrite=overwrite,
    )
    with IndexPack(output) as pack:
        click.echo(
            f"Wrote {output}: collections={pack.num_collections} "
            f"segments={pack.num_segments} layout={pack.layout_hash.hex()}"
        )


if __name__ == "__main__":
    main()
