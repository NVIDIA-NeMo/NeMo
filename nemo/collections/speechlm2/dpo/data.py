# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Finite direct-Lhotse preference data contract for SpeechLM2 DPO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.distributed as dist
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    source_id: str
    prompt: str | dict[str, str]
    chosen: str
    rejected: str
    audio: torch.Tensor
    active: bool


@dataclass(frozen=True)
class PreferenceBatch:
    global_step: int
    dpo_pass: int
    source_shard: int
    pairs: tuple[PreferencePair, ...]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _preference(cut: Any) -> dict[str, Any]:
    supervisions = list(getattr(cut, "supervisions", ()) or ())
    if len(supervisions) != 1:
        raise ValueError(f"{getattr(cut, 'id', '<unknown>')}: expected exactly one supervision")
    value = dict(getattr(supervisions[0], "custom", {}) or {}).get("dpo_preference")
    if not isinstance(value, dict):
        raise ValueError(f"{cut.id}: missing supervision.custom.dpo_preference")
    return value


def _validate(cut: Any, value: dict[str, Any]) -> None:
    required = ("record_id", "source_id", "chosen", "rejected", "audio_sha256")
    missing = [key for key in required if not str(value.get(key, "")).strip()]
    prompt = value.get("prompt")
    if isinstance(prompt, str):
        prompt_ok = bool(prompt.strip())
    elif isinstance(prompt, Mapping):
        prompt_ok = isinstance(prompt.get("system", ""), str) and isinstance(prompt.get("user"), str) and bool(prompt["user"].strip())
    else:
        prompt_ok = False
    if not prompt_ok:
        missing.append("prompt")
    if missing:
        raise ValueError(f"{cut.id}: missing DPO fields {missing}")
    checks = {
        "cut_id": str(cut.id) == str(value["record_id"]),
        "ordered": str(value["chosen"]).strip() != str(value["rejected"]).strip(),
        "same_audio_control": value.get("same_audio_rejected_control") is True,
        "hard_valid": value.get("hard_invalid") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"{cut.id}: invalid DPO pair contract {checks}")
    recording = getattr(cut, "recording", None)
    sources = list(getattr(recording, "sources", ()) or ())
    audio_id = str(value["audio_sha256"])
    if (
        recording is None
        or str(getattr(recording, "id", "")) != audio_id
        or len(sources) != 1
        or Path(str(getattr(sources[0], "source", ""))).stem != audio_id
    ):
        raise ValueError(f"{cut.id}: Lhotse recording does not bind the declared audio identity")


class FiniteLhotsePreferenceCorpus:
    """Read each direct Lhotse preference cut exactly once on its owning rank."""

    def __init__(self, cfg: Any) -> None:
        self.path = Path(str(cfg.cuts_path))
        self.expected_rows = int(cfg.expected_rows)
        self.pairs_per_update = int(cfg.pairs_per_update)
        self.source_shards = int(cfg.source_shards)
        self.world_size = int(cfg.world_size)
        self.shuffle = bool(cfg.shuffle)
        self.cycle = bool(cfg.cycle)
        if self.shuffle or self.cycle:
            raise ValueError("DPO finite Lhotse contract requires shuffle=false and cycle=false")
        if self.expected_rows != self.pairs_per_update * self.source_shards:
            raise ValueError("expected_rows must equal pairs_per_update * source_shards")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

    def load_rank_shards(self, rank: int) -> tuple[list[list[PreferencePair]], dict[str, Any]]:
        from lhotse import CutSet

        cuts = CutSet.from_file(self.path)
        local: list[list[PreferencePair]] = [[] for _ in range(self.source_shards)]
        ids: list[str] = []
        reads = 0
        for position, cut in enumerate(cuts):
            if position >= self.expected_rows:
                raise ValueError(f"finite Lhotse source has more than {self.expected_rows} rows")
            value = _preference(cut)
            _validate(cut, value)
            pair_id = str(value["record_id"])
            ids.append(pair_id)
            reads += 1
            if position % self.world_size != rank:
                continue
            waveform = np.asarray(cut.load_audio(), dtype=np.float32)
            if waveform.ndim == 2:
                waveform = waveform[0] if waveform.shape[0] == 1 else waveform.mean(axis=0)
            waveform = np.ascontiguousarray(waveform.reshape(-1), dtype=np.float32)
            if not waveform.size or not np.isfinite(waveform).all():
                raise ValueError(f"{pair_id}: invalid decoded waveform")
            # Both completions below receive this one decoded tensor.  The
            # manifest's audio digest identifies the staged recording (checked
            # in ``_validate``).  It is deliberately not recomputed after WAV
            # decoding: the historical staging digest predates PCM rounding.
            local[position // self.pairs_per_update].append(
                PreferencePair(
                    pair_id=pair_id,
                    source_id=str(value["source_id"]),
                    prompt=value["prompt"] if isinstance(value["prompt"], str) else dict(value["prompt"]),
                    chosen=str(value["chosen"]),
                    rejected=str(value["rejected"]),
                    audio=torch.from_numpy(waveform),
                    active=True,
                )
            )
        if reads != self.expected_rows or len(set(ids)) != self.expected_rows:
            raise ValueError(f"finite Lhotse corpus mismatch: reads={reads} unique_ids={len(set(ids))}")
        expected_rank_counts = [len(range(rank, self.pairs_per_update, self.world_size)) for rank in range(self.world_size)]
        wanted = expected_rank_counts[rank]
        if any(len(shard) != wanted for shard in local):
            raise ValueError(f"rank {rank} did not receive the expected direct-Lhotse slots")
        padded: list[list[PreferencePair]] = []
        for shard in local:
            entries = list(shard)
            while len(entries) < (self.pairs_per_update + self.world_size - 1) // self.world_size:
                last = entries[-1]
                entries.append(PreferencePair(**{**last.__dict__, "active": False}))
            padded.append(entries)
        return padded, {
            "cuts_path": str(self.path), "direct": True, "shuffle": False, "cycle": False, "pair_reads": reads,
            "exhausted_exactly": True, "active_pair_ids_sha256": _stable_hash(ids), "source_shards": self.source_shards,
            "pairs_per_update": self.pairs_per_update, "rank": rank, "rank_local_slots": len(padded[0]),
        }


class _Schedule(IterableDataset):
    def __init__(self, shards: list[list[PreferencePair]]) -> None:
        super().__init__()
        self.shards = shards

    def __iter__(self) -> Iterator[PreferenceBatch]:
        for dpo_pass in (1, 2):
            for source_shard, pairs in enumerate(self.shards, 1):
                yield PreferenceBatch(
                    global_step=(dpo_pass - 1) * len(self.shards) + source_shard,
                    dpo_pass=dpo_pass,
                    source_shard=source_shard,
                    pairs=tuple(pairs),
                )


class FiniteLhotsePreferenceDataModule(LightningDataModule):
    """A finite, ordered, noncycling two-pass DPO schedule over direct Lhotse."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.corpus = FiniteLhotsePreferenceCorpus(cfg)
        self.shards: list[list[PreferencePair]] | None = None
        self.receipt: dict[str, Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.shards is not None:
            return
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if world_size != self.corpus.world_size:
            raise RuntimeError(f"DPO config world_size={self.corpus.world_size}, runtime={world_size}")
        self.shards, self.receipt = self.corpus.load_rank_shards(rank)

    def train_dataloader(self) -> DataLoader:
        if self.shards is None:
            raise RuntimeError("call setup before train_dataloader")
        return DataLoader(_Schedule(self.shards), batch_size=None, num_workers=0)

    def local_shards(self) -> list[list[PreferencePair]]:
        if self.shards is None:
            raise RuntimeError("data module has not been set up")
        return self.shards
