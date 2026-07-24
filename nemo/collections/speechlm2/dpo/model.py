# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Native Lightning DPO model for finite same-audio SpeechLM preference pairs."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel

from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.speechlm2.dpo.data import PreferenceBatch, PreferencePair
from nemo.collections.speechlm2.dpo.objective import dpo_pair_objective
from nemo.collections.speechlm2.dpo.surface import configure_partial_acoustic_surface, named_selected_parameters
from nemo.collections.speechlm2.models.salm_automodel import SALMAutomodel


@dataclass(frozen=True)
class _EncodedCompletion:
    input_ids: tuple[int, ...]
    completion_mask: tuple[bool, ...]
    answer_tokens: int


def _field(value: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _ids(value: Any) -> list[int]:
    return torch.as_tensor(value).detach().cpu().to(torch.long).reshape(-1).tolist()


def _dialog_turns(prompt: str | Mapping[str, str], completion: str, audio_tag: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create prompt-formatter turns without flattening structured system/user prompts."""

    if isinstance(prompt, str):
        system = None
        user_text = prompt
    elif isinstance(prompt, Mapping) and isinstance(prompt.get("system", ""), str) and isinstance(prompt.get("user"), str):
        system = str(prompt.get("system", ""))
        user_text = str(prompt["user"])
    else:
        raise ValueError("DPO prompt must be a nonempty string or {system, user} mapping")
    user_text = user_text if audio_tag in user_text else f"{audio_tag}\n{user_text}"
    context: list[dict[str, Any]] = []
    if system is not None:
        context.append({"role": "system", "slots": {"message": system}})
    context.append({"role": "user", "slots": {"message": user_text}})
    answered = [*context, {"role": "assistant", "slots": {"message": completion}}]
    empty_answer = [*context, {"role": "assistant", "slots": {"message": ""}}]
    return answered, empty_answer


def _encode_completion(formatter: Any, *, prompt: str | Mapping[str, str], completion: str, audio_tag: str, audio_tag_id: int) -> _EncodedCompletion:
    answered, empty_answer = _dialog_turns(prompt, completion, audio_tag)
    failures: list[str] = []
    for style, turns, prefix_turns in (("slots", answered, empty_answer),):
        try:
            try:
                encoded = formatter.encode_dialog(turns=turns, enable_thinking=False)
            except TypeError:
                encoded = formatter.encode_dialog(turns=turns)
            input_ids = _ids(_field(encoded, ("input_ids",)))
            raw_mask = _field(encoded, ("mask", "loss_mask", "answer_mask", "labels_mask"))
            mask = [] if raw_mask is None else [bool(x) for x in torch.as_tensor(raw_mask).detach().cpu().reshape(-1).tolist()]
            if len(mask) != len(input_ids) or not any(mask):
                try:
                    prefix = formatter.encode_dialog(turns=prefix_turns, enable_thinking=False)
                except TypeError:
                    prefix = formatter.encode_dialog(turns=prefix_turns)
                prefix_ids = _ids(_field(prefix, ("input_ids",)))
                shared = next((index for index, pair in enumerate(zip(input_ids, prefix_ids)) if pair[0] != pair[1]), min(len(input_ids), len(prefix_ids)))
                mask = [index >= shared for index in range(len(input_ids))]
            if not any(mask) or audio_tag_id not in input_ids:
                raise ValueError("empty completion mask or absent audio locator")
            return _EncodedCompletion(tuple(input_ids), tuple(mask), sum(mask))
        except Exception as error:  # noqa: BLE001
            failures.append(f"{style}:{type(error).__name__}:{error}")
    raise RuntimeError("prompt formatter failed to encode DPO completion: " + " | ".join(failures))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _local(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


class DPOSALMAutomodel(SALMAutomodel):
    """SALMAutomodel with standard manual-optimization DPO update semantics.

    Reference values are captured exactly once through the grad-enabled policy
    path before the first AdamW update, detached to FP32 scalars, and reused
    for both explicit ordered passes.  Manual optimization exists only to
    retain r2's bounded 55-pair FSDP accumulation; optimizer, backward,
    clipping, checkpoint, and distributed state remain native Lightning/PyTorch
    lifecycle operations.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.automatic_optimization = False
        self._surface = None
        self._references: dict[int, list[tuple[float, float]]] = {}
        self._encoded: dict[str, tuple[_EncodedCompletion, _EncodedCompletion]] = {}
        self._checkpoint_steps = {int(step) for step in self.cfg.dpo.checkpoint_steps}
        self._expected_updates = int(self.cfg.dpo.expected_updates)
        if not 1 <= self._expected_updates <= 2 * int(self.cfg.dpo.source_shards):
            raise ValueError("DPO expected_updates must be within the finite two-pass schedule")
        self._output_root = Path(str(self.cfg.dpo.output_root))
        self._initial_checkpoint = Path(str(self.cfg.dpo.source_checkpoint))
        self._metrics: list[dict[str, Any]] = []

    def configure_model(self, *args: Any, **kwargs: Any) -> None:
        super().configure_model(*args, **kwargs)
        if getattr(self, "_dpo_checkpoint_loaded", False):
            return
        from torch.distributed.checkpoint import load

        if not self._initial_checkpoint.joinpath(".metadata").is_file():
            raise FileNotFoundError(self._initial_checkpoint / ".metadata")
        state = {"state_dict": self.state_dict()}
        load(state, checkpoint_id=str(self._initial_checkpoint))
        incompatible = self.load_state_dict(state["state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict Hero2 model DCP load failed: {incompatible}")
        self._dpo_checkpoint_loaded = True

    def configure_optimizers(self):
        self._surface = configure_partial_acoustic_surface(self)
        dpo = self.cfg.dpo
        optimizer = torch.optim.AdamW(
            list(named_selected_parameters(self)),
            lr=float(dpo.learning_rate),
            betas=tuple(float(item) for item in dpo.optimizer.betas),
            eps=float(dpo.optimizer.eps),
            weight_decay=float(dpo.optimizer.weight_decay),
            foreach=bool(dpo.optimizer.foreach),
            fused=bool(dpo.optimizer.fused),
        )
        if len(optimizer.param_groups) != 1 or len(optimizer.param_groups[0]["params"]) != 269:
            raise RuntimeError("AdamW does not own the declared 269-tensor DPO surface")
        return optimizer

    def on_fit_start(self) -> None:
        super().on_fit_start()
        data_module = self.trainer.datamodule
        shards = data_module.local_shards()
        self._capture_initial_references(shards)
        if self.global_rank == 0:
            _write_json(
                self._output_root / "TRAJECTORY.json",
                {
                    "schema": "speechlm2.dpo.finite_lhotse.v1", "reference_path": "grad_enabled_policy_detached_once",
                    "lhotse": data_module.receipt, "global_updates": self._expected_updates, "explicit_passes": 2,
                    "checkpoint_steps": sorted(self._checkpoint_steps), "surface": self._surface.__dict__, "lora": False,
                },
            )

    def _encoded_pair(self, pair: PreferencePair) -> tuple[_EncodedCompletion, _EncodedCompletion]:
        if pair.pair_id not in self._encoded:
            formatter = PromptFormatter.resolve(self.cfg.prompt_format)(self.tokenizer)
            self._encoded[pair.pair_id] = (
                _encode_completion(formatter, prompt=pair.prompt, completion=pair.chosen, audio_tag=self.audio_locator_tag, audio_tag_id=int(self.audio_locator_tag_id)),
                _encode_completion(formatter, prompt=pair.prompt, completion=pair.rejected, audio_tag=self.audio_locator_tag, audio_tag_id=int(self.audio_locator_tag_id)),
            )
        return self._encoded[pair.pair_id]

    def _completion_logprob(self, encoded: _EncodedCompletion, audio: torch.Tensor) -> torch.Tensor:
        device = self.device
        batch = {
            "input_ids": torch.tensor([encoded.input_ids], dtype=torch.long, device=device),
            "loss_mask": torch.tensor([encoded.completion_mask], dtype=torch.bool, device=device),
            "audios": audio.reshape(1, -1).to(device=device, dtype=torch.float32),
            "audio_lens": torch.tensor([audio.numel()], dtype=torch.long, device=device),
        }
        # This code deliberately does not enter no_grad: reference capture must
        # exercise the same grad-enabled policy path as the DPO branch.
        prepared = self.prepare_inputs(batch)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
            outputs = self(prepared["input_embeds"], attention_mask=prepared["attention_mask"], **prepared.get("llm_kwargs", {}))
        targets = prepared["target_ids"]
        with loss_parallel():
            losses = F.cross_entropy(outputs["logits"].float().reshape(-1, outputs["logits"].size(-1)), targets.reshape(-1), reduction="none", ignore_index=-100).reshape(targets.shape)
        mask = targets != -100
        if not bool(mask.any()):
            raise RuntimeError("DPO completion contains no target tokens after native prepare_inputs")
        return -(losses * mask).sum()

    def _policy_pair(self, pair: PreferencePair) -> tuple[torch.Tensor, torch.Tensor]:
        chosen, rejected = self._encoded_pair(pair)
        return self._completion_logprob(chosen, pair.audio), self._completion_logprob(rejected, pair.audio)

    def _force_reshard(self) -> None:
        modules: list[torch.nn.Module] = []
        seen: set[int] = set()
        for _, module in self.named_modules():
            get_state = getattr(module, "_get_fsdp_state", None)
            if callable(get_state) and (group := getattr(get_state(), "_fsdp_param_group", None)) is not None and id(group) not in seen:
                seen.add(id(group))
                modules.append(module)
        for module in reversed(modules):
            module.reshard()

    def _capture_initial_references(self, shards: list[list[PreferencePair]]) -> None:
        self.eval()
        for source_shard, pairs in enumerate(shards, 1):
            values: list[tuple[float, float]] = []
            for pair in pairs:
                chosen, rejected = self._policy_pair(pair)
                if not chosen.requires_grad or not rejected.requires_grad:
                    raise RuntimeError("initial DPO reference was not captured through the grad-enabled policy path")
                values.append((float(chosen.detach().float().cpu()), float(rejected.detach().float().cpu())))
                del chosen, rejected
            self._references[source_shard] = values
            self._force_reshard()
        if len(self._references) != len(shards):
            raise RuntimeError("incomplete initial DPO reference cache")

    def _digest_surface(self) -> str:
        digest = hashlib.sha256()
        for parameter in named_selected_parameters(self):
            value = _local(parameter).detach().contiguous().view(torch.uint8).cpu()
            digest.update(memoryview(value.numpy()))
        return digest.hexdigest()

    def training_step(self, batch: PreferenceBatch, batch_idx: int):
        del batch_idx
        if batch.global_step != len(self._metrics) + 1:
            raise RuntimeError("finite DPO update schedule drift")
        references = self._references.get(batch.source_shard)
        if references is None or len(references) != len(batch.pairs):
            raise RuntimeError("DPO reference-cache/source-shard mismatch")
        # Retain Lightning's optimizer wrapper so its normal manual-optimization
        # bookkeeping advances ``trainer.global_step`` once per AdamW update.
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        self.eval()
        scale = float(self.cfg.dpo.world_size) / float(self.cfg.dpo.pairs_per_update)
        before = self._digest_surface()
        local_loss_sum = 0.0
        local_margin_sum = 0.0
        local_active = 0
        for pair, reference in zip(batch.pairs, references, strict=True):
            chosen, rejected = self._policy_pair(pair)
            ref_chosen = torch.tensor(reference[0], dtype=torch.float32, device=self.device)
            ref_rejected = torch.tensor(reference[1], dtype=torch.float32, device=self.device)
            objective = dpo_pair_objective(
                chosen_policy_logp=chosen, rejected_policy_logp=rejected,
                chosen_reference_logp=ref_chosen, rejected_reference_logp=ref_rejected, beta=float(self.cfg.dpo.beta),
            )
            loss = objective.loss * (scale if pair.active else 0.0)
            self.manual_backward(loss)
            if pair.active:
                local_loss_sum += float(objective.loss.detach().cpu())
                local_margin_sum += float(objective.margin.detach().cpu())
                local_active += 1
            del chosen, rejected, objective, loss
            self._force_reshard()
        grads = [parameter.grad for parameter in named_selected_parameters(self) if parameter.grad is not None]
        if not grads or not all(bool(torch.isfinite(_local(grad)).all()) for grad in grads):
            raise RuntimeError("DPO backward produced missing or nonfinite gradients")
        self.clip_gradients(optimizer, gradient_clip_val=float(self.cfg.dpo.gradient_clip_norm), gradient_clip_algorithm="norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self._force_reshard()
        after = self._digest_surface()
        changed = before != after
        health = torch.tensor([local_loss_sum, local_margin_sum, float(local_active)], device=self.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(health, op=dist.ReduceOp.SUM)
        if int(health[2].item()) != int(self.cfg.dpo.pairs_per_update) or not changed:
            raise RuntimeError("DPO update did not cover exactly one shard or did not move selected parameters")
        record = {
            "global_step": batch.global_step, "dpo_pass": batch.dpo_pass, "source_shard": batch.source_shard,
            "active_pairs": int(health[2].item()), "mean_loss": float(health[0].item() / health[2].item()),
            "mean_margin": float(health[1].item() / health[2].item()), "surface_digest_before": before,
            "surface_digest_after": after, "surface_changed": changed,
        }
        self._metrics.append(record)
        self.log("dpo/loss", record["mean_loss"], on_step=True, prog_bar=True, batch_size=1)
        self.log("dpo/margin", record["mean_margin"], on_step=True, batch_size=1)
        if self.global_rank == 0:
            _write_json(self._output_root / "steps" / f"s{batch.global_step:02d}.json", record)
        return record

    def on_train_batch_end(self, outputs: Any, batch: PreferenceBatch, batch_idx: int) -> None:
        del outputs, batch_idx
        if batch.global_step not in self._checkpoint_steps:
            return
        from torch.distributed.checkpoint import save

        destination = self._output_root / "checkpoints" / f"s{batch.global_step:02d}"
        if self.global_rank == 0:
            destination.mkdir(parents=True, exist_ok=False)
        if dist.is_initialized():
            dist.barrier()
        optimizer = self.optimizers().optimizer
        save({"state_dict": self.state_dict()}, checkpoint_id=str(destination / "model_weights.dcp"))
        save({"model": self.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_id=str(destination / "training_state.dcp"))
        if dist.is_initialized():
            dist.barrier()
        local_ready = all(
            path.is_file()
            for path in (
                destination / "model_weights.dcp" / ".metadata",
                destination / "training_state.dcp" / ".metadata",
            )
        )
        ready_across_ranks = torch.tensor(int(local_ready), device=self.device, dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(ready_across_ranks, op=dist.ReduceOp.MIN)
        if not bool(ready_across_ranks.item()):
            raise RuntimeError(f"incomplete DPO checkpoint at {destination}")
        if self.global_rank == 0:
            ready = {
                "step": batch.global_step, "model_weights": str(destination / "model_weights.dcp"),
                "training_state": str(destination / "training_state.dcp"), "passed": True,
            }
            _write_json(destination / "CHECKPOINT_READY.json", ready)
        if dist.is_initialized():
            dist.barrier()

    def on_train_end(self) -> None:
        if len(self._metrics) != self._expected_updates:
            raise RuntimeError(f"DPO completed {len(self._metrics)} rather than {self._expected_updates} updates")
        if self.global_rank == 0:
            _write_json(self._output_root / "global_compact.json", {"status": "passed", "steps": self._metrics, "surface": self._surface.__dict__})
            _write_json(self._output_root / "DONE.json", {"status": "passed", "updates": len(self._metrics)})
