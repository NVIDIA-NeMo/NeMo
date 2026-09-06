#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the two EasyMagpie vLLM-Omni stages."""
from __future__ import annotations

import argparse
import os
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


def _replica_count(stage: Any) -> int:
    count = int(getattr(stage.runtime, "num_replicas", 1))
    if not 1 <= count <= 16:
        raise ValueError(f"stage {stage.stage_id} num_replicas must be in 1..=16")
    return count


@contextmanager
def _stage_mps_priority(stage_id: int) -> Iterator[None]:
    key = f"EASYMAGPIE_STAGE{stage_id}_MPS_CLIENT_PRIORITY"
    value = os.environ.get(key)
    if value is None:
        yield
        return
    if value not in ("0", "1"):
        raise ValueError(f"{key} must be 0 or 1")

    old_value = os.environ.get("CUDA_MPS_CLIENT_PRIORITY")
    os.environ["CUDA_MPS_CLIENT_PRIORITY"] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("CUDA_MPS_CLIENT_PRIORITY", None)
        else:
            os.environ["CUDA_MPS_CLIENT_PRIORITY"] = old_value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--stage0-handshake", default="tcp://127.0.0.1:62100")
    parser.add_argument("--stage1-handshake", default="tcp://127.0.0.1:62101")
    parser.add_argument("--log-stats", action="store_true")
    return parser.parse_args()


def _supervise_stages(start_stage: Callable[[int], Any]) -> None:
    managers = []
    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for stage_id in (0, 1):
            if stopping:
                break
            managers.append(start_stage(stage_id))

        while not stopping:
            for stage_id, manager in enumerate(managers):
                finished = manager.finished_procs()
                if finished:
                    raise RuntimeError(f"EasyMagpie stage {stage_id} exited unexpectedly: {finished}")
            time.sleep(0.25)
    finally:
        for manager in reversed(managers):
            manager.shutdown()


def main() -> None:
    args = _parse_args()

    import vllm_plugin_easymagpie_omni

    vllm_plugin_easymagpie_omni.register()

    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_omni_kv_config_for_stage,
    )
    from vllm_omni.engine.stage_engine_core_proc_manager import StageEngineCoreProcManager
    from vllm_omni.engine.stage_init_utils import (
        build_engine_args_dict,
        build_vllm_config,
        get_stage_connector_spec,
        inject_omni_kv_connector_config,
        load_omni_transfer_config_for_model,
        prepare_engine_environment,
    )
    from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs

    config_path, stage_configs, _ = load_and_resolve_stage_configs(
        args.model,
        None,
        {},
        trust_remote_code=True,
        deploy_config_path=args.deploy_config,
    )
    stages = {stage.stage_id: stage for stage in stage_configs}
    missing = {0, 1} - stages.keys()
    if missing:
        raise ValueError(f"deploy config is missing stages: {sorted(missing)}")
    if any(stages[index].stage_type == "diffusion" for index in (0, 1)):
        raise ValueError("the Rust transport requires two LLM stages")
    replica_counts = {stage_id: _replica_count(stage) for stage_id, stage in stages.items()}
    if replica_counts[1] != 1:
        raise ValueError("the Rust transport requires exactly one Stage 1 replica")

    prepare_engine_environment()
    transfer_config = load_omni_transfer_config_for_model(args.model, config_path)
    handshakes = {0: args.stage0_handshake, 1: args.stage1_handshake}

    def start_stage(stage_id: int) -> Any:
        stage_config = stages[stage_id]
        connector_config = resolve_omni_kv_config_for_stage(transfer_config, stage_id)
        connector_spec = get_stage_connector_spec(
            omni_transfer_config=transfer_config,
            stage_id=stage_id,
            async_chunk=True,
        )
        engine_args = build_engine_args_dict(
            stage_config,
            args.model,
            stage_connector_spec=connector_spec,
            cli_tokenizer=None,
        )
        inject_omni_kv_connector_config(engine_args, connector_config, stage_id)
        vllm_config, executor_class = build_vllm_config(
            stage_config,
            args.model,
            stage_connector_spec=connector_spec,
            engine_args_dict=engine_args,
            headless=True,
        )
        with _stage_mps_priority(stage_id):
            return StageEngineCoreProcManager(
                local_engine_count=replica_counts[stage_id],
                start_index=0,
                local_start_index=0,
                vllm_config=vllm_config,
                local_client=True,
                handshake_address=handshakes[stage_id],
                executor_class=executor_class,
                log_stats=args.log_stats,
                omni_stage_id=stage_id,
            )

    _supervise_stages(start_stage)


if __name__ == "__main__":
    main()
