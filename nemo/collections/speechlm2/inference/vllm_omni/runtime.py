# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Process-scoped vLLM-Omni engines on one background asyncio loop.

``AsyncOmni`` is asynchronous and its engines are expensive, so exactly one
:class:`OmniRuntime` is built per process and shared by every stream. It owns
a daemon thread running a dedicated event loop, the independent Nemotron and
EarTTS engines, and the deploy-YAML overrides they were started with.

Nemotron and EarTTS get separate one-stage engines so NeMo can hand tokens
between them and give EarTTS a classifier-free-guidance companion request
without duplicating the much larger Nemotron request.

Which components exist is decided here, at construction, and read off the
runtime afterwards (``llm_engine``/``tts_engine`` being None) -- callers do not
pass backend flags around.
"""

import asyncio
import logging as stdlib_logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml

from nemo.collections.speechlm2.inference.vllm_omni import default_deploy_yaml, default_eartts_deploy_yaml
from nemo.collections.speechlm2.inference.vllm_omni.checkpoint import EARTTS_SUBDIR
from nemo.utils import logging


class _ExpectedJanusShutdownFilter(stdlib_logging.Filter):
    """Hide only vLLM-Omni's expected output-queue shutdown traceback."""

    def filter(self, record: stdlib_logging.LogRecord) -> bool:
        if "[AsyncOmni] final_output_loop failed." not in record.getMessage():
            return True
        exc = record.exc_info[1] if record.exc_info else None
        return not (
            exc is not None
            and exc.__class__.__module__.startswith("janus")
            and exc.__class__.__name__
            in {
                "ShutDown",
                "SyncQueueShutDown",
                "AsyncQueueShutDown",
            }
        )


class OmniRuntime:
    """Long-lived split AsyncOmni engines + one background asyncio loop.

    Constructed once by the inference wrapper and shared across streams.
    Nemotron and EarTTS have independent one-stage engines so NeMo can hand
    tokens between them and create an EarTTS CFG companion without duplicating
    the much larger Nemotron request.
    """

    def __init__(
        self,
        wrapper_dir: str,
        *,
        stage_configs_path: str | None = None,
        eartts_stage_configs_path: str | None = None,
        stage_overrides: dict | None = None,
        eartts_stage_overrides: dict | None = None,
        log_stats: bool = False,
        stage_init_timeout: int = 600,
        enable_llm: bool = True,
        enable_tts: bool = True,
    ) -> None:
        if not enable_llm and not enable_tts:
            raise ValueError("OmniRuntime requires at least one enabled component")
        self.enable_llm = bool(enable_llm)
        self.enable_tts = bool(enable_tts)

        llm_yaml = Path(stage_configs_path) if stage_configs_path else default_deploy_yaml()
        tts_yaml = Path(eartts_stage_configs_path) if eartts_stage_configs_path else default_eartts_deploy_yaml()
        required_yamls = []
        if self.enable_llm:
            required_yamls.append(llm_yaml)
        if self.enable_tts:
            required_yamls.append(tts_yaml)
        for deploy_yaml in required_yamls:
            if not deploy_yaml.is_file():
                raise FileNotFoundError(f"Deploy YAML not found: {deploy_yaml}")

        # Accept single-pipeline override keys as well: ``stage_0`` addresses
        # the Nemotron engine and ``stage_1`` the EarTTS engine's stage 0.
        llm_overrides, legacy_tts_overrides = self._split_stage_overrides(stage_overrides)
        if eartts_stage_overrides is None:
            eartts_stage_overrides = legacy_tts_overrides
        self._llm_stage_yaml_path = (
            self._maybe_write_overridden_yaml(llm_yaml, llm_overrides, prefix="nemotron_") if self.enable_llm else None
        )
        self._tts_stage_yaml_path = (
            self._maybe_write_overridden_yaml(tts_yaml, eartts_stage_overrides, prefix="eartts_")
            if self.enable_tts
            else None
        )
        self._wrapper_dir = wrapper_dir
        self._eartts_dir = os.path.join(wrapper_dir, EARTTS_SUBDIR)
        self._shutdown = False

        # Start the background loop in a daemon thread first; ``AsyncOmni``
        # is constructed *on* that loop (its ``__init__`` allocates
        # ``asyncio.Condition`` / ``asyncio.Queue`` and the orchestrator
        # binds them to the current event loop, so the engine must be
        # built from inside that loop's thread).
        self._loop = asyncio.new_event_loop()
        self._ready_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._loop_runner,
            name="OmniRuntimeLoop",
            daemon=True,
        )
        self._thread.start()
        self._ready_evt.wait()

        # Register in this process before constructing the engine:
        # ``AsyncOmniEngine.__init__`` resolves ``model_type`` before loading
        # plugin groups. An unregistered model type selects the default diffusion
        # pipeline, which expects ``model_index.json``. Entry points register the
        # same pipeline in spawned stage processes.
        from vllm_omni import AsyncOmni

        from nemo.collections.speechlm2.inference.vllm_omni.register import register_nemo_voicechat

        register_nemo_voicechat()

        logging.info(
            "Creating split AsyncOmni engines from wrapper=%s (Nemotron) and %s (EarTTS) ...",
            wrapper_dir,
            self._eartts_dir,
        )

        async def _build_engines() -> tuple[Any | None, Any | None]:
            llm_engine = None
            tts_engine = None
            try:
                if self.enable_llm:
                    llm_engine = AsyncOmni(
                        model=wrapper_dir,
                        stage_configs_path=str(self._llm_stage_yaml_path),
                        log_stats=log_stats,
                        stage_init_timeout=stage_init_timeout,
                    )
                if self.enable_tts:
                    tts_engine = AsyncOmni(
                        model=self._eartts_dir,
                        stage_configs_path=str(self._tts_stage_yaml_path),
                        log_stats=log_stats,
                        stage_init_timeout=stage_init_timeout,
                    )
                return llm_engine, tts_engine
            except BaseException:
                if llm_engine is not None:
                    llm_engine.shutdown()
                raise

        fut = asyncio.run_coroutine_threadsafe(_build_engines(), self._loop)
        self.llm_engine, self.tts_engine = fut.result()
        logging.info(
            "Split AsyncOmni ready (Nemotron=%s, EarTTS=%s)",
            f"{self.llm_engine.num_stages} stage" if self.llm_engine is not None else "native",
            f"{self.tts_engine.num_stages} stage" if self.tts_engine is not None else "native",
        )

    # ------------------------------------------------------------------ #
    #  YAML override                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_stage_overrides(
        stage_overrides: dict | None,
    ) -> tuple[dict | None, dict | None]:
        if not stage_overrides:
            return None, None
        common = dict(stage_overrides.get("common", {}) or {})
        llm: dict[str, Any] = {}
        tts: dict[str, Any] = {}
        if common:
            llm["common"] = common
            tts["common"] = common
        if stage_overrides.get("stage_0"):
            llm["stage_0"] = dict(stage_overrides["stage_0"])
        if stage_overrides.get("stage_1"):
            tts["stage_0"] = dict(stage_overrides["stage_1"])
        return llm or None, tts or None

    @staticmethod
    def _maybe_write_overridden_yaml(
        deploy_yaml: Path,
        stage_overrides: dict | None,
        *,
        prefix: str,
    ) -> Path:
        """Apply per-stage overrides to the deploy YAML, write to a tmp file.

        ``stage_overrides`` shape::

            {
                "common": {<flat keys applied to every stage>},
                "stage_0": {<flat keys for stage 0>},
                "stage_1": {<flat keys for stage 1>},
            }

        Returns the path that ``AsyncOmni`` should load; the original YAML
        is returned untouched when no overrides are supplied.
        """
        if not stage_overrides:
            return deploy_yaml

        with open(deploy_yaml, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        common = stage_overrides.get("common", {}) or {}
        per_stage = {int(k.split("_", 1)[1]): v for k, v in stage_overrides.items() if k.startswith("stage_") and v}

        for stage in cfg.get("stages", []):
            for key, value in common.items():
                stage[key] = value
            sid = int(stage.get("stage_id", -1))
            for key, value in per_stage.get(sid, {}).items():
                stage[key] = value

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=prefix,
            delete=False,
        )
        yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
        tmp.close()
        logging.info(f"Wrote overridden stage config to {tmp.name}")
        return Path(tmp.name)

    # ------------------------------------------------------------------ #
    #  Background loop                                                   #
    # ------------------------------------------------------------------ #

    def _loop_runner(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready_evt.set()
        try:
            self._loop.run_forever()
        finally:
            # ``run_forever`` returns when ``loop.stop()`` is called from
            # ``shutdown``. Tear down any pending tasks before closing.
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
            except RuntimeError:
                pass
            try:
                self._loop.close()
            except Exception:
                pass

    def submit(self, coro):
        """Schedule a coroutine on the background loop, return the concurrent ``Future``."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        """Stop both engines and the background loop."""
        if self._shutdown:
            return
        self._shutdown = True

        async def _shutdown_engines() -> None:
            for name in ("tts_engine", "llm_engine"):
                engine = getattr(self, name, None)
                if engine is None:
                    continue
                try:
                    final_output_task = getattr(engine, "final_output_task", None)
                    if final_output_task is not None and not final_output_task.done():
                        final_output_task.cancel()
                        await asyncio.gather(
                            final_output_task,
                            return_exceptions=True,
                        )
                        engine.final_output_task = None
                    engine.shutdown()
                except Exception as exc:
                    logging.warning(f"{name}.shutdown() raised: {exc!r}")

        shutdown_filter = _ExpectedJanusShutdownFilter()
        async_omni_logger = stdlib_logging.getLogger("vllm_omni.entrypoints.async_omni")
        async_omni_logger.addFilter(shutdown_filter)
        root_handlers = list(stdlib_logging.getLogger().handlers)
        for handler in root_handlers:
            handler.addFilter(shutdown_filter)
        try:
            self.submit(_shutdown_engines()).result(timeout=120)
        except Exception as exc:
            logging.warning(f"Split AsyncOmni shutdown raised: {exc!r}")
        finally:
            async_omni_logger.removeFilter(shutdown_filter)
            for handler in root_handlers:
                handler.removeFilter(shutdown_filter)
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        self._thread.join(timeout=10)
