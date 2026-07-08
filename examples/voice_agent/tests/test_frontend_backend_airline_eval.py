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

"""Offline checks for the Frontend/Backend airline evaluator prototype."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

nemo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(nemo_root))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_scorer_module():
    path = nemo_root / "examples" / "voice_agent" / "evaluation" / "score_frontend_backend_airline.py"
    return _load_module("score_frontend_backend_airline", path)


def _load_frontend_backend_scenario_module():
    # The scenario dataclasses only need NoiseConfig for typing/defaults.
    # Importing the full voice-agent utils package also imports live
    # ASR/diarization runtime dependencies, which are not needed here.
    utils_module = types.ModuleType("nemo.agents.voice_agent.utils")
    audio_module = types.ModuleType("nemo.agents.voice_agent.utils.audio")
    audio_module.NoiseConfig = type("NoiseConfig", (), {})
    utils_module.audio = audio_module
    sys.modules.setdefault("nemo.agents.voice_agent.utils", utils_module)
    sys.modules.setdefault("nemo.agents.voice_agent.utils.audio", audio_module)

    classes_module = _load_module(
        "nemo.agents.voice_agent.evaluation.scenarios.classes",
        nemo_root / "nemo" / "agents" / "voice_agent" / "evaluation" / "scenarios" / "classes.py",
    )
    registry = {}
    scenarios_module = types.ModuleType("nemo.agents.voice_agent.evaluation.scenarios")

    def register_eval_scenario(cls):
        if not issubclass(cls, classes_module.Scenario):
            raise ValueError(f"Class {cls.__name__} is not a subclass of Scenario")
        registry[getattr(cls, "name", cls.__name__)] = cls
        return cls

    def get_eval_scenario(name: str, **kwargs):
        if name not in registry:
            return None
        return registry[name](**kwargs)

    scenarios_module.register_eval_scenario = register_eval_scenario
    scenarios_module.get_eval_scenario = get_eval_scenario
    scenarios_module.list_eval_scenarios = lambda: list(registry)
    scenarios_module.Scenario = classes_module.Scenario
    scenarios_module.__path__ = []
    sys.modules["nemo.agents.voice_agent.evaluation.scenarios"] = scenarios_module
    sys.modules.setdefault(
        "nemo.agents.voice_agent.evaluation.scenarios.data",
        types.ModuleType("nemo.agents.voice_agent.evaluation.scenarios.data"),
    )

    scenario_module = _load_module(
        "nemo.agents.voice_agent.evaluation.scenarios.data.frontend_backend_airline",
        nemo_root
        / "nemo"
        / "agents"
        / "voice_agent"
        / "evaluation"
        / "scenarios"
        / "data"
        / "frontend_backend_airline.py",
    )
    return scenarios_module, scenario_module


scenarios, frontend_backend_airline = _load_frontend_backend_scenario_module()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_frontend_backend_airline_scenarios_are_registered():
    cases = frontend_backend_airline.load_frontend_backend_airline_cases()
    names = {case["name"] for case in cases}

    registered = set(scenarios.list_eval_scenarios())
    assert names <= registered

    scenario = scenarios.get_eval_scenario("frontend_backend_airline__fragmented_booking_window_seat")
    assert scenario is not None
    assert scenario.reference_answer is None
    assert "window seat" in scenario.get_user_prompt().lower()


def test_scorer_passes_with_transcript_tool_results_and_query(tmp_path):
    scorer = _load_scorer_module()
    case = next(
        case
        for case in frontend_backend_airline.load_frontend_backend_airline_cases()
        if case["name"] == "frontend_backend_airline__fragmented_booking_window_seat"
    )
    scenario_dir = tmp_path / case["name"]

    _write_json(
        scenario_dir / "conversation_log.seglst.json",
        [
            {"speaker": "user", "words": "I want to book a flight."},
            {
                "speaker": "agent",
                "words": (
                    "I found flight options and booked the first flight. "
                    "Your booking is confirmed with a window seat. Your PNR is XYZ123."
                ),
            },
        ],
    )
    _write_json(
        scenario_dir / "bot_logs_agent" / "llm_context.json",
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "call_backend",
                            "arguments": json.dumps(
                                {
                                    "query": (
                                        "The user wants a new flight from New York JFK to San Francisco SFO "
                                        "on 2026-05-01 and wants the first option with a window seat."
                                    )
                                }
                            ),
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "call_backend",
                "content": json.dumps({"type": "tool_result", "tool": "flight_search", "status": "success"}),
            },
            {
                "role": "tool",
                "name": "call_backend",
                "content": json.dumps({"type": "tool_result", "tool": "booking", "status": "success"}),
            },
        ],
    )

    result = scorer.score_scenario(case, scenario_dir)
    assert result["intent_achievement"]["pass"] is True
    assert result["intent_achievement"]["tool_result_pass"] is True
    assert result["rephrased_query_to_backend"]["pass"] is True


def test_scorer_reports_missing_tool_trace_as_partial_evidence(tmp_path):
    scorer = _load_scorer_module()
    case = next(
        case
        for case in frontend_backend_airline.load_frontend_backend_airline_cases()
        if case["name"] == "frontend_backend_airline__fragmented_search_price_sort"
    )
    scenario_dir = tmp_path / case["name"]

    _write_json(
        scenario_dir / "conversation_log.seglst.json",
        [
            {
                "speaker": "agent",
                "words": "I found flights from Los Angeles LAX to Seattle SEA for May second.",
            }
        ],
    )

    result = scorer.score_scenario(case, scenario_dir)
    assert result["intent_achievement"]["transcript_pass"] is True
    assert result["intent_achievement"]["observable_pass"] is True
    assert result["intent_achievement"]["pass"] is False
    assert result["intent_achievement"]["tool_result_pass"] is None
    assert result["rephrased_query_to_backend"]["pass"] is None
