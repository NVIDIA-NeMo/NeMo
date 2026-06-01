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

"""Thinker/Talker airline scenarios for the voice-agent evaluator.

These scenarios are intentionally data-driven so new fragmented-turn cases can
be added by editing ``data/thinker_talker_airline_cases.json`` without adding a
new Python class each time.
"""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from nemo.agents.voice_agent.evaluation import get_eval_data_root
from nemo.agents.voice_agent.evaluation.scenarios import register_eval_scenario
from nemo.agents.voice_agent.evaluation.scenarios.classes import Actions, Persona, Resources, Scenario, Task


@cache
def load_thinker_talker_airline_cases() -> list[dict[str, Any]]:
    """Load the data-driven Thinker/Talker airline case catalog."""
    path = get_eval_data_root() / "thinker_talker_airline_cases.json"
    payload = json.loads(path.read_text())
    cases = payload.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty 'cases' list")
    return cases


class ThinkerTalkerAirlineBaseScenario(Scenario):
    """Base scenario for testing an external Thinker/Talker voice agent.

    The NeMo evaluator still owns the simulated-user side and the audio bridge.
    The agent-under-test is expected to be the Thinker/Talker WebSocket endpoint
    from the nemotron voice-agent repo. That external endpoint currently does
    not expose NeMo's evaluation summary actions, so these scenarios leave
    ``reference_answer`` unset and rely on the post-run scorer for task and
    rephrased-query metrics.
    """

    case: dict[str, Any] = {}
    reference_answer = None
    max_duration = 480

    @property
    def description(self) -> str:
        return str(self.case.get("description", "Thinker/Talker airline evaluation scenario."))

    @property
    def user_persona(self) -> Persona:
        user = self.case["user"]
        return Persona(
            role=user["role"],
            name=user["name"],
            background=user["background"],
            personality=user["personality"],
        )

    @property
    def user_task(self) -> Task:
        return Task(goal=self.case["user"]["goal"])

    @property
    def user_actions(self) -> Actions:
        user = self.case["user"]
        return Actions(
            instructions=list(user.get("instructions", [])),
            guidelines=list(user.get("guidelines", []))
            + [
                "Keep the conversation moving after the agent asks a question.",
                "Once your stated goal is complete, say a clear goodbye.",
            ],
        )

    @property
    def user_resources(self) -> Resources:
        return Resources()

    @property
    def agent_persona(self) -> Persona:
        return Persona(
            role="airline voice agent",
            name="Thinker/Talker",
            background=(
                "You are an airline voice agent that can search flights, create new bookings, "
                "and check PNR status. You are being evaluated through a live audio bridge."
            ),
            personality="Concise, helpful, and direct.",
        )

    @property
    def agent_task(self) -> Task:
        return Task(
            goal=(
                "Help the caller complete the requested new booking, flight search, or PNR status task. "
                "Preserve details that arrive across separate turns and long pauses."
            )
        )

    @property
    def agent_actions(self) -> Actions:
        return Actions(
            instructions=[
                "Answer greetings briefly.",
                "For flight search, new booking, and PNR status, gather missing details and complete the task.",
                "When details arrive across later turns, merge them with the existing task context.",
                "When the task is complete, summarize the result and say goodbye if the caller is done.",
            ],
            guidelines=[
                "Do not treat a later seat, meal, date, route, or PNR fragment as a new unrelated task.",
                "For a new booking, do not book until the user confirms.",
                "Keep spoken answers short and suitable for text-to-speech.",
            ],
        )

    @property
    def agent_resources(self) -> Resources:
        # The external Thinker/Talker endpoint owns its tool schema. This field is
        # useful when running the same scenario against a NeMo-compatible bot but
        # is ignored by the current external /api/ws endpoint.
        return Resources()


def _class_name_from_case_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("__", 1)[-1].split("_")) + "Scenario"


def _register_data_driven_cases() -> None:
    seen: set[str] = set()
    for case in load_thinker_talker_airline_cases():
        name = str(case.get("name") or "").strip()
        if not name:
            raise ValueError("Every Thinker/Talker airline case must define a non-empty name")
        if name in seen:
            raise ValueError(f"Duplicate Thinker/Talker airline case name: {name}")
        seen.add(name)
        attrs = {
            "name": name,
            "case": case,
            "max_duration": int(case.get("max_duration") or ThinkerTalkerAirlineBaseScenario.max_duration),
            "__module__": __name__,
        }
        register_eval_scenario(type(_class_name_from_case_name(name), (ThinkerTalkerAirlineBaseScenario,), attrs))


_register_data_driven_cases()

__all__ = ["ThinkerTalkerAirlineBaseScenario", "load_thinker_talker_airline_cases"]
