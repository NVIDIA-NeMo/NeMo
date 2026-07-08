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

"""Offline checks for evaluator compatibility helpers."""

from __future__ import annotations

from nemo.agents.voice_agent.evaluation import utils


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def test_judge_scenario_uses_available_evidence_without_reference(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["payload"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            '{"score": 1.0, "reason": "passed", '
            '"nl_assertion_verdicts": [{"index": 1, "passed": true, "reason": "matched"}]}'
        )

    monkeypatch.setattr(utils.requests, "post", fake_post)
    judge = utils.LLMJudge(url="http://judge", model="test-model", api_key="token", timeout=7)

    result = judge.judge_scenario(
        conversation=[{"role": "user", "text": "I need to change my flight."}],
        agent_context_history=[{"role": "assistant", "tool_calls": [{"function": {"name": "call_thinker"}}]}],
        user_context_history=[{"role": "user", "content": "I need to change my flight."}],
        nl_assertions=["The agent routed the request to an internal task handler."],
    )

    user_content = captured["payload"]["messages"][1]["content"]
    assert "<reference>" not in user_content
    assert "<prediction>" not in user_content
    assert "<conversation>" in user_content
    assert "<agent_context_history>" in user_content
    assert "<user_context_history>" in user_content
    assert "<nl_assertions>" in user_content
    assert captured["timeout"] == 7
    assert result["score"] == 1.0
    assert result["nl_assertion_pass_count"] == 1
    assert result["nl_assertion_total"] == 1
    assert result["nl_assertion_pass_rate"] == 1.0


def test_judge_scenario_returns_zero_without_evidence(monkeypatch):
    def fail_post(*args, **kwargs):
        raise AssertionError("judge endpoint should not be called without evidence")

    monkeypatch.setattr(utils.requests, "post", fail_post)
    judge = utils.LLMJudge(url="http://judge", model="test-model", api_key="token")

    result = judge.judge_scenario()

    assert result["score"] == 0.0
    assert "No judge evidence" in result["reason"]
