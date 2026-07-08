#!/usr/bin/env python3
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

"""Post-run scorer for Frontend/Backend airline voice-agent evaluations.

The generic NeMo evaluator captures the conversation, audio, and any available
agent context. This scorer adds Frontend/Backend-specific metrics:

* intent achievement from spoken outcome plus observed backend tool results
* rephrased-query accuracy from observed ``call_backend.query`` arguments

The current external Frontend/Backend WebSocket endpoint does not expose NeMo's
``get_context_history`` or ``get_scenario_summary`` actions. In that case this
script still scores transcript-observable task achievement and reports tool and
query telemetry as missing rather than silently passing it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_CASES = Path(__file__).resolve().parent / "data" / "frontend_backend_airline_cases.json"
# Accept legacy trace names so older Frontend/Backend artifacts remain scorable
# after the public example rename.
BACKEND_TOOL_NAMES = frozenset({"call_backend", "call_thinker"})
TRACE_CANDIDATES = (
    "backend_lifecycle.json",
    "backend_lifecycle_events.json",
    "thinker_lifecycle.json",
    "thinker_lifecycle_events.json",
    "agent_trace.json",
    "bot_logs_agent/backend_lifecycle.json",
    "bot_logs_agent/backend_lifecycle_events.json",
    "bot_logs_agent/thinker_lifecycle.json",
    "bot_logs_agent/thinker_lifecycle_events.json",
    "bot_logs_agent/agent_trace.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Frontend/Backend airline evaluation results.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session-dir", type=Path, help="Evaluation session directory containing scenario folders.")
    group.add_argument(
        "--scenario-dir",
        type=Path,
        action="append",
        help="One scenario result directory. May be repeated.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help=f"Case catalog JSON (default: {DEFAULT_CASES})",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write frontend_backend_airline_score.json per scenario and "
            "frontend_backend_airline_summary.json per session."
        ),
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    scenario_dirs = args.scenario_dir or _discover_scenario_dirs(args.session_dir, cases)
    results = []
    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        case = cases.get(scenario_name)
        if case is None:
            results.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_directory": str(scenario_dir),
                    "error": f"No Frontend/Backend airline case named {scenario_name!r}",
                }
            )
            continue
        result = score_scenario(case, scenario_dir)
        results.append(result)
        if args.write:
            _write_json(scenario_dir / "frontend_backend_airline_score.json", result)

    summary = summarize(results)
    payload = {"summary": summary, "results": results}
    if args.write and args.session_dir:
        _write_json(args.session_dir / "frontend_backend_airline_summary.json", payload)
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=args.pretty))
    return 0


def score_scenario(case: dict[str, Any], scenario_dir: Path) -> dict[str, Any]:
    """Score one scenario directory against one case definition."""
    transcript = _load_transcript(scenario_dir)
    agent_text = " ".join(segment["words"] for segment in transcript if segment.get("speaker") == "agent")
    expected = case.get("expected", {})
    intent_expected = expected.get("intent", {})
    rephrased_expected = expected.get("rephrased_query", {})
    telemetry = _load_telemetry(scenario_dir)

    agent_term_groups = intent_expected.get("required_agent_terms", [])
    agent_term_matches = [_any_alias_present(agent_text, group) for group in agent_term_groups]
    transcript_accuracy = _mean_bool(agent_term_matches)

    required_tool_results = intent_expected.get("required_tool_results", [])
    tool_result_matches = [
        _has_required_tool_result(telemetry["tool_results"], required) for required in required_tool_results
    ]
    tool_results_observed = bool(telemetry["tool_results"])
    tool_result_accuracy = _mean_bool(tool_result_matches) if tool_results_observed or required_tool_results else None

    required_query_groups = rephrased_expected.get("required_alias_groups", [])
    best_query, query_group_matches = _best_query_match(telemetry["backend_queries"], required_query_groups)
    query_accuracy = _mean_bool(query_group_matches) if telemetry["backend_queries"] else None

    transcript_pass = bool(agent_term_matches) and all(agent_term_matches)
    tool_result_pass = None if required_tool_results and not tool_results_observed else all(tool_result_matches)
    rephrased_query_pass = (
        None if required_query_groups and not telemetry["backend_queries"] else all(query_group_matches)
    )

    # Strict end-to-end intent achievement requires spoken evidence and tool-level
    # evidence. Observable pass is useful while the external agent lacks a trace export.
    intent_achievement_pass = transcript_pass and tool_result_pass is True
    observable_intent_achievement_pass = transcript_pass and tool_result_pass in {True, None}

    return {
        "scenario_name": case["name"],
        "scenario_directory": str(scenario_dir),
        "description": case.get("description", ""),
        "intent_achievement": {
            "pass": intent_achievement_pass,
            "observable_pass": observable_intent_achievement_pass,
            "transcript_pass": transcript_pass,
            "transcript_accuracy": transcript_accuracy,
            "agent_term_matches": _alias_match_details(agent_text, agent_term_groups),
            "tool_result_pass": tool_result_pass,
            "tool_result_accuracy": tool_result_accuracy,
            "required_tool_results": required_tool_results,
            "observed_tool_results": telemetry["tool_results"],
        },
        "rephrased_query_to_backend": {
            "pass": rephrased_query_pass,
            "accuracy": query_accuracy,
            "required_alias_groups": required_query_groups,
            "best_query": best_query,
            "best_query_matches": _alias_match_details(best_query or "", required_query_groups),
            "observed_queries": telemetry["backend_queries"],
        },
        "evidence": {
            "transcript_segments": len(transcript),
            "agent_text_chars": len(agent_text),
            "context_file_found": telemetry["context_file_found"],
            "trace_files_found": telemetry["trace_files_found"],
            "tool_results_observed": tool_results_observed,
            "backend_queries_observed": bool(telemetry["backend_queries"]),
        },
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [result for result in results if "error" not in result]
    intent = [result["intent_achievement"] for result in valid]
    rephrased = [result["rephrased_query_to_backend"] for result in valid]
    return {
        "scenario_count": len(valid),
        "errors": [result for result in results if "error" in result],
        "intent_achievement_pass_rate": _rate(item["pass"] for item in intent),
        "observable_intent_achievement_pass_rate": _rate(item["observable_pass"] for item in intent),
        "transcript_intent_pass_rate": _rate(item["transcript_pass"] for item in intent),
        "tool_result_pass_rate": _rate(
            item["tool_result_pass"] for item in intent if item["tool_result_pass"] is not None
        ),
        "rephrased_query_pass_rate": _rate(item["pass"] for item in rephrased if item["pass"] is not None),
        "missing_tool_trace_count": sum(1 for result in valid if not result["evidence"]["tool_results_observed"]),
        "missing_backend_query_count": sum(
            1 for result in valid if not result["evidence"]["backend_queries_observed"]
        ),
    }


def _load_cases(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    cases = payload.get("cases", [])
    return {case["name"]: case for case in cases}


def _discover_scenario_dirs(session_dir: Path, cases: dict[str, dict[str, Any]]) -> list[Path]:
    return [session_dir / name for name in cases if (session_dir / name).is_dir()]


def _load_transcript(scenario_dir: Path) -> list[dict[str, Any]]:
    path = scenario_dir / "conversation_log.seglst.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _load_telemetry(scenario_dir: Path) -> dict[str, Any]:
    context_path = scenario_dir / "bot_logs_agent" / "llm_context.json"
    roots: list[Any] = []
    trace_files = []
    if context_path.exists():
        roots.append(_load_json(context_path))
    for rel_path in TRACE_CANDIDATES:
        path = scenario_dir / rel_path
        if path.exists():
            trace_files.append(str(path))
            roots.append(_load_json(path))

    queries: list[str] = []
    tool_results: list[dict[str, Any]] = []
    for root in roots:
        _collect_telemetry(root, queries, tool_results)

    return {
        "context_file_found": context_path.exists(),
        "trace_files_found": trace_files,
        "backend_queries": _unique_nonempty(queries),
        "tool_results": _dedupe_dicts(tool_results),
    }


def _collect_telemetry(node: Any, queries: list[str], tool_results: list[dict[str, Any]]) -> None:
    if isinstance(node, list):
        for item in node:
            _collect_telemetry(item, queries, tool_results)
        return
    if not isinstance(node, dict):
        _collect_from_string(node, queries, tool_results)
        return

    marker = node.get("marker")
    if marker in {"BackendStarted", "ThinkerStarted"} and node.get("query"):
        queries.append(str(node["query"]))
    payload = node.get("payload")
    if isinstance(payload, dict) and payload.get("type") == "tool_result":
        tool_results.append(_minimal_tool_result(payload))

    _collect_backend_query_from_tool_call(node, queries)
    if node.get("type") == "tool_result" and node.get("tool"):
        tool_results.append(_minimal_tool_result(node))

    for value in node.values():
        _collect_telemetry(value, queries, tool_results)


def _collect_backend_query_from_tool_call(node: dict[str, Any], queries: list[str]) -> None:
    function = node.get("function")
    if isinstance(function, dict) and function.get("name") in BACKEND_TOOL_NAMES:
        arguments = _decode_json_maybe(function.get("arguments"))
        if isinstance(arguments, dict) and arguments.get("query"):
            queries.append(str(arguments["query"]))

    if node.get("name") in BACKEND_TOOL_NAMES:
        arguments = _decode_json_maybe(node.get("arguments"))
        if isinstance(arguments, dict) and arguments.get("query"):
            queries.append(str(arguments["query"]))

    function_call = node.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name") in BACKEND_TOOL_NAMES:
        arguments = _decode_json_maybe(function_call.get("arguments"))
        if isinstance(arguments, dict) and arguments.get("query"):
            queries.append(str(arguments["query"]))


def _collect_from_string(value: Any, queries: list[str], tool_results: list[dict[str, Any]]) -> None:
    if not isinstance(value, str):
        return
    decoded = _decode_json_maybe(value)
    if decoded is not value:
        _collect_telemetry(decoded, queries, tool_results)


def _minimal_tool_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in ("type", "tool", "status", "context")
        if key in payload and payload[key] is not None
    }


def _has_required_tool_result(observed: list[dict[str, Any]], required: dict[str, Any]) -> bool:
    for item in observed:
        if all(item.get(key) == value for key, value in required.items()):
            return True
    return False


def _best_query_match(queries: list[str], alias_groups: list[list[str]]) -> tuple[str | None, list[bool]]:
    if not alias_groups:
        return (queries[-1] if queries else None), []
    best_query = None
    best_matches: list[bool] = [False] * len(alias_groups)
    best_score = -1
    for query in queries:
        matches = [_any_alias_present(query, group) for group in alias_groups]
        score = sum(matches)
        if score > best_score:
            best_query = query
            best_matches = matches
            best_score = score
    return best_query, best_matches


def _alias_match_details(text: str, alias_groups: list[list[str]]) -> list[dict[str, Any]]:
    normalized = _normalize_text(text)
    details = []
    for group in alias_groups:
        matched_aliases = [alias for alias in group if _normalize_text(alias) in normalized]
        details.append({"aliases": group, "matched": bool(matched_aliases), "matched_aliases": matched_aliases})
    return details


def _any_alias_present(text: str, aliases: list[str]) -> bool:
    normalized = _normalize_text(text)
    return any(_normalize_text(alias) in normalized for alias in aliases)


def _normalize_text(text: str) -> str:
    lowered = str(text).lower()
    # Keep alphanumerics only, then pad with spaces so phrase checks do not
    # accidentally match across word boundaries.
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return f" {' '.join(cleaned.split())} "


def _mean_bool(values: list[bool]) -> float:
    if not values:
        return 1.0
    return sum(1 for value in values if value) / len(values)


def _rate(values) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _unique_nonempty(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        clean = str(value).strip()
        if clean and clean not in seen:
            result.append(clean)
            seen.add(clean)
    return result


def _dedupe_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _decode_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
