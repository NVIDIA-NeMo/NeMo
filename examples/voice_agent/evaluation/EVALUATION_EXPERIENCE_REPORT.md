# Voice Agent Evaluation Experience Report

This note captures what we learned while using the NeMo voice-agent evaluator
for airline voice-agent evaluation, with emphasis on the EVA airline domain.

The runs were exploratory. The goal was not to prove final model quality, but to
check whether the evaluator can help us iteratively build, debug, and compare
agentic voice agents.

## Scope

We focused on the `eva_airline` domain in the NeMo evaluator:

- structured scenario DBs,
- airline tools,
- DB-state scoring,
- transcript and audio capture,
- final scenario state capture,
- per-scenario logs for debugging.

We also confirmed that evaluating other agent architectures will require custom
supported scenarios, compatible tool/state adapters, and scoring hooks. Help in
building those adapters would be valuable because the evaluator is strongest
when the agent under test can expose the same task state that the scenario
expects.

## Runs Completed

### EVA Airline Smoke Run

Run directory:

`examples/voice_agent/evaluation/eval_results_eva_airline_smoke_tmux/eval_20260602_114138`

Scenarios run: `5`

Scenarios:

- `eva_airline__voluntary_date_change`
- `eva_airline__cancellation_refund`
- `eva_airline__irrops_cancellation`
- `eva_airline__missed_flight_standby`
- `eva_airline__escalation_edge_case`

Results:

| Scenario | Strict success | DB-state match | Turns | Duration |
| --- | ---: | ---: | ---: | ---: |
| `eva_airline__voluntary_date_change` | `False` | `False` | `14` | `647.6s` |
| `eva_airline__cancellation_refund` | `N/A` | `True` | `12` | `313.0s` |
| `eva_airline__irrops_cancellation` | `N/A` | `False` | `7` | `906.1s` |
| `eva_airline__missed_flight_standby` | `N/A` | `False` | `6` | `906.1s` |
| `eva_airline__escalation_edge_case` | `N/A` | `False` | `28` | `906.1s` |

Aggregate:

- Total scenarios: `5`
- Total turns: `67`
- Total duration: `3678.9s`
- Overall strict success: `0/1`, because only one of these scenarios had a
  `reference_answer`
- DB-state match rate: `1/5`
- Overall latency P50: `20826.1ms`
- Overall latency P95: `125514.2ms`

What this showed:

- The EVA evaluation harness runs end to end.
- Prompt rendering, tool registration, shared scenario DB seeding, tool execution,
  final DB pullback, transcript logging, audio logging, and metrics generation
  all worked.
- The agent successfully called EVA tools such as `GetReservationTool`,
  `GetFlightStatusTool`, `SearchRebookingOptionsTool`, `CancelReservationTool`,
  `ProcessRefundTool`, and `AssignSeatTool`.
- The harness completed cleanly with evaluator exit status `0`.

### EVA Airline Full Run

Status: completed.

Run directory:

`examples/voice_agent/evaluation/eval_results_eva_airline_full_tmux/eval_20260602_142402`

Started: `2026-06-02 14:24:02`

Completed: `2026-06-03 02:40:43`

Scenarios run: `50`

Command shape:

```bash
python run_evaluation.py \
  --user-url ws://localhost:8876 \
  --agent-url ws://localhost:8875 \
  --domain eva_airline \
  --duration 900 \
  --pause 1 \
  --judge-url "" \
  --output-dir examples/voice_agent/evaluation/eval_results_eva_airline_full_tmux
```

Aggregate results:

- Total scenarios: `50`
- Total turns: `376`
- Total duration: `43648.3s` (about `12.1` hours)
- Strict success rate: `100.00%` (`1/1` scenarios with `reference_answer`)
- DB-state match rate: `16.00%` (`8/50` scenarios with
  `expected_scenario_db`)
- Overall latency P50: `12002.0ms`
- Overall latency P95: `61809.0ms`

DB-state passing scenarios:

| Scenario | Strict success | Turns | Duration |
| --- | ---: | ---: | ---: |
| `eva_airline__voluntary_date_change` | `True` | `12` | `578.5s` |
| `eva_airline__cancellation_refund` | `N/A` | `14` | `449.7s` |
| `eva_airline__5_2_5` | `N/A` | `4` | `906.0s` |
| `eva_airline__7_1_1` | `N/A` | `8` | `254.9s` |
| `eva_airline__7_2_2` | `N/A` | `6` | `906.0s` |
| `eva_airline__7_2_8` | `N/A` | `4` | `906.0s` |
| `eva_airline__7_3_1` | `N/A` | `6` | `906.0s` |
| `eva_airline__7_4_1` | `N/A` | `6` | `906.0s` |

## Why DB-State Match Was Low

The low `16%` DB-state match rate is not a harness startup failure. The run
completed all 50 scenarios and wrote `all_summary.txt`, `all_metrics.json`, and
`all_latencies.csv`. The low score reflects that the final scenario databases
usually did not reach EVA's expected end state.

Observed failure pattern across the 42 DB-state failures:

| Failure signal | Count |
| --- | ---: |
| DB-state failures | `42` |
| Failures with no write actions recorded | `38` |
| Failures with at least one write action recorded | `4` |
| Failures that ran to about the `900s` cap | `41` |
| Timed-out failures with `6` or fewer turns | `29` |

Main causes visible from the artifacts:

- Most failed scenarios never reached a state-mutating tool call. Their
  `final_agent_response.json` had an empty `actions` list, so reservations,
  refunds, vouchers, standby lists, or travel credits stayed unchanged.
- Many scenarios stalled early in the dialogue, often around authentication or
  missing confirmation details. Several conversation logs show repeated
  confirmation-number prompts followed by `[INTERRUPTED]` user turns.
- Some failures were partial task completion. Example: an IRROPS cancellation
  case issued a meal voucher and rebooked, but the expected final DB also needed
  a seat assignment. The diff showed `seat` expected as `21C` but actual `null`.
- Some failures used the wrong path or wrong object. Example: one escalation
  case rebooked and assigned a seat, but the final reservation had mismatched
  fare class, fare paid, and journey fields compared with the expected DB.
- Some failures were long reasoning stalls. In the missed-flight standby case,
  the agent produced a very long spoken reasoning trace, consumed most of the
  scenario time, and never performed the expected standby/state update.
- Common DB diffs were in `reservations`, `journeys`, `meal_vouchers`,
  `refunds`, and `travel_credits`. Frequent field-level mismatches included
  `status`, `segments`, `available_seats`, `bookings`, `seat`, and
  `bags_checked`.

Interpretation:

- DB-state scoring is strict and useful: it catches incomplete or wrong
  world-state changes even when the conversation appears plausible.
- The low score mostly points to agent execution reliability under voice
  conditions: authentication recovery, concise reasoning, tool selection,
  confirmation, and completing state changes before timeout.
- Some expected-state diffs may also expose scenario/tool-policy ambiguity. For
  example, if the agent reaches a plausible but different final state, the
  evaluation should tell us whether that path is acceptable or whether the
  scenario expects a single canonical state.

## What Worked Well

- The scenario abstraction is useful. It can inject user persona, agent persona,
  task instructions, tools, and shared state per scenario.
- EVA DB-state scoring is valuable because it evaluates the final world state,
  not just whether a particular action list was emitted.
- Per-scenario artifacts are strong for debugging:
  - `conversation_log.txt`
  - `conversation_log.seglst.json`
  - `conversation_log.wav`
  - `bot_logs_agent/llm_context.json`
  - `bot_logs_user/llm_context.json`
  - `final_agent_response.json`
  - `final_scenario_db.json`
  - `metrics.json`
- The bridge-pulled `get_scenario_summary` pattern is better than relying on the
  LLM to call a final summary tool correctly.

## Metric Ownership Summary

| Metric or signal | Source | Feedback bucket |
| --- | --- | --- |
| DB-state match against `expected_scenario_db` | Already supported by EVA evaluator | Existing metric |
| Strict success from `reference_answer` | Already supported by evaluator | Existing metric |
| Per-turn and aggregate latency | Already supported by evaluator | Existing metric |
| Transcript, audio, final response, and final DB artifacts | Already supported by evaluator artifacts | Existing evidence |
| End-to-end task or intent achievement | Added/prototyped as custom post-run scoring | Added metric |
| Internal handoff/query-fidelity accuracy | Added/prototyped as custom trace-based scoring | Added metric |
| User-observed outcome and transcript-evidence split | Added/prototyped scoring direction | Added metric |
| Action-path or trace-based pass/fail | Added/prototyped scoring direction | Added metric |

## Issues And Feedback

### Feedback On Existing EVA Metrics

#### DB-State Match: Primary Signal

DB-state match is the strongest existing EVA metric because it grades reality,
not the conversation. It asks whether the agent actually changed the shared
world state to the expected outcome.

This is a harder bar than "the agent said the right thing" or "the agent called
a tool". A voice agent can sound competent, ask the right questions, call a
reasonable tool, and still leave the reservation unchanged or changed to the
wrong final state. DB-state match catches those cases.

In the full run, the DB-state result was:

- `8/50` scenarios matched the expected DB.
- `42/50` scenarios did not match the expected DB.
- `38/42` DB failures had no write actions in `final_agent_response.json`.
- `4/42` DB failures had at least one write action, but the final DB still did
  not match the expected DB.

How to explain the low score:

- It is not an evaluator startup failure. The run completed all scenarios and
  produced final metrics.
- It mostly means the agent did not finish the world-state mutation needed by
  the scenario.
- The dominant pattern was no write action before timeout, not a small mismatch
  after a successful task.

#### Make The DB Diff Actionable

The raw `db_state_diff` is too low-level for routine iteration. A human should
not need to read 42 field-by-field JSON diffs to understand the run.

The evaluator should roll DB-state failures into six diagnostic buckets. These
buckets can overlap; for example, a scenario can be both a timeout and an
authentication loop.

| Bucket | Meaning | Verified examples from this run |
| --- | --- | --- |
| No write | No state-mutating tool action was recorded, so the DB stayed unchanged. | `38/42` DB failures had an empty action list. In `eva_airline__7_2_6`, the expected state cancelled reservation `M62JCV` and removed a travel credit record, but actual state still had the reservation `confirmed` and no write action was recorded. |
| Partial write | Some required state changes happened, but not all. | `eva_airline__irrops_cancellation` wrote `issue_meal_voucher` and `rebook_flight`, but the DB diff still showed the expected seat `21C` with actual `null`. |
| Wrong write | The agent mutated state, but chose the wrong final object, fare class, fare, route, or policy path. | `eva_airline__escalation_edge_case` wrote `rebook_flight` and `assign_seat`, but the final DB had mismatched fare class and fare paid, and journey seat-availability diffs. |
| Timeout | The scenario ran to the practical duration cap before the task reached the expected final state. | `40` DB failures were at `>=899s`; including the near-cap escalation run at `892.7s`, `41` failures were effectively timeout-heavy. |
| Auth or identifier failure | The conversation got stuck while collecting or validating confirmation numbers, names, flight numbers, airport codes, or seat preferences. | `eva_airline__1_1_4` repeatedly asked for confirmation/name, used `Johansen` while the scenario expected `Johansson`, then transferred instead of completing the expected rebooking. |
| User-simulator drift | The simulated user response made the outcome harder to interpret or did not provide the decisive information the agent asked for. | In `eva_airline__1_1_3`, after the agent presented multiple options and asked which one to process, the user repeatedly said "Please process the change" without naming an option. |

Suggested dashboard fields:

- scenario name,
- DB match,
- strict success,
- duration,
- turn count,
- final action types,
- bucket labels,
- one-line expected state,
- one-line actual state,
- highest-signal transcript quote or artifact pointer.

This would turn DB-state scoring from "the DB hash did not match" into a
developer-actionable failure report.

#### Strict Success: Useful But Sparse

Strict success from `reference_answer` is useful when a scenario has a textual
or summary answer that must be matched. In this full EVA run, it is too sparse
to be a meaningful headline metric:

- Only `1/50` scenarios had a `reference_answer`.
- That one scenario passed, so the reported strict success was `100.00%`.
- Because coverage was only one scenario, `100.00%` does not mean the domain is
  performing well overall.

Recommendation: keep strict success, but report its denominator prominently and
avoid treating it as domain accuracy until many more EVA scenarios define
reference answers.

#### Latency: Valuable But Separate

Latency should be read as an operational metric, not blended directly into task
accuracy:

- P50 latency was `12002.0ms`.
- P95 latency was `61809.0ms`.
- The full 50-scenario run took about `12.1` hours.
- Many failures reached the `900s` scenario cap.

A timeout failure is not the same as a task-policy failure. If both are lumped
into one "accuracy" number, the metric mixes two questions:

- Could the agent decide and act correctly?
- Could the voice pipeline complete the interaction fast enough?

Recommendation: group timeout-heavy failures separately before interpreting
policy, reasoning, or tool-selection accuracy.

### Feedback On Added Or Prototyped Metrics

The additional metrics explored in this work cover gaps that DB-state and
reference-answer scoring do not fully capture:

- end-to-end task or intent achievement,
- internal handoff/query-fidelity accuracy,
- user-observed outcome,
- transcript evidence,
- action-path or trace-based pass/fail.

Feedback on added/prototyped metrics:

- Keep these metrics separate from EVA's built-in DB-state score. They answer
  different questions and should not be collapsed into one generic "accuracy"
  number.
- Intent achievement asks: did the agent accomplish what the user came to do?
  This fills a gap because DB-state cannot always tell whether the agent pursued
  the right goal, and transcript-only judging can under-score runs where state
  changed correctly but the transcript missed the final confirmation.
- Intent achievement should prefer structured state evidence whenever available.
  Transcript-only intent scoring is useful for observability, but it should be
  treated as a separate evidence channel.
- Internal handoff/query-fidelity scoring requires structured trace export. For
  each custom agent architecture, the evaluator needs a documented way to read
  the relevant planner request, normalized user request, tool arguments, or
  equivalent internal handoff artifact.
- Custom scenarios should define expected slots and values explicitly, so the
  scorer can report expected versus observed values instead of only pass/fail.
- These added metrics will be most useful when paired with custom supported
  scenarios and compatible tool/state adapters. Help for creating those
  scenarios, adapters, and custom post-run scorers would be valuable.

### Prompt Material Used In Evaluation

The evaluator does not use one monolithic prompt. It renders per-side system
prompts from scenario objects:

- `Persona`: role, name, background, personality
- `Task`: goal and optional background
- `Actions`: ordered instructions and persistent guidelines
- `Resources`: tools and additional information

For EVA airline runs, the runtime prompts came from the existing `eva_airline`
scenario definitions and were saved per run under each scenario's
`scenario_config/user_prompt.txt` and `scenario_config/agent_prompt.txt`. We did
not add or change EVA prompts for the smoke or full-domain runs.

No API keys or `.env` values are committed. The live runs sourced environment
variables from local files outside this report and only committed scenario/docs
content.

### Dataset And Registration Clarity

The repository contains `50` EVA airline dataset rows and scenario fixtures, and
`run_evaluation.py --list` currently lists `50` registered `eva_airline`
scenarios. The README should stay aligned with the runtime registry so users
know whether they are running seed scenarios, generated EVA scenarios, or the
entire domain.

### Custom Scenario And Tool Support

The EVA domain works well because its scenarios, tools, shared state, and
DB-state scorer are aligned. Other agent architectures will need the same level
of compatibility to get meaningful end-to-end scores.

Valuable support would include:

- templates for adding custom voice-agent domains,
- guidance for mapping external tool schemas to evaluator tools,
- trace/state export examples for non-NeMo agents,
- helpers for writing custom post-run scorers,
- documentation for choosing between action-list, DB-state, transcript, and LLM
  judge scoring.

## Practical Constraints On Iteration

### Voice Latency Is A Major Practical Constraint

The full EVA run took about `12.1` hours for 50 scenarios. That is an overnight
run, not a tight development loop. Latency was high:

- P50: `12002.0ms`
- P95: `61809.0ms`

Many failed scenarios ran to the `900s` cap. For iterative development, we need
shorter smoke scenarios, faster model settings, or a text/debug mode that
preserves the same scenario and scoring logic.

One verified failure mode was long spoken reasoning. In
`eva_airline__missed_flight_standby`, the agent produced a single response that
lasted about `547s`. It reasoned aloud about the reservation, possible missed
flight policy, and candidate actions, but did not perform the expected state
update before timeout. This should be reported as a latency/reasoning-control
failure before it is interpreted as a pure policy failure.

### User Simulator Can Drift

The user simulator sometimes responded in ways that made scenario outcomes
harder to interpret. For high-signal evaluations, user prompts should include
stricter guardrails for how to evaluate options, when to repeat identifiers, and
when to stop.

Verified examples:

- In `eva_airline__1_1_3`, the agent presented multiple flight options and asked
  which one to process. The user replied "Please process the change" instead of
  selecting a specific flight. The agent still proceeded, but this makes failure
  attribution ambiguous: did the agent infer too much, or did the simulated user
  fail to provide the requested decision?
- Many failed conversations had repeated `[INTERRUPTED]` user turns. This often
  prevented the agent from receiving the exact identifier, seat preference, or
  final confirmation needed to complete the task.

Recommendation: scenario prompts should define tighter simulator behavior:

- how to choose among options,
- when to repeat an identifier,
- how many times to tolerate repeated clarification,
- when to give up,
- what the user should say after the task is completed.

### ASR And Spoken Identifiers Are Still Risky

Airline scenarios rely on confirmation numbers, flight numbers, airport codes,
and PNRs. Even with instructions to spell identifiers, ASR can misrecognize them.
A single wrong character can send the agent to the wrong reservation or block
authentication entirely.

Verified examples:

- `eva_airline__2_1_2`: the scenario confirmation number was `PP248Z`, spoken as
  `P, P, two, four, eight, Z`. The agent transcript showed "PP2 for eight Z" and
  expanded it as `P, P, 2, F, O, R, 8, Z`, turning one six-character code into an
  eight-character interpretation.
- `eva_airline__1_1_4`: the scenario last name was `Johansson`, but the
  conversation repeatedly used `Johansen`. The run eventually recorded a
  `transfer_to_agent` action instead of the expected return-flight rebooking.
- `eva_airline__7_2_6`: the expected confirmation number was `M62JCV`. The user
  eventually provided "M, six, two, J, C, V. DeSilva." at about `594s`, but the
  scenario timed out before the agent completed the expected cancellation and
  travel-credit state change.

When failures involve identifiers, the first debugging step should be comparing:

- user simulator text in `bot_logs_user/llm_context.json`
- agent-side ASR text in `bot_logs_agent/llm_context.json`
- final tool arguments in `bot_logs_agent/llm_context.json`
- scenario prompt values in `scenario_config/user_prompt.txt`

For future runs, identifiers should be normalized before scoring where possible:

- map spoken digits such as "two" to `2`,
- avoid treating "four" as `for`,
- preserve repeated letters,
- validate expected identifier length before querying tools,
- ask a concise confirmation question before performing state-changing actions.

## Evidence Pointers

Use these artifacts when answering follow-up questions:

| Question | Artifact to inspect |
| --- | --- |
| Did the full run complete? | `eval_results_eva_airline_full_tmux/eval_20260602_142402/evaluation_log.txt` and `all_summary.txt` |
| Where did `8/50` DB-state match come from? | `all_summary.txt` and per-scenario `metrics.json` files |
| Why was strict success `100%` but not meaningful? | `all_summary.txt`, which reports `1/1 scenarios with reference_answer` |
| Which scenarios had no writes? | Per-scenario `final_agent_response.json`; empty `actions` means no recorded final action |
| What changed in the DB? | Per-scenario `metrics.json`, field `db_state_diff`, plus `final_scenario_db.json` |
| What did the user and agent actually say? | Per-scenario `conversation_log.txt` and `conversation_log.seglst.json` |
| What did the scenario expect the user to provide? | Per-scenario `scenario_config/user_prompt.txt` |
| What did the agent/tool path finally report? | Per-scenario `final_agent_response.json` and `bot_logs_agent/llm_context.json` |
| Where are latency numbers? | `all_summary.txt`, `all_latencies.csv`, and per-scenario `metrics.json` |

Specific examples worth keeping handy:

- `eva_airline__2_1_2`: identifier confusion for `PP248Z`.
- `eva_airline__1_1_4`: `Johansson` versus `Johansen`, repeated auth loop,
  transfer instead of expected rebooking.
- `eva_airline__7_2_6`: late correct identifier, timeout, no cancellation write.
- `eva_airline__irrops_cancellation`: partial write; voucher and rebooking
  recorded, expected seat assignment missing.
- `eva_airline__missed_flight_standby`: long spoken reasoning response before
  timeout.

## Recommended Next Steps

1. Keep the README scenario counts aligned with `run_evaluation.py --list`.

2. Add an EVA smoke helper script that starts or documents isolated ports,
   selected scenarios, output directory, and no-judge mode.

3. Add a compact dashboard or script that groups DB-state failures by:
   - no write actions,
   - wrong write action,
   - partial write action,
   - timeout,
   - authentication/identifier failure,
   - user-simulator drift.

4. Improve voice-run efficiency:
   - add a text/debug mode for faster iteration,
   - reduce default smoke duration,
   - add early-stop detection for stalled authentication loops,
   - cap accidental long reasoning speech before it consumes the scenario.

5. Improve scoring diagnostics:
   - summarize `db_state_diff` at a higher level,
   - list expected versus actual action/state changes,
   - separate state, transcript, observed-user, and action-path scoring.

6. Build custom scenario/tool support for other agent architectures:
   - define compatible scenario catalogs,
   - map tools and state into evaluator-visible artifacts,
   - add post-run scorers where DB-state scoring is not directly available.

7. Use a staged evaluation sequence:
   - 1-2 scenario smoke run,
   - 5 scenario domain smoke run,
   - focused failure-regression batch,
   - full 50-scenario EVA run only after the smaller batches are stable.

## Bottom Line

The evaluator is useful for iterative voice-agent development. It gave us
actionable failures that normal tool-call tests would miss: missing world-state
updates, partial tool execution, authentication loops, user-observed outcome
gaps, latency stalls, and timeout behavior.

The main work now is to improve structured failure diagnostics, make runs faster
to iterate on, and provide better support for custom scenarios and tool/state
adapters.
