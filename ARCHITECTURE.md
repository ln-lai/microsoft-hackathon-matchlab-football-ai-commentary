# Architecture and Agent execution

MatchLab Football AI Commentary combines a frozen position predictor with two Azure model roles and a synchronized browser replay. The two roles use separate requests to one configured GPT-5-mini deployment.

## Evidence and handoff

1. The backend constructs evidence from the selected synthetic scene, player and timestamp. Highlight reasons describe the preceding second. Position forecasts use past tracking inputs only.
2. **Match Analyst Agent** selects and orders allowed fact IDs. The server checks the selection and renders the explanation from computed evidence.
3. **Commentary Reviewer Agent** receives that explanation, the proposed speech script and the evidence. A strict response schema asks for player, number, time and forecast-wording checks. The server validates the scope and types and accepts a pass only when all checks pass.
4. The response includes `agent_trace` and, on successful review, `reviewed_script`. The UI displays the roles and exposes the trace in **How this moment was made**.

`demo/reviewed_insights.py` is the lightweight execution harness. It uses a fixed sequential handoff, successful-review caching, request limits, timeouts and explicit fallback states. There is no autonomous planner, function-calling loop or separate Agent Framework dependency.

## States and failure behavior

| Result | Meaning |
| --- | --- |
| Analyst completed | Azure selected facts that passed server validation. |
| Analyst local fallback | A deterministic local explanation was used; no successful cloud analysis is claimed. |
| Reviewer passed | The response matched the evidence scope and all four review checks passed. |
| Reviewer cached | A previously successful review was reused for the same evidence and content fingerprint. |
| Reviewer rejected or unavailable | A check failed, the service failed, or a limit prevented completion. No approved script is returned. |
| Reviewer not run | Cloud analysis did not succeed, so the second role did not run. |

Review is an additional inspectable assessment, not a guarantee that every audio clip has been approved. The speech renderer remains constrained by validated facts. Existing cached audio and cache-only replay preparation can bypass the text roles. A script review also does not establish an audio transcript match.

## Configuration and limits

Both roles share `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `INSIGHT_MAX_CALLS_PER_MINUTE`, `INSIGHT_MAX_CALLS_TOTAL`, `INSIGHT_TIMEOUT_SECONDS` and `INSIGHT_CACHE_SECONDS`. No second deployment is needed. An uncached successful pair uses two text requests; cached reuse can use none.

The Reviewer has a maximum 10-second request timeout, also bounded by the configured timeout and the remaining 26-second workflow budget. It has no automatic rewrite/retry loop. Text limits are shared within one server process and reset on restart. Speech has its own persistent attempt ledger. Run one worker and preserve the limits described in [RUNNING.md](RUNNING.md).

The interface uses **Coach** for readability. API requests retain `audience: "analyst"`; **Fan** uses `"fan"`. Player focus is independent of audience depth.

## Replay and speech

The browser advances by elapsed time. Text and audio preparation do not drive its playback clock. Each clip carries an event anchor and clip boundaries used by annotations, captions and prepared speech. User playback gestures unlock audio; pause/resume and context cancellation prevent stale or overlapping playback. Missing prepared audio leaves the replay and captions usable.

TTS is preferred for new clips when `AZURE_TTS_DEPLOYMENT` is configured. The shipped voice uses built-in Alloy. The optional GPT Audio route remains in the code with transcript checks; it is not the voice used by the current prepared clips.

## Reproduce the Agent checks

After installing the development requirements, this focused harness needs no checkpoint or Azure credentials and makes no cloud calls:

```sh
.venv/bin/python -m pytest demo/test_reviewed_insights.py -q
```

The nine cases cover two distinct calls, cached reuse, wrong player or timestamp, failed checks, shared quota, timeout and unconfigured fallback. [Saved real Azure results](docs/two-agent-proof.json) separately show one Fan and one Coach request completing both roles without a cache hit. Those two examples establish integration, not a statistical accuracy or latency benchmark.
