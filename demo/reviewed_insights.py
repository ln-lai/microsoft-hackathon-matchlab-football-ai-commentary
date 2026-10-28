"""Two small LLM roles with a fixed handoff, not autonomous tool orchestration.

The existing Match Analyst selects grounded facts. A separate Reviewer model
invocation checks the rendered analysis and exact speech script. Both share
the existing request allowance, timeout and server-only credentials. Cached
speech/playback never waits for this optional review.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import time
import uuid

from demo.azure_insights import AzureInsightService
from demo.commentary import make_script


class ReviewedInsightService(AzureInsightService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reviews = OrderedDict()
        self._review_successes = 0

    def get_status(self):
        return {**super().get_status(), "agent_roles": ["Match Analyst", "Commentary Reviewer"],
                "orchestration": "Two LLM roles, sequential fixed handoff; no autonomous tool selection",
                "review_successes": self._review_successes}

    def generate(self, evidence, audience="fan", language="en"):
        started = self._clock()
        result = super().generate(evidence, audience, language)
        e = result["evidence"]
        script = make_script(e, audience, result) if e.get("highlight") else result["text"]
        fingerprint = hashlib.sha256(json.dumps([e, audience, result["text"], script], sort_keys=True).encode()).hexdigest()
        trace = {"run_id": uuid.uuid4().hex, "architecture": "Two LLM roles with a fixed handoff",
                 "analyst": {"role": "Match Analyst Agent", "status": "completed" if result["mode"] == "azure" else "local_fallback",
                             "cached": result["cached"], "fact_ids": result["fact_ids"],
                             "elapsed_ms": round((self._clock()-started)*1000), "reason": result.get("reason")},
                 "reviewer": {"role": "Commentary Reviewer Agent", "status": "not_run"},
                 "scope": {k:e[k] for k in ("match_id", "player_id", "anchor_frame", "anchor_time_s", "target_time_s")}}
        result["agent_trace"] = trace
        result["reviewed_script"] = None
        if result["mode"] != "azure":
            trace["reviewer"]["reason"] = "Analysis used local fallback; no successful two-agent claim"
            return result
        with self._lock:
            cached = self._reviews.get(fingerprint)
            if cached and self._clock()-cached[0] < self._cache_seconds:
                trace["reviewer"] = copy.deepcopy(cached[1])
                trace["reviewer"]["cached"] = True
                self._reviews.move_to_end(fingerprint)
                result["reviewed_script"] = script
                return result
        def incomplete(reason, status="unavailable"):
            trace["reviewer"].update(status=status, reason=reason, cached=False)
            return result
        if not self._request_lock.acquire(blocking=False):
            return incomplete("busy")
        try:
            now = self._clock()
            with self._lock:
                while self._recent and now-self._recent[0] >= 60:
                    self._recent.popleft()
                if self._calls >= self._total:
                    return incomplete("budget_exhausted")
                if len(self._recent) >= self._per_minute:
                    return incomplete("rate_limited")
                self._recent.append(now)
                self._calls += 1
            remaining = min(10.0, 26.0-(now-started), self._timeout)
            if remaining <= 0:
                return incomplete("timeout")
            properties = {"verdict":{"type":"string","enum":["pass","reject"]},
                          "player_id":{"type":"string"}, "anchor_frame":{"type":"integer"},
                          "target_time_s":{"type":"number"},
                          "player_correct":{"type":"boolean"}, "numbers_correct":{"type":"boolean"},
                          "time_correct":{"type":"boolean"}, "forecast_labeled":{"type":"boolean"}}
            payload = {"model":self._deployment, "max_completion_tokens":384,
                "response_format":{"type":"json_schema","json_schema":{"name":"commentary_review","strict":True,
                    "schema":{"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}}},
                "messages":[{"role":"system","content":
                    "You are the Commentary Reviewer Agent, independent of the Match Analyst. Check the proposed analysis and exact speech script against supplied evidence. "
                    "Verify player identity including spelled-out jersey numbers; all numeric claims including rounded/spoken metres; past-versus-future timing; and forecasts labeled as model estimates. "
                    "A fan script may omit distances; omission is not an error. A valid script can say space opened in the past but will tighten in the forecast. "
                    "Do not rewrite or add facts. Return the required JSON. Copy the evidence player_id, anchor_frame and target_time_s. "
                    "Set each check honestly and verdict pass only if all four checks pass. Never treat a model forecast as recorded truth."},
                    {"role":"user","content":json.dumps({"audience":audience,"evidence":e,"analysis":result["text"],"speech_script":script},allow_nan=False)}]}
            if self._reasoning_effort:
                payload["reasoning_effort"] = self._reasoning_effort
            try:
                raw = self._transport(self._endpoint,{"Content-Type":"application/json","api-key":self._key},payload,remaining)
                choice = raw["choices"][0]
                content = choice["message"]["content"]
                if choice.get("finish_reason") != "stop" or not isinstance(content,str) or len(content)>2048:
                    return incomplete("invalid_response")
                review = json.loads(content)
                checks = ["player_correct","numbers_correct","time_correct","forecast_labeled"]
                if (not isinstance(review,dict) or set(review)!=set(properties) or
                    review["player_id"]!=e["player_id"] or type(review["anchor_frame"]) is not int or
                    review["anchor_frame"]!=e["anchor_frame"] or isinstance(review["target_time_s"],bool) or
                    review["target_time_s"]!=e["target_time_s"] or any(type(review[k]) is not bool for k in checks)):
                    return incomplete("invalid_response")
                if self._clock()-started > 26:
                    return incomplete("timeout")
                if review["verdict"]!="pass" or not all(review[k] for k in checks):
                    return incomplete("evidence_check_failed", "rejected")
            except Exception:
                return incomplete("service_unavailable")
            trace["reviewer"].update(status="passed", cached=False, checked_at_unix=round(time.time()),
                                    checks=checks, elapsed_ms=round((self._clock()-now)*1000),
                                    script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                                    evidence_sha256=fingerprint)
            result["reviewed_script"] = script
            with self._lock:
                self._review_successes += 1
                self._reviews[fingerprint] = (self._clock(), copy.deepcopy(trace["reviewer"]))
                while len(self._reviews)>128:
                    self._reviews.popitem(last=False)
            return result
        finally:
            self._request_lock.release()
