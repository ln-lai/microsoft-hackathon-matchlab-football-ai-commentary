"""Microsoft Foundry evidence selection for the synthetic replay demo.

The remote model selects and orders *existing, computed facts*. It does not
author unchecked tactical assertions. Returned text is rendered from those
facts, so a cloud response cannot invent a pass, goal, probability or distance.
This module never loads future truth, training files, or model weights.

Official v1 API reference (both supported resource hostnames):
https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle

Required environment: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT and
AZURE_OPENAI_API_KEY. Optional cost guards: INSIGHT_MAX_CALLS_PER_MINUTE (6),
INSIGHT_MAX_CALLS_TOTAL (100), INSIGHT_TIMEOUT_SECONDS (10),
INSIGHT_CACHE_SECONDS (1800). Counters are per process and reset on restart;
deploy one process/replica, or impose an external shared quota when scaling.
No retries are performed, and failed remote attempts also consume the quota.
"""
from collections import OrderedDict, deque
import copy
import hashlib
import http.client
import json
import math
import os
import re
import threading
import time
from urllib.parse import urlsplit


PROVIDER = "Microsoft Foundry / Azure OpenAI"
LOCAL_PROVIDER = "Local geometry summary"
FIELDS = {"match_id", "anchor_frame", "anchor_time_s", "target_time_s", "synthetic",
          "player_id", "team", "current_nearest_opponent_m", "predicted_nearest_opponent_m",
          "predicted_displacement_m", "model_name", "checkpoint_sha256"}
HIGHLIGHT_FIELDS = {"id", "kind", "analysis_mode", "window_start_s", "window_end_s",
                    "previous_nearest_opponent_m", "current_nearest_opponent_m",
                    "observed_gap_change_m", "observed_displacement_m", "selection_score"}
CONTENT_VERSION = "temporal-evidence-v3"
LIMITATION = ("This is a one-second model projection on synthetic data. Nearest-opponent "
              "distance alone does not establish a safe pass, offside status or scoring chance.")
REASONS = {
    "not_configured": "Microsoft service is not configured; this is a local geometry summary.",
    "invalid_configuration": "Microsoft service configuration needs correction; this is a local geometry summary.",
    "rate_limited": "The Microsoft service request limit has been reached; this is a local geometry summary.",
    "budget_exhausted": "The server's Microsoft service call allowance is exhausted; this is a local geometry summary.",
    "busy": "A Microsoft request is already running; this is a local geometry summary.",
    "service_unavailable": "The Microsoft service could not complete this request; this is a local geometry summary.",
    "invalid_response": "The Microsoft response did not pass evidence checks; this is a local geometry summary.",
}


def _endpoint_url(value):
    """Accept only an HTTPS Azure resource, never user URLs or redirects."""
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password or
            parsed.port not in (None, 443) or parsed.query or parsed.fragment or
            not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}\.(?:openai\.azure\.com|services\.ai\.azure\.com)", host) or
            parsed.path.rstrip("/") not in ("", "/openai/v1")):
        raise ValueError("Endpoint must be an HTTPS Azure OpenAI or Foundry resource root or /openai/v1")
    return f"https://{host}/openai/v1/chat/completions"


def _env_number(env, name, default, minimum, maximum, integer=True):
    value = (int if integer else float)(env.get(name, default))
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("Invalid service limit")
    return value


def validate_evidence(evidence):
    """Whitelist causal predictions and optional observed-past selection facts."""
    if not isinstance(evidence, dict) or set(evidence) not in (FIELDS, FIELDS | {"highlight"}):
        raise ValueError("Evidence must contain exactly the documented causal evidence fields")
    result = copy.deepcopy(evidence)
    for key in ("match_id", "player_id"):
        if not isinstance(result[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", result[key]):
            raise ValueError(f"Invalid {key}")
    if result["synthetic"] is not True or result["team"] not in ("home", "away"):
        raise ValueError("This demo accepts only synthetic evidence and a known team")
    if result["model_name"] != "ST-GNN + TCN":
        raise ValueError("Evidence must identify the frozen prediction model")
    if not isinstance(result["checkpoint_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", result["checkpoint_sha256"]):
        raise ValueError("Invalid checkpoint hash")
    if type(result["anchor_frame"]) is not int or not 98 <= result["anchor_frame"] <= 10_000_000:
        raise ValueError("Invalid anchor frame")
    for key in ("anchor_time_s", "target_time_s", "current_nearest_opponent_m",
                "predicted_nearest_opponent_m", "predicted_displacement_m"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1_000_000:
            raise ValueError(f"Invalid numeric evidence: {key}")
        result[key] = float(value)
    if not math.isclose(result["target_time_s"] - result["anchor_time_s"], 1.0, abs_tol=1e-7):
        raise ValueError("Evidence horizon must be exactly one second")
    if "highlight" in result:
        observed = result["highlight"]
        if not isinstance(observed, dict) or set(observed) != HIGHLIGHT_FIELDS:
            raise ValueError("Unknown highlight evidence fields")
        if (not isinstance(observed["id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", observed["id"]) or
                observed["kind"] not in ("space", "pressure", "movement") or
                observed["analysis_mode"] != "retrospective"):
            raise ValueError("Invalid observed highlight identity or mode")
        numeric_keys = HIGHLIGHT_FIELDS - {"id", "kind", "analysis_mode"}
        for key in numeric_keys:
            value = observed[key]
            minimum = -1_000_000 if key == "observed_gap_change_m" else 0
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not minimum <= value <= 1_000_000:
                raise ValueError(f"Invalid observed evidence: {key}")
            observed[key] = float(value)
        if (not math.isclose(observed["window_end_s"], result["anchor_time_s"], abs_tol=1e-7) or
                not math.isclose(observed["window_end_s"] - observed["window_start_s"], 1.0, abs_tol=1e-7)):
            raise ValueError("Observed selection evidence must end at the anchor and cover its previous second")
        delta = observed["current_nearest_opponent_m"] - observed["previous_nearest_opponent_m"]
        if (not math.isclose(observed["current_nearest_opponent_m"], result["current_nearest_opponent_m"], abs_tol=1e-4) or
                not math.isclose(delta, observed["observed_gap_change_m"], abs_tol=1e-7) or
                not math.isclose(abs(delta) + .35 * observed["observed_displacement_m"], observed["selection_score"], abs_tol=1e-7)):
            raise ValueError("Observed selection values must agree with the measured evidence and score")
        kind = "space" if delta >= .5 else "pressure" if delta <= -.5 else "movement"
        if observed["kind"] != kind or (abs(delta) < .5 and observed["observed_displacement_m"] < 1.5):
            raise ValueError("Observed highlight does not meet its stated detector threshold")
    return result


def evidence_facts(evidence, audience):
    """Keep observed past, current measurements and model estimates separate."""
    label = evidence["player_id"].replace("_", " ")
    before = evidence["current_nearest_opponent_m"]
    after = evidence["predicted_nearest_opponent_m"]
    change = after - before
    direction = "increases" if change >= 0 else "decreases"
    if audience == "fan":
        separation = (f"{label}'s distance to the nearest opponent is projected to {('grow' if change >= 0 else 'shrink')} "
                      f"from {before:.1f} m to {after:.1f} m over the next second.")
        movement = f"{label} is projected to move {evidence['predicted_displacement_m']:.1f} m from the current position."
    else:
        separation = (f"For {label}, minimum distance to all opposing players {direction} from {before:.2f} m "
                      f"to {after:.2f} m at +1 s (change {change:+.2f} m), using both teams' model-predicted positions.")
        movement = (f"{label}'s predicted displacement is {evidence['predicted_displacement_m']:.2f} m "
                    f"between t={evidence['anchor_time_s']:.2f} s and t={evidence['target_time_s']:.2f} s.")
    rounded_current = round(before)
    current_distance = ("less than a metre" if rounded_current == 0 else
                        "about a metre" if rounded_current == 1 else f"about {rounded_current} metres")
    current = (f"At this moment, {label}'s nearest opponent is {before:.2f} m away."
               if audience == "analyst" else
               f"At this moment, {label}'s nearest opponent is {current_distance} away.")
    facts = {"separation": separation, "movement": movement, "current": current}
    if "highlight" in evidence:
        h = evidence["highlight"]
        delta = h["observed_gap_change_m"]
        action = "opened" if delta >= 0 else "lost"
        if h["kind"] == "movement":
            observed = (f"{label} just moved {h['observed_displacement_m']:.0f} metres over the previous second."
                        if audience == "fan" else
                        f"Observed from t={h['window_start_s']:.2f} to {h['window_end_s']:.2f} s: {label} moved {h['observed_displacement_m']:.2f} m; nearest-opponent gap changed {delta:+.2f} m.")
        elif audience == "fan":
            amount = round(abs(delta))
            rounded = ("about a metre" if amount == 1 else f"about {amount} metres") if abs(delta) >= 1 else "a little"
            observed = f"{label} just {action} {rounded} of space to the nearest opponent."
        else:
            observed = (f"Observed from t={h['window_start_s']:.2f} to {h['window_end_s']:.2f} s: {label}'s nearest-opponent gap "
                        f"changed from {h['previous_nearest_opponent_m']:.2f} m to {h['current_nearest_opponent_m']:.2f} m ({delta:+.2f} m).")
        facts["observed"] = observed
        if audience == "fan":
            projected = ("stay almost unchanged" if abs(change) < .1 else
                         "grow a little" if 0 <= change < 1 else "shrink a little" if -1 < change < 0 else
                         "grow" if change >= 0 else "shrink")
            facts["separation"] = f"The model expects that gap to {projected} over the next second."
    return facts


def _https_post(url, headers, payload, timeout):
    """Bounded HTTPS call without proxy rewriting, redirects, retries or logging."""
    parsed = urlsplit(url)
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=timeout)
    try:
        connection.request("POST", parsed.path, body=json.dumps(payload, allow_nan=False).encode("utf-8"), headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ConnectionError("Microsoft service returned an unsuccessful status")
        body = response.read(65_537)
        if len(body) > 65_536:
            raise ValueError("Microsoft response exceeded size limit")
        return json.loads(body)
    finally:
        connection.close()


class AzureInsightService:
    def __init__(self, env=None, *, transport=None, clock=None):
        env = os.environ if env is None else env
        self._transport = transport or _https_post
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._recent = deque()
        self._cache = OrderedDict()
        self._calls = 0
        self._successes = 0
        self._last_reason = None
        self._configuration_reason = None
        self._endpoint = None
        self._key = None
        self._deployment = None
        self._reasoning_effort = None
        # Initialize defaults even if optional configuration is malformed.
        self._per_minute, self._total, self._timeout, self._cache_seconds = 6, 100, 10, 1800
        try:
            self._per_minute = _env_number(env, "INSIGHT_MAX_CALLS_PER_MINUTE", 6, 1, 60)
            self._total = _env_number(env, "INSIGHT_MAX_CALLS_TOTAL", 100, 1, 10_000)
            self._timeout = _env_number(env, "INSIGHT_TIMEOUT_SECONDS", 10, 1, 30, False)
            self._cache_seconds = _env_number(env, "INSIGHT_CACHE_SECONDS", 1800, 1, 86_400)
            effort = env.get("AZURE_OPENAI_REASONING_EFFORT", "")
            if effort and effort not in {"none", "minimal", "low", "medium", "high"}:
                raise ValueError("Invalid reasoning effort")
            self._reasoning_effort = effort or None
            required = [str(env.get(key, "")).strip() for key in
                        ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_KEY")]
            if not all(required):
                self._configuration_reason = "not_configured"
            else:
                endpoint, deployment, key = required
                self._endpoint = _endpoint_url(endpoint)
                if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", deployment) or not 8 <= len(key) <= 512 or any(ord(ch) < 32 for ch in key):
                    raise ValueError("Invalid service credentials")
                self._deployment, self._key = deployment, key
        except (TypeError, ValueError):
            self._configuration_reason = "invalid_configuration"

    def get_status(self):
        with self._lock:
            return {"provider": PROVIDER, "configured": self._configuration_reason is None,
                    "live_verified": self._successes > 0, "successful_calls": self._successes,
                    "requests_used": self._calls, "requests_remaining": max(0, self._total - self._calls),
                    "max_calls_per_minute": self._per_minute,
                    "quota_scope": "one server process; resets on restart", "language": "en",
                    "reason": self._configuration_reason or self._last_reason,
                    "capability": "Cloud model selects and orders computed facts; server renders validated evidence only."}

    def _result(self, evidence, audience, selected, *, mode, reason=None, cached=False):
        facts = evidence_facts(evidence, audience)
        text = " ".join(facts[key] for key in selected)
        if "highlight" not in evidence:
            text += " " + LIMITATION
        return {"provider": PROVIDER if mode == "azure" else LOCAL_PROVIDER,
                "mode": mode, "text": text,
                "stages": {"observed": facts.get("observed", ""),
                           "current": facts["current"],
                           "forecast": facts["separation"]},
                "observed_available": "highlight" in evidence,
                "limitation": LIMITATION, "content_version": CONTENT_VERSION,
                "evidence": evidence, "fact_ids": list(selected), "grounded": True,
                "grounding_method": "server-rendered statements from computed numerical evidence",
                "audience": audience, "language": "en", "cached": cached,
                "reason": reason, "notice": REASONS.get(reason) if reason else "Microsoft service selected the facts and their order; numbers are computed locally."}

    def generate(self, evidence, audience="fan", language="en"):
        evidence = validate_evidence(evidence)
        if audience not in ("fan", "analyst") or language != "en":
            raise ValueError("Choose fan or analyst audience; this demo currently supports English")
        defaults = ("observed", "separation") if "highlight" in evidence else ("separation", "movement")
        fallback = lambda reason: self._result(evidence, audience, defaults, mode="local", reason=reason)
        if self._configuration_reason:
            return fallback(self._configuration_reason)
        cache_key = hashlib.sha256(json.dumps([CONTENT_VERSION, evidence, audience, language], sort_keys=True, allow_nan=False).encode()).hexdigest()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and self._clock() - cached[0] < self._cache_seconds:
                self._cache.move_to_end(cache_key)
                result = copy.deepcopy(cached[1])
                result["cached"] = True
                return result
        if not self._request_lock.acquire(blocking=False):
            return fallback("busy")
        try:
            now = self._clock()
            with self._lock:
                while self._recent and now - self._recent[0] >= 60:
                    self._recent.popleft()
                if self._calls >= self._total:
                    return fallback("budget_exhausted")
                if len(self._recent) >= self._per_minute:
                    return fallback("rate_limited")
                self._recent.append(now)
                self._calls += 1
            facts = evidence_facts(evidence, audience)
            # Azure supports strict schemas but not array minItems/maxItems or
            # uniqueItems. Bound the allowed IDs here; retain the independent
            # length, uniqueness and required-highlight checks below.
            allowed_ids = ["observed", "separation"] if "highlight" in evidence else ["separation", "movement"]
            response_format = {"type": "json_schema", "json_schema": {
                "name": "grounded_fact_selection", "strict": True,
                "schema": {"type": "object", "properties": {
                    "fact_ids": {"type": "array", "items": {"type": "string", "enum": allowed_ids}}},
                    "required": ["fact_ids"], "additionalProperties": False}}}
            payload = {"model": self._deployment, "max_completion_tokens": 256,
                       "response_format": response_format, "messages": [
                           {"role": "system", "content": "You are an editor for a synthetic football replay. Select the most useful supplied fact first for the requested audience. Return only a JSON object with the key fact_ids, containing one or two distinct IDs from available_facts. When highlight selection evidence is present, include observed and separation: observed explains why this retrospective highlight was selected over its PAST second; current describes a measurement at the anchor; separation and movement describe MODEL estimates for the NEXT second. Never confuse an observed increase with a forecast increase. Do not invent or rewrite text, numbers or facts. There is no future ground truth. A projection is not a confirmed outcome. For fans prioritize an easily understood change; for analysts prioritize the most informative measured change. No other keys."},
                           {"role": "user", "content": json.dumps({"audience": audience, "language": language, "evidence": evidence, "available_facts": facts}, allow_nan=False)}]}
            if self._reasoning_effort:
                payload["reasoning_effort"] = self._reasoning_effort
            try:
                raw = self._transport(self._endpoint,
                                      {"Content-Type": "application/json", "api-key": self._key},
                                      payload, self._timeout)
            except Exception:
                # Third-party errors can contain URLs, headers or response bodies;
                # never interpolate them into logs or public responses.
                with self._lock:
                    self._last_reason = "service_unavailable"
                return fallback("service_unavailable")
            try:
                choice = raw["choices"][0]
                content = choice["message"]["content"]
                if choice.get("finish_reason") != "stop" or not isinstance(content, str) or len(content) > 4096:
                    raise ValueError("Incomplete or oversized selection")
                selection = json.loads(content)
                if not isinstance(selection, dict) or set(selection) != {"fact_ids"}:
                    raise ValueError("Unexpected selection fields")
                ids = selection["fact_ids"]
                if (not isinstance(ids, list) or not 1 <= len(ids) <= 2 or
                        any(not isinstance(item, str) or item not in allowed_ids for item in ids) or len(set(ids)) != len(ids)):
                    raise ValueError("Unknown or duplicate facts")
                if "highlight" in evidence and set(ids) != {"observed", "separation"}:
                    raise ValueError("A highlight explanation must include its observed reason and separate forecast")
            except (KeyError, TypeError, ValueError, IndexError):
                with self._lock:
                    self._last_reason = "invalid_response"
                return fallback("invalid_response")
            result = self._result(evidence, audience, ids, mode="azure")
            with self._lock:
                self._successes += 1
                self._last_reason = None
                self._cache[cache_key] = (self._clock(), copy.deepcopy(result))
                self._cache.move_to_end(cache_key)
                while len(self._cache) > 256:
                    self._cache.popitem(last=False)
            return result
        finally:
            self._request_lock.release()
