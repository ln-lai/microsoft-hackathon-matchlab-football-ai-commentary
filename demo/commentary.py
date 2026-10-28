"""Bounded Azure audio commentary from validated replay evidence.

Audio is generated in advance from validated scripts, then cached. GPT audio
must also pass a provider-transcript check; TTS input is labelled separately.
This is not a
live broadcast or an unrestricted public speech endpoint. Credentials stay on
the server. Neither the script nor the speech request includes future truth.
"""
from collections import deque
import base64
import copy
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit
import wave

from demo.azure_insights import _endpoint_url, _env_number, validate_evidence

CONTENT_VERSION = "matchlab-commentary-v3"
PROVIDER = "Microsoft Foundry / Azure OpenAI"
ROOT = Path(__file__).resolve().parent
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()


def spoken_number(value):
    value = round(float(value), 1)
    whole, tenth = divmod(int(round(abs(value) * 10)), 10)
    if whole < 20:
        result = WORDS[whole]
    elif whole < 100:
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        result = tens[whole // 10] + (" " + WORDS[whole % 10] if whole % 10 else "")
    else:
        raise ValueError("Spoken geometry is outside this demo's range")
    return result + (" point " + WORDS[tenth] if tenth else "")


def spoken_metres(value):
    rounded = round(float(value), 1)
    return spoken_number(rounded) + (" metre" if rounded == 1 else " metres")


def commentary_energy(evidence, audience):
    """Delivery intensity uses observed motion, never the forecast or future truth."""
    past = evidence["highlight"]
    return ("lively" if audience == "fan" and
            (abs(past["observed_gap_change_m"]) >= 1.0 or past["observed_displacement_m"] >= 3.0)
            else "calm")


def commentary_prosody(energy):
    if energy == "lively":
        return ("Use lively, warm sports-commentary energy, a clear lift on the observed change, "
                "then measured emphasis on the word model. Keep it brisk without shouting.")
    return "Use calm, attentive sports commentary with a clear, brisk pace and measured emphasis on the word model."


def normalize_transcript(text):
    if not isinstance(text, str) or len(text) > 2000:
        raise ValueError("Invalid speech transcript")
    def number(match):
        raw = match.group()
        sign = "minus " if raw.startswith("-") else "plus " if raw.startswith("+") else ""
        raw = raw.lstrip("+-")
        whole, dot, fraction = raw.partition(".")
        return sign + spoken_number(int(whole)) + (" point " + " ".join(WORDS[int(x)] for x in fraction) if dot else "")
    text = re.sub(r"(?<![\w.])[+-]?\d+(?:\.\d+)?\b", number, text.lower())
    text = text.replace("meters", "metres").replace("meter", "metre").replace("’", "'")
    return " ".join(re.findall(r"[a-z]+", text))


def make_script(evidence, audience, insight=None):
    evidence = validate_evidence(evidence)
    if audience not in {"fan", "analyst"} or not evidence.get("highlight"):
        raise ValueError("Commentary requires a validated discovered highlight")
    team, number = evidence["player_id"].rsplit("_", 1)
    label = f"{team.title()} {spoken_number(int(number))}"
    past = evidence["highlight"]
    delta = past["observed_gap_change_m"]
    projected = evidence["predicted_nearest_opponent_m"]
    now = evidence["current_nearest_opponent_m"]
    if audience == "analyst":
        if abs(delta) >= .05:
            observed = f"{label}. The gap {'grew' if delta >= 0 else 'fell'} by {spoken_metres(abs(delta))}."
        else:
            observed = f"{label}. The gap barely changed."
        forecast = f"The model projects {spoken_metres(projected)} next second."
    else:
        lively = commentary_energy(evidence, audience) == "lively"
        observed = ((f"Look at {label}, space just opened up!" if lively else f"{label} just gained a little space.") if delta >= .5 else
                    (f"{label}, that space just tightened!" if lively else f"{label}'s space just narrowed.") if delta <= -.5 else
                    f"{label} just covered ground!")
        trend = "a similar" if abs(projected-now) < .1 else "a wider" if projected > now else "a tighter"
        forecast = f"The model predicts {trend} gap next second."
    clauses = {"observed": observed, "separation": forecast}
    # Spoken chronology stays past -> forecast. The existing Agent still
    # selects/orders the evidence card and its validated provenance is kept
    # in prepare(); it cannot reorder an already prepared narration.
    return " ".join(clauses[key] for key in ("observed", "separation"))


def _post_audio(url, headers, payload, timeout):
    parsed = urlsplit(url)
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=timeout)
    try:
        connection.request("POST", parsed.path, body=json.dumps(payload, allow_nan=False).encode(), headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ConnectionError("Audio service unavailable")
        raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError("Oversized audio response")
        return json.loads(raw)
    finally:
        connection.close()


def commentary_cache_key(evidence, highlight, audience, language, text):
    # ARM and x86 inference can differ by a few millionths of a metre.
    # Quantize only cache identity to millimetres; retain exact evidence and
    # the rendered script (including every spoken number) in the response.
    def stable(value):
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, dict):
            return {key: stable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value
    identity = [CONTENT_VERSION, "evidence-locked-speech", stable(evidence), highlight["id"],
                evidence["player_id"], language, audience, text,
                float(highlight["clip_start_s"]), float(highlight["clip_end_s"])]
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _post_tts(url, headers, payload, timeout):
    """Exact-script TTS; bounded binary response, no redirects or retries."""
    parsed = urlsplit(url)
    connection = http.client.HTTPSConnection(parsed.hostname, timeout=timeout)
    try:
        path = parsed.path + ("?" + parsed.query if parsed.query else "")
        connection.request("POST", path, body=json.dumps(payload, allow_nan=False).encode(), headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise ConnectionError("TTS service unavailable")
        raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError("Oversized TTS response")
        return raw
    finally:
        connection.close()


class CommentaryService:
    def __init__(self, env=None, *, transport=None, tts_transport=None, clock=None):
        env = os.environ if env is None else env
        self._transport = transport or _post_audio
        self._tts_transport = tts_transport or _post_tts
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._recent = deque()
        self._calls = 0
        self._successes = 0
        self._reason = None
        self._config_reason = None
        self._endpoint = self._key = self._deployment = None
        self._tts_deployment = self._tts_endpoint = None
        self._per_minute, self._total, self._timeout = 3, 60, 25
        default_cache = "/home/data/the-match-lab-commentary" if env.get("WEBSITE_SITE_NAME") else str(ROOT / "artifacts/commentary-cache")
        self._cache_dir = Path(env.get("COMMENTARY_CACHE_DIR", default_cache))
        self._seed_dir = Path(env.get("COMMENTARY_SEED_DIR", str(ROOT / "commentary_seed")))
        try:
            self._per_minute = _env_number(env, "COMMENTARY_MAX_CALLS_PER_MINUTE", 3, 1, 12)
            self._total = _env_number(env, "COMMENTARY_MAX_CALLS_TOTAL", 60, 1, 200)
            self._timeout = _env_number(env, "COMMENTARY_TIMEOUT_SECONDS", 25, 1, 30, False)
            endpoint = env.get("AZURE_OPENAI_ENDPOINT", "")
            self._deployment = env.get("AZURE_AUDIO_DEPLOYMENT", "")
            self._tts_deployment = env.get("AZURE_TTS_DEPLOYMENT", "")
            self._key = env.get("AZURE_OPENAI_API_KEY", "")
            if not all((endpoint, self._deployment or self._tts_deployment, self._key)):
                self._config_reason = "not_configured"
            else:
                self._endpoint = _endpoint_url(endpoint)
                if (any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name) for name in (self._deployment, self._tts_deployment) if name) or
                        not 8 <= len(self._key) <= 512 or any(ord(char) < 32 for char in self._key)):
                    raise ValueError("Invalid audio configuration")
                if self._tts_deployment:
                    self._tts_endpoint = ("https://" + urlsplit(self._endpoint).hostname +
                                          f"/openai/deployments/{self._tts_deployment}/audio/speech?api-version=2025-04-01-preview")
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            ledger = self._cache_dir / "usage.json"
            if ledger.exists():
                self._calls = max(0, int(json.loads(ledger.read_text())["calls"]))
        except (OSError, ValueError, TypeError, KeyError):
            self._config_reason = "invalid_configuration"

    def get_status(self):
        return {"provider": PROVIDER, "configured": self._config_reason is None,
                "live_verified": self._successes > 0, "successful_calls": self._successes,
                "requests_used": self._calls, "requests_remaining": max(0, self._total-self._calls),
                "max_calls_per_minute": self._per_minute, "reason": self._config_reason or self._reason,
                "delivery": "pre-generated, evidence-checked synchronized commentary",
                "uncached_synthesis": "OpenAI TTS" if self._tts_deployment else "GPT Audio Mini",
                "content_version": CONTENT_VERSION, "quota_scope": "persistent server cache; one worker"}

    def _read(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            return None
        for directory in (self._cache_dir, self._seed_dir):
            try:
                meta = json.loads((directory / f"{key}.json").read_text())
                if not isinstance(meta, dict):
                    continue
                audio = (directory / f"{key}.wav").read_bytes()
                if (meta.get("cache_key") != key or meta.get("content_version") != CONTENT_VERSION or
                        hashlib.sha256(audio).hexdigest() != meta.get("audio_sha256")):
                    continue
                return meta, audio
            except (OSError, ValueError):
                continue
        return None

    def audio(self, key):
        value = self._read(key)
        return value[1] if value else None

    def prepare(self, evidence, highlight, audience="fan", language="en", *, insight=None, cache_only=False):
        if language != "en" or highlight["id"] != evidence.get("highlight", {}).get("id"):
            raise ValueError("Commentary must match a known English highlight")
        if (highlight.get("player_id") != evidence.get("player_id") or
                highlight.get("anchor_frame") != evidence.get("anchor_frame") or
                not math.isclose(float(highlight.get("time_s", -1)), evidence.get("anchor_time_s", -2), abs_tol=1e-7)):
            raise ValueError("Highlight, player and time must match commentary evidence")
        chosen = insight.get("fact_ids") if isinstance(insight, dict) else None
        accepted_insight = (isinstance(insight, dict) and insight.get("evidence") == evidence and
                            insight.get("audience") == audience and insight.get("language") == language and
                            insight.get("mode") in ("azure", "local") and isinstance(chosen, list) and
                            len(chosen) == 2 and all(isinstance(item, str) for item in chosen) and
                            set(chosen) == {"observed", "separation"})
        insight = insight if accepted_insight else None
        text = make_script(evidence, audience, insight)
        energy = commentary_energy(evidence, audience)
        clip_start, clip_end = float(highlight["clip_start_s"]), float(highlight["clip_end_s"])
        if not (math.isfinite(clip_start) and math.isfinite(clip_end) and
                0 <= clip_start <= evidence["highlight"]["window_start_s"] and
                evidence["target_time_s"] <= clip_end and 0 < clip_end-clip_start <= 12):
            raise ValueError("Invalid commentary clip")
        key = commentary_cache_key(evidence, highlight, audience, language, text)
        base = {"status": "unavailable", "cache_key": key, "content_version": CONTENT_VERSION,
                "match_id": evidence["match_id"], "highlight_id": highlight["id"],
                "player_id": evidence["player_id"], "audience": audience, "language": language,
                "text": text, "provider": PROVIDER, "cached": False,
                "energy": energy, "energy_basis": "observed geometry over the previous second",
                "prosody": commentary_prosody(energy),
                "clip_start_s": clip_start, "clip_end_s": clip_end,
                "anchor_time_s": evidence["anchor_time_s"], "cue_timebase": "replay",
                "cues": [{"start_s": clip_start, "end_s": clip_end, "text": text}],
                "disclosure": "AI-generated commentary · prepared in advance",
                "script_method": "Validated rendering of observed highlight evidence and model forecast",
                "evidence_source": (insight or {}).get("mode", "local"),
                "fact_ids": (insight or {}).get("fact_ids", [])}
        cached = self._read(key)
        if cached:
            result = copy.deepcopy(cached[0]); result["cached"] = True
            return result
        if cache_only:
            # Internal read-only probe: prepared speech should never wait for
            # a fresh text-model response or consume a generation allowance.
            return {**base, "reason": "cache_miss"}
        def unavailable(reason):
            self._reason = reason
            return {**base, "reason": reason, "notice": "Commentary audio is unavailable. Replay, subtitles and evidence remain available."}
        if self._config_reason:
            return unavailable(self._config_reason)
        if clip_end-evidence["anchor_time_s"] < 2.5:
            result = unavailable("clip_too_short")
            result["notice"] = "This boundary clip is too short for spoken commentary. Replay with captions and evidence remains available."
            return result
        if not self._request_lock.acquire(blocking=False):
            return unavailable("busy")
        try:
            # Recheck after acquiring generation lock, before consuming quota.
            cached = self._read(key)
            if cached:
                return {**cached[0], "cached": True}
            now = self._clock()
            with self._lock:
                while self._recent and now-self._recent[0] >= 60:
                    self._recent.popleft()
                if self._calls >= self._total:
                    return unavailable("budget_exhausted")
                if len(self._recent) >= self._per_minute:
                    return unavailable("rate_limited")
                self._calls += 1
                self._recent.append(now)
                # Persist attempts before the paid call. Failure is counted too.
                ledger = self._cache_dir / "usage.json"
                tmp = self._cache_dir / "usage.tmp"
                tmp.write_text(json.dumps({"calls": self._calls}))
                tmp.replace(ledger)
            payload = {"model": self._deployment, "modalities": ["text", "audio"], "temperature": 0,
                       "audio": {"voice": "alloy", "format": "wav"}, "max_completion_tokens": 600,
                       "messages": [{"role": "system", "content":
                           "You are a speech synthesizer. Read only the exact supplied script, then stop immediately. "
                           "Speak in a generic sports commentator voice; do not imitate any real person. "
                           + commentary_prosody(energy) + " "
                           "Do not add any introduction, ending, reaction, sound effect, music, or extra words. Never rewrite numbers."},
                           {"role": "user", "content": "Read exactly and then stop:\n" + text}]}
            try:
                headers = {"Content-Type": "application/json", "api-key": self._key}
                if self._tts_deployment:
                    # Exact TTS is preferred for new clips; validated cached GPT
                    # clips remain reusable under the same narration contract.
                    audio = self._tts_transport(self._tts_endpoint, headers,
                                                {"model": "tts-1", "input": text, "voice": "alloy",
                                                 "response_format": "wav", "speed": 1.15}, self._timeout)
                    transcript = text
                    raw = None
                else:
                    raw = self._transport(self._endpoint, headers, payload, self._timeout)
            except Exception:
                return unavailable("service_unavailable")
            try:
                if not self._tts_deployment:
                    choice = raw["choices"][0]
                    generated = choice["message"]["audio"]
                    transcript = generated["transcript"]
                    if choice.get("finish_reason") != "stop" or normalize_transcript(transcript) != normalize_transcript(text):
                        return unavailable("transcript_mismatch")
                    audio = base64.b64decode(generated["data"], validate=True)
                with wave.open(io.BytesIO(audio), "rb") as reader:
                    channels, width, rate = reader.getnchannels(), reader.getsampwidth(), reader.getframerate()
                    if channels not in (1, 2) or width != 2 or not 8000 <= rate <= 48000:
                        raise ValueError("Unexpected audio encoding")
                    # Streaming WAV headers can declare an unknown data length.
                    # Validate actual PCM bytes and write an accurate browser WAV.
                    pcm = reader.readframes(rate * 13)
                    if len(pcm) % (channels * width):
                        raise ValueError("Partial PCM frame")
                    duration = len(pcm)/(rate*channels*width)
                lead_s = evidence["anchor_time_s"] - clip_start
                if not math.isfinite(duration) or not .5 <= duration <= clip_end-evidence["anchor_time_s"]:
                    return unavailable("audio_duration_mismatch")
                speech_duration = duration
                pcm = bytes(round(lead_s*rate)*channels*width) + pcm
                duration = len(pcm)/(rate*channels*width)
                output = io.BytesIO()
                with wave.open(output, "wb") as writer:
                    writer.setnchannels(channels); writer.setsampwidth(width); writer.setframerate(rate)
                    writer.writeframes(pcm)
                audio = output.getvalue()
            except (KeyError, IndexError, TypeError, ValueError, wave.Error, EOFError):
                return unavailable("invalid_audio")
            result = {**base, "status": "ready", "audio_url": f"/api/commentary/audio/{key}.wav",
                      "duration_s": duration, "transcript": transcript, "transcript_verified": not bool(self._tts_deployment),
                      "transcript_source": "synthesis_input" if self._tts_deployment else "provider_audio_transcript",
                      "speech_duration_s": speech_duration, "speech_start_s": evidence["anchor_time_s"],
                      "audio_sha256": hashlib.sha256(audio).hexdigest(), "reason": None,
                      "voice": "alloy", "model": "tts" if self._tts_deployment else "gpt-audio-mini",
                      "model_version": "001" if self._tts_deployment else "2025-12-15",
                      "delivery_control": "tts_default_voice" if self._tts_deployment else "gpt_prosody_instruction",
                      "cues": [{"start_s": clip_start, "end_s": evidence["anchor_time_s"],
                                "text": "Follow " + evidence["player_id"].replace("_", " ") + " into this highlight."},
                               {"start_s": evidence["anchor_time_s"], "end_s": clip_start+duration, "text": text}]}
            # WAV first; publish metadata last so incomplete writes are never used.
            (self._cache_dir / f"{key}.wav").write_bytes(audio)
            temp = self._cache_dir / f"{key}.tmp"
            temp.write_text(json.dumps(result, indent=2))
            temp.replace(self._cache_dir / f"{key}.json")
            self._successes += 1
            self._reason = None
            return result
        except OSError:
            return unavailable("cache_unavailable")
        finally:
            self._request_lock.release()
