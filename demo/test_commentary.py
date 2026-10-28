"""Offline speech boundary and API tests; generated WAVs are mock silence.

These tests prove synchronization metadata, validation and failure isolation.
They are deliberately not evidence of a live Azure voice or audible playback.
"""
import base64
import copy
import io
import json
import threading
import wave
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from demo import backend, commentary
from demo.azure_insights import AzureInsightService
from demo.commentary import CommentaryService, make_script, normalize_transcript


@pytest.fixture
def evidence():
    return {"match_id": "space-run", "anchor_frame": 300, "anchor_time_s": 12.0,
            "target_time_s": 13.0, "synthetic": True, "player_id": "Home_9", "team": "home",
            "current_nearest_opponent_m": 2.25, "predicted_nearest_opponent_m": 3.1,
            "predicted_displacement_m": 4.2, "model_name": "ST-GNN + TCN", "checkpoint_sha256": "a" * 64,
            "highlight": {"id": "space-run-300-8", "kind": "space", "analysis_mode": "retrospective",
                          "window_start_s": 11.0, "window_end_s": 12.0,
                          "previous_nearest_opponent_m": 1.0, "current_nearest_opponent_m": 2.25,
                          "observed_gap_change_m": 1.25, "observed_displacement_m": 4.0,
                          "selection_score": 2.65}}


@pytest.fixture
def highlight():
    return {"id": "space-run-300-8", "anchor_frame": 300, "time_s": 12.0,
            "player_id": "Home_9", "clip_start_s": 9.0, "clip_end_s": 19.0}


@pytest.fixture
def environment(tmp_path):
    return {"AZURE_OPENAI_ENDPOINT": "https://commentary-test.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "never-a-real-key", "AZURE_AUDIO_DEPLOYMENT": "mock-audio",
            "COMMENTARY_CACHE_DIR": str(tmp_path / "cache"), "COMMENTARY_SEED_DIR": str(tmp_path / "seed")}


def wav_bytes(duration=2.0, channels=1, sample_width=2, rate=8000):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(sample_width)
        audio.setframerate(rate)
        audio.writeframes(b"\0" * round(duration * rate) * channels * sample_width)
    return output.getvalue()


def completion(text, *, duration=2.0, finish="stop", audio=None):
    return {"choices": [{"finish_reason": finish, "message": {"audio": {
        "transcript": text, "data": base64.b64encode(wav_bytes(duration) if audio is None else audio).decode()}}}]}


def script_from_payload(payload):
    content = payload["messages"][-1]["content"]
    return content.removeprefix("Read exactly and then stop:\n")


def transport_spy(calls, **changes):
    def transport(url, headers, payload, timeout):
        calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return completion(script_from_payload(payload), **changes)
    return transport


def test_bounded_audio_response_is_transcript_verified_and_scheduled_to_replay(environment, evidence, highlight):
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    result = service.prepare(evidence, highlight)
    assert result["status"] == "ready" and result["transcript_verified"]
    assert result["speech_duration_s"] == 2.0 and result["duration_s"] == 5.0
    assert result["cues"][-1] == {"start_s": 12.0, "end_s": 14.0, "text": result["text"]}
    assert result["cues"][0]["start_s"] == 9.0 and result["cues"][0]["end_s"] == 12.0
    assert result["cue_timebase"] == "replay"
    assert "prepared in advance" in result["disclosure"]
    request = calls[0]
    assert request["payload"]["model"] == "mock-audio"
    assert request["payload"]["audio"] == {"voice": "alloy", "format": "wav"}
    assert request["payload"]["max_completion_tokens"] == 600
    assert request["timeout"] <= 30
    assert "exact supplied script" in request["payload"]["messages"][0]["content"]
    assert service.audio(result["cache_key"]) == wav_bytes(duration=5.0)
    assert service.prepare(evidence, highlight)["cached"] and len(calls) == 1
    public = json.dumps(result) + json.dumps(service.get_status())
    assert "never-a-real-key" not in public and "commentary-test" not in public


@pytest.mark.parametrize("delta,displacement,forecast,observed_phrase,forecast_phrase,energy", [
    (1.25, 4.0, 1.5, "space just opened up", "a tighter gap", "lively"),
    (-1.25, 2.0, 3.0, "space just tightened", "a wider gap", "lively"),
    (.6, 1.8, 2.26, "just gained a little space", "a similar gap", "calm"),
    (-.6, 1.8, 1.5, "space just narrowed", "a tighter gap", "calm"),
    (.1, 1.8, 3.0, "just covered ground", "a wider gap", "calm"),
])
def test_fan_script_and_energy_follow_observed_evidence_with_separate_modal_forecast(
        environment, evidence, highlight, delta, displacement, forecast, observed_phrase, forecast_phrase, energy):
    evidence["highlight"].update(
        kind="space" if delta >= .5 else "pressure" if delta <= -.5 else "movement",
        previous_nearest_opponent_m=evidence["current_nearest_opponent_m"]-delta,
        observed_gap_change_m=delta, observed_displacement_m=displacement,
        selection_score=abs(delta)+.35*displacement)
    evidence["predicted_nearest_opponent_m"] = forecast
    calls = []
    result = CommentaryService(env=environment, transport=transport_spy(calls)).prepare(evidence, highlight)
    assert result["status"] == "ready" and result["energy"] == energy
    assert observed_phrase in result["text"] and forecast_phrase in result["text"]
    assert "The model predicts" in result["text"] and "next second" in result["text"]
    assert len(result["text"].split()) <= 20  # Word budget; actual WAV duration is independently checked.
    assert result["energy_basis"] == "observed geometry over the previous second"
    assert result["delivery_control"] == "gpt_prosody_instruction"
    system = calls[0]["payload"]["messages"][0]["content"]
    assert result["prosody"] in system and "do not imitate any real person" in system
    assert "extra words" in system and result["transcript"] == result["text"]
    assert not any(word in result["text"].lower() for word in ("pass", "goal", "defender", "will"))
    assert result["cues"][-1]["start_s"] == evidence["anchor_time_s"]
    assert result["cues"][-1]["text"] == result["text"]


def test_forecast_magnitude_cannot_manufacture_excited_delivery(evidence):
    evidence["highlight"].update(kind="movement", previous_nearest_opponent_m=2.2,
                                 observed_gap_change_m=.05, observed_displacement_m=1.6,
                                 selection_score=.05+.35*1.6)
    assert commentary.commentary_energy(evidence, "fan") == "calm"
    evidence["predicted_nearest_opponent_m"] = 20.0
    evidence["predicted_displacement_m"] = 20.0
    assert commentary.commentary_energy(evidence, "fan") == "calm"
    evidence["highlight"].update(observed_displacement_m=3.2, selection_score=.05+.35*3.2)
    assert commentary.commentary_energy(evidence, "fan") == "lively"
    assert commentary.commentary_energy(evidence, "analyst") == "calm"


def test_analyst_spoken_units_match_rounded_evidence_without_one_metres(evidence):
    evidence["highlight"].update(previous_nearest_opponent_m=1.25, observed_gap_change_m=1.0,
                                 selection_score=1.0+.35*4.0)
    evidence["predicted_nearest_opponent_m"] = 1.0
    script = make_script(evidence, "analyst")
    assert "grew by one metre." in script and "projects one metre next second." in script
    assert "one metres" not in script
    evidence["predicted_nearest_opponent_m"] = 1.1
    assert "projects one point one metres" in make_script(evidence, "analyst")


def test_new_speech_version_rejects_old_seed_and_retains_attempt_ledger(environment, evidence, highlight):
    service = CommentaryService(env=environment, transport=transport_spy([]))
    ready = service.prepare(evidence, highlight)
    assert ready["content_version"] == "matchlab-commentary-v3"
    old = {**ready, "content_version": "matchlab-commentary-v2"}
    (service._cache_dir / f"{ready['cache_key']}.json").write_text(json.dumps(old))
    calls = []
    restarted = CommentaryService(env=environment, transport=transport_spy(calls))
    assert restarted.get_status()["requests_used"] == 1
    result = restarted.prepare(evidence, highlight)
    assert result["status"] == "ready" and not result["cached"] and len(calls) == 1
    assert restarted.get_status()["requests_used"] == 2


def test_default_speech_limit_remains_three_calls_per_minute(environment):
    assert CommentaryService(env=environment).get_status()["max_calls_per_minute"] == 3


@pytest.mark.parametrize("mutation", ["future_truth", "past_after_anchor", "inconsistent_gap", "missing_highlight"])
def test_invalid_evidence_never_reaches_paid_transport(environment, evidence, highlight, mutation):
    if mutation == "future_truth":
        evidence["future_truth"] = [[1, 2]]
    elif mutation == "past_after_anchor":
        evidence["highlight"]["window_end_s"] = 13.0
    elif mutation == "inconsistent_gap":
        evidence["highlight"]["observed_gap_change_m"] = -5.0
    else:
        del evidence["highlight"]
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    with pytest.raises(ValueError):
        service.prepare(evidence, highlight)
    assert calls == [] and service.get_status()["requests_used"] == 0


@pytest.mark.parametrize("changes", [
    {"id": "space-run-301-8"}, {"player_id": "Home_7"}, {"anchor_frame": 301},
    {"time_s": 12.04}, {"clip_start_s": -1.0, "clip_end_s": 9.0},
    {"clip_start_s": 13.0, "clip_end_s": 23.0}, {"clip_start_s": 9.0, "clip_end_s": 9.0},
    {"clip_start_s": 9.0, "clip_end_s": 25.0}, {"clip_end_s": float("nan")},
])
def test_stale_or_mismatched_clip_is_rejected_before_paid_transport(environment, evidence, highlight, changes):
    highlight.update(changes)
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    with pytest.raises(ValueError):
        service.prepare(evidence, highlight)
    assert not calls


@pytest.mark.parametrize("audience,language", [("coach", "en"), ("fan", "zh")])
def test_unsupported_audience_or_language_is_not_a_speech_endpoint(environment, evidence, highlight, audience, language):
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    with pytest.raises(ValueError):
        service.prepare(evidence, highlight, audience, language)
    assert not calls


@pytest.mark.parametrize("mutation,reason", [
    ("wrong_words", "transcript_mismatch"), ("wrong_number", "transcript_mismatch"),
    ("too_long", "audio_duration_mismatch"), ("too_long_after_anchor", "audio_duration_mismatch"), ("too_short", "audio_duration_mismatch"),
    ("invalid_wav", "invalid_audio"), ("wrong_encoding", "invalid_audio"),
    ("not_finished", "transcript_mismatch"), ("non_string_transcript", "invalid_audio"),
    ("invalid_choice", "invalid_audio"),
])
def test_invalid_generated_speech_fails_closed_but_keeps_subtitles(environment, evidence, highlight, mutation, reason):
    def transport(url, headers, payload, timeout):
        text = script_from_payload(payload)
        if mutation == "wrong_words":
            return completion(text + " A goal is guaranteed.")
        if mutation == "wrong_number":
            return completion(text.replace("nine", "seven"))
        if mutation == "too_long":
            return completion(text, duration=10.5)
        if mutation == "too_long_after_anchor":
            return completion(text, duration=7.5)
        if mutation == "too_short":
            return completion(text, duration=.25)
        if mutation == "invalid_wav":
            return completion(text, audio=b"not a wave")
        if mutation == "wrong_encoding":
            return completion(text, audio=wav_bytes(sample_width=1))
        if mutation == "not_finished":
            return completion(text, finish="length")
        if mutation == "non_string_transcript":
            return completion(7)
        return {"choices": [None]}
    service = CommentaryService(env=environment, transport=transport)
    result = service.prepare(evidence, highlight)
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["text"] == make_script(evidence, "fan")
    assert "audio_url" not in result and service.audio(result["cache_key"]) is None
    assert result["cues"][0]["start_s"] == 9.0
    assert service.get_status()["requests_used"] == 1 and not service.get_status()["live_verified"]


def test_timeout_is_sanitized_and_failed_attempt_survives_service_restart(environment, evidence, highlight):
    calls = []
    def timeout(*args):
        calls.append(1)
        raise TimeoutError("private endpoint and secret key must never leak")
    env = {**environment, "COMMENTARY_MAX_CALLS_TOTAL": "1"}
    service = CommentaryService(env=env, transport=timeout)
    result = service.prepare(evidence, highlight)
    assert result["status"] == "unavailable" and result["reason"] == "service_unavailable"
    assert result["text"] and result["cues"] and "secret" not in json.dumps(result)
    assert service.get_status()["requests_used"] == 1
    restarted = CommentaryService(env=env, transport=timeout)
    assert restarted.prepare(evidence, highlight)["reason"] == "budget_exhausted"
    assert restarted.get_status()["requests_used"] == 1 and len(calls) == 1


def test_minute_limit_counts_failures_and_recovers_without_bypassing_total(environment, evidence, highlight):
    now, calls = [0.0], []
    def failure(*args):
        calls.append(1)
        raise TimeoutError()
    service = CommentaryService(env={**environment, "COMMENTARY_MAX_CALLS_PER_MINUTE": "1", "COMMENTARY_MAX_CALLS_TOTAL": "2"},
                                transport=failure, clock=lambda: now[0])
    assert service.prepare(evidence, highlight)["reason"] == "service_unavailable"
    assert service.prepare(evidence, highlight, "analyst")["reason"] == "rate_limited"
    now[0] = 60.1
    assert service.prepare(evidence, highlight, "analyst")["reason"] == "service_unavailable"
    now[0] = 120.2
    assert service.prepare(evidence, highlight)["reason"] == "budget_exhausted"
    assert len(calls) == 2


def test_busy_request_does_not_queue_or_charge_another_request(environment, evidence, highlight):
    entered, release = threading.Event(), threading.Event()
    def transport(url, headers, payload, timeout):
        entered.set()
        assert release.wait(3)
        return completion(script_from_payload(payload))
    service = CommentaryService(env=environment, transport=transport)
    thread = threading.Thread(target=lambda: service.prepare(evidence, highlight))
    thread.start()
    try:
        assert entered.wait(3)
        assert service.prepare(evidence, highlight, "analyst")["reason"] == "busy"
        assert service.get_status()["requests_used"] == 1
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()


def test_cache_identity_separates_focus_audience_clip_timing_and_content_version(environment, evidence, highlight, monkeypatch):
    calls = []
    service = CommentaryService(env={**environment, "COMMENTARY_MAX_CALLS_PER_MINUTE": "12"}, transport=transport_spy(calls))
    first = service.prepare(evidence, highlight)
    analyst = service.prepare(evidence, highlight, "analyst")
    focused = copy.deepcopy(evidence)
    focused["player_id"] = "Home_7"
    focused["highlight"]["id"] = "space-run-300-6"
    second = service.prepare(focused, {**highlight, "id": "space-run-300-6", "player_id": "Home_7"})
    retimed = service.prepare(evidence, {**highlight, "clip_start_s": 10.0, "clip_end_s": 20.0})
    monkeypatch.setattr(commentary, "CONTENT_VERSION", "test-next-content-version")
    versioned = service.prepare(evidence, highlight)
    assert len({r["cache_key"] for r in (first, analyst, second, retimed, versioned)}) == 5
    assert len(calls) == 5
    assert retimed["clip_start_s"] == 10.0 and retimed["cues"][0]["start_s"] == 10.0
    assert retimed["cues"][-1]["start_s"] == 12.0
    assert first["text"] != second["text"]


def test_audio_hash_tampering_and_corrupt_metadata_do_not_return_ready_audio(environment, evidence, highlight):
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    ready = service.prepare(evidence, highlight)
    key = ready["cache_key"]
    (service._cache_dir / f"{key}.wav").write_bytes(b"corrupt audio")
    assert service.audio(key) is None
    repaired = service.prepare(evidence, highlight)
    assert repaired["status"] == "ready" and not repaired["cached"] and len(calls) == 2
    (service._cache_dir / f"{key}.json").write_text("[]")
    assert service.audio(key) is None


@pytest.mark.parametrize("key", ["../usage", "not-a-key", "a" * 63, "A" * 64, "0" * 64])
def test_unknown_audio_keys_cannot_access_files(environment, key):
    assert CommentaryService(env=environment).audio(key) is None


def test_speech_transcript_normalization_accepts_equivalent_digits_but_not_different_numbers():
    assert normalize_transcript("Home 9's gap grew by 3.3 meters.") == normalize_transcript("Home nine's gap grew by three point three metres.")
    assert normalize_transcript("Home 9's gap grew by 3.3 meters.") != normalize_transcript("Home nine's gap grew by three point four metres.")
    assert normalize_transcript("The model projects -3.3 metres.") != normalize_transcript("The model projects three point three metres.")
    assert normalize_transcript("The model projects 3.34 metres.") != normalize_transcript("The model projects three point three metres.")


def test_streaming_wav_unknown_lengths_are_measured_and_rewritten_for_browser(environment, evidence, highlight):
    source = bytearray(wav_bytes(duration=2.5))
    assert source[:4] == b"RIFF" and source[36:40] == b"data"
    source[4:8] = (0xFFFFFFFF).to_bytes(4, "little")
    source[40:44] = (0xFFFFFFFF).to_bytes(4, "little")
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls, audio=bytes(source)))
    result = service.prepare(evidence, highlight)
    assert result["status"] == "ready" and result["speech_duration_s"] == 2.5 and result["duration_s"] == 5.5
    browser_audio = service.audio(result["cache_key"])
    assert browser_audio != bytes(source)
    with wave.open(io.BytesIO(browser_audio), "rb") as reader:
        assert reader.getnframes() == 44_000
        assert len(reader.readframes(reader.getnframes())) == 88_000
    assert int.from_bytes(browser_audio[4:8], "little") == len(browser_audio) - 8


def test_narration_pcm_begins_at_highlight_anchor_after_silent_build_up(environment, evidence, highlight):
    source = bytearray(wav_bytes(duration=2))
    source[44:] = b"\x01\x00" * 16000
    service = CommentaryService(env=environment, transport=transport_spy([], audio=bytes(source)))
    result = service.prepare(evidence, highlight)
    assert result["status"] == "ready"
    with wave.open(io.BytesIO(service.audio(result["cache_key"])), "rb") as reader:
        build_up = reader.readframes(24000)
        narration = reader.readframes(16000)
    assert build_up == b"\0" * 48000
    assert narration == bytes(source[44:])
    assert result["cues"][-1]["start_s"] == evidence["anchor_time_s"]


def test_streaming_wav_partial_pcm_frame_is_rejected(environment, evidence, highlight):
    source = bytearray(wav_bytes(duration=2))
    source[4:8] = (0xFFFFFFFF).to_bytes(4, "little")
    source[40:44] = (0xFFFFFFFF).to_bytes(4, "little")
    source.append(0)
    service = CommentaryService(env=environment, transport=transport_spy([], audio=bytes(source)))
    result = service.prepare(evidence, highlight)
    assert result["status"] == "unavailable" and result["reason"] == "invalid_audio"


def test_agent_fact_order_cannot_reorder_spoken_chronology_or_insert_unchecked_prose(evidence):
    insight = {"evidence": evidence, "audience": "fan", "language": "en", "mode": "azure",
               "fact_ids": ["separation", "observed"], "text": "Home nine scored seven goals!"}
    script = make_script(evidence, "fan", insight)
    assert script.startswith("Look at Home nine, space just opened up!") and script.endswith("next second.")
    assert script == make_script(evidence, "fan", {**insight, "fact_ids": ["observed", "separation"]})
    assert "scored" not in script and "goals" not in script
    stale = {**insight, "evidence": {**evidence, "player_id": "Home_7"}}
    assert make_script(evidence, "fan", stale) == make_script(evidence, "fan")
    assert make_script(evidence, "analyst", insight) == make_script(evidence, "analyst")


def test_agent_order_does_not_invalidate_voice_cache_or_change_evidence_numbers(environment, evidence, highlight):
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    insight = {"evidence": evidence, "audience": "analyst", "language": "en", "mode": "azure",
               "fact_ids": ["separation", "observed"], "text": "The gap is ninety metres."}
    first = service.prepare(evidence, highlight, "analyst", insight=insight)
    second = service.prepare(evidence, highlight, "analyst", insight={**insight, "fact_ids": ["observed", "separation"]})
    assert first["status"] == "ready" and second["cached"] and len(calls) == 1
    assert first["cache_key"] == second["cache_key"] and first["text"] == second["text"]
    assert first["text"].startswith("Home nine. The gap grew by one point two metres.")
    assert "three point one metres next second" in first["text"] and "ninety" not in first["text"]
    assert first["evidence_source"] == "azure" and first["fact_ids"] == ["separation", "observed"]


@pytest.mark.parametrize("player,label", [("Home_9", "Home nine"), ("Home_7", "Home seven")])
def test_analyst_name_is_a_clear_separate_phrase_and_changed_audio_cannot_reuse_possessive_cache(evidence, highlight, player, label):
    evidence["player_id"] = highlight["player_id"] = player
    script = make_script(evidence, "analyst")
    assert script.startswith(label + ". The gap grew by one point two metres.")
    assert script.endswith("The model projects three point one metres next second.")
    old_script = script.replace(label + ". The gap", label + "'s gap", 1)
    assert commentary.commentary_cache_key(evidence,highlight,"analyst","en",script) != commentary.commentary_cache_key(evidence,highlight,"analyst","en",old_script)
    assert make_script(evidence,"fan") == f"Look at {label}, space just opened up! The model predicts a wider gap next second."


@pytest.mark.parametrize("fact_ids", [["observed"], ["observed", "observed"], ["invented", "observed"], [{}, "observed"], "observed"])
def test_malformed_agent_selection_uses_safe_default_script(evidence, fact_ids):
    insight = {"evidence": evidence, "audience": "fan", "language": "en", "fact_ids": fact_ids}
    assert make_script(evidence, "fan", insight) == make_script(evidence, "fan")


@pytest.mark.parametrize("changes", [
    {"audience": "analyst"}, {"language": "zh"}, {"evidence": {}},
    {"fact_ids": ["invented", "observed"]},
])
def test_rejected_agent_context_is_not_reported_as_cloud_authored_script(environment, evidence, highlight, changes):
    insight = {"evidence": evidence, "audience": "fan", "language": "en", "mode": "azure",
               "fact_ids": ["separation", "observed"], **changes}
    service = CommentaryService(env=environment, transport=transport_spy([]))
    result = service.prepare(evidence, highlight, insight=insight)
    assert result["status"] == "ready" and result["text"] == make_script(evidence, "fan")
    assert result["evidence_source"] == "local"
    assert result["fact_ids"] in ([], ["observed", "separation"])


def test_packaged_verified_seed_is_usable_without_a_live_paid_service(environment, evidence, highlight):
    service = CommentaryService(env=environment, transport=transport_spy([]))
    ready = service.prepare(evidence, highlight)
    seed = Path(environment["COMMENTARY_SEED_DIR"])
    seed.mkdir()
    for extension in ("json", "wav"):
        filename = f"{ready['cache_key']}.{extension}"
        (seed / filename).write_bytes((service._cache_dir / filename).read_bytes())
    offline = CommentaryService(env={"COMMENTARY_CACHE_DIR": str(seed.parent / "offline-cache"), "COMMENTARY_SEED_DIR": str(seed)},
                                transport=lambda *args: pytest.fail("Seed playback attempted a paid call"))
    result = offline.prepare(evidence, highlight)
    assert result["status"] == "ready" and result["cached"]
    assert offline.audio(result["cache_key"]) == service.audio(result["cache_key"])
    assert offline.get_status()["requests_used"] == 0


def test_failed_persistent_usage_write_blocks_the_paid_call(environment, evidence, highlight, monkeypatch):
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    original = Path.write_text
    def fail_usage(path, *args, **kwargs):
        if path.name == "usage.tmp":
            raise OSError("simulated full disk")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", fail_usage)
    result = service.prepare(evidence, highlight)
    assert result["status"] == "unavailable" and result["reason"] == "cache_unavailable"
    assert not calls


@pytest.mark.parametrize("ledger", ["not json", "[]", '{"calls":"invalid"}'])
def test_corrupt_persistent_usage_ledger_fails_closed(environment, evidence, highlight, ledger):
    cache = Path(environment["COMMENTARY_CACHE_DIR"])
    cache.mkdir()
    (cache / "usage.json").write_text(ledger)
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    assert service.prepare(evidence, highlight)["reason"] == "invalid_configuration"
    assert not calls


def test_audio_transport_rejects_redirect_and_oversized_body_without_leaking_connection(monkeypatch):
    state = {"status": 302, "body": b"{}", "closed": False}
    class FakeResponse:
        status = property(lambda self: state["status"])
        def read(self, amount):
            assert amount == 4_000_001
            return state["body"][:amount]
    class FakeConnection:
        def __init__(self, host, timeout):
            assert host == "mock.openai.azure.com" and timeout == 3
        def request(self, method, path, body, headers):
            assert method == "POST" and path == "/openai/v1/chat/completions"
        def getresponse(self):
            return FakeResponse()
        def close(self):
            state["closed"] = True
    monkeypatch.setattr(commentary.http.client, "HTTPSConnection", FakeConnection)
    with pytest.raises(ConnectionError):
        commentary._post_audio("https://mock.openai.azure.com/openai/v1/chat/completions", {}, {}, 3)
    assert state["closed"]
    state.update(status=200, body=b"x" * 4_000_010, closed=False)
    with pytest.raises(ValueError, match="Oversized"):
        commentary._post_audio("https://mock.openai.azure.com/openai/v1/chat/completions", {}, {}, 3)
    assert state["closed"]


@pytest.fixture
def api_client(monkeypatch, environment):
    paid_calls = {"insight": [], "audio": []}
    def insight_transport(url, headers, payload, timeout):
        paid_calls["insight"].append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"fact_ids":["observed","separation"]}'}}]}
    speech = CommentaryService(env=environment, transport=transport_spy(paid_calls["audio"]))
    insight = AzureInsightService(env={**environment, "AZURE_OPENAI_DEPLOYMENT": "mock-insight"}, transport=insight_transport)
    monkeypatch.setattr(backend, "CommentaryService", lambda: speech)
    monkeypatch.setattr(backend, "AzureInsightService", lambda: insight)
    with TestClient(backend.app) as client:
        yield client, paid_calls, speech


def api_body(client, player="Home_9"):
    h = client.get("/api/highlights", params={"match_id": "space-run", "player_id": player}).json()["highlights"][0]
    return {"match_id": "space-run", "highlight_id": h["id"], "player_id": player, "audience": "fan", "language": "en"}


@pytest.mark.parametrize("changes", [
    {"match_id": "missing"}, {"highlight_id": "missing"}, {"player_id": "Home_7"},
    {"player_id": "Home_99"}, {"match_id": "transition"}, {"language": "zh"},
    {"evidence": {"future_truth": [1, 2]}}, {"text": "Say anything"},
])
def test_invalid_api_candidates_cannot_spend_insight_or_audio_budget(api_client, changes):
    client, calls, speech = api_client
    response = client.post("/api/commentary", json={**api_body(client), **changes})
    assert response.status_code == 422
    assert calls == {"insight": [], "audio": []}
    assert speech.get_status()["requests_used"] == 0


@pytest.mark.parametrize("cache_only", [False, True])
def test_cached_api_audio_bypasses_even_a_failed_text_service(api_client, monkeypatch, cache_only):
    client, calls, speech = api_client
    body = api_body(client)
    first = client.post("/api/commentary", json=body).json()
    assert first["status"] == "ready"
    def text_service_must_not_run(*args, **kwargs):
        raise AssertionError("Cached speech waited for text generation")
    monkeypatch.setattr(client.app.state.azure_insights, "generate", text_service_must_not_run)
    second = client.post("/api/commentary", json={**body, "cache_only": cache_only}).json()
    assert second["status"] == "ready" and second["cached"]
    assert second["audio_sha256"] == first["audio_sha256"]
    assert len(calls["insight"]) == len(calls["audio"]) == 1


def test_public_cache_only_miss_keeps_evidence_without_agent_audio_or_quota_use(api_client, monkeypatch):
    client, calls, speech = api_client
    body = api_body(client)
    original_agent = client.app.state.azure_insights.generate
    def forbidden_agent(*args, **kwargs):
        pytest.fail("Cache-only miss reached the fact-selection Agent")
    monkeypatch.setattr(client.app.state.azure_insights, "generate", forbidden_agent)
    response = client.post("/api/commentary", json={**body, "cache_only": True})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "unavailable" and result["reason"] == "cache_miss"
    assert result["text"] and result["cues"] and "audio_url" not in result
    assert calls == {"insight": [], "audio": []}
    assert speech.get_status()["requests_used"] == 0
    assert not (speech._cache_dir / "usage.json").exists()
    # The probe must not disable ordinary, explicitly requested generation.
    monkeypatch.setattr(client.app.state.azure_insights, "generate", original_agent)
    generated = client.post("/api/commentary", json={**body, "cache_only": False}).json()
    assert generated["status"] == "ready"
    assert len(calls["insight"]) == len(calls["audio"]) == 1


@pytest.mark.parametrize("invalid", ["true", "false", 1, 0, None, {}, []])
def test_public_cache_only_flag_requires_a_real_json_boolean(api_client, invalid):
    client, calls, speech = api_client
    response = client.post("/api/commentary", json={**api_body(client), "cache_only": invalid})
    assert response.status_code == 422
    assert calls == {"insight": [], "audio": []} and speech.get_status()["requests_used"] == 0


def test_cache_probe_miss_never_generates_or_consumes_budget(environment, evidence, highlight):
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    result = service.prepare(evidence, highlight, cache_only=True)
    assert result["reason"] == "cache_miss" and result["status"] == "unavailable"
    assert not calls and service.get_status()["requests_used"] == 0


def test_api_delivers_verified_audio_and_two_focus_contents(api_client):
    client, calls, speech = api_client
    first = client.post("/api/commentary", json=api_body(client)).json()
    second = client.post("/api/commentary", json=api_body(client, "Home_7")).json()
    assert first["status"] == second["status"] == "ready"
    assert first["player_id"] == "Home_9" and second["player_id"] == "Home_7"
    assert first["highlight_id"] != second["highlight_id"] and first["text"] != second["text"]
    assert "space just opened" in first["text"] and "space just tightened" in second["text"]
    audio = client.get(first["audio_url"])
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert "immutable" in audio.headers["cache-control"]
    assert audio.content == speech.audio(first["cache_key"])
    assert client.get("/api/commentary/audio/unknown.wav").status_code == 404
    assert client.get("/api/commentary/audio/" + "0" * 64 + ".wav").status_code == 404


def test_api_audio_timeout_preserves_replay_prediction_and_explanation(api_client):
    client, calls, speech = api_client
    def timeout(*args):
        raise TimeoutError("private upstream error")
    speech._transport = timeout
    body = api_body(client)
    response = client.post("/api/commentary", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "unavailable" and result["reason"] == "service_unavailable"
    assert result["text"] and result["cues"] and "Replay, subtitles and evidence remain available" in result["notice"]
    assert client.get("/api/matches/space-run").status_code == 200
    anchor = int(body["highlight_id"].split("-")[-2])
    assert client.get("/api/predict", params={"match_id": "space-run", "anchor_frame": anchor}).status_code == 200
    insight = client.post("/api/insights", json={**body, "anchor_frame": anchor})
    assert insight.status_code == 200 and insight.json()["grounded"]


def test_tts_only_configuration_sends_exact_script_and_reports_unverified_input_transcript(environment, evidence, highlight):
    calls = []
    env = {key: value for key, value in environment.items() if key != "AZURE_AUDIO_DEPLOYMENT"}
    env["AZURE_TTS_DEPLOYMENT"] = "matchlab-tts"
    def tts(url, headers, payload, timeout):
        calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return wav_bytes(duration=2.0)
    service = CommentaryService(env=env, tts_transport=tts,
                                transport=lambda *args: pytest.fail("TTS-only request unexpectedly invoked GPT audio"))
    assert service.get_status()["configured"]
    result = service.prepare(evidence, highlight)
    assert result["status"] == "ready"
    assert result["model"] == "tts" and result["model_version"] == "001"
    assert result["transcript_source"] == "synthesis_input" and result["transcript_verified"] is False
    assert result["delivery_control"] == "tts_default_voice"
    assert result["transcript"] == result["text"] == make_script(evidence, "fan")
    assert result["speech_start_s"] == 12.0 and result["speech_duration_s"] == 2.0
    assert result["duration_s"] == 5.0 and result["cues"][-1]["start_s"] == 12.0
    assert service.audio(result["cache_key"]) == wav_bytes(duration=5.0)
    request = calls[0]
    assert request["url"] == "https://commentary-test.openai.azure.com/openai/deployments/matchlab-tts/audio/speech?api-version=2025-04-01-preview"
    assert request["headers"]["api-key"] == environment["AZURE_OPENAI_API_KEY"]
    assert request["payload"] == {"model": "tts-1", "input": result["text"], "voice": "alloy", "response_format": "wav", "speed": 1.15}
    assert request["timeout"] <= 30
    assert service.prepare(evidence, highlight)["cached"] and len(calls) == 1
    assert "never-a-real-key" not in json.dumps(result)


def test_enabling_tts_preserves_cached_gpt_audio_and_its_actual_provenance(environment, evidence, highlight):
    gpt_calls, tts_calls = [], []
    gpt = CommentaryService(env=environment, transport=transport_spy(gpt_calls))
    original = gpt.prepare(evidence, highlight)
    original_audio = gpt.audio(original["cache_key"])
    def tts(*args):
        tts_calls.append(1)
        return wav_bytes()
    mixed = CommentaryService(env={**environment, "AZURE_TTS_DEPLOYMENT": "matchlab-tts"}, tts_transport=tts,
                              transport=lambda *args: pytest.fail("Validated cached GPT clip was regenerated"))
    cached = mixed.prepare(evidence, highlight)
    assert cached["status"] == "ready" and cached["cached"]
    assert cached["cache_key"] == original["cache_key"] and cached["audio_url"] == original["audio_url"]
    assert cached["model"] == "gpt-audio-mini" and cached["model_version"] == "2025-12-15"
    assert cached["transcript_verified"] and cached["transcript_source"] == "provider_audio_transcript"
    assert cached["audio_sha256"] == original["audio_sha256"]
    assert mixed.audio(cached["cache_key"]) == original_audio
    assert len(gpt_calls) == 1 and not tts_calls
    # Only a new audience/content variant invokes exact-script TTS.
    fresh = mixed.prepare(evidence, highlight, "analyst")
    assert fresh["status"] == "ready" and not fresh["cached"] and fresh["model"] == "tts"
    assert fresh["cache_key"] != cached["cache_key"] and len(tts_calls) == 1
    assert mixed.get_status()["requests_used"] == 2


@pytest.mark.parametrize("output,reason", [
    (b"not WAV data", "invalid_audio"), (None, "invalid_audio"), ({"wrong": "type"}, "invalid_audio"),
    (wav_bytes(duration=.2), "audio_duration_mismatch"),
    (wav_bytes(duration=7.5), "audio_duration_mismatch"),
    (wav_bytes(sample_width=1), "invalid_audio"),
])
def test_tts_invalid_wave_fails_closed_keeps_captions_and_counts_attempt(environment, evidence, highlight, output, reason):
    service = CommentaryService(env={**environment, "AZURE_TTS_DEPLOYMENT": "matchlab-tts"},
                                tts_transport=lambda *args: output,
                                transport=lambda *args: pytest.fail("Invalid TTS response must not trigger an unbudgeted GPT retry"))
    result = service.prepare(evidence, highlight)
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["text"] == make_script(evidence, "fan") and result["cues"]
    assert "audio_url" not in result and service.audio(result["cache_key"]) is None
    assert service.get_status()["requests_used"] == 1


def test_tts_timeout_uses_shared_minute_total_and_persistent_failure_budget(environment, evidence, highlight):
    now, calls = [0.0], []
    def fail(*args):
        calls.append(1)
        raise TimeoutError("private TTS error with credentials")
    env = {**environment, "AZURE_TTS_DEPLOYMENT": "matchlab-tts", "COMMENTARY_MAX_CALLS_PER_MINUTE": "1", "COMMENTARY_MAX_CALLS_TOTAL": "2"}
    service = CommentaryService(env=env, tts_transport=fail, clock=lambda: now[0])
    result = service.prepare(evidence, highlight)
    assert result["reason"] == "service_unavailable" and "credentials" not in json.dumps(result)
    assert service.prepare(evidence, highlight, "analyst")["reason"] == "rate_limited"
    now[0] = 61
    assert service.prepare(evidence, highlight, "analyst")["reason"] == "service_unavailable"
    restarted = CommentaryService(env=env, tts_transport=fail)
    assert restarted.prepare(evidence, highlight)["reason"] == "budget_exhausted"
    assert restarted.get_status()["requests_used"] == 2 and len(calls) == 2


def test_tts_boundary_clip_too_short_does_not_spend_budget(environment, evidence, highlight):
    highlight["clip_end_s"] = evidence["anchor_time_s"] + 2.0
    service = CommentaryService(env={**environment, "AZURE_TTS_DEPLOYMENT": "matchlab-tts"},
                                tts_transport=lambda *args: pytest.fail("Too-short clip consumed paid TTS"),
                                transport=lambda *args: pytest.fail("Too-short clip consumed paid GPT audio"))
    result = service.prepare(evidence, highlight)
    assert result["status"] == "unavailable" and result["reason"] == "clip_too_short"
    assert result["text"] and result["cues"] and "boundary clip" in result["notice"]
    assert service.get_status()["requests_used"] == 0


def test_tts_invalid_evidence_never_reaches_synthesis(environment, evidence, highlight):
    evidence["future_truth"] = [[99, 99]]
    service = CommentaryService(env={**environment, "AZURE_TTS_DEPLOYMENT": "matchlab-tts"},
                                tts_transport=lambda *args: pytest.fail("Invalid evidence reached TTS"))
    with pytest.raises(ValueError):
        service.prepare(evidence, highlight)
    assert service.get_status()["requests_used"] == 0


@pytest.mark.parametrize("deployment", ["bad/deployment", "bad deployment", "x\nheader", "x" * 129])
def test_invalid_tts_configuration_cannot_reach_transport(environment, evidence, highlight, deployment):
    service = CommentaryService(env={**environment, "AZURE_TTS_DEPLOYMENT": deployment},
                                tts_transport=lambda *args: pytest.fail("Invalid TTS configuration reached network"))
    assert service.prepare(evidence, highlight)["reason"] == "invalid_configuration"


def test_null_gpt_response_still_fails_closed_when_tts_is_not_configured(environment, evidence, highlight):
    service = CommentaryService(env=environment, transport=lambda *args: None)
    result = service.prepare(evidence, highlight)
    assert result["status"] == "unavailable" and result["reason"] == "invalid_audio"


def test_tts_binary_transport_includes_api_version_and_rejects_redirects_or_oversize(monkeypatch):
    state = {"status": 200, "body": wav_bytes(), "closed": False}
    class FakeResponse:
        status = property(lambda self: state["status"])
        def read(self, amount):
            assert amount == 4_000_001
            return state["body"][:amount]
    class FakeConnection:
        def __init__(self, host, timeout):
            assert host == "mock.openai.azure.com" and timeout == 3
        def request(self, method, path, body, headers):
            assert method == "POST"
            assert path == "/openai/deployments/mock-tts/audio/speech?api-version=2025-04-01-preview"
            assert json.loads(body)["input"] == "Read only this."
        def getresponse(self):
            return FakeResponse()
        def close(self):
            state["closed"] = True
    monkeypatch.setattr(commentary.http.client, "HTTPSConnection", FakeConnection)
    url = "https://mock.openai.azure.com/openai/deployments/mock-tts/audio/speech?api-version=2025-04-01-preview"
    assert commentary._post_tts(url, {}, {"input": "Read only this."}, 3) == wav_bytes()
    assert state["closed"]
    state.update(status=302, body=b"", closed=False)
    with pytest.raises(ConnectionError):
        commentary._post_tts(url, {}, {"input": "Read only this."}, 3)
    assert state["closed"]
    state.update(status=200, body=b"x" * 4_000_010, closed=False)
    with pytest.raises(ValueError, match="Oversized TTS"):
        commentary._post_tts(url, {}, {"input": "Read only this."}, 3)
    assert state["closed"]


def test_millionth_metre_cpu_noise_reuses_audio_without_rounding_raw_evidence_or_script(environment, evidence, highlight):
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    first = service.prepare(evidence, highlight)
    noisy = copy.deepcopy(evidence)
    # Deliberately away from millimetre rounding boundaries. Mirror the scale
    # of measured ARM/x86 differences and retain all evidence equalities.
    epsilon = .000001
    noisy["predicted_nearest_opponent_m"] += 3 * epsilon
    noisy["predicted_displacement_m"] -= epsilon
    noisy["current_nearest_opponent_m"] += epsilon
    noisy["highlight"]["current_nearest_opponent_m"] += epsilon
    noisy["highlight"]["observed_gap_change_m"] += epsilon
    noisy["highlight"]["selection_score"] += epsilon
    exact_noisy = copy.deepcopy(noisy)
    repeated = service.prepare(noisy, highlight)
    assert repeated["status"] == "ready" and repeated["cached"]
    assert repeated["cache_key"] == first["cache_key"]
    assert repeated["audio_sha256"] == first["audio_sha256"] and len(calls) == 1
    assert noisy == exact_noisy and noisy != evidence
    assert noisy["predicted_nearest_opponent_m"] == evidence["predicted_nearest_opponent_m"] + 3 * epsilon
    assert repeated["text"] == make_script(exact_noisy, "fan") == first["text"]


def test_centimetre_prediction_change_invalidates_audio_even_if_fan_script_is_unchanged(environment, evidence, highlight):
    calls = []
    service = CommentaryService(env=environment, transport=transport_spy(calls))
    first = service.prepare(evidence, highlight)
    changed = copy.deepcopy(evidence)
    changed["predicted_nearest_opponent_m"] += .01
    exact_changed = copy.deepcopy(changed)
    assert make_script(changed, "fan") == make_script(evidence, "fan")
    second = service.prepare(changed, highlight)
    assert second["status"] == "ready" and not second["cached"]
    assert second["cache_key"] != first["cache_key"] and len(calls) == 2
    assert changed == exact_changed


@pytest.mark.parametrize("dimension", ["script", "scene", "focus", "audience", "language", "content_version"])
def test_canonical_cache_identity_keeps_semantic_dimensions_separate(evidence, highlight, monkeypatch, dimension):
    text = make_script(evidence, "fan")
    before = commentary.commentary_cache_key(evidence, highlight, "fan", "en", text)
    changed, clip = copy.deepcopy(evidence), copy.deepcopy(highlight)
    audience, language = "fan", "en"
    if dimension == "script":
        text = "The model estimate remains uncertain. " + text
    elif dimension == "scene":
        changed["match_id"] = "transition"
        changed["highlight"]["id"] = clip["id"] = "transition-300-8"
    elif dimension == "focus":
        changed["player_id"] = clip["player_id"] = "Home_7"
        changed["highlight"]["id"] = clip["id"] = "space-run-300-6"
    elif dimension == "audience":
        audience = "analyst"
    elif dimension == "language":
        # This tests identity only; the public request still rejects languages
        # other than English before synthesis, as tested separately above.
        language = "es"
    else:
        monkeypatch.setattr(commentary, "CONTENT_VERSION", "test-next-semantic-content")
    after = commentary.commentary_cache_key(changed, clip, audience, language, text)
    assert after != before
