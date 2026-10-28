"""Integration checks for the demo contract, not new model accuracy evaluation."""
from dataclasses import replace
import json

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from demo.backend import (DEFAULT_CHECKPOINT, EXPECTED_CHECKPOINT_SHA256, Predictor,
                          app, geometric_insight, historical_input)
from demo.synthetic import PLAYER_IDS, TEAMS, make_matches
from model import STGNN_TCN, constant_velocity


@pytest.fixture(scope="module")
def scenarios():
    return make_matches()


@pytest.fixture(scope="module")
def engine():
    return Predictor()


def test_synthetic_replay_timestamp_identity_contract(scenarios):
    second = make_matches()
    assert len(scenarios) == 3
    for key, match in scenarios.items():
        assert match.positions.shape == (1001, 23, 2)
        assert match.available.shape == (1001, 23)
        assert len(set(match.player_ids)) == 22
        assert match.player_ids == PLAYER_IDS
        np.testing.assert_allclose(np.diff(match.timestamps), .04, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(match.positions, second[key].positions)
        assert np.isfinite(match.positions[match.available]).all()
        payload = match.public_payload()
        assert payload["source"] == "synthetic"
        assert payload["positions"][500][22] is None
        assert not payload["available"][500][22]
        assert len(payload["players"]) == 22
        assert [p["id"] for p in payload["players"]] == list(PLAYER_IDS)
        assert [p["team"] for p in payload["players"]] == [0] * 11 + [1] * 11
        json.dumps(payload, allow_nan=False)


def test_causal_source_rate_velocity_and_mask(scenarios):
    match = scenarios["space-run"]
    x, mask, team = historical_input(match, 300)
    idx = 300 + np.arange(-98, 1, 2)
    expected = (match.positions[idx] - match.positions[idx - 1]) / .04
    assert x.shape == mask.shape == (50, 23, 4)
    assert x.dtype == np.float32 and mask.dtype == bool
    np.testing.assert_allclose(x[..., :2], match.positions[idx], atol=4e-6)
    np.testing.assert_allclose(x[..., 2:], expected, atol=1e-6)
    np.testing.assert_array_equal(team, TEAMS)
    # This is deliberately different from using the preceding 12.5 Hz sample.
    wrong = (match.positions[idx] - match.positions[idx - 2]) / .08
    assert np.max(np.abs(expected - wrong)) > .001
    edge_x, edge_mask, _ = historical_input(match, 98)
    assert edge_mask[0, :, :2].all()
    assert not edge_mask[0, :, 2:].any()
    assert edge_mask[-1, :22].all()
    assert (edge_x[~edge_mask] == 0).all()


def test_missing_ball_preserves_masks_and_zero_fill(scenarios, engine):
    match = scenarios["space-run"]
    x, mask, _ = historical_input(match, 505)
    assert not mask[-1, 22].any()
    assert mask[-1, :22].all()
    assert (x[~mask] == 0).all()
    assert np.isfinite(x).all()
    output = engine.predict(match, 505)
    assert np.isfinite(output["predicted_xy"]).all()
    json.dumps(output, allow_nan=False)


def test_actual_checkpoint_forward_matches_standalone_model(scenarios, engine):
    saved = torch.load(DEFAULT_CHECKPOINT, map_location="cpu", weights_only=True)
    cfg, norm = saved["config"], saved["normalization"]
    independent = STGNN_TCN(norm["mean"], norm["std"],
                            **{key: cfg[key] for key in ["width", "dropout", "distance_scale_m"]})
    independent.load_state_dict(saved["state_dict"])
    independent.eval()
    x, mask, team = historical_input(scenarios["transition"], 400)
    with torch.inference_mode():
        expected = independent(torch.tensor(x[None]), torch.tensor(mask[None]), torch.tensor(team[None]))[0].numpy()
        baseline = constant_velocity(torch.tensor(x[None]), torch.tensor(mask[None]))[0].numpy()
    actual = engine.predict(scenarios["transition"], 400)
    np.testing.assert_allclose(actual["predicted_xy"], expected, rtol=0, atol=1e-6)
    assert np.max(np.abs(expected - baseline)) > .001, "Inference must actually use learned weights"
    assert actual["model"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    np.testing.assert_array_equal(engine.model.mean.numpy(), saved["state_dict"]["mean"].numpy())
    np.testing.assert_array_equal(engine.model.std.numpy(), saved["state_dict"]["std"].numpy())


def test_future_positions_and_timestamps_cannot_change_prediction(scenarios, engine):
    match, anchor = scenarios["space-run"], 300
    altered_positions = match.positions.copy()
    altered_positions[anchor + 1:] = np.nan
    altered_available = match.available.copy()
    altered_available[anchor + 1:] = False
    altered_times = match.timestamps.copy()
    altered_times[anchor + 1:] += 5000
    poisoned = replace(match, positions=altered_positions, available=altered_available, timestamps=altered_times)
    x1, m1, _ = historical_input(match, anchor)
    x2, m2, _ = historical_input(poisoned, anchor)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(m1, m2)
    first, second = engine.predict(match, anchor), engine.predict(poisoned, anchor)
    np.testing.assert_array_equal(first["predicted_xy"], second["predicted_xy"])
    assert first["insight"] == second["insight"]
    assert first["target_time_s"] == second["target_time_s"]
    assert not ({"actual_xy", "future_xy", "target_xy", "Y"} & first.keys())


def test_invalid_histories_and_roster_rejected(scenarios):
    match = scenarios["space-run"]
    for anchor in (-1, 0, 97, 1001, True, 1.5):
        with pytest.raises(ValueError):
            historical_input(match, anchor)
    with pytest.raises(ValueError, match="IDs/order"):
        historical_input(replace(match, player_ids=tuple(reversed(match.player_ids))), 300)
    times = match.timestamps.copy()
    times[299] += .01
    with pytest.raises(ValueError, match="timestamps"):
        historical_input(replace(match, timestamps=times), 300)
    available = match.available.copy()
    available[300, 0] = False
    with pytest.raises(ValueError, match="Every player"):
        historical_input(replace(match, available=available), 300)


def test_geometric_insight_is_recomputable(scenarios, engine):
    output = engine.predict(scenarios["space-run"], 350)
    current = np.asarray(output["current_xy"])
    predicted = np.asarray(output["predicted_xy"])
    insight = output["insight"]
    index = output["player_ids"].index(insight["player_id"])
    now = np.linalg.norm(current[index] - current[11:22], axis=-1).min()
    future = np.linalg.norm(predicted[index] - predicted[11:22], axis=-1).min()
    assert insight["current_nearest_opponent_m"] == pytest.approx(now)
    assert insight["predicted_nearest_opponent_m"] == pytest.approx(future, abs=5e-6)


def test_all_scenarios_have_aligned_start_middle_end_predictions(scenarios, engine):
    for match in scenarios.values():
        for anchor in (98, 500, 975):
            out = engine.predict(match, anchor)
            assert out["target_frame"] == anchor + 25
            assert out["target_time_s"] == pytest.approx(match.timestamps[anchor + 25])
            assert out["anchor_time_s"] == match.timestamps[anchor]
            assert out["player_ids"] == list(match.player_ids)
            np.testing.assert_allclose(out["current_xy"], match.positions[anchor, :22], atol=4e-6)
            assert np.shape(out["predicted_xy"]) == (22, 2)
            assert np.isfinite(out["predicted_xy"]).all()


def test_http_api_lifecycle_prediction_and_validation():
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["model_loaded"] is True
        assert health["device"] == "cpu"
        assert health["synthetic"] is True
        assert health["model_selection"]["used_test_for_selection"] is False
        matches = client.get("/api/matches").json()["matches"]
        assert len(matches) == 3
        replay = client.get("/api/matches/space-run")
        assert replay.status_code == 200
        prediction = client.get("/api/predict", params={"match_id": "space-run", "anchor_frame": 300})
        assert prediction.status_code == 200
        assert prediction.json()["target_frame"] == 325
        assert len(prediction.json()["predicted_xy"]) == 22
        assert client.get("/api/matches/unknown").status_code == 422
        for anchor, status in [(0, 409), (97, 409), (976, 409), (1001, 422), (-1, 422), ("1.5", 422)]:
            response = client.get("/api/predict", params={"match_id": "space-run", "anchor_frame": anchor})
            assert response.status_code == status, response.text
        assert client.get("/api/predict", params={"match_id": "missing", "anchor_frame": 300}).status_code == 422


def test_modified_checkpoint_rejected(tmp_path):
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"not the frozen checkpoint")
    with pytest.raises(ValueError, match="hash"):
        Predictor(path)


def test_discovery_has_recomputable_causal_evidence_and_distinct_moments(scenarios):
    from demo.backend import discover_highlights
    for match in scenarios.values():
        result = discover_highlights(match)
        assert len(result['highlights']) == 3
        assert result == discover_highlights(replace(match, events=()))
        for i, h in enumerate(result['highlights']):
            anchor, player = h['anchor_frame'], match.player_ids.index(h['player_id'])
            assert 98 <= anchor <= len(match.timestamps) - 26
            assert h['time_s'] == match.timestamps[anchor]
            assert player not in (0, 11)
            opponents = slice(11, 22) if player < 11 else slice(0, 11)
            now = np.linalg.norm(match.positions[anchor, opponents] - match.positions[anchor, player], axis=1).min()
            before = np.linalg.norm(match.positions[anchor - 25, opponents] - match.positions[anchor - 25, player], axis=1).min()
            e = h['evidence']
            assert e['current_nearest_opponent_m'] == pytest.approx(now)
            assert e['observed_gap_change_m'] == pytest.approx(now - before)
            assert h['score'] == pytest.approx(abs(now - before) + .35 * e['observed_displacement_m'])
            for other in result['highlights'][i + 1:]:
                assert abs(h['time_s'] - other['time_s']) >= 6
        json.dumps(result, allow_nan=False)


def test_highlights_api_and_model_replay_alignment():
    with TestClient(app) as client:
        assert client.get('/api/highlights', params={'match_id': 'unknown'}).status_code == 422
        for scenario in client.get('/api/matches').json()['matches']:
            response = client.get('/api/highlights', params={'match_id': scenario['id']})
            assert response.status_code == 200
            for h in response.json()['highlights']:
                prediction = client.get('/api/predict', params={'match_id': scenario['id'], 'anchor_frame': h['anchor_frame']}).json()
                assert h['player_id'] in prediction['player_ids']
                assert prediction['target_time_s'] == pytest.approx(h['time_s'] + 1)
                assert 'actual_xy' not in prediction


def test_focus_discovery_changes_actual_moments_and_has_bounded_context(scenarios):
    from demo.backend import discover_highlights
    match = scenarios["space-run"]
    first, second = [discover_highlights(match, player_id) for player_id in ("Home_9", "Home_7")]
    assert first["focus_player_id"] == "Home_9" and second["focus_player_id"] == "Home_7"
    assert first["highlights"] and second["highlights"]
    assert {h["anchor_frame"] for h in first["highlights"]} != {h["anchor_frame"] for h in second["highlights"]}
    for result in (first, second):
        assert result["analysis_mode"] == "retrospective"
        for item in result["highlights"]:
            assert item["player_id"] == result["focus_player_id"]
            assert 0 <= item["clip_start_s"] < item["time_s"] < item["clip_end_s"] <= match.duration_s
            assert item["clip_end_s"] - item["clip_start_s"] <= 10.0 + 1e-8
            if 3 <= item["time_s"] <= match.duration_s - 7:
                assert item["clip_end_s"] - item["clip_start_s"] == pytest.approx(10.0)
            assert item["evidence"]["window_end_s"] == item["time_s"]
            assert item["evidence"]["window_start_s"] == pytest.approx(item["time_s"] - 1.0)
    # The same focus is stable and not a relabelled version of the global list.
    assert first == discover_highlights(match, "Home_9")
    assert first["highlights"][0]["evidence"] != second["highlights"][0]["evidence"]


def test_discovery_does_not_invent_highlights_for_static_or_excluded_players(scenarios):
    from demo.backend import discover_highlights
    match = scenarios["space-run"]
    still = replace(match, positions=np.repeat(match.positions[:1], len(match.timestamps), axis=0))
    result = discover_highlights(still, "Home_9")
    assert result["highlights"] == [] and "threshold" in result["empty_reason"]
    keeper = discover_highlights(match, "Home_1")
    assert keeper["highlights"] == [] and "Goalkeepers are excluded" in keeper["empty_reason"]
    with pytest.raises(ValueError, match="Unknown player"):
        discover_highlights(match, "Home_99")


def test_focus_api_validates_player_and_restores_all_players():
    with TestClient(app) as client:
        default = client.get("/api/highlights", params={"match_id": "space-run"}).json()
        assert default["focus_player_id"] is None
        assert client.get("/api/highlights", params={"match_id": "space-run", "player_id": "Home_99"}).status_code == 422
        focused = client.get("/api/highlights", params={"match_id": "space-run", "player_id": "Home_7"}).json()
        assert focused["focus_player_id"] == "Home_7"
        assert all(item["player_id"] == "Home_7" for item in focused["highlights"])
        assert client.get("/api/highlights", params={"match_id": "space-run"}).json() == default
