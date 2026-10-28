"""FastAPI replay demo. Run: python -m uvicorn demo.backend:app --host 127.0.0.1 --port 8000.

Only the procedural replay is exposed. Inference uses the frozen validation-
selected checkpoint and past observations; future replay positions never enter
the model or the prediction API response. No training/test files are opened.
"""
from contextlib import asynccontextmanager
import hashlib
import os
from pathlib import Path
import threading
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool
import numpy as np
import torch

from model import STGNN_TCN
from demo.synthetic import FPS, PLAYER_IDS, TEAMS, Match, make_matches
from demo.reviewed_insights import ReviewedInsightService as AzureInsightService
from demo.commentary import CommentaryService


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHECKPOINT_SHA256 = "4ecea9ccfe5c632afe5da0be7e76150ca31ea85142fac0dd0d8aa0cea6233923"
DEFAULT_CHECKPOINT = ROOT / "runs/three_seed_stability/seed_20261026/best.pt"
MODEL_NAME = "ST-GNN + TCN"
SELECTION = {"seed": 20261026, "epoch": 36, "criterion": "lowest validation mean position error among the three fixed-seed runs",
             "validation_error_m": 0.2855924807984957, "used_test_for_selection": False}


def historical_input(match: Match, anchor: int):
    """Return raw metre features + masks using only indices <= anchor.

    Each of the 50 points uses the previous *raw* 25 Hz frame for velocity,
    matching the original training adapter. At the initial source frame no
    predecessor exists, so only those velocity slots are masked, never filled
    with a forward difference. The model performs its stored normalization.
    """
    if isinstance(anchor, bool) or not isinstance(anchor, (int, np.integer)):
        raise ValueError("anchor_frame must be an integer")
    if anchor < 98:
        raise ValueError("50 history samples require anchor_frame >= 98")
    if anchor >= len(match.timestamps):
        raise ValueError("anchor_frame is outside this replay")
    if tuple(match.player_ids) != PLAYER_IDS:
        raise ValueError("Player IDs/order do not match the fixed 22-player contract")
    idx = anchor + np.arange(-98, 1, 2)
    current = np.array(match.positions[idx], dtype=np.float64, copy=True)
    valid = np.array(match.available[idx], dtype=bool, copy=True) & np.isfinite(current).all(-1)
    x = np.zeros((50, 23, 4), dtype=np.float32)
    mask = np.zeros_like(x, dtype=bool)
    x[..., :2] = np.where(valid[..., None], current, 0.0)
    mask[..., :2] = valid[..., None]
    has_predecessor = idx > 0
    previous_idx = idx[has_predecessor] - 1
    previous = np.asarray(match.positions[previous_idx], dtype=np.float64)
    previous_valid = np.asarray(match.available[previous_idx], dtype=bool) & np.isfinite(previous).all(-1)
    dt = match.timestamps[idx[has_predecessor]] - match.timestamps[previous_idx]
    if not np.allclose(dt, 1 / FPS, rtol=0, atol=1e-7):
        raise ValueError("History must use contiguous 25 Hz source timestamps")
    if not np.allclose(np.diff(match.timestamps[idx]), 2 / FPS, rtol=0, atol=1e-7):
        raise ValueError("Input timestamps must be exactly 12.5 Hz")
    velocities = (current[has_predecessor] - previous) / dt[:, None, None]
    velocity_valid = valid[has_predecessor] & previous_valid & np.isfinite(velocities).all(-1)
    speed = np.linalg.norm(velocities, axis=-1)
    velocity_valid[:, :22] &= speed[:, :22] <= 15.0
    velocity_valid[:, 22] &= speed[:, 22] <= 60.0
    x[has_predecessor, :, 2:] = np.where(velocity_valid[..., None], velocities, 0.0)
    mask[has_predecessor, :, 2:] = velocity_valid[..., None]
    if not mask[-1, :22].all():
        raise ValueError("Every player needs a valid current position and causal velocity")
    return x, mask, TEAMS.copy()


def geometric_insight(current, predicted, player_ids):
    """A transparent distance indicator, not a tactical outcome/confidence claim."""
    now = np.linalg.norm(current[:11, None] - current[None, 11:22], axis=-1).min(-1)
    after = np.linalg.norm(predicted[:11, None] - predicted[None, 11:22], axis=-1).min(-1)
    # Exclude the goalkeeper. Positive delta indicates projected separation.
    chosen = int(np.argmax(after[1:] - now[1:]) + 1)
    change = float(after[chosen] - now[chosen])
    direction = "increases" if change >= 0 else "decreases"
    headline = "Projected separation grows" if change >= .3 else "Watch the nearest defender"
    return {"player_id": player_ids[chosen], "headline": headline,
            "explanation": f"For Home {chosen + 1:02d}, nearest-opponent distance {direction} from {now[chosen]:.1f} m to {after[chosen]:.1f} m at +1 s. This compares all predicted opponent positions; it does not establish a safe pass, offside status or scoring chance.",
            "current_nearest_opponent_m": float(now[chosen]),
            "predicted_nearest_opponent_m": float(after[chosen]),
            "method": "Euclidean distance to nearest opponent at current time and at predicted +1 s; largest increase among home outfield players."}


def highlight_candidates(match):
    """Compute meaningful observed changes independently of the chosen focus.

    Discovery scans the replay retrospectively, but each candidate's evidence
    uses only its previous second. No model output, future outcome or authored
    event label affects ranking. Prediction remains a separate causal call.
    """
    candidates = []
    for anchor in range(100, len(match.timestamps) - FPS, 12):
        before = anchor - FPS
        valid = match.available[[before, anchor], :22].all(axis=0)
        current, previous = match.positions[anchor, :22], match.positions[before, :22]
        valid &= np.isfinite(current).all(axis=1) & np.isfinite(previous).all(axis=1)
        if not valid.all():
            continue
        gaps_now = np.linalg.norm(current[:11, None] - current[None, 11:], axis=-1)
        gaps_before = np.linalg.norm(previous[:11, None] - previous[None, 11:], axis=-1)
        now = np.concatenate((gaps_now.min(axis=1), gaps_now.min(axis=0)))
        then = np.concatenate((gaps_before.min(axis=1), gaps_before.min(axis=0)))
        displacement = np.linalg.norm(current - previous, axis=1)
        for player in [*range(1, 11), *range(12, 22)]:
            delta = float(now[player] - then[player])
            score = abs(delta) + .35 * float(displacement[player])
            # Avoid promoting low-amplitude background motion into an event.
            # These are explicit geometric thresholds, not learned importance.
            if abs(delta) < .5 and float(displacement[player]) < 1.5:
                continue
            kind = "space" if delta >= .5 else "pressure" if delta <= -.5 else "movement"
            headline = {"space": "Space opens up", "pressure": "The gap is closing", "movement": "A run worth watching"}[kind]
            candidates.append({"id": f"{match.id}-{anchor}-{player}", "anchor_frame": anchor,
                               "time_s": float(match.timestamps[anchor]), "player_id": match.player_ids[player],
                               "kind": kind, "headline": headline, "score": score,
                               "clip_start_s": max(float(match.timestamps[0]), float(match.timestamps[anchor]) - 3.0),
                               "clip_end_s": min(float(match.timestamps[-1]), float(match.timestamps[anchor]) + 7.0),
                               "analysis_mode": "retrospective",
                               "evidence": {"previous_nearest_opponent_m": float(then[player]),
                                            "current_nearest_opponent_m": float(now[player]),
                                            "observed_gap_change_m": delta,
                                            "observed_displacement_m": float(displacement[player]),
                                            "window_s": 1.0,
                                            "window_start_s": float(match.timestamps[before]),
                                            "window_end_s": float(match.timestamps[anchor])}})
    return candidates


def discover_highlights(match, player_id=None, *, candidates=None):
    """Rank actual candidates for all players or one explicitly followed player."""
    if player_id is not None and player_id not in match.player_ids:
        raise ValueError("Unknown player ID")
    candidates = highlight_candidates(match) if candidates is None else candidates
    eligible = [item for item in candidates if player_id is None or item["player_id"] == player_id]
    picked = []
    for candidate in sorted(eligible, key=lambda item: (-item["score"], item["anchor_frame"], item["player_id"])):
        if all(abs(candidate["time_s"] - other["time_s"]) >= 6 for other in picked):
            picked.append(dict(candidate))
        if len(picked) == 3:
            break
    for rank, item in enumerate(picked, 1):
        item["rank"] = rank
    empty_reason = None
    if not picked:
        empty_reason = ("No related outfield highlights were found for this player. Goalkeepers are excluded from this movement detector."
                        if player_id in (match.player_ids[0], match.player_ids[11]) else
                        "No related moments meet the detector's movement or gap-change threshold in this replay.")
    return {"match_id": match.id, "focus_player_id": player_id, "highlights": picked,
            "source": "observed_geometry", "analysis_mode": "retrospective", "empty_reason": empty_reason,
            "method": "Retrospective replay scan every 0.48 s. Candidates require at least 0.5 m absolute nearest-opponent gap change or 1.5 m displacement over the previous second. Score = absolute gap change + 0.35 × displacement. Outfield players only, filtered by follow choice; at least 6 s between picks. Each candidate uses only its past second. Score is not confidence or match importance."}


class Predictor:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT):
        self.checkpoint = Path(checkpoint)
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if self.sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("Checkpoint hash does not match the frozen validation-selected model")
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        cfg, stats = saved["config"], saved["normalization"]
        torch.set_num_threads(min(4, int(cfg["cpu_threads"])))
        self.device = "cpu"
        self.model = STGNN_TCN(stats["mean"], stats["std"],
                              **{key: cfg[key] for key in ("width", "dropout", "distance_scale_m")})
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.lock = threading.Lock()

    def predict(self, match, anchor):
        # Only metadata length is checked for the future target; its values are
        # neither read by preprocessing nor returned with predictions.
        if anchor + FPS >= len(match.timestamps):
            raise ValueError("No replay frame exists at exactly +1 second")
        x, mask, team = historical_input(match, anchor)
        xt = torch.from_numpy(x[None])
        mt = torch.from_numpy(mask[None])
        tt = torch.from_numpy(team[None])
        with self.lock, torch.inference_mode():
            start = time.perf_counter()
            prediction = self.model(xt, mt, tt)[0].cpu().numpy()
            elapsed_ms = (time.perf_counter() - start) * 1000
        if prediction.shape != (22, 2) or not np.isfinite(prediction).all():
            raise RuntimeError("The model produced an invalid prediction")
        current = x[-1, :22, :2].astype(np.float64)
        now = float(match.timestamps[anchor])
        return {"match_id": match.id, "anchor_frame": int(anchor), "anchor_time_s": now,
                "target_frame": int(anchor + FPS), "target_time_s": round(now + 1.0, 8),
                "player_ids": list(match.player_ids), "current_xy": current.tolist(),
                "predicted_xy": prediction.tolist(),
                "model": {"name": MODEL_NAME, "device": self.device,
                          "checkpoint_sha256": self.sha256, "inference_ms": elapsed_ms,
                          "latency_scope": "CPU forward pass only; excludes preprocessing and HTTP"},
                "insight": geometric_insight(current, prediction, match.player_ids)}


@asynccontextmanager
async def lifespan(app):
    app.state.matches = make_matches()
    app.state.highlights, app.state.focus_highlights, app.state.highlight_index = {}, {}, {}
    for key, match in app.state.matches.items():
        candidates = highlight_candidates(match)
        app.state.highlights[key] = discover_highlights(match, candidates=candidates)
        app.state.focus_highlights[key] = {player: discover_highlights(match, player, candidates=candidates)
                                           for player in match.player_ids}
        # Only server-discovered, selected moments are addressable by public APIs.
        groups = [app.state.highlights[key], *app.state.focus_highlights[key].values()]
        app.state.highlight_index[key] = {item["id"]: item for group in groups for item in group["highlights"]}
    app.state.predictor = Predictor(os.environ.get("DEMO_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    app.state.azure_insights = AzureInsightService()
    app.state.commentary = CommentaryService()
    yield


app = FastAPI(title="THE MATCH LAB — Football Match Intelligence", version="1.6.0", lifespan=lifespan)


@app.get("/api/health")
def health(request: Request):
    engine = request.app.state.predictor
    return {"version": app.version, "model_loaded": True, "device": engine.device, "checkpoint_sha256": engine.sha256,
            "model_name": MODEL_NAME, "synthetic": True, "model_selection": SELECTION,
            "services": {"azure_openai": request.app.state.azure_insights.get_status(),
                         "commentary": request.app.state.commentary.get_status()}}


@app.get("/api/matches")
def matches(request: Request):
    return {"matches": [match.summary() for match in request.app.state.matches.values()]}


def find_match(request, match_id):
    match = request.app.state.matches.get(match_id)
    if match is None:
        raise HTTPException(status_code=422, detail="Unknown synthetic match ID")
    return match


@app.get("/api/matches/{match_id}")
def replay(match_id: str, request: Request):
    return find_match(request, match_id).public_payload()


@app.get("/api/highlights")
def highlights(request: Request, match_id: str, player_id: str | None = None):
    match = find_match(request, match_id)
    if player_id is not None and player_id not in match.player_ids:
        raise HTTPException(status_code=422, detail="Unknown player ID")
    return (request.app.state.highlights[match_id] if player_id is None else
            request.app.state.focus_highlights[match_id][player_id])


def find_highlight(request, match_id, highlight_id, player_id=None):
    """Resolve a known server-discovered moment; never accept caller evidence."""
    match = find_match(request, match_id)
    if player_id is not None and player_id not in match.player_ids:
        raise HTTPException(status_code=422, detail="Unknown player ID")
    item = request.app.state.highlight_index[match_id].get(highlight_id)
    if item is None:
        raise HTTPException(status_code=422, detail="Unknown highlight for this replay")
    if player_id is not None and item["player_id"] != player_id:
        raise HTTPException(status_code=422, detail="Highlight does not belong to the selected player")
    return item


@app.get("/api/predict")
def predict(request: Request, match_id: str, anchor_frame: int = Query(..., ge=0)):
    match = find_match(request, match_id)
    if anchor_frame >= len(match.timestamps):
        raise HTTPException(status_code=422, detail="anchor_frame is outside this replay")
    try:
        return request.app.state.predictor.predict(match, anchor_frame)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class InsightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    match_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    anchor_frame: int = Field(ge=98, le=10_000_000, strict=True)
    player_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    audience: Literal["fan", "analyst"] = "fan"
    language: Literal["en"] = "en"
    highlight_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


def player_evidence(prediction, player_id):
    """Only current and model-predicted values enter the Microsoft request."""
    try:
        index = prediction["player_ids"].index(player_id)
    except ValueError as exc:
        raise ValueError("Unknown player ID") from exc
    current = np.asarray(prediction["current_xy"], dtype=np.float64)
    projected = np.asarray(prediction["predicted_xy"], dtype=np.float64)
    opponents = slice(11, 22) if index < 11 else slice(0, 11)
    return {"match_id": prediction["match_id"], "anchor_frame": prediction["anchor_frame"],
            "anchor_time_s": prediction["anchor_time_s"], "target_time_s": prediction["target_time_s"],
            "synthetic": True, "player_id": player_id, "team": "home" if index < 11 else "away",
            "current_nearest_opponent_m": float(np.linalg.norm(current[opponents] - current[index], axis=1).min()),
            "predicted_nearest_opponent_m": float(np.linalg.norm(projected[opponents] - projected[index], axis=1).min()),
            "predicted_displacement_m": float(np.linalg.norm(projected[index] - current[index])),
            "model_name": MODEL_NAME, "checkpoint_sha256": prediction["model"]["checkpoint_sha256"]}


def build_insight_evidence(match, prediction, player_id, highlight=None):
    """Join independently labelled observed history with a causal prediction.

    Discovery may choose moments across the replay. Its attached numerical
    reason is recomputed from <= anchor; no future observed positions enter
    the model, the prompt or this evidence object.
    """
    evidence = player_evidence(prediction, player_id)
    if highlight is None:
        return evidence
    anchor = prediction["anchor_frame"]
    if (highlight["anchor_frame"] != anchor or highlight["player_id"] != player_id or
            prediction["match_id"] != match.id):
        raise ValueError("Highlight, player and prediction must refer to the same moment")
    index = match.player_ids.index(player_id)
    opponents = slice(11, 22) if index < 11 else slice(0, 11)
    previous = match.positions[anchor - FPS, :22]
    current = match.positions[anchor, :22]
    previous_gap = float(np.linalg.norm(previous[opponents] - previous[index], axis=1).min())
    current_gap = float(np.linalg.norm(current[opponents] - current[index], axis=1).min())
    displacement = float(np.linalg.norm(current[index] - previous[index]))
    gap_change = current_gap - previous_gap
    evidence["highlight"] = {"id": highlight["id"], "kind": highlight["kind"],
                             "analysis_mode": "retrospective",
                             "window_start_s": float(match.timestamps[anchor - FPS]),
                             "window_end_s": evidence["anchor_time_s"],
                             "previous_nearest_opponent_m": previous_gap,
                             "current_nearest_opponent_m": current_gap,
                             "observed_gap_change_m": gap_change,
                             "observed_displacement_m": displacement,
                             "selection_score": abs(gap_change) + .35 * displacement}
    return evidence


@app.post("/api/insights")
def insights(body: InsightRequest, request: Request):
    match = find_match(request, body.match_id)
    if body.player_id not in match.player_ids:
        raise HTTPException(status_code=422, detail="Unknown player ID")
    if body.anchor_frame >= len(match.timestamps):
        raise HTTPException(status_code=422, detail="anchor_frame is outside this replay")
    highlight = find_highlight(request, body.match_id, body.highlight_id, body.player_id) if body.highlight_id else None
    if highlight and highlight["anchor_frame"] != body.anchor_frame:
        raise HTTPException(status_code=422, detail="Highlight anchor does not match the requested moment")
    try:
        prediction = request.app.state.predictor.predict(match, body.anchor_frame)
        evidence = build_insight_evidence(match, prediction, body.player_id, highlight)
        return request.app.state.azure_insights.generate(evidence, body.audience, body.language)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class CommentaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    match_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    highlight_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    player_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    audience: Literal["fan", "analyst"] = "fan"
    language: Literal["en"] = "en"
    cache_only: StrictBool = False


@app.post("/api/commentary")
def commentary(body: CommentaryRequest, request: Request):
    match = find_match(request, body.match_id)
    highlight = find_highlight(request, body.match_id, body.highlight_id, body.player_id)
    try:
        prediction = request.app.state.predictor.predict(match, highlight["anchor_frame"])
        evidence = build_insight_evidence(match, prediction, body.player_id, highlight)
        cached = request.app.state.commentary.prepare(evidence, highlight, body.audience, body.language, cache_only=True)
        if cached["status"] == "ready" or body.cache_only:
            return cached
        insight = request.app.state.azure_insights.generate(evidence, body.audience, body.language)
        return request.app.state.commentary.prepare(evidence, highlight, body.audience, body.language, insight=insight)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/commentary/audio/{key}.wav")
def commentary_audio(key: str, request: Request):
    content = request.app.state.commentary.audio(key)
    if content is None:
        raise HTTPException(status_code=404, detail="Commentary audio is not available")
    return Response(content=content, media_type="audio/wav", headers={"Cache-Control": "public, max-age=86400, immutable"})


WEB = Path(__file__).parent / "web"


@app.get("/", include_in_schema=False)
def index():
    if not (WEB / "index.html").is_file():
        raise HTTPException(status_code=503, detail="Replay interface has not been built")
    return FileResponse(WEB / "index.html")


# Serve a small local UI without a second development server.
app.mount("/", StaticFiles(directory=str(WEB), check_dir=False), name="web")
