# Running MatchLab Football AI Commentary

This guide applies to the version 1.6.1 source package. Run commands from the repository root, which contains `model.py` and `demo/`.

**The trained checkpoint is not included.** This repository supports source review and checkpoint-free service tests, but the complete web application requires the compatible frozen checkpoint described below. A public download is not provided in this package. To try the working product without local setup, use the [Azure-hosted demonstration](https://nextsecond-football-27oct26.azurewebsites.net/).

The package also excludes original tracking data, pre-generated commentary audio, private Azure configuration and deployment credentials. Its three replays are generated synthetic data; no original match files are needed to generate them.

## 1. Install the application dependencies

Use Python 3.12. The application runs inference on CPU; CUDA, Node.js, an Azure SDK and a frontend build step are not required.

```sh
git clone https://github.com/ln-lai/microsoft-hackathon-matchlab-football-ai-commentary.git
cd microsoft-hackathon-matchlab-football-ai-commentary
python3.12 -m venv .venv
.venv/bin/python -m pip install -r demo/requirements-dev.txt
.venv/bin/python -m pip check
```

The development requirements include the runtime requirements:

| Dependency | Pinned version | Purpose |
| --- | --- | --- |
| FastAPI | 0.162.0 | Web API and static interface |
| Uvicorn | 0.54.0 | Application server |
| NumPy | 2.5.4 | Replay data and geometry |
| PyTorch | 2.14.1 | Frozen CPU prediction model |
| HTTPX | 0.28.1 | Test client |
| pytest | 9.1.1 | Test suite |

For runtime only, install `demo/requirements.txt`. These files pin direct dependencies; they are not a complete lockfile for transitive dependencies. Commands above use macOS/Linux paths. On Windows, use `.venv\Scripts\python.exe` and equivalent environment-variable commands.

## 2. Run tests without a checkpoint or Azure

These tests inject mock service transports and do not request paid model inference:

```sh
.venv/bin/python -m pytest demo/test_azure_insights.py demo/test_commentary.py demo/test_reviewed_insights.py -q -k 'not api_ and not public_cache_only'
```

The audited package passed **178 tests**, with **21 deselected**. Both exclusions matter: the `public_cache_only` tests start the application and therefore require the checkpoint even though their names do not contain `api_`.

Without the checkpoint, starting the complete web server fails during startup with a missing-file error. There is no random-weight substitute or checkpoint-free web mode.

## 3. Start the complete local application

Use only a checkpoint you are authorized to access. The server requires this exact SHA-256:

```text
4ecea9ccfe5c632afe5da0be7e76150ca31ea85142fac0dd0d8aa0cea6233923
```

It must contain the frozen model's `state_dict`, `config` and training normalization. The server checks the hash before loading it and rejects a different file. Do not change the hash check to use arbitrary weights.

Either place the file at the default repository-relative path:

```text
runs/three_seed_stability/seed_20261026/best.pt
```

or point the server to its private location:

```sh
export DEMO_CHECKPOINT='/absolute/path/to/authorized/best.pt'
.venv/bin/python -m uvicorn demo.backend:app --host 127.0.0.1 --port 8000 --workers 1
```

Open [the local application](http://127.0.0.1:8000/). Its [health endpoint](http://127.0.0.1:8000/api/health) reports the application version, checkpoint hash and actual service configuration. Run one worker: the service locks and rate limits have a single-process contract.

Leave the optional `AZURE_*` variables unset for an offline model-service configuration. With the checkpoint available, this provides:

- All three 40-second synthetic replays, continuous playback and highlight selection.
- Follow-player filtering, pitch focus and evidence for the selected player.
- Frozen CPU forecasts at +1 second and recorded-outcome comparison.
- Clearly labelled local evidence summaries, short commentary text and captions.

It does not synthesize speech. With no separately supplied valid audio cache, speech returns `unavailable` / `not_configured`; a cache-only lookup returns `cache_miss`. Replay and evidence continue. There is no browser speech fallback.

## 4. Optional Azure evidence selection and speech

The application reads process environment variables. Copying `.env.example` to `.env` does **not** load it automatically; export the edited values in the shell or supply them through your server environment. Remove unused placeholder values. Keep secrets, weights and generated audio outside version control.

Use existing deployments in your own authorized Azure resource. This package does not create infrastructure or grant access to the demonstration's services.

| Capability | Required environment variables |
| --- | --- |
| Cloud evidence selection | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` |
| Exact-script TTS | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_TTS_DEPLOYMENT` |
| GPT Audio route | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_AUDIO_DEPLOYMENT` |

Deployment values are your Azure **deployment names**, not interchangeable model labels. The speech implementation supports the existing TTS / GPT Audio routes in `demo/commentary.py`; arbitrary models are not a supported substitute. When TTS is configured it is preferred for new clips. GPT Audio output must match the supplied script's provider transcript. TTS metadata identifies its transcript as synthesis input, not independently verified recognition. Both routes use a generic built-in voice.

The two Agent roles use the same text deployment and allowance; an uncached successful pair consumes two text requests. Coach uses the existing API value `audience: "analyst"`. A standalone `coach` value is not accepted. See [ARCHITECTURE.md](ARCHITECTURE.md) for status semantics and the review boundary.

Cloud evidence selection is optional for speech: if it is absent or fails, validated local facts remain available. A configured deployment or quota alone does not prove a successful audio call; check the returned status and actually listen to a prepared clip.

### Prepare speech before expecting to hear it

No ready audio ships in this source package. The browser prepares the selected highlight in the background, including the initially selected highlight. Selecting a different scene, focus, highlight or explanation mode can request new preparation. These uncached requests can call configured Azure services. The AI commentary switch controls playback; it is not a guarantee that preparation requests are disabled.

1. Select a scene, followed player and Fan or Coach mode.
2. Select a highlight and wait for **AI commentary ready**.
3. Click **Play this highlight** with AI commentary enabled. The click permits browser audio playback. Use 1× speed for the normal demonstration.
4. To hear several moments in **Play full replay · 40s**, prepare the desired highlights in that same scene, focus and mode first.

Full-replay bulk lookups use `cache_only: true`: they read prepared clips and never synthesize missing speech. An unready or late clip is skipped while the replay and captions continue. Select and prepare the highlight, then replay it; audio is not inserted into the middle of an already-passed moment. Coach and Fan clips have separate cache identities, as do different players, scenes, highlights and content versions. Some boundary clips are too short for speech and remain caption-only.

For API integration, discover valid IDs with `GET /api/highlights?match_id=space-run&player_id=Home_9`, then send `POST /api/commentary` with `match_id`, the returned `highlight_id`, its `player_id`, `audience` (`fan` or `analyst`) and `language: "en"`. Adding the strict JSON boolean `cache_only: true` makes this a read-only lookup. Omitting it permits bounded generation on a cache miss. Arbitrary scripts are not accepted.

The default local cache is `demo/artifacts/commentary-cache/`; `COMMENTARY_CACHE_DIR` can select a private writable directory. `COMMENTARY_SEED_DIR` can point to separately supplied verified audio/metadata pairs, but none are included here. Valid prepared audio can be replayed without live Azure credentials.

Keep the existing request guards: speech defaults to **3 attempts per minute** and **60 total attempts** recorded in the persistent cache's `usage.json`; failures count. Evidence selection defaults to **6 attempts per minute** and **100 per process**, resetting on restart. Do not delete the speech ledger to bypass its limit. These are request safeguards, not a monetary spending cap. Service errors, timeouts and exhausted limits retain local evidence and captions.

## 5. Run the complete test suite

The complete suite needs the exact checkpoint at the **default path**, because some test fixtures call `Predictor()` directly. Setting `DEMO_CHECKPOINT` alone redirects server startup but does not redirect those fixtures. A private local symlink can be used where supported:

```sh
export DEMO_CHECKPOINT='/absolute/path/to/authorized/best.pt'
mkdir -p runs/three_seed_stability/seed_20261026
ln -s "$DEMO_CHECKPOINT" runs/three_seed_stability/seed_20261026/best.pt
.venv/bin/python -m pytest demo/test_backend.py demo/test_azure_insights.py demo/test_insight_api.py demo/test_commentary.py demo/test_reviewed_insights.py -q
```

Skip the symlink command if the correct file is already at that path. Keep Azure service variables unset when running tests; cloud responses are mocked. The audited package passed **238 tests**. A Starlette/HTTPX deprecation warning was emitted without failing the suite; no dependency upgrade was made during the audit.

## Verification scope

On 28 October 2026, the source package was copied to a temporary directory and tested with Python 3.12.14 and the dependency versions above. The checkpoint-free tests, complete suite with the existing private checkpoint, and `pip check` passed. With Azure variables removed, ASGI startup and routes confirmed the page, three replays, 22-player forecasts, local explanations and caption-only speech fallback with zero paid attempts. The `DEMO_CHECKPOINT` override was checked separately from the default-path test setup.

This audit reused installed dependencies; it did not verify a fresh download from a package index. Uvicorn reached application startup, but the audit sandbox blocked binding a new local port, so route checks used FastAPI's TestClient. Mocked audio tests do not establish live Azure availability or audible browser playback. The public deployment's prepared audio and deployment validation are separate from this source-only local-running audit.
