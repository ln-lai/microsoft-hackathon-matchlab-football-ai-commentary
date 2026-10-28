# Release validation

Release **1.6.0** adds Fan/Coach presentation and two Agent roles. Its current source passed **238 backend tests**, including nine focused Reviewer cases, and nine controlled playback checks on 28 October 2026. The two roles also completed two recorded uncached Azure runs (one Fan and one Coach); see [proof records](docs/two-agent-proof.json). These are integration examples, not an accuracy benchmark.

The following browser and audio checks established the preceding 1.5.0 replay baseline. They are not a claim that every browser check was rerun after the presentation update. The [hosted demo](https://nextsecond-football-27oct26.azurewebsites.net/) returned HTTP 200 with the model loaded in a read-only health check at 04:24 UTC that day.

## Public release check

At approximately 04:58 UTC on 28 October 2026, the public health endpoint returned **1.6.0** and both Agent roles. All three served frontend files matched the release package. A real 22-player prediction and a prepared Home 09 audio-file hash also passed verification. This deployment check used no new text or speech generation. The previously retained 1.5.0 deployment package remains available privately for rollback.

## Established replay baseline

- **229 backend tests passed**, covering model input boundaries, highlight filtering, evidence validation, caching, request rejection and recovery from service failures. Cloud responses in this suite are mocked.
- **Nine controlled playback race checks passed** against the released JavaScript.
- Local browser acceptance completed a full 40-second replay with three sequential commentary clips. It also covered pause/resume, seeking, changing player or scene, commentary/caption/volume controls, the optional analyst step mode, empty highlight results, missing cached speech and delayed prediction responses.
- Public browser checks completed a 40-second replay and a separate highlight with actual audio playback. The highlight test checked pause/resume, commentary off while the picture continued, and captions off. Public full-replay checks sampled the first voice and the completed final voice; every intervening public voice was not separately sampled. All three were sampled locally.
- Public API checks matched the released frontend assets and all 13 prepared audio files, verified differing Home 09 and Home 07 highlights plus an honest empty result, and confirmed cache-only requests did not trigger new synthesis. A separate local check exercised Home 06.
- Desktop and phone layouts were checked at 1492 × 748, 1280 × 720 and 390 × 844.

## Reproducing checks

Use the commands in [RUNNING.md](RUNNING.md). The checkpoint-free service subset uses mocked responses and does not establish live Azure speech. Full backend tests require the exact frozen checkpoint. The checkpoint-free subset for current source includes the nine Reviewer cases and passes 178 tests; the full five-file suite passes 238. Browser and live service acceptance are distinct from those automated checks.

These results describe the tested prototype, not a production load benchmark, continuous monitoring guarantee or unseen-match accuracy claim.
