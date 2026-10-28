# THE MATCH LAB Project Pitch

## Tagline

Follow your player. Hear the story. Inspect the evidence.

## Short pitch

THE MATCH LAB makes football movement understandable through continuous replay, synchronized commentary, and personal player focus. Fan mode tells the story; Coach mode exposes the evidence and next-second forecast. On Azure, GPT-5-mini powers a Match Analyst Agent that selects computed facts, followed by a Commentary Reviewer Agent checking the player, numbers, timing, and forecast wording through a lightweight sequential harness. Azure OpenAI Alloy supplies pre-generated, cached speech. The demo uses three synthetic 40-second scenes. Viewers can inspect review status and compare predicted positions with recorded outcomes, turning a pitch full of dots into an explanation they can follow.

## Extended pitch

Football tracking can show every movement without helping a viewer understand what matters. THE MATCH LAB turns that movement into a story: discover a highlight, watch it unfold, hear the evidence, and follow the player you care about.

The experience serves two perspectives. Fan mode offers clear, concise commentary and visual focus. Coach mode brings numerical evidence and next-second predictions into view. My Focus supports all 22 players, changing the recommended highlights, pitch emphasis, and explanation. Viewers can watch a complete scenario or replay one moment, with captions and annotations coordinated to the action.

Two specialist AI roles make the explanation inspectable. Using Azure-hosted GPT-5-mini, the Match Analyst Agent selects relevant computed facts. A separate Commentary Reviewer Agent request checks the explanation and proposed speech script for player identity, numerical claims, timing, and clear forecast wording. A lightweight harness coordinates this fixed handoff, validates responses, shares request limits, and records the outcome. Both roles have completed successfully in real Azure calls; the interface distinguishes successful review from unavailable review or local fallback.

Azure App Service hosts the application, while Azure OpenAI Alloy provides pre-generated, cached English narration synchronized with the replay. Pausing or changing scenes stops the corresponding sound, and unavailable audio leaves picture, captions, and evidence usable.

The demonstration contains three clearly labelled synthetic 40-second tracking scenarios. A frozen ST-GNN + TCN checkpoint, trained on three Metrica sample matches, predicts positions one second ahead. The replay reveals the recorded outcome at the matching time, separating observed movement from a model estimate.

THE MATCH LAB combines the accessibility of commentary with evidence viewers can inspect, giving fans a clearer story and coaches a focused way to explore movement.

## Live product

[Try THE MATCH LAB](https://nextsecond-football-27oct26.azurewebsites.net/)

Release 1.6.0 is deployed on the public demo. The two-agent workflow has also completed real Azure verification runs.

[Source code and setup](https://github.com/ln-lai/the-match-lab)
