# Fraud Ring Board — visual demo

A single self-contained HTML page (`index.html`) that visually explains *why* a graph
neural network can catch fraud that a tabular model (XGBoost / LightGBM) misses.

It's a synthetic, illustrative scenario — 20 transactions, 10 fraudulent, 8 of them
routed through one shared device — not a run of the models in this repo. See the
page's own footnotes for exactly which numbers are real (pulled from `README.md`)
and which are staged for the walkthrough.

No build step, no dependencies, no server-side code. It's one HTML file with inlined
CSS/JS, plus two Google Fonts loaded over the network.

## How to run it

**Option 1 — just open it (fastest)**

```bash
open demo/index.html          # macOS
# or double-click demo/index.html in Finder
```

**Option 2 — serve it locally (closer to how you'd share/present it)**

```bash
cd demo
python3 -m http.server 8000
```

Then open `http://localhost:8000` in a browser.

## Showing the demo

1. Load the page. Panel A (XGBoost) and Panel B (GraphSAGE) start identical — every
   transaction scored from its own features, ring members look clean (green).
2. Click **Run message passing →**. Watch the pulses travel from the 8 ring
   transactions into the shared device node (`DVC-771`) in Panel B.
3. The ring flips red in Panel B only, and the tally below updates from 2/10 to
   10/10 caught — Panel A never changes, which is the point.
4. Click **Replay ↺** to reset Panel B and run it again.

Works offline except for the two Google Fonts requests (Fraunces, Public Sans, IBM
Plex Mono) — if you're fully offline the page still works, it just falls back to
system fonts.
