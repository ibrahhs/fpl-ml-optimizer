# FPL 25/26 Optimizer v3

- Live FPL data (auto-updated for 25/26 via official public endpoints)
- Single-GW optimizer + captain/vice
- Multi-GW planner with horizon EP
- Player ID directory (download)
- Transfers vs current team: suggest best 1–2 moves by marginal EP gain

## Deploy
Push files to GitHub (root), then deploy on Streamlit Community Cloud → `app.py`.

## Notes
- Budget is in tenths (1000 = £100.0).
- Chip logic kept heuristic for clarity. Extendable to multi-GW chip search.
