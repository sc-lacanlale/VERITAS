# VERITAS inference app

Upload one image. Each face is classified independently (OpenForensics Full VERITAS). Fake faces get a localization mask.

**Teammates: start at [`../APP_SETUP.md`](../APP_SETUP.md)** — download `full_veritas_defacto_best.zip`, put it in `models/`, then start backend and frontend.

```text
Next.js (localhost:3000)
        │  POST /predict
        ↓
FastAPI (localhost:8000)
        │
Face detector → crop / letterbox 600px → Full VERITAS → masks
```

The app prefers `models/full_veritas_defacto_best.zip` (from `OF_04_full`). Weights are not invented and are not in git.
