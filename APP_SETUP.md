# VERITAS app — teammate setup

The UI classifies each detected face as real or fake (OpenForensics Full VERITAS) and draws a localization mask on fakes.

You need **Python 3.10+**, **Node 18+**, and the **~800 MB checkpoint**. The zip is not in git.

## 1. Get the code

```bash
git clone https://github.com/sc-lacanlale/VERITAS.git
cd VERITAS
```

## 2. Download the model

Ask the team for `full_veritas_defacto_best.zip` (Kaggle output from `OF_04_full`, or the copy already used locally).

Put it here, exact name:

```text
models/full_veritas_defacto_best.zip
```

A Chrome download named `full_veritas_defacto_best (1).zip` also works — rename it to `full_veritas_defacto_best.zip`.

Optional: `MODEL_CHECKPOINT=C:\path\to\full_veritas_defacto_best.zip`

Do **not** commit the zip.

## 3. Backend

```bash
cd veritas-app/backend
pip install -r requirements.txt
python app.py
```

Wait for `VERITAS loaded` and `Inference service ready`. First start can take a minute. GPU is used if CUDA is installed; otherwise CPU.

Check: http://localhost:8000/health

You should see `"ok": true`, `"checkpoint": "full_veritas_defacto_best.zip"`.

## 4. Frontend

In a second terminal:

```bash
cd veritas-app/frontend
npm install
npm run dev
```

Open http://localhost:3000

Upload a `.jpg` / `.png` / `.webp` → **Analyze Image**.

## 5. If it fails

| Symptom | Fix |
| --- | --- |
| `/health` says `MODEL_NOT_CONFIGURED` | Zip missing or wrong folder. Path must be `models/full_veritas_defacto_best.zip` |
| Port 8000 in use | Stop the other `python app.py` |
| Frontend cannot reach API | Backend must be on port 8000. Frontend is 3000 |
| First load is slow | Normal (~800 MB weights) |

Tests (no checkpoint needed):

```bash
cd veritas-app/backend
pytest tests/test_inference.py -q
```
