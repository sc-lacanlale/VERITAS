# VERITAS inference app

Minimal face-level inference UI. Upload one image; every detected face is classified independently. Fake faces also get a manipulation mask projected back onto the original photo.

```text
Next.js (localhost:3000)
        │  POST /predict
        ↓
FastAPI (localhost:8000)
        ↓
Face detector → crop / letterbox 600px → Full VERITAS → masks
```

The checkpoint is **not** invented. It loads the trained `full_veritas` weights from `models/`.

## 1. Checkpoint

Place the trained file at one of:

```text
models/full_veritas_best.zip
models/full_veritas_best.pt
```

or set:

```bash
MODEL_CHECKPOINT=C:\path\to\full_veritas_best.zip
```

If Kaggle gave you an extracted zip (`full_veritas_best.pt/data.pkl` inside), leave it as `.zip`. The backend repacks it on first load.

Training protocol used here: EfficientNet-B7, multi-stream + segmentation, `image_size=600`, `face_margin=0.1`, threshold `0.5`.

## 2. Backend

```bash
cd veritas-app/backend
pip install -r requirements.txt
python app.py
```

Wait until it prints `VERITAS loaded` and `Inference service ready`. First load can take a minute (repack + ~800 MB weights). CUDA is used if available, otherwise CPU.

Health check: http://localhost:8000/health

## 3. Frontend

```bash
cd veritas-app/frontend
npm install
npm run dev
```

Open http://localhost:3000, choose a `.jpg` / `.png` / `.webp`, click **Analyze Image**.

## 4. What you should see

- Full image with boxes, face IDs, real/fake labels, and fake-face masks
- A button per face
- Selected face crop + mask, classification, fake probability

No faces → `No faces detected.` (HTTP 200, empty list). One fake face does **not** label the whole image fake.

## 5. Tests

```bash
cd veritas-app/backend
pytest tests/test_inference.py -q
```

These cover one face, many faces, mixed real/fake, no faces, mask projection, invalid upload, and the API schema. They use a stub model so they do not need the 800 MB checkpoint.

## Not included

Auth, database, video, webcam, batch jobs, or a polished UI. Another frontend can replace `frontend/` and keep `POST /predict`.
