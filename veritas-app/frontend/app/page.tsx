"use client";

import { useMemo, useState } from "react";
import { FaceList } from "@/components/FaceList";
import { FaceResultCard } from "@/components/FaceResult";
import { ImageUploader } from "@/components/ImageUploader";
import { ImageViewer } from "@/components/ImageViewer";
import { predictImage } from "@/lib/api";
import type { PredictionResponse } from "@/types/veritas";

export default function Page() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const selected = useMemo(
    () => result?.faces.find((f) => f.face_id === selectedId) ?? null,
    [result, selectedId]
  );

  function onFile(next: File) {
    setFile(next);
    setResult(null);
    setSelectedId(null);
    setError(null);
    setPreview(URL.createObjectURL(next));
  }

  async function onAnalyze() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const data = await predictImage(file);
      setResult(data);
      setSelectedId(data.faces[0]?.face_id ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <h1>VERITAS</h1>
      <p className="muted" style={{ textAlign: "center" }}>
        Upload an image. Each face is classified independently.
      </p>
      {result && (
        <p style={{ textAlign: "center" }}>
          <button
            type="button"
            onClick={() => {
              setFile(null);
              setPreview(null);
              setResult(null);
              setSelectedId(null);
            }}
          >
            Upload Another Image
          </button>
        </p>
      )}
      <ImageUploader onFile={onFile} onAnalyze={onAnalyze} busy={busy} hasFile={!!file} />
      {error && <p className="error">{error}</p>}
      {result && result.faces.length === 0 && <p>No faces detected.</p>}
      <ImageViewer
        overlay={result?.overlay_image}
        originalUrl={preview || undefined}
        faces={result?.faces ?? []}
        selectedId={selectedId}
      />
      {result && <FaceList faces={result.faces} selectedId={selectedId} onSelect={setSelectedId} />}
      <FaceResultCard face={selected} />
    </main>
  );
}
