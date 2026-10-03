"use client";

import type { FaceResult } from "@/types/veritas";

type Props = {
  faces: FaceResult[];
  selectedId: number | null;
  onSelect: (id: number) => void;
};

export function FaceList({ faces, selectedId, onSelect }: Props) {
  if (faces.length === 0) return null;
  return (
    <div className="card">
      <p>DETECTED FACES</p>
      <div className="row">
        {faces.map((f) => (
          <button
            key={f.face_id}
            type="button"
            className={`face-btn ${f.classification.label} ${selectedId === f.face_id ? "active" : ""}`}
            onClick={() => onSelect(f.face_id)}
          >
            Face {f.face_id}
            <br />
            {f.classification.label.toUpperCase()} · {(f.classification.fake_probability * 100).toFixed(1)}%
          </button>
        ))}
      </div>
    </div>
  );
}
