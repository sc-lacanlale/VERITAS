"use client";

import type { FaceResult } from "@/types/veritas";

type Props = {
  overlay?: string;
  originalUrl?: string;
  faces: FaceResult[];
  selectedId: number | null;
};

export function ImageViewer({ overlay, originalUrl, faces, selectedId }: Props) {
  const src = overlay || originalUrl;
  if (!src) return null;
  return (
    <div className="card">
      <p>FULL IMAGE</p>
      <img className="full" src={src} alt="VERITAS overlay" />
      <p className="muted">
        {faces.length} face{faces.length === 1 ? "" : "s"}
        {selectedId !== null ? ` · selected Face ${selectedId}` : ""}
      </p>
    </div>
  );
}
