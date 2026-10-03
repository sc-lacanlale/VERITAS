"use client";

import type { FaceResult } from "@/types/veritas";
import { FaceMaskViewer } from "./FaceMaskViewer";

type Props = { face: FaceResult | null };

export function FaceResultCard({ face }: Props) {
  if (!face) return null;
  const { classification, segmentation } = face;
  return (
    <div className="card">
      <p>SELECTED FACE {face.face_id}</p>
      <FaceMaskViewer crop={face.crop} cropMask={segmentation.crop_mask} />
      <p>
        Classification: <strong>{classification.label.toUpperCase()}</strong>
      </p>
      <p>Fake Probability: {(classification.fake_probability * 100).toFixed(1)}%</p>
      <p>
        Segmentation:{" "}
        {segmentation.available
          ? classification.label === "fake" && segmentation.mask
            ? "Manipulation detected"
            : "Available (no fake-region mask for this face)"
          : "Not available"}
      </p>
    </div>
  );
}
