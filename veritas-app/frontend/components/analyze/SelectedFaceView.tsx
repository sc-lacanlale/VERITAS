"use client";

import { useState } from "react";
import type { FaceResult } from "@/types/veritas";

interface SelectedFaceViewProps {
  face: FaceResult;
  onBack: () => void;
  /** True when the result came from the baseline (classification only). */
  isBaseline?: boolean;
}

export function SelectedFaceView({ face, onBack, isBaseline = false }: SelectedFaceViewProps): React.ReactElement {
  const isFake = face.classification.label === "fake";
  const score = face.classification.fake_probability;

  // Prefer the mask-overlay crop for fakes, otherwise the plain crop
  const cropMask = face.segmentation?.crop_mask;
  const crop = face.crop;
  const [showOverlay, setShowOverlay] = useState(true);
  const imgSrc = (showOverlay ? cropMask : null) || crop || undefined;

  // Real explainability asset from the model's segmentation output.
  const binary = face.segmentation?.crop_binary ?? null;
  const coverage = face.segmentation?.coverage;
  const coveragePct =
    typeof coverage === "number" ? Math.round(coverage * 1000) / 10 : null;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 border border-border rounded-lg px-3 py-1.5 text-sm font-semibold text-navy hover:bg-[#F4F6FF] transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="19" y1="12" x2="5" y2="12" />
            <polyline points="12 19 5 12 12 5" />
          </svg>
          Back to full image
        </button>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {/* Toggle the red mask overlay so the plain input crop can be viewed too */}
          {cropMask && crop && (
            <button
              type="button"
              role="switch"
              aria-checked={showOverlay}
              onClick={() => setShowOverlay((v) => !v)}
              className="flex items-center gap-2.5 border border-border bg-white rounded-full pl-3 pr-1.5 py-1 text-xs font-semibold text-navy hover:bg-[#F4F6FF] transition-colors"
            >
              Mask overlay
              <span
                className="relative inline-block w-8 h-[18px] rounded-full transition-colors"
                style={{ backgroundColor: showOverlay ? "#3333FF" : "#D9D9D9" }}
              >
                <span
                  className="absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white shadow transition-all"
                  style={{ left: showOverlay ? "16px" : "2px" }}
                />
              </span>
            </button>
          )}
          <span className="text-xs text-brand bg-[#EAF0FE] px-3 py-1 rounded-md font-medium">
            Face {face.face_id} selected
          </span>
        </div>
      </div>

      {imgSrc && (
        <img
          src={imgSrc}
          alt={`Face ${face.face_id}`}
          className="rounded-lg mx-auto block max-h-[340px] w-auto object-contain"
        />
      )}

      {/* Prediction banner */}
      <div
        className="mt-4 rounded-xl px-4 py-3 sm:px-5 sm:py-3 flex items-center gap-3 sm:gap-5 relative overflow-hidden border"
        style={{
          backgroundColor: isFake ? "#FFE9E9" : "#E9F5F1",
          borderColor: isFake ? "#FFD0D0" : "#C4F0E3",
        }}
      >
        {/* Left accent bar */}
        <div
          className="absolute left-0 top-0 bottom-0 w-1.5 rounded-l-xl"
          style={{ backgroundColor: isFake ? "#FF3131" : "#2C8833" }}
        />

        {/* Icon — aligned with the FAKE/REAL text row */}
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ml-2"
          style={{ backgroundColor: isFake ? "#FAD4D1" : "#C4F0E3" }}
        >
          {isFake ? (
            <img src="/icons/fake-icon.svg" alt="" width={40} height={40} className="object-contain" />
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2C8833" strokeWidth="2.5">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          )}
        </div>

        {/* Prediction */}
        <div className="flex-1">
          <p className="text-[11px] sm:text-xs text-muted uppercase tracking-wide">
            Face {face.face_id} — Model prediction
          </p>
          <p className="text-2xl sm:text-3xl font-extrabold leading-tight" style={{ color: "#142849" }}>
            {isFake ? "FAKE" : "REAL"}
          </p>
          <p className="text-xs text-muted mt-0.5">
            {isFake ? "Predicted manipulated" : "Predicted authentic"}
          </p>
        </div>

        {/* Score */}
        <div className="flex-[1.4] min-w-[260px]">
          <div className="flex items-center gap-1.5">
            <span className="text-xs sm:text-sm text-muted">Model Score</span>
            <img src="/icons/info-icon.svg" alt="info" width={16} height={16} className="object-contain opacity-70" />
          </div>
          <p className="text-2xl sm:text-3xl font-extrabold leading-tight" style={{ color: "#142849" }}>
            {score.toFixed(3)} <span className="text-sm sm:text-base font-normal text-muted">/ 1.000</span>
          </p>
          <div className="h-2.5 bg-[#D9D9D9] rounded-full mt-2 overflow-hidden w-full">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min(score * 100, 100)}%`,
                backgroundColor: isFake ? "#FF3131" : "#2C8833",
              }}
            />
          </div>
          <div className="flex justify-between text-sm text-muted mt-1">
            <span>0</span>
            <span>1</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 mt-3">
        <span className="w-4 h-4 rounded" style={{ backgroundColor: "#FFE9E9", border: "1px solid #FF3131" }} />
        <span className="text-xs text-muted">{isBaseline ? "Predicted fake" : "Predicted fake / mask overlay"}</span>
      </div>

      {/* Baseline has no segmentation head, so there is no mask to show */}
      {isBaseline && (
        <div className="mt-6 border-t border-border pt-5">
          <h3 className="text-base font-bold text-navy">Explainability</h3>
          <div className="mt-3 rounded-xl border border-border bg-[#FAFBFD] p-4 flex items-start gap-3">
            <img src="/icons/info-icon.svg" alt="" width={18} height={18} className="object-contain opacity-70 mt-0.5" />
            <p className="text-xs text-muted leading-relaxed">
              Baseline EfficientNetB7 is classification-only, so it gives a score and verdict for
              Face {face.face_id} but no localization mask. Run the same image with VERITAS to see
              which region was flagged.
            </p>
          </div>
        </div>
      )}

      {/* Explainability — only for fake faces, using the model's real segmentation output */}
      {isFake && binary && (
        <div className="mt-6 border-t border-border pt-5">
          <h3 className="text-base font-bold text-navy">Explainability</h3>
          <p className="text-xs text-muted mt-0.5">
            Segmentation mask for Face {face.face_id}
            {coveragePct !== null ? ` · ${coveragePct}% of the face crop flagged` : ""}
          </p>

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-[1fr_1.2fr] gap-4 items-start">
            <figure className="m-0">
              <img src={binary} alt="predicted binary mask" className="rounded-lg w-full object-contain bg-black" />
              <figcaption className="text-[11px] font-semibold text-navy mt-1.5">Predicted binary mask</figcaption>
            </figure>

            <div className="text-xs text-muted leading-relaxed">
              <p className="font-semibold text-navy mb-1">How to read this</p>
              <p>
                White areas mark the pixels the model localized as manipulated (probability of at least
                0.5). The mask shows where the model places the manipulation, not how strong the evidence
                is. It was trained on generated masks, not hand-annotated boundaries.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
