"use client";

import { useState, useRef } from "react";
import type { PredictionResponse, FaceResult } from "@/types/veritas";
import { FaceCard } from "./FaceCard";
import { SelectedFaceView } from "./SelectedFaceView";
import { SummaryReportModal } from "./SummaryReportModal";
import { LoadingOverlay } from "@/components/ui/LoadingOverlay";

interface AnalysisResultsProps {
  result: PredictionResponse;
  fileName: string;
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  onReset: () => void;
}

export function AnalysisResults({
  result,
  fileName,
  selectedId,
  onSelect,
  onReset,
}: AnalysisResultsProps): React.ReactElement {
  const [reportOpen, setReportOpen] = useState(false);
  const [transitioning, setTransitioning] = useState(false);
  const imagePanelRef = useRef<HTMLDivElement>(null);
  const selected: FaceResult | null =
    result.faces.find((f) => f.face_id === selectedId) ?? null;

  const overlaySrc = result.overlay_image || undefined;

  // Brief loading state (~450ms) when switching between faces / back to full image
  function runWithTransition(action: () => void, scroll = false): void {
    setTransitioning(true);
    action();
    if (scroll && typeof window !== "undefined" && window.innerWidth < 1024) {
      requestAnimationFrame(() => {
        imagePanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
    window.setTimeout(() => setTransitioning(false), 450);
  }

  // On mobile (< lg), scroll the image panel into view after selecting a face
  function handleSelect(id: number | null): void {
    runWithTransition(() => onSelect(id), id !== null);
  }

  return (
    <div className="flex flex-col gap-6 font-inter w-full">
      {/* Header */}
      <div className="text-left">
        <h1 className="text-3xl font-extrabold text-navy tracking-[-0.002rem] text-left">Analysis Results</h1>
        <p className="text-sm text-muted mt-1 text-left">Review each detected face and select one to inspect its result.</p>
      </div>

      {result.faces.length === 0 ? (
        <div className="bg-white border border-border rounded-xl p-8 text-center">
          <p className="text-sm text-muted">No faces detected in this image.</p>
          <button
            onClick={onReset}
            className="mt-4 inline-flex items-center gap-2 bg-brand text-white px-5 py-2.5 rounded-lg font-semibold text-sm hover:bg-[#2222cc] transition-colors"
          >
            Analyze another image
          </button>
        </div>
      ) : (
        <div className="flex flex-col lg:flex-row gap-5 items-start">
          {/* Left: image / selected face */}
          <div ref={imagePanelRef} className="bg-white border border-border rounded-xl p-5 flex-[2] w-full scroll-mt-20 min-h-[320px]">
            {transitioning ? (
              <LoadingOverlay variant="inline" label="Loading..." />
            ) : selected ? (
              <SelectedFaceView face={selected} onBack={() => runWithTransition(() => onSelect(null))} />
            ) : (
              <>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-base font-bold text-navy">Image overview</h2>
                  <span className="text-xs text-muted bg-[#F1F2F5] px-3 py-1 rounded-md">No face selected</span>
                </div>
                {overlaySrc && (
                  <img
                    src={overlaySrc}
                    alt="analysis overlay"
                    className="rounded-lg mx-auto block max-h-[420px] w-auto object-contain"
                  />
                )}
                <div className="flex items-center gap-5 mt-4">
                  <div className="flex items-center gap-2">
                    <span className="w-4 h-4 rounded bg-[#C4F0E3] border border-[#2C8833]" />
                    <span className="text-xs text-muted">Predicted real</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-4 h-4 rounded bg-[#FEEBEA] border border-[#C05852]" />
                    <span className="text-xs text-muted">Predicted fake / mask overlay</span>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Right: detected faces + actions */}
          <div className="flex flex-col gap-4 flex-1 w-full">
            <div className="bg-white border border-border rounded-xl p-5">
              <h2 className="text-base font-bold text-navy mb-1">Detected faces</h2>
              <p className="text-xs text-muted mb-4">Choose a face to inspect.</p>
              <div className="flex flex-col gap-3">
                {result.faces.map((face) => (
                  <FaceCard
                    key={face.face_id}
                    face={face}
                    active={face.face_id === selectedId}
                    onClick={() => handleSelect(face.face_id)}
                  />
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <button
                onClick={onReset}
                className="flex items-center justify-center gap-2 bg-brand text-white px-5 py-2.5 rounded-lg font-semibold text-sm hover:bg-[#2222cc] transition-colors"
              >
                <img src="/icons/image-icon.svg" alt="" width={18} height={18} className="object-contain invert brightness-0" style={{ filter: "brightness(0) invert(1)" }} />
                Analyze another image
              </button>
              <button
                onClick={() => setReportOpen(true)}
                className="flex items-center justify-center gap-2 bg-[#EAF0FE] text-brand px-5 py-2.5 rounded-lg font-semibold text-sm hover:bg-[#dde6fd] transition-colors"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="6 9 6 2 18 2 18 9" />
                  <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2" />
                  <rect x="6" y="14" width="12" height="8" />
                </svg>
                Export analysis results
              </button>
            </div>
          </div>
        </div>
      )}

      {reportOpen && (
        <SummaryReportModal
          result={result}
          fileName={fileName}
          onClose={() => setReportOpen(false)}
        />
      )}
    </div>
  );
}
