"use client";

import type { FaceResult } from "@/types/veritas";

interface FaceCardProps {
  face: FaceResult;
  active: boolean;
  onClick: () => void;
}

export function FaceCard({ face, active, onClick }: FaceCardProps): React.ReactElement {
  const isFake = face.classification.label === "fake";
  const cropSrc = face.crop || undefined;

  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-4 p-3 rounded-xl border text-left transition-colors w-full
        ${active ? "border-brand bg-[#F4F6FF]" : "border-border bg-[#FAFBFD] hover:bg-[#F4F6FF]"}`}
    >
      {cropSrc && (
        <img src={cropSrc} alt={`Face ${face.face_id}`} className="w-16 h-16 rounded-lg object-cover flex-shrink-0" />
      )}
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-navy">Face {face.face_id}</span>
          <span
            className={`text-[11px] font-bold px-2 py-0.5 rounded-md
              ${isFake ? "bg-[#FEEBEA] text-danger" : "bg-[#E9F5F1] text-positive"}`}
          >
            {isFake ? "FAKE" : "REAL"}
          </span>
        </div>
        <span className="text-[11px] text-muted">Model score</span>
        <span className="text-sm font-bold text-navy">
          {face.classification.fake_probability.toFixed(3)}
        </span>
      </div>
    </button>
  );
}
