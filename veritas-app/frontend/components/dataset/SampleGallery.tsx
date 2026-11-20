"use client";

import { useState } from "react";

interface Sample {
  id: string;
  image: string;
  maskOverlay: string;
  label: "Real" | "Fake" | "Scenario";
  description: string;
}

const SAMPLES: Sample[] = [
  {
    id: "01",
    image: "/samples/01_0.jpg_IN_1.jpg.png",
    maskOverlay: "/samples/01_0.jpg_IN_1.jpg_mask_overlay.png",
    label: "Fake",
    description: "Fake sample with its pre-generated localization mask",
  },
  {
    id: "02",
    image: "/samples/02_0.jpg_IN_10.jpg.png",
    maskOverlay: "/samples/02_0.jpg_IN_10.jpg_mask_overlay.png",
    label: "Fake",
    description: "Fake sample with its pre-generated localization mask",
  },
  {
    id: "03",
    image: "/samples/03_0.jpg_IN_50.jpg.png",
    maskOverlay: "/samples/03_0.jpg_IN_50.jpg_mask_overlay.png",
    label: "Fake",
    description: "Fake sample with its pre-generated localization mask",
  },
  {
    id: "04",
    image: "/samples/04_0.jpg_IN_100.jpg.png",
    maskOverlay: "/samples/04_0.jpg_IN_100.jpg_mask_overlay.png",
    label: "Fake",
    description: "Fake sample with its pre-generated localization mask",
  },
  {
    id: "05",
    image: "/samples/05_0.jpg_IN_110.jpg.png",
    maskOverlay: "/samples/05_0.jpg_IN_110.jpg_mask_overlay.png",
    label: "Fake",
    description: "Fake sample with its pre-generated localization mask",
  },
  {
    id: "06",
    image: "/samples/mixed_two_faces.png",
    maskOverlay: "/samples/mixed_two_faces.png",
    label: "Scenario",
    description: "Two-face scenario; only one stored crop is scored",
  },
];

interface ImageModalProps {
  sample: Sample | null;
  showMask: boolean;
  onClose: () => void;
  onToggleMask: () => void;
}

function ImageModal({ sample, showMask, onClose, onToggleMask }: ImageModalProps): React.ReactElement | null {
  if (!sample) return null;
  const labelClass =
    sample.label === "Fake"
      ? "bg-[#FEEBEA] text-danger"
      : sample.label === "Real"
        ? "bg-[#EAF3FE] text-brand"
        : "bg-[#F1EFFD] text-[#5A4AA3]";

  return (
    <div
      className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-white border-b border-border p-4 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span
                className={`px-2 py-1 rounded text-xs font-bold ${labelClass}`}
              >
                {sample.label}
              </span>
              <h3 className="text-lg font-bold text-navy">Sample {sample.id}</h3>
            </div>
            <p className="text-xs text-muted mt-1">{sample.description}</p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg hover:bg-[#F9FAFD] flex items-center justify-center transition-colors flex-shrink-0"
            aria-label="Close"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M15 5L5 15M5 5L15 15"
                stroke="#142849"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>

        <div className="p-6">
          <div className="relative bg-[#F9FAFD] rounded-xl overflow-hidden">
            <img
              src={showMask && sample.label === "Fake" ? sample.maskOverlay : sample.image}
              alt={`Sample ${sample.id}`}
              className="w-full h-auto"
            />
          </div>

          {sample.label === "Fake" && (
            <div className="mt-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="show-mask"
                  checked={showMask}
                  onChange={onToggleMask}
                  className="w-4 h-4 rounded border-border accent-brand"
                />
                <label htmlFor="show-mask" className="text-sm text-navy cursor-pointer">
                  Show localization mask overlay
                </label>
              </div>
              <p className="text-xs text-muted">
                {showMask ? "Mask overlay visible" : "Original image"}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export function SampleGallery(): React.ReactElement {
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const [showMask, setShowMask] = useState(false);

  const handleOpenSample = (sample: Sample): void => {
    setSelectedSample(sample);
    setShowMask(false);
  };

  const handleClose = (): void => {
    setSelectedSample(null);
    setShowMask(false);
  };

  const getLabelClass = (label: Sample["label"]): string => {
    if (label === "Fake") return "bg-[#FEEBEA] text-danger";
    if (label === "Real") return "bg-[#EAF3FE] text-brand";
    return "bg-[#F1EFFD] text-[#5A4AA3]";
  };

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {SAMPLES.map((sample) => (
          <button
            key={sample.id}
            onClick={() => handleOpenSample(sample)}
            className="group relative bg-[#F9FAFD] border border-[#DADADA] rounded-xl overflow-hidden hover:shadow-lg transition-all duration-200"
          >
            <div className="aspect-square overflow-hidden">
              <img
                src={sample.image}
                alt={`Sample ${sample.id}`}
                className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-105"
              />
            </div>
            <div className="p-3 bg-white border-t border-[#DADADA]">
              <div className="flex items-center gap-2 mb-1">
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold ${getLabelClass(sample.label)}`}
                >
                  {sample.label}
                </span>
                <span className="text-[10px] font-semibold text-muted">#{sample.id}</span>
              </div>
              <p className="text-[11px] text-navy text-left line-clamp-2">
                {sample.description}
              </p>
            </div>
            <div className="absolute top-2 right-2 bg-white/90 backdrop-blur-sm rounded-lg px-2 py-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-[10px] font-semibold text-navy">Click to view</p>
            </div>
          </button>
        ))}
      </div>

      <ImageModal
        sample={selectedSample}
        showMask={showMask}
        onClose={handleClose}
        onToggleMask={() => setShowMask(!showMask)}
      />
    </>
  );
}
