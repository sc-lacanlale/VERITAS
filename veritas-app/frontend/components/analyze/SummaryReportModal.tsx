"use client";

import type { PredictionResponse } from "@/types/veritas";
import { PrintReport } from "./PrintReport";

interface SummaryReportModalProps {
  result: PredictionResponse;
  fileName: string;
  onClose: () => void;
}

export function SummaryReportModal({
  result,
  fileName,
  onClose,
}: SummaryReportModalProps): React.ReactElement {
  const timestamp = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
  const inferenceMs = result.stages?.inference.ms ?? null;
  const fakeCount = result.faces.filter((f) => f.classification.label === "fake").length;

  function handlePrint(): void {
    window.print();
  }

  return (
    <>
    {/* Print-only forensic report */}
    <PrintReport result={result} fileName={fileName} />

    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 no-print">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl w-full max-w-[880px] max-h-[90vh] overflow-y-auto shadow-xl">
        {/* Header — navy band with white logo + text */}
        <div
          className="flex items-start justify-between gap-8 p-6 rounded-t-2xl"
          style={{ backgroundColor: "#142849" }}
        >
          <div className="flex flex-col gap-2">
            <img src="/icons/veritas-logo-vector.svg" alt="VERITAS" className="h-8 w-auto self-start" />
            <h2 className="text-lg font-bold text-white leading-tight">
              Forensic Analysis Summary Report
            </h2>
          </div>
          <div className="flex items-start gap-3">
            <div className="text-right text-[11px] text-[#c9d2e3]">
              <p>Sample timestamp: {timestamp}</p>
              <p>Sample case hash: 0x8F2A...9C1B</p>
            </div>
            <button onClick={onClose} className="text-[#c9d2e3] hover:text-white">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-6 flex flex-col gap-6">
          {/* Metadata + annotated image */}
          <div>
            <h3 className="text-sm font-bold text-navy mb-3">Case &amp; Image Metadata</h3>
            <div className="flex flex-col lg:flex-row gap-4 items-start">
              {result.overlay_image && (
                <figure className="m-0 w-full lg:w-[44%] flex-shrink-0">
                  <img
                    src={result.overlay_image}
                    alt="input image with detected faces"
                    className="rounded-lg w-full object-contain border border-border"
                  />
                  <figcaption className="text-[11px] text-muted mt-1.5">Input image with detected faces</figcaption>
                </figure>
              )}
              <div className="grid grid-cols-2 gap-3 flex-1 w-full">
                <MetaBox label="File Name" value={fileName} />
                <MetaBox label="Dimensions" value={`${result.image.width}×${result.image.height}`} />
                <MetaBox label="Faces Detected" value={String(result.faces.length)} />
                <MetaBox label="Processing time" value={inferenceMs !== null ? `${inferenceMs} ms` : "—"} />
                <MetaBox label="Flagged as manipulated" value={String(fakeCount)} />
                <MetaBox label="Detector" value={result.detector ?? "—"} />
              </div>
            </div>
          </div>

          {/* Results table */}
          <div>
            <h3 className="text-sm font-bold text-navy mb-3">Multi-Face Results Table</h3>
            <div className="border border-border rounded-xl overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-[#F6F7F9] text-left text-[11px] text-muted uppercase">
                    <th className="px-4 py-2 font-semibold">Face ID</th>
                    <th className="px-4 py-2 font-semibold">Crop</th>
                    <th className="px-4 py-2 font-semibold">Model Score</th>
                    <th className="px-4 py-2 font-semibold">Verdict</th>
                    <th className="px-4 py-2 font-semibold">Mask Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.faces.map((face) => {
                    const isFake = face.classification.label === "fake";
                    const cropSrc = face.crop || undefined;
                    return (
                      <tr key={face.face_id} className="border-t border-border">
                        <td className="px-4 py-3 font-semibold text-navy">Face {face.face_id}</td>
                        <td className="px-4 py-3">
                          {cropSrc && <img src={cropSrc} alt="" className="w-9 h-9 rounded object-cover" />}
                        </td>
                        <td className="px-4 py-3 text-navy">{face.classification.fake_probability.toFixed(3)}</td>
                        <td className="px-4 py-3">
                          <span
                            className={`text-[11px] font-bold px-2 py-0.5 rounded-md
                              ${isFake ? "bg-[#FEEBEA] text-danger" : "bg-[#E9F5F1] text-positive"}`}
                          >
                            {isFake ? "FAKE" : "REAL"}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-muted text-xs">
                          {isFake ? "Manipulated" : "Zero Mask"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Footer */}
          <div className="flex items-end justify-between gap-4 pt-2">
            <div>
              <p className="text-[11px] text-muted mb-1">Verification Signature · placeholder</p>
              <div className="w-48 h-10 border border-border rounded-lg bg-[#FAFBFD]" />
            </div>
            <button
              onClick={handlePrint}
              className="flex items-center gap-2 bg-brand text-white px-5 py-2.5 rounded-lg font-semibold text-sm hover:bg-[#2222cc] transition-colors"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Download PDF
            </button>
          </div>
        </div>
      </div>
    </div>
    </>
  );
}

function MetaBox({ label, value }: { label: string; value: string }): React.ReactElement {
  return (
    <div className="border border-border rounded-lg px-3 py-2 bg-[#FAFBFD]">
      <p className="text-[10px] text-muted uppercase tracking-wide">{label}</p>
      <p className="text-sm text-navy font-medium truncate">{value}</p>
    </div>
  );
}
