"use client";

import { useRef, useState, useMemo } from "react";

interface UploadPanelProps {
  file: File | null;
  busy: boolean;
  error: string | null;
  onFile: (file: File) => void;
  onAnalyze: () => void;
}

const PIPELINE_STEPS = [
  { n: 1, title: "File validation", desc: "Check file format, size and integrity." },
  { n: 2, title: "Face detection", desc: "Detect and crop face region(s)." },
  { n: 3, title: "Preprocessing", desc: "Align, normalize and prepare input." },
  { n: 4, title: "Model inference", desc: "Run the trained model for detection and segmentation." },
];

export function UploadPanel({ file, busy, error, onFile, onAnalyze }: UploadPanelProps): React.ReactElement {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  // Preview is derived from the validated `file` prop set by the parent,
  // so rejected files never show a preview.
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  function handleDrop(e: React.DragEvent): void {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) onFile(dropped);
  }

  return (
    <div className="flex flex-col gap-6 font-inter w-full">
      {/* Header */}
      <div className="text-left">
        <h1 className="text-3xl font-extrabold text-navy tracking-[-0.002rem] text-left">Analyze Image</h1>
        <p className="text-sm text-muted mt-1 text-left">
          Upload a facial image to detect potential manipulation and generate a localization map.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-5 items-start">
        {/* Upload card */}
        <div className="bg-white border border-border rounded-xl p-6 flex-[2] w-full">
          <h2 className="text-base font-bold text-navy mb-4">Upload image</h2>

          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-4 transition-colors min-h-[360px] p-6 relative overflow-hidden
              ${dragging ? "border-brand bg-[#F1EFFD]" : "border-border bg-[#FAFBFD]"}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onFile(f);
                e.target.value = "";
              }}
            />

            {preview ? (
              <>
                <img
                  src={preview}
                  alt="preview"
                  className="max-h-[300px] max-w-full rounded-lg object-contain"
                />
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  className="flex items-center gap-2 bg-white border border-border text-navy px-5 py-2 rounded-lg font-semibold text-sm hover:bg-[#F4F6FF] transition-colors"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                    <path d="M21 3v5h-5" />
                  </svg>
                  Change image
                </button>
              </>
            ) : (
              <>
                <img src="/icons/image-icon.svg" alt="" width={40} height={40} className="object-contain opacity-70" />
                <p className="text-sm font-medium text-navy">Drag an image here</p>
                <div className="flex items-center gap-3 w-full max-w-[200px]">
                  <div className="h-px bg-border flex-1" />
                  <span className="text-xs text-muted">OR</span>
                  <div className="h-px bg-border flex-1" />
                </div>
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  className="flex items-center gap-2 bg-brand text-white px-6 py-3 rounded-lg font-semibold text-sm hover:bg-[#2222cc] transition-colors"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                  </svg>
                  Upload image
                </button>
                <p className="text-[11px] text-muted">Supported formats: JPG, JPEG, PNG, WebP | Max size: 10 MB</p>
              </>
            )}
          </div>

          <div className="h-px bg-border my-5" />

          <div className="flex items-center justify-between gap-3">
            <span className="text-sm text-muted truncate max-w-[70%]" title={file?.name}>
              {file ? file.name : "No image selected"}
            </span>
            <button
              type="button"
              onClick={onAnalyze}
              disabled={!file || busy}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm transition-colors
                ${file && !busy
                  ? "bg-brand text-white hover:bg-[#2222cc]"
                  : "bg-[#EDEEF2] text-[#A5AAB5] cursor-not-allowed"}`}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              {busy ? "Analyzing..." : "Analyze image"}
            </button>
          </div>

          {error && <p className="text-danger text-sm mt-3">{error}</p>}
        </div>

        {/* Pipeline card */}
        <div className="bg-white border border-border rounded-xl p-6 flex-1 w-full">
          <h2 className="text-base font-bold text-navy mb-1">Analysis Pipeline</h2>
          <p className="text-xs text-muted mb-5">Steps performed after image upload</p>
          <div className="flex flex-col gap-5">
            {PIPELINE_STEPS.map((step, idx) => (
              <div key={step.n} className="flex gap-3 relative">
                {idx < PIPELINE_STEPS.length - 1 && (
                  <div className="absolute left-[15px] top-8 bottom-[-20px] w-px bg-border" />
                )}
                <div className="w-8 h-8 rounded-full bg-[#EAF3FE] text-brand font-bold text-sm flex items-center justify-center flex-shrink-0 z-10">
                  {step.n}
                </div>
                <div>
                  <p className="text-sm font-semibold text-navy">{step.title}</p>
                  <p className="text-xs text-muted leading-snug">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
