"use client";

import { useRef, useState, useMemo, useEffect } from "react";
import type { AnalyzePhase, InlineError, StageDetailMap, StageStateMap, StepKey, StepStatus } from "@/app/(workspace)/analyze/page";

interface UploadPanelProps {
  file: File | null;
  busy: boolean;
  phase: AnalyzePhase;
  stageStates: StageStateMap;
  stageDetails: StageDetailMap;
  error: string | null;
  inlineError: InlineError | null;
  onDismissError: () => void;
  onFile: (file: File) => void;
  onAnalyze: () => void;
}

interface PipelineStep {
  key: StepKey;
  n: number;
  title: string;
  desc: string;
}

const PIPELINE_STEPS: PipelineStep[] = [
  { key: "validation", n: 1, title: "File validation", desc: "Check file format, size and integrity." },
  { key: "face_detection", n: 2, title: "Face detection", desc: "Detect and crop face region(s)." },
  { key: "preprocessing", n: 3, title: "Preprocessing", desc: "Align, normalize and prepare input." },
  { key: "inference", n: 4, title: "Model inference", desc: "Run the trained model for detection and segmentation." },
  { key: "explainability", n: 5, title: "Explainability map", desc: "Generate the Grad-XMage attribution map." },
];

export function UploadPanel({ file, busy, phase, stageStates, stageDetails, error, inlineError, onDismissError, onFile, onAnalyze }: UploadPanelProps): React.ReactElement {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  // Preview derived from the validated `file` prop set by the parent.
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  /**
   * Real per-step state comes straight from live server stage events.
   * Each step is waiting until it's its turn, active while actually running,
   * and done once the backend reports that stage complete.
   */
  function stepStateFor(index: number): StepStatus {
    return stageStates[PIPELINE_STEPS[index].key];
  }

  // The step currently running, for the compact progress header.
  const activeStep = PIPELINE_STEPS.find((s) => stageStates[s.key] === "active") ?? null;

  // A failure is present if any step is in the error state.
  const hasFailure = PIPELINE_STEPS.some((s) => stageStates[s.key] === "error");

  function handleDrop(e: React.DragEvent): void {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) onFile(dropped);
  }

  const fileExt = file?.name.split(".").pop()?.toUpperCase() ?? "";

  // Real progress from live stage states: each done step = 1 unit,
  // the currently active step counts as half.
  const doneCount = PIPELINE_STEPS.filter((_, i) => stepStateFor(i) === "done").length;
  const hasActive = PIPELINE_STEPS.some((_, i) => stepStateFor(i) === "active");
  const progressPct = busy
    ? ((doneCount + (hasActive ? 0.5 : 0)) / PIPELINE_STEPS.length) * 100
    : 0;
  const activeIndex = activeStep ? PIPELINE_STEPS.indexOf(activeStep) : -1;
  const headerLine = activeStep ? activeStep.title : "Analyzing image";

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

          {/* When a file is selected, show it full-bleed in the card (no dashed box) */}
          {preview ? (
            <>
              <img
                src={preview}
                alt="preview"
                className="w-full max-h-[320px] rounded-xl object-cover"
              />

              {/* File info row with Replace image */}
              <div className="mt-4 border border-border rounded-xl bg-[#F6F8FF] p-3 flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-[#142849] flex items-center justify-center flex-shrink-0">
                  <img src="/icons/image-icon.svg" alt="" width={20} height={20} className="object-contain" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-semibold text-navy truncate" title={file?.name}>
                    {file?.name}
                  </p>
                  <p className="text-[11px] text-muted">{fileExt} image, ready to analyze</p>
                </div>
                {phase === "idle" && (
                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    className="text-brand font-semibold text-sm hover:underline flex-shrink-0"
                  >
                    Replace image
                  </button>
                )}
              </div>
            </>
          ) : (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl flex flex-col items-center justify-center gap-4 transition-colors min-h-[240px] sm:min-h-[300px] lg:min-h-[360px] p-6
                ${dragging ? "border-brand bg-[#F1EFFD]" : "border-border bg-[#FAFBFD]"}`}
            >
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
            </div>
          )}

          {/* Hidden input */}
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

          {/* Inline error card (e.g. no face detected / face not clear enough) */}
          {inlineError?.variant === "card" && !busy && (
            <div
              className="mt-4 rounded-xl border p-4 flex items-start gap-3"
              style={{ backgroundColor: "#FFF2F2", borderColor: "#FFD0D0" }}
            >
              <div
                className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
                style={{ backgroundColor: "#FDE3E3" }}
              >
                {inlineError.icon ? (
                  <img src={`/icons/${inlineError.icon}.svg`} alt="" width={18} height={18} className="object-contain" />
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF3131" strokeWidth="2.5">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-bold text-navy">{inlineError.title}</p>
                <p className="text-[11px] font-semibold mb-1" style={{ color: "#FF3131" }}>
                  Analysis stopped at face detection
                </p>
                <p className="text-xs text-muted leading-snug">{inlineError.message}</p>
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  className="mt-3 inline-flex items-center gap-2 bg-brand text-white px-4 py-2 rounded-lg font-semibold text-sm hover:bg-[#2222cc] transition-colors"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                  Upload another image
                </button>
              </div>
            </div>
          )}

          {/* Progress bar while analyzing */}
          {busy && (
            <div className="mt-4 border border-border rounded-xl bg-white p-3 flex items-center gap-3">
              <div className="relative w-6 h-6 flex-shrink-0">
                <div className="absolute inset-0 rounded-full border-2 border-[#E5E7EB]" />
                <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-brand animate-spin" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-semibold text-navy">{headerLine}</p>
                <p className="text-[11px] text-muted">
                  {activeIndex >= 0
                    ? `Step ${activeIndex + 1} of ${PIPELINE_STEPS.length}: ${activeStep?.title}`
                    : `${doneCount} of ${PIPELINE_STEPS.length} steps complete`}
                </p>
              </div>
              <div className="w-28 h-2 bg-[#E5E7EB] rounded-full overflow-hidden flex-shrink-0">
                <div
                  className="h-full bg-brand rounded-full transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          )}

          <div className="h-px bg-border my-5" />

          <div className="flex items-center justify-between gap-3">
            <span className="text-sm text-muted truncate max-w-[70%]" title={file?.name}>
              {busy ? "Analysis in progress" : file ? `${file.name.length > 28 ? "1 image selected" : file.name}` : "No image selected"}
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

        {/* Pipeline / Processing status card */}
        <div className="bg-white border border-border rounded-xl p-6 flex-1 w-full">
          <h2 className="text-base font-bold text-navy mb-1">
            {busy ? "Processing Status" : hasFailure ? "Analysis Stopped" : "Analysis Pipeline"}
          </h2>
          <p className="text-xs text-muted mb-5">
            {busy
              ? `Live progress for ${file?.name ?? "image"}`
              : hasFailure
                ? "The analysis could not be completed"
                : "Steps performed after image upload"}
          </p>

          <div className="flex flex-col gap-5">
            {PIPELINE_STEPS.map((step, idx) => {
              const state = stepStateFor(idx);
              // Show live statuses while busy, or whenever a failure is present.
              const showStatus = busy || hasFailure;
              const done = showStatus && state === "done";
              const inProgress = showStatus && state === "active";
              const errored = showStatus && state === "error";
              const skipped = showStatus && state === "skipped";
              const detail = stageDetails[step.key];

              let status = "";
              if (errored) status = detail || "Error";
              else if (skipped) status = "Skipped";
              else if (done) status = detail || "Done";
              else if (inProgress) status = "In progress";
              else if (busy) status = "Waiting";

              return (
                <div key={step.n} className="flex gap-3 relative">
                  {idx < PIPELINE_STEPS.length - 1 && (
                    <div
                      className={`absolute left-[15px] top-8 bottom-[-20px] w-px ${done ? "bg-[#C4F0E3]" : "bg-border"}`}
                    />
                  )}

                  {/* Step indicator */}
                  {errored ? (
                    <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 z-10" style={{ backgroundColor: "#FDE3E3" }}>
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF3131" strokeWidth="2.5">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </div>
                  ) : done ? (
                    <div className="w-8 h-8 rounded-full bg-[#C4F0E3] flex items-center justify-center flex-shrink-0 z-10">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2C8833" strokeWidth="2.5">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    </div>
                  ) : inProgress ? (
                    <div className="relative w-8 h-8 flex-shrink-0 z-10">
                      <div className="absolute inset-0 rounded-full border-2 border-[#E5E7EB]" />
                      <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-brand animate-spin" />
                    </div>
                  ) : skipped ? (
                    <div className="w-8 h-8 rounded-full bg-[#EDEEF2] text-[#A5AAB5] font-bold text-sm flex items-center justify-center flex-shrink-0 z-10">
                      {step.n}
                    </div>
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-[#EAF3FE] text-brand font-bold text-sm flex items-center justify-center flex-shrink-0 z-10">
                      {step.n}
                    </div>
                  )}

                  <div>
                    <p className={`text-sm font-semibold ${skipped ? "text-[#A5AAB5]" : "text-navy"}`}>{step.title}</p>
                    {showStatus ? (
                      <p
                        className={`text-xs leading-snug font-medium ${
                          errored
                            ? "text-[#FF3131]"
                            : skipped
                              ? "text-[#A5AAB5]"
                              : done
                                ? "text-positive"
                                : inProgress
                                  ? "text-brand"
                                  : "text-muted"
                        }`}
                      >
                        {status}
                      </p>
                    ) : (
                      <p className="text-xs text-muted leading-snug">{step.desc}</p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Floating toast for unsupported-file errors */}
      {inlineError?.variant === "toast" && (
        <div className="fixed bottom-6 right-6 z-[200] w-[320px] max-w-[calc(100vw-2rem)]">
          <div
            className="rounded-xl border bg-white shadow-lg p-4 flex items-start gap-3"
            style={{ borderColor: "#FFD0D0" }}
          >
            <div
              className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: "#FDE3E3" }}
            >
              {inlineError.icon ? (
                <img src={`/icons/${inlineError.icon}.svg`} alt="" width={18} height={18} className="object-contain" />
              ) : (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF3131" strokeWidth="2.5">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-bold text-navy">{inlineError.title}</p>
              <p className="text-xs text-muted leading-snug">{inlineError.message}</p>
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="mt-2 text-brand font-semibold text-sm hover:underline"
              >
                Upload another image
              </button>
            </div>
            <button
              type="button"
              onClick={onDismissError}
              className="text-muted hover:text-navy flex-shrink-0"
              aria-label="Dismiss"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
