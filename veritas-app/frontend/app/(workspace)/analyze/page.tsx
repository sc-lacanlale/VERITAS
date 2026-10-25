"use client";

import { useMemo, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { predictImageStream, type StreamStageName } from "@/lib/api";
import type { PredictionResponse } from "@/types/veritas";
import { UploadPanel } from "@/components/analyze/UploadPanel";
import { AnalysisResults } from "@/components/analyze/AnalysisResults";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { NoticeDialog } from "@/components/ui/NoticeDialog";
import { LoadingOverlay } from "@/components/ui/LoadingOverlay";
import { useUnsavedGuard } from "@/lib/useUnsavedGuard";

const ALLOWED_EXT = ["jpg", "jpeg", "png", "webp"];
const MAX_BYTES = 10 * 1024 * 1024; // 10 MB

interface NoticeState {
  open: boolean;
  tone: "error" | "info" | "success";
  title: string;
  message: string;
  iconSrc?: string;
}

const NO_NOTICE: NoticeState = { open: false, tone: "info", title: "", message: "" };

/** Inline error shown inside the upload panel instead of a center dialog. */
export interface InlineError {
  /** "toast" = small floating card (unsupported file); "card" = in-panel block. */
  variant: "toast" | "card";
  title: string;
  message: string;
  /** Icon name under /public/icons without extension. */
  icon?: string;
}

/**
 * Real processing phases.
 * - idle: nothing running
 * - validating: client-side file checks (format/size)
 * - uploading: request in flight, server is detecting/preprocessing/inferring
 * - done: response received, stages reflect real counts
 * - failed: validation or request error
 */
export type AnalyzePhase = "idle" | "validating" | "uploading" | "preparing" | "done" | "failed";

/** Live per-step state used to drive the processing UI. */
export type StepStatus = "waiting" | "active" | "done" | "error" | "skipped";

/** The five pipeline steps in order. "validation" is client-side; the rest are server stages. */
export type StepKey = "validation" | StreamStageName;

export type StageStateMap = Record<StepKey, StepStatus>;

/** Short human-readable detail per step (e.g. "2 faces detected", "18.4 ms"). */
export type StageDetailMap = Partial<Record<StepKey, string>>;

const INITIAL_STAGES: StageStateMap = {
  validation: "waiting",
  face_detection: "waiting",
  preprocessing: "waiting",
  inference: "waiting",
  explainability: "waiting",
};

// Ordered keys so we can mark everything after a failing step as "skipped".
const STEP_ORDER: StepKey[] = [
  "validation",
  "face_detection",
  "preprocessing",
  "inference",
  "explainability",
];

/**
 * Build a pipeline state where `failedKey` errored and all later steps are
 * skipped. Steps before the failure keep their current state (default done).
 */
function buildFailedStages(failedKey: StepKey, prior: Partial<StageStateMap> = {}): StageStateMap {
  const idx = STEP_ORDER.indexOf(failedKey);
  const next = {} as StageStateMap;
  STEP_ORDER.forEach((key, i) => {
    if (i < idx) next[key] = prior[key] ?? "done";
    else if (i === idx) next[key] = "error";
    else next[key] = "skipped";
  });
  return next;
}

// Minimum time (ms) a step stays visibly "active" so fast stages don't flash by.
const MIN_STEP_MS = 550;

// Time (ms) to hold on the fully-completed pipeline before switching to results,
// so the final explainability step is clearly visible as done.
const FINAL_HOLD_MS = 1500;

// Time (ms) to show the "preparing results" loading overlay before the results view.
const PREPARING_MS = 1500;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export default function AnalyzePage(): React.ReactElement {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<AnalyzePhase>("idle");
  const [stageStates, setStageStates] = useState<StageStateMap>(INITIAL_STAGES);
  const [stageDetails, setStageDetails] = useState<StageDetailMap>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const [notice, setNotice] = useState<NoticeState>(NO_NOTICE);
  const [inlineError, setInlineError] = useState<InlineError | null>(null);
  const [pendingHref, setPendingHref] = useState<string | null>(null);

  useMemo(
    () => result?.faces.find((f) => f.face_id === selectedId) ?? null,
    [result, selectedId]
  );

  // There is unsaved work when a file is chosen or results are showing
  const hasWork = Boolean(file || result);

  const handleNavigateAttempt = useCallback((href: string) => {
    setPendingHref(href);
  }, []);

  useUnsavedGuard(hasWork, handleNavigateAttempt);

  // Returns null if valid, otherwise a short pipeline label + long modal message.
  function validateFile(f: File): { label: string; message: string } | null {
    const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
    if (!ALLOWED_EXT.includes(ext)) {
      return {
        label: "Unsupported file",
        message: `"${f.name}" is not a supported format. Please upload a JPG, JPEG, PNG, or WebP image.`,
      };
    }
    if (f.size > MAX_BYTES) {
      return {
        label: "File too large",
        message: `"${f.name}" is too large. The maximum allowed size is 10 MB.`,
      };
    }
    return null;
  }

  function handleFile(f: File): void {
    const err = validateFile(f);
    if (err) {
      // Reflect the failure inline on the pipeline: validation errored, rest skipped.
      setStageStates(buildFailedStages("validation"));
      setStageDetails({ validation: err.label });
      setInlineError({
        variant: "toast",
        title: err.label,
        message: "Please upload a valid image file.",
        icon: "red-image-icon",
      });
      return;
    }
    // A new valid file clears any previous error state and returns to the
    // fresh, ready-to-analyze state (so "Replace image" shows again).
    setStageStates({ ...INITIAL_STAGES });
    setStageDetails({});
    setInlineError(null);
    setPhase("idle");
    setFile(f);
  }

  // Server stages in the order they are reported.
  const SERVER_ORDER: StreamStageName[] = [
    "face_detection",
    "preprocessing",
    "inference",
    "explainability",
  ];

  // Turn the real stage payload into a short, meaningful label.
  function detailFor(name: StreamStageName, data?: Record<string, unknown>): string {
    const num = (k: string): number => Number(data?.[k] ?? 0);
    switch (name) {
      case "face_detection": {
        const status = data?.status as string | undefined;
        if (status === "none") return "No face found";
        if (status === "low_quality") return "Insufficient facial exposure";
        const n = Number(data?.analyzable ?? data?.faces_detected ?? 0);
        return n === 0 ? "No face found" : `${n} face${n === 1 ? "" : "s"} detected`;
      }
      case "preprocessing": {
        const n = num("faces_prepared");
        return `${n} face${n === 1 ? "" : "s"} aligned · ${num("ms")} ms`;
      }
      case "inference": {
        const n = num("faces_scored");
        return `${n} face${n === 1 ? "" : "s"} scored · ${num("ms")} ms`;
      }
      case "explainability": {
        const n = num("maps_generated");
        return n === 0 ? "No manipulated regions" : `${n} localization map${n === 1 ? "" : "s"}`;
      }
      default:
        return "Done";
    }
  }

  async function handleAnalyze(): Promise<void> {
    if (!file) return;
    setBusy(true);
    setStageStates({ ...INITIAL_STAGES });
    setStageDetails({});
    setInlineError(null);

    // Step 1 is a genuine client-side check that runs before the request.
    setPhase("validating");
    setStageStates((s) => ({ ...s, validation: "active" }));
    const validationError = validateFile(file);
    if (validationError) {
      setStageStates(buildFailedStages("validation"));
      setStageDetails({ validation: validationError.label });
      setPhase("failed");
      setBusy(false);
      setInlineError({
        variant: "toast",
        title: validationError.label,
        message: "Please upload a valid image file.",
        icon: "red-image-icon",
      });
      return;
    }
    await sleep(MIN_STEP_MS);
    const ext = file.name.split(".").pop()?.toUpperCase() ?? "image";
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    setStageDetails((d) => ({ ...d, validation: `${ext} · ${sizeKb} KB OK` }));
    setStageStates((s) => ({ ...s, validation: "done", face_detection: "active" }));

    setPhase("uploading");

    // Buffer real events, then replay them with a minimum dwell so each step
    // is actually visible. The data shown is the real backend telemetry.
    const pending: { name: StreamStageName; data?: Record<string, unknown> }[] = [];
    let streamDone = false; // true once the SSE stream has fully finished
    let draining = false; // true while the drain loop is running

    // Processes the queue one step at a time. Stays alive until the stream is
    // finished AND the queue is empty, so a late event (e.g. explainability
    // arriving after inference's dwell) is never dropped.
    const drain = async (): Promise<void> => {
      if (draining) return;
      draining = true;
      while (!streamDone || pending.length > 0) {
        const evt = pending.shift();
        if (!evt) {
          // Nothing queued yet but the stream may still send more — wait briefly.
          await sleep(30);
          continue;
        }
        const idx = SERVER_ORDER.indexOf(evt.name);
        setStageStates((prev) => ({ ...prev, [evt.name]: "active" }));
        await sleep(MIN_STEP_MS);
        setStageDetails((d) => ({ ...d, [evt.name]: detailFor(evt.name, evt.data) }));
        setStageStates((prev) => {
          const next = { ...prev, [evt.name]: "done" as StepStatus };
          const upcoming = SERVER_ORDER[idx + 1];
          if (upcoming && next[upcoming] === "waiting") next[upcoming] = "active";
          return next;
        });
      }
      draining = false;
    };

    const draining$ = drain();

    try {
      const data = await predictImageStream(file, {
        onStage: (evt) => {
          if (evt.status === "done") {
            pending.push({ name: evt.name, data: evt.data });
          }
        },
      });

      // The stream has delivered everything; let the drain finish the queue.
      streamDone = true;
      await draining$;

      const detectionStatus = data.detection?.status ?? (data.faces?.length ? "ok" : "none");
      // Only stop when there are genuinely no faces to show. A "low_quality"
      // verdict still returns analyzable faces, so we proceed to results.
      if (!data.faces || data.faces.length === 0) {
        setPhase("failed");
        // Validation passed, but detection failed: error on detection, rest skipped.
        setStageStates(buildFailedStages("face_detection", { validation: "done" }));

        if (detectionStatus === "low_quality") {
          setStageDetails({ validation: "File OK", face_detection: "Insufficient facial exposure" });
          setInlineError({
            variant: "card",
            title: "Face not clear enough",
            message:
              "A face was found, but too much of it is covered or turned away to analyze. Upload an image with an unobstructed, front-facing face.",
            icon: "no-face-icon",
          });
        } else {
          setStageDetails({ validation: "File OK", face_detection: "No face found" });
          setInlineError({
            variant: "card",
            title: "No face detected",
            message:
              "VERITAS could not find a face in this image. Upload an image where at least one face is clearly visible.",
            icon: "no-face-icon",
          });
        }
        setBusy(false);
        return;
      }

      // Hold on the completed pipeline so the final step (explainability) is
      // clearly visible as "done".
      await sleep(FINAL_HOLD_MS);

      // Show a brief loading state while we transition to the results view.
      setPhase("preparing");
      await sleep(PREPARING_MS);

      setPhase("done");
      setResult(data);
      setSelectedId(null);
    } catch (e) {
      // Let the drain loop terminate instead of spinning forever.
      streamDone = true;
      await draining$;
      setPhase("failed");
      // Error on whichever step was in progress; mark the rest skipped.
      setStageStates((prev) => {
        const failingKey =
          STEP_ORDER.find((k) => prev[k] === "active") ??
          STEP_ORDER.find((k) => prev[k] === "waiting") ??
          "inference";
        return buildFailedStages(failingKey, prev);
      });
      setInlineError({
        variant: "card",
        title: "Analysis failed",
        message:
          e instanceof Error
            ? e.message
            : "Something went wrong while analyzing the image. Please try again.",
        icon: "red-image-icon",
      });
    } finally {
      // Safety net: ensure the drain loop is never left running.
      streamDone = true;
      setBusy(false);
    }
  }

  function reset(): void {
    setFile(null);
    setResult(null);
    setSelectedId(null);
    setPhase("idle");
    setStageStates({ ...INITIAL_STAGES });
    setStageDetails({});
    setInlineError(null);
  }

  function confirmLeave(): void {
    const href = pendingHref;
    setPendingHref(null);
    reset();
    if (href) router.push(href);
  }

  return (
    <>
      {phase === "preparing" && <LoadingOverlay label="Preparing results..." />}

      {result ? (
        <AnalysisResults
          result={result}
          fileName={file?.name ?? "image.jpg"}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onReset={reset}
        />
      ) : (
        <UploadPanel
          file={file}
          busy={busy}
          phase={phase}
          stageStates={stageStates}
          stageDetails={stageDetails}
          error={null}
          inlineError={inlineError}
          onDismissError={() => setInlineError(null)}
          onFile={handleFile}
          onAnalyze={handleAnalyze}
        />
      )}

      {/* Leave / exit confirmation */}
      <ConfirmDialog
        open={pendingHref !== null}
        tone="warning"
        title="Leave this page?"
        message="You have an image in progress. If you leave now, your current analysis will be lost. Are you sure you want to exit?"
        confirmLabel="Yes, leave"
        cancelLabel="Stay"
        onConfirm={confirmLeave}
        onCancel={() => setPendingHref(null)}
      />

      {/* Errors / info notices */}
      <NoticeDialog
        open={notice.open}
        tone={notice.tone}
        title={notice.title}
        message={notice.message}
        iconSrc={notice.iconSrc}
        onClose={() => setNotice(NO_NOTICE)}
      />
    </>
  );
}

