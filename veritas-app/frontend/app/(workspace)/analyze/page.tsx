"use client";

import { useMemo, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { predictImage } from "@/lib/api";
import type { PredictionResponse } from "@/types/veritas";
import { UploadPanel } from "@/components/analyze/UploadPanel";
import { AnalysisResults } from "@/components/analyze/AnalysisResults";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import { NoticeDialog } from "@/components/ui/NoticeDialog";
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

export default function AnalyzePage(): React.ReactElement {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const [notice, setNotice] = useState<NoticeState>(NO_NOTICE);
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

  function validateFile(f: File): string | null {
    const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
    if (!ALLOWED_EXT.includes(ext)) {
      return `"${f.name}" is not a supported format. Please upload a JPG, JPEG, PNG, or WebP image.`;
    }
    if (f.size > MAX_BYTES) {
      return `"${f.name}" is too large. The maximum allowed size is 10 MB.`;
    }
    return null;
  }

  function handleFile(f: File): void {
    const err = validateFile(f);
    if (err) {
      setNotice({ open: true, tone: "error", title: "Unsupported file", message: err });
      return;
    }
    setFile(f);
  }

  async function handleAnalyze(): Promise<void> {
    if (!file) return;
    setBusy(true);
    try {
      const data = await predictImage(file);
      if (!data.faces || data.faces.length === 0) {
        setNotice({
          open: true,
          tone: "info",
          title: "No faces detected",
          message:
            "We could not detect any faces in this image. Try uploading a clearer photo with at least one visible face.",
          iconSrc: "/icons/no-face-icon.svg",
        });
        setBusy(false);
        return;
      }
      setResult(data);
      setSelectedId(null);
    } catch (e) {
      setNotice({
        open: true,
        tone: "error",
        title: "Analysis failed",
        message:
          e instanceof Error
            ? e.message
            : "Something went wrong while analyzing the image. Please try again.",
      });
    } finally {
      setBusy(false);
    }
  }

  function reset(): void {
    setFile(null);
    setResult(null);
    setSelectedId(null);
  }

  function confirmLeave(): void {
    const href = pendingHref;
    setPendingHref(null);
    reset();
    if (href) router.push(href);
  }

  return (
    <>
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
          error={null}
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
