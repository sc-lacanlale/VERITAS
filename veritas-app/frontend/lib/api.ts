import type { PredictionResponse } from "@/types/veritas";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export type HealthInfo = {
  ok: boolean;
  checkpoint?: string;
  checkpoint_kind?: string;
  classification_threshold?: number;
  emit_masks?: boolean;
};

export async function fetchHealth(): Promise<HealthInfo | null> {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", file);
  const response = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    let message = "Prediction failed";
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      /* keep default */
    }
    throw new Error(message);
  }
  return response.json();
}

/** Backend stage names emitted over the SSE stream, in order. */
export type StreamStageName =
  | "face_detection"
  | "preprocessing"
  | "inference"
  | "explainability";

export interface StreamStageEvent {
  name: StreamStageName;
  status: "active" | "done";
  data?: Record<string, unknown>;
}

export interface PredictStreamHandlers {
  onStage?: (event: StreamStageEvent) => void;
}

/**
 * Streaming variant of predictImage using Server-Sent Events.
 * Invokes onStage as each real pipeline stage progresses, and resolves with
 * the final full prediction payload. Falls back cleanly to an error if the
 * stream reports one. The non-streaming predictImage remains available.
 */
export async function predictImageStream(
  file: File,
  handlers: PredictStreamHandlers = {}
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${BACKEND_URL}/predict/stream`, {
    method: "POST",
    body: formData,
    headers: { Accept: "text/event-stream" },
  });

  if (!response.ok || !response.body) {
    let message = "Prediction failed";
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      /* keep default */
    }
    throw new Error(message);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: PredictionResponse | null = null;
  let streamError: string | null = null;

  // Parse one SSE record ("event:" + "data:" lines separated by a blank line).
  const handleRecord = (raw: string): void => {
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of raw.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(dataLines.join("\n"));
    } catch {
      return;
    }
    if (eventName === "stage") {
      handlers.onStage?.(parsed as StreamStageEvent);
    } else if (eventName === "result") {
      result = parsed as PredictionResponse;
    } else if (eventName === "error") {
      streamError = (parsed as { detail?: string }).detail || "Inference failed";
    }
  };

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const record = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      if (record.trim()) handleRecord(record);
    }
  }
  if (buffer.trim()) handleRecord(buffer);

  if (streamError) throw new Error(streamError);
  if (!result) throw new Error("Prediction failed");
  return result;
}
