"use client";

import { useEffect, useState } from "react";
import { fetchHealth, type HealthInfo } from "@/lib/api";

type Tone = "good" | "info" | "warn" | "muted";

const TONES: Record<Tone, { bg: string; text: string; dot: string }> = {
  good: { bg: "bg-[#E9F5F1]", text: "text-positive", dot: "bg-positive" },
  info: { bg: "bg-[#EAF0FE]", text: "text-brand", dot: "bg-brand" },
  warn: { bg: "bg-[#FEF4DD]", text: "text-warning", dot: "bg-warning" },
  muted: { bg: "bg-[#F1F2F5]", text: "text-muted", dot: "bg-[#A5AAB5]" },
};

function StatusPill({ tone, label }: { tone: Tone; label: string }): React.ReactElement {
  const t = TONES[tone];
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] font-semibold px-2 py-0.5 rounded-full whitespace-nowrap ${t.bg} ${t.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${t.dot}`} />
      {label}
    </span>
  );
}

/** Overview card listing both trained models and their live status from the backend. */
export function TrainedModelsCard(): React.ReactElement {
  // undefined = still checking, null = backend unreachable
  const [health, setHealth] = useState<HealthInfo | null | undefined>(undefined);

  useEffect(() => {
    let alive = true;
    fetchHealth().then((h) => alive && setHealth(h));
    return () => {
      alive = false;
    };
  }, []);

  const checking = health === undefined;
  const offline = health === null;

  const veritas: { tone: Tone; label: string } = checking
    ? { tone: "muted", label: "Checking..." }
    : offline
      ? { tone: "warn", label: "Backend offline" }
      : health.ok
        ? { tone: "good", label: "Ready" }
        : { tone: "warn", label: "Not configured" };

  const baseline: { tone: Tone; label: string } = checking
    ? { tone: "muted", label: "Checking..." }
    : offline
      ? { tone: "warn", label: "Backend offline" }
      : !health.baseline?.available
        ? { tone: "muted", label: "Not found" }
        : health.baseline.loaded
          ? { tone: "good", label: "Ready" }
          : { tone: "info", label: "Available" };

  return (
    <div className="bg-white border border-border rounded-xl p-6 flex-1 min-w-[220px] w-full">
      <h2 className="text-base font-bold text-navy mb-1">Trained Models</h2>
      <p className="text-xs text-muted mb-4">Models available for image analysis</p>

      <div className="flex flex-col gap-3">
        {/* Full VERITAS */}
        <div className="rounded-xl border border-[#C9D3FB] bg-[#F4F6FF] p-4 flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-navy whitespace-nowrap">Full VERITAS</span>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-[#EAF0FE] text-brand">Default</span>
            </div>
            <StatusPill {...veritas} />
          </div>
          <p className="text-[11px] text-muted leading-snug">Multi-stream EfficientNet-B7 + DeepLabV3+ · score and localization mask</p>
          {health?.checkpoint && (
            <p className="text-[11px] text-muted flex items-center gap-1.5">
              <img src="/icons/file-icon.svg" alt="" width={12} height={12} className="object-contain opacity-70" />
              <span className="truncate">{health.checkpoint}</span>
            </p>
          )}
        </div>

        {/* Baseline */}
        <div className="rounded-xl border border-border bg-[#FAFBFD] p-4 flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <span className="text-sm font-bold text-navy whitespace-nowrap">Baseline B7</span>
            <StatusPill {...baseline} />
          </div>
          <p className="text-[11px] text-muted leading-snug">RGB-only EfficientNet-B7 · score only, no mask</p>
          {health?.baseline?.checkpoint && (
            <p className="text-[11px] text-muted flex items-center gap-1.5">
              <img src="/icons/file-icon.svg" alt="" width={12} height={12} className="object-contain opacity-70" />
              <span className="truncate">{health.baseline.checkpoint}</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
