"use client";

import { useRef } from "react";
import type { BaselineStatus } from "@/app/(workspace)/analyze/page";
import type { ModelKey } from "@/types/veritas";

interface ModelSelectorProps {
  value: ModelKey;
  onChange: (model: ModelKey) => void;
  baseline: BaselineStatus;
  /** Locks the selector while an analysis is running. */
  disabled?: boolean;
}

interface ModelOption {
  key: ModelKey;
  title: string;
  subtitle: string;
  streams: string[];
  outputs: { label: string; on: boolean }[];
}

const OPTIONS: ModelOption[] = [
  {
    key: "veritas",
    title: "Full VERITAS",
    subtitle: "Detects and localizes the manipulated region",
    streams: ["RGB", "SRM", "FFT"],
    outputs: [
      { label: "Verdict", on: true },
      { label: "Score", on: true },
      { label: "Mask", on: true },
    ],
  },
  {
    key: "baseline",
    title: "Baseline B7",
    subtitle: "Classifies each face as real or fake",
    streams: ["RGB"],
    outputs: [
      { label: "Verdict", on: true },
      { label: "Score", on: true },
      { label: "Mask", on: false },
    ],
  },
];

/** Small picture of what the model returns: a face with a score bar, plus a mask for VERITAS. */
function OutputGlyph({ kind, muted }: { kind: ModelKey; muted: boolean }): React.ReactElement {
  const face = muted ? "#A5AAB5" : "#142849";
  const accent = muted ? "#C9CDD5" : kind === "veritas" ? "#FF3131" : "#3333FF";
  return (
    <svg width="56" height="56" viewBox="0 0 56 56" fill="none" aria-hidden="true" className="model-glyph">
      {/* Face crop frame */}
      <rect x="6" y="4" width="44" height="38" rx="8" fill="#FFFFFF" stroke={muted ? "#DADADA" : "#C9D3FB"} />
      {/* Face */}
      <ellipse cx="28" cy="23" rx="11" ry="13" stroke={face} strokeWidth="1.8" />
      <circle cx="24" cy="21" r="1.4" fill={face} />
      <circle cx="32" cy="21" r="1.4" fill={face} />
      <path d="M24.5 28.5c2 1.6 5 1.6 7 0" stroke={face} strokeWidth="1.6" strokeLinecap="round" />
      {kind === "veritas" ? (
        <>
          {/* Localized region */}
          <path d="M30 15c4-1 7 2 7 6s-2 6-5 6-5-2-5-5 0-6 3-7z" fill={accent} fillOpacity="0.35" stroke={accent} strokeWidth="1.2" strokeDasharray="2 1.5" />
          {/* Corner brackets = localization */}
          <path d="M9 12V8h4M47 12V8h-4M9 34v4h4M47 34v4h-4" stroke={accent} strokeWidth="1.4" strokeLinecap="round" />
        </>
      ) : null}
      {/* Score bar */}
      <rect x="6" y="46" width="44" height="5" rx="2.5" fill="#E5E7EB" />
      <rect x="6" y="46" width={kind === "veritas" ? 34 : 28} height="5" rx="2.5" fill={accent} />
    </svg>
  );
}

export function ModelSelector({ value, onChange, baseline, disabled = false }: ModelSelectorProps): React.ReactElement {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const isUnavailable = (key: ModelKey): boolean => key === "baseline" && baseline.available !== true;

  // Arrow keys move between the options, like a native radio group.
  function handleKey(e: React.KeyboardEvent, index: number): void {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    const step = e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1;
    for (let i = 1; i <= OPTIONS.length; i++) {
      const next = (index + step * i + OPTIONS.length) % OPTIONS.length;
      if (!isUnavailable(OPTIONS[next].key)) {
        onChange(OPTIONS[next].key);
        refs.current[next]?.focus();
        return;
      }
    }
  }

  // While analyzing, the choice is locked: show one compact line instead of both cards.
  if (disabled) {
    const opt = OPTIONS.find((o) => o.key === value) ?? OPTIONS[0];
    return (
      <div className="mt-5 flex items-center gap-3 rounded-xl border border-border bg-[#FAFBFD] px-3 py-2.5">
        <span className="w-2 h-2 rounded-full bg-brand flex-shrink-0" />
        <span className="text-[11px] uppercase tracking-wide text-muted font-semibold">Model</span>
        <span className="text-sm font-bold text-navy">{opt.title}</span>
        <span className="hidden sm:flex items-center gap-1">
          {opt.streams.map((s) => (
            <span key={s} className="text-[10px] font-semibold px-1.5 py-0.5 rounded border border-[#C9D3FB] text-navy bg-white">
              {s}
            </span>
          ))}
        </span>
        <svg className="ml-auto flex-shrink-0 text-muted" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-label="Locked during analysis">
          <rect x="5" y="11" width="14" height="10" rx="2" />
          <path d="M8 11V7a4 4 0 0 1 8 0v4" />
        </svg>
      </div>
    );
  }

  return (
    <div className="mt-5">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-1 sm:gap-3 mb-3">
        <div>
          <h3 className="text-sm font-bold text-navy">Choose a model</h3>
          <p className="text-[11px] text-muted">Pick which trained model analyzes this image.</p>
        </div>
        <span className="hidden sm:inline text-[11px] text-muted">You can switch and re-run after results</span>
      </div>

      <div role="radiogroup" aria-label="Model" className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {OPTIONS.map((opt, i) => {
          const active = value === opt.key;
          const unavailable = isUnavailable(opt.key);
          const locked = disabled || unavailable;
          return (
            <button
              key={opt.key}
              ref={(el) => {
                refs.current[i] = el;
              }}
              type="button"
              role="radio"
              aria-checked={active}
              aria-disabled={locked}
              tabIndex={active ? 0 : -1}
              onClick={() => !locked && onChange(opt.key)}
              onKeyDown={(e) => !disabled && handleKey(e, i)}
              className={`group relative flex flex-col gap-3 p-3 sm:p-4 rounded-xl border text-left transition-all duration-200 w-full outline-none
                focus-visible:ring-2 focus-visible:ring-brand/40
                ${active
                  ? "border-brand bg-gradient-to-br from-[#F4F6FF] to-white shadow-[0_6px_18px_-10px_rgba(51,51,255,0.55)]"
                  : "border-border bg-[#FAFBFD]"}
                ${locked
                  ? active ? "cursor-default" : "cursor-not-allowed"
                  : active ? "" : "hover:border-[#C9D3FB] hover:bg-[#F4F6FF] hover:-translate-y-0.5"}`}
            >
              {/* Selected check */}
              <span
                className={`absolute top-3 right-3 w-5 h-5 rounded-full flex items-center justify-center transition-all duration-200
                  ${active ? "bg-brand scale-100" : "border-2 border-[#C9CDD5] bg-white scale-90"}`}
              >
                {active && (
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" strokeWidth="3.5">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}
              </span>

              <div className="flex flex-wrap items-start gap-3 pr-6">
                <div
                  className={`w-16 h-16 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors
                    ${unavailable ? "bg-[#F1F2F5]" : active ? "bg-[#EAF0FE]" : "bg-white border border-border"}`}
                >
                  <OutputGlyph kind={opt.key} muted={unavailable} />
                </div>
                {/* Inline min-width: workspace.css resets min-width to 0 on mobile */}
                <div className="flex-1" style={{ minWidth: 130 }}>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`text-sm font-bold whitespace-nowrap ${unavailable ? "text-[#A5AAB5]" : "text-navy"}`}>{opt.title}</span>
                    {opt.key === "veritas" && (
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-[#EAF0FE] text-brand whitespace-nowrap">Default</span>
                    )}
                  </div>
                  <p style={{ wordBreak: "normal" }} className={`text-[11px] leading-snug mt-0.5 ${unavailable ? "text-[#A5AAB5]" : "text-muted"}`}>
                    {opt.subtitle}
                  </p>
                  {/* Input streams */}
                  <div className="flex flex-wrap items-center gap-1 mt-1.5">
                    {opt.streams.map((s) => (
                      <span
                        key={s}
                        className={`text-[10px] font-semibold px-1.5 py-0.5 rounded border whitespace-nowrap flex-shrink-0
                          ${unavailable ? "border-border text-[#A5AAB5]" : "border-[#C9D3FB] text-navy bg-white"}`}
                      >
                        {s}
                      </span>
                    ))}
                    <span className="text-[10px] text-muted ml-0.5 whitespace-nowrap">input</span>
                  </div>
                </div>
              </div>

              {/* What you get */}
              <div className={`flex flex-wrap items-center gap-x-3 gap-y-1.5 pt-3 border-t ${active ? "border-[#DCE3FD]" : "border-border"}`}>
                <span className="text-[10px] uppercase tracking-wide text-muted font-semibold whitespace-nowrap">Output</span>
                {opt.outputs.map((o) => (
                  <span
                    key={o.label}
                    className={`flex items-center gap-1 text-[11px] font-medium whitespace-nowrap
                      ${unavailable ? "text-[#A5AAB5]" : o.on ? "text-navy" : "text-[#A5AAB5] line-through"}`}
                  >
                    {o.on ? (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={unavailable ? "#A5AAB5" : "#2C8833"} strokeWidth="3">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    ) : (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#A5AAB5" strokeWidth="3">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    )}
                    {o.label}
                  </span>
                ))}
              </div>

              {/* Baseline checkpoint state */}
              {opt.key === "baseline" && (
                <div
                  style={{ wordBreak: "normal" }}
                  className={`flex items-start gap-1.5 text-[11px] leading-snug -mt-1 ${
                    baseline.available === true ? "text-muted" : "text-[#9D803E]"
                  }`}
                >
                  {baseline.available === null ? (
                    <>
                      <span className="w-3 h-3 rounded-full border-2 border-[#E5E7EB] border-t-[#9D803E] animate-spin" />
                      Checking for checkpoint...
                    </>
                  ) : baseline.available ? (
                    <>
                      <img src="/icons/file-icon.svg" alt="" width={12} height={12} className="object-contain opacity-70" />
                      <span className="truncate">{baseline.checkpoint ?? "Checkpoint found"}</span>
                    </>
                  ) : (
                    <>
                      <svg className="flex-shrink-0 mt-0.5" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <rect x="5" y="11" width="14" height="10" rx="2" />
                        <path d="M8 11V7a4 4 0 0 1 8 0v4" />
                      </svg>
                      <span>Add a baseline checkpoint to models/ to enable</span>
                    </>
                  )}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
