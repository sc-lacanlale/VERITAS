"use client";

type NoticeTone = "error" | "info" | "success";

interface NoticeDialogProps {
  open: boolean;
  tone?: NoticeTone;
  title: string;
  message: string;
  buttonLabel?: string;
  iconSrc?: string; // optional custom icon (overrides the default tone icon)
  onClose: () => void;
}

const TONE_MAP: Record<NoticeTone, { iconBg: string; stroke: string; icon: "x" | "info" | "check" }> = {
  error:   { iconBg: "#FFE9E9", stroke: "#FF3131", icon: "x" },
  info:    { iconBg: "#EAF3FE", stroke: "#2B5CAE", icon: "info" },
  success: { iconBg: "#E9F5F1", stroke: "#2C8833", icon: "check" },
};

function Icon({ kind, stroke }: { kind: "x" | "info" | "check"; stroke: string }): React.ReactElement {
  if (kind === "x") {
    return (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="2.2">
        <circle cx="12" cy="12" r="10" />
        <line x1="15" y1="9" x2="9" y2="15" />
        <line x1="9" y1="9" x2="15" y2="15" />
      </svg>
    );
  }
  if (kind === "check") {
    return (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="2.5">
        <polyline points="20 6 9 17 4 12" />
      </svg>
    );
  }
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth="2.2">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

export function NoticeDialog({
  open,
  tone = "info",
  title,
  message,
  buttonLabel = "Got it",
  iconSrc,
  onClose,
}: NoticeDialogProps): React.ReactElement | null {
  if (!open) return null;
  const t = TONE_MAP[tone];

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 font-inter">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative bg-white rounded-2xl w-full max-w-[420px] shadow-xl p-6 flex flex-col items-center text-center gap-4">
        <div
          className="w-14 h-14 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: t.iconBg }}
        >
          {iconSrc ? (
            <img src={iconSrc} alt="" width={28} height={28} className="object-contain" />
          ) : (
            <Icon kind={t.icon} stroke={t.stroke} />
          )}
        </div>
        <h2 className="text-lg font-bold text-navy">{title}</h2>
        <p className="text-sm text-muted leading-relaxed">{message}</p>
        <button
          onClick={onClose}
          className="w-full px-5 py-2.5 rounded-lg font-semibold text-sm text-white bg-brand hover:bg-[#2222cc] transition-colors mt-2"
        >
          {buttonLabel}
        </button>
      </div>
    </div>
  );
}
