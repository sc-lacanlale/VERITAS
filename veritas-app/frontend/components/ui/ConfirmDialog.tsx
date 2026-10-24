"use client";

type DialogTone = "warning" | "danger" | "info";

interface ConfirmDialogProps {
  open: boolean;
  tone?: DialogTone;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

const TONE_MAP: Record<DialogTone, { ring: string; iconBg: string; stroke: string; confirmBg: string; confirmHover: string }> = {
  warning: {
    ring: "#FEF4DD",
    iconBg: "#FEF4DD",
    stroke: "#9D803E",
    confirmBg: "#3333FF",
    confirmHover: "#2222cc",
  },
  danger: {
    ring: "#FFE9E9",
    iconBg: "#FFE9E9",
    stroke: "#FF3131",
    confirmBg: "#FF3131",
    confirmHover: "#e02020",
  },
  info: {
    ring: "#EAF3FE",
    iconBg: "#EAF3FE",
    stroke: "#2B5CAE",
    confirmBg: "#3333FF",
    confirmHover: "#2222cc",
  },
};

export function ConfirmDialog({
  open,
  tone = "warning",
  title,
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  onConfirm,
  onCancel,
}: ConfirmDialogProps): React.ReactElement | null {
  if (!open) return null;
  const t = TONE_MAP[tone];

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4 font-inter">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onCancel} />

      {/* Dialog */}
      <div className="relative bg-white rounded-2xl w-full max-w-[420px] shadow-xl p-6 flex flex-col items-center text-center gap-4">
        <div
          className="w-14 h-14 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: t.iconBg }}
        >
          <img src="/icons/exit-icon.svg" alt="" width={28} height={28} className="object-contain" />
        </div>

        <h2 className="text-lg font-bold text-navy">{title}</h2>
        <p className="text-sm text-muted leading-relaxed">{message}</p>

        <div className="flex flex-col-reverse sm:flex-row gap-3 w-full mt-2">
          <button
            onClick={onCancel}
            className="flex-1 px-5 py-2.5 rounded-lg font-semibold text-sm border border-border text-navy bg-white hover:bg-[#F4F6FF] transition-colors"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 px-5 py-2.5 rounded-lg font-semibold text-sm text-white transition-colors"
            style={{ backgroundColor: t.confirmBg }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = t.confirmHover)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = t.confirmBg)}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
