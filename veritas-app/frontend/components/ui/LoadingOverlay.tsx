"use client";

interface LoadingOverlayProps {
  label?: string;
  /** "overlay" covers the viewport; "inline" fills its parent area */
  variant?: "overlay" | "inline";
}

export function LoadingOverlay({
  label = "Loading...",
  variant = "overlay",
}: LoadingOverlayProps): React.ReactElement {
  const wrapper =
    variant === "overlay"
      ? "fixed top-[56px] right-0 bottom-0 left-0 lg:top-0 lg:left-[220px] z-[300] flex items-center justify-center bg-[#F9FAFD]/80 backdrop-blur-sm"
      : "flex items-center justify-center w-full py-24";

  return (
    <div className={`${wrapper} font-inter`}>
      <div className="flex flex-col items-center gap-4">
        <div className="relative w-12 h-12">
          <div className="absolute inset-0 rounded-full border-4 border-[#E5E7EB]" />
          <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-brand animate-spin" />
        </div>
        <p className="text-sm font-medium text-muted">{label}</p>
      </div>
    </div>
  );
}
