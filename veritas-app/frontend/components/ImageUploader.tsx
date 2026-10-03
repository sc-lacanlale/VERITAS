"use client";

type Props = {
  onFile: (file: File) => void;
  onAnalyze: () => void;
  busy: boolean;
  hasFile: boolean;
};

export function ImageUploader({ onFile, onAnalyze, busy, hasFile }: Props) {
  return (
    <div className="card">
      <p>Upload Image</p>
      <input
        type="file"
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        disabled={busy}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
      <div style={{ marginTop: 12 }}>
        <button type="button" disabled={!hasFile || busy} onClick={onAnalyze}>
          {busy ? "Analyzing image..." : "Analyze Image"}
        </button>
      </div>
    </div>
  );
}
