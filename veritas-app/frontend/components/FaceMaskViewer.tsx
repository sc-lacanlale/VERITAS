"use client";

type Props = {
  crop?: string;
  cropMask?: string | null;
};

export function FaceMaskViewer({ crop, cropMask }: Props) {
  const src = cropMask || crop;
  if (!src) return <p className="muted">No crop available.</p>;
  return <img className="crop" src={src} alt="Selected face" />;
}
