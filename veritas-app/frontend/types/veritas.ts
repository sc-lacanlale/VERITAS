export interface FaceResult {
  face_id: number;
  bbox: [number, number, number, number];
  detector_confidence: number;
  classification: {
    label: "real" | "fake";
    fake_probability: number;
  };
  segmentation: {
    available: boolean;
    mask: string | null;
    crop_mask?: string | null;
    /** Binary thresholded mask as a standalone PNG (fake faces). */
    crop_binary?: string | null;
    /** Fraction of the face crop flagged as manipulated (0..1). */
    coverage?: number;
  };
  crop?: string;
}

export interface PredictionStages {
  face_detection: { faces_detected: number };
  preprocessing: { faces_prepared: number; ms: number };
  inference: { faces_scored: number; ms: number };
  explainability: { maps_generated: number; available: boolean };
}

export interface PredictionResponse {
  image: {
    width: number;
    height: number;
  };
  faces: FaceResult[];
  overlay_image?: string;
  detector?: string;
  stages?: PredictionStages;
  /** Face-detection quality verdict. */
  detection?: {
    status: "ok" | "none" | "low_quality";
    detected: number;
    analyzable: number;
  };
}
