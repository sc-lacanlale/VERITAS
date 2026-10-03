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
  };
  crop?: string;
}

export interface PredictionResponse {
  image: {
    width: number;
    height: number;
  };
  faces: FaceResult[];
  overlay_image?: string;
  detector?: string;
}
