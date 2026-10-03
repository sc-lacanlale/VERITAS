import type { PredictionResponse } from "@/types/veritas";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

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
