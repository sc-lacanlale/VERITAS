"use client";

import type { PredictionResponse } from "@/types/veritas";

interface PrintReportProps {
  result: PredictionResponse;
  fileName: string;
}

/**
 * Print-only forensic report. Hidden on screen (see print.css),
 * only visible when the browser print dialog runs.
 */
export function PrintReport({ result, fileName }: PrintReportProps): React.ReactElement {
  const timestamp = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";
  const fakeCount = result.faces.filter((f) => f.classification.label === "fake").length;
  const firstFace = result.faces[0];

  return (
    <div id="print-report" className="print-report">
      {/* Header band */}
      <div className="pr-header">
        <img src="/icons/veritas-logo-vector.svg" alt="VERITAS" className="pr-logo-img" />
        <div className="pr-header-right">
          FACIAL MANIPULATION<br />DETECTION &amp; LOCALIZATION
        </div>
      </div>

      <div className="pr-body">
      {/* Title */}
      <p className="pr-eyebrow">FORENSIC ANALYSIS / SUMMARY REPORT</p>
      <h1 className="pr-title">Image Analysis Report</h1>
      <p className="pr-subtitle">Face-level detection results and analysis metadata</p>

      <div className="pr-meta-row">
        <span><strong>Report ID</strong> &nbsp; VRT-{Date.now().toString().slice(-8)}</span>
        <span><strong>Analysis timestamp</strong> &nbsp; {timestamp}</span>
      </div>

      <hr className="pr-rule" />

      {/* 01 Metadata */}
      <h2 className="pr-section"><span className="pr-num">01</span> CASE &amp; IMAGE METADATA</h2>
      <div className="pr-grid">
        <div>
          <p className="pr-label">SOURCE FILE</p>
          <p className="pr-value">{fileName}</p>
        </div>
        <div>
          <p className="pr-label">IMAGE DIMENSIONS</p>
          <p className="pr-value">{result.image.width} × {result.image.height} px</p>
        </div>
        <div>
          <p className="pr-label">FACES DETECTED</p>
          <p className="pr-value">{result.faces.length}</p>
        </div>
        <div>
          <p className="pr-label">FACE DETECTOR</p>
          <p className="pr-value">{result.detector ?? "—"}</p>
        </div>
      </div>

      <hr className="pr-rule" />

      {/* 02 Summary */}
      <h2 className="pr-section"><span className="pr-num">02</span> ANALYSIS SUMMARY</h2>
      <div className="pr-callout">
        <p className="pr-callout-title">
          {fakeCount > 0
            ? `${fakeCount} face${fakeCount > 1 ? "s" : ""} classified as manipulated.`
            : "No faces classified as manipulated."}
        </p>
        {firstFace && (
          <p className="pr-callout-body">
            Face {firstFace.face_id} classified as {firstFace.classification.label.toUpperCase()},
            with a model score of {firstFace.classification.fake_probability.toFixed(3)}.
          </p>
        )}
      </div>

      <hr className="pr-rule" />

      {/* 03 Results */}
      <h2 className="pr-section"><span className="pr-num">03</span> FACE-LEVEL RESULTS</h2>
      <table className="pr-table">
        <thead>
          <tr>
            <th>FACE ID</th>
            <th>FACE CROP</th>
            <th>MODEL SCORE</th>
            <th>CLASSIFICATION</th>
            <th>MASK STATUS</th>
          </tr>
        </thead>
        <tbody>
          {result.faces.map((face) => {
            const isFake = face.classification.label === "fake";
            return (
              <tr key={face.face_id}>
                <td className="pr-face-id">Face {face.face_id}</td>
                <td>{face.crop && <img src={face.crop} alt="" className="pr-crop" />}</td>
                <td>{face.classification.fake_probability.toFixed(3)}</td>
                <td className={isFake ? "pr-fake" : "pr-real"}>{isFake ? "FAKE" : "REAL"}</td>
                <td>{isFake ? "Manipulated" : "Zero Mask"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="pr-note">
        Model score is shown as reported; it is not presented as a calibrated confidence percentage.
      </p>

      <hr className="pr-rule" />

      {/* 04 Review */}
      <h2 className="pr-section"><span className="pr-num">04</span> REVIEW &amp; VERIFICATION</h2>
      <p className="pr-review-line">Reviewer notes: [Enter observations or verification details.]</p>
      <div className="pr-sign-row">
        <div>
          <div className="pr-sign-line" />
          <p className="pr-sign-label"><strong>Reviewed by</strong></p>
          <p className="pr-sign-sub">Name and designation</p>
        </div>
        <div>
          <div className="pr-sign-line" />
          <p className="pr-sign-label"><strong>Signature / Date</strong></p>
          <p className="pr-sign-sub">Complete after review</p>
        </div>
      </div>
      </div>
    </div>
  );
}
