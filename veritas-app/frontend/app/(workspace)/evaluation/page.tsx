// ── Types ────────────────────────────────────────────────────────────────────
interface MetricCardProps {
  label: string;
  value: string;
  bg: string;
  icon?: React.ReactNode;
}

interface SopResultRowProps {
  model: string;
  accuracy: string;
  precision: string;
  recall: string;
  f1: string;
  highlight?: boolean;
}

interface SegmentationMetricProps {
  metric: string;
  value: string;
  description: string;
}

// ── Metric card ───────────────────────────────────────────────────────────────
function MetricCard({ label, value, bg, icon }: MetricCardProps): React.ReactElement {
  return (
    <div className={`${bg} rounded-xl p-4 flex flex-col gap-2`}>
      {icon && <span className="flex-shrink-0">{icon}</span>}
      <p className="text-xs font-semibold text-navy">{label}</p>
      <p className="text-2xl font-extrabold text-navy tracking-tight">{value}</p>
    </div>
  );
}

// ── SOP result row ────────────────────────────────────────────────────────────
function SopResultRow({ model, accuracy, precision, recall, f1, highlight }: SopResultRowProps): React.ReactElement {
  const rowClass = highlight 
    ? "bg-[#EAF3FE] border-l-4 border-brand font-semibold" 
    : "bg-white";
  
  return (
    <tr className={`border-b border-border last:border-0 ${rowClass}`}>
      <td className="py-3 px-4 text-xs text-navy">{model}</td>
      <td className="py-3 px-4 text-xs text-navy text-center">{accuracy}</td>
      <td className="py-3 px-4 text-xs text-navy text-center">{precision}</td>
      <td className="py-3 px-4 text-xs text-navy text-center">{recall}</td>
      <td className="py-3 px-4 text-xs text-navy text-center">{f1}</td>
    </tr>
  );
}

// ── Segmentation metric ───────────────────────────────────────────────────────
function SegmentationMetric({ metric, value, description }: SegmentationMetricProps): React.ReactElement {
  return (
    <div className="flex items-start gap-3 p-3 bg-[#F9FAFD] rounded-lg">
      <div className="flex-1">
        <p className="text-xs font-semibold text-navy">{metric}</p>
        <p className="text-[11px] text-muted mt-0.5">{description}</p>
      </div>
      <p className="text-base font-bold text-brand">{value}</p>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function EvaluationPage(): React.ReactElement {
  return (
    <div className="flex flex-col gap-6 font-inter w-full">

      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-navy m-0 tracking-[-0.002rem]">
          Evaluation
        </h1>
        <p className="text-sm text-muted mt-1">
          Test results of the two deployed models, Baseline EfficientNetB7 and VERITAS (Final), on the OpenForensics Test set (n = 10,905)
        </p>
      </div>

      {/* Key metrics overview (VERITAS) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          bg="bg-[#EAF3FE] border border-[#DADADA]"
          label="VERITAS Accuracy"
          value="89.56%"
        />
        <MetricCard
          bg="bg-white border border-border"
          label="VERITAS Precision"
          value="0.846"
        />
        <MetricCard
          bg="bg-white border border-border"
          label="VERITAS Recall"
          value="0.969"
        />
        <MetricCard
          bg="bg-white border border-border"
          label="VERITAS F1-Score"
          value="0.903"
        />
      </div>

      {/* SOP 1: Classification Performance */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">SOP 1: Classification Performance</h2>
        <p className="text-xs text-muted mb-4">
          Same OpenForensics Test set for both models; checkpoint chosen by Validation accuracy
        </p>

        <div className="border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-2.5 bg-[#F9FAFD] border-b border-border">
            <p className="text-xs font-bold text-navy">Table A · Classification metrics</p>
            <p className="text-[11px] text-muted">Accuracy, precision, recall and F1 on the 10,905 Test images</p>
          </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-xs font-bold text-navy">Model</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">Accuracy</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">Precision</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">Recall</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">F1-Score</th>
              </tr>
            </thead>
            <tbody>
              <SopResultRow
                model="Baseline EfficientNetB7"
                accuracy="86.69%"
                precision="0.804"
                recall="0.972"
                f1="0.880"
              />
              <SopResultRow
                model="VERITAS (Final)"
                accuracy="89.56%"
                precision="0.846"
                recall="0.969"
                f1="0.903"
                highlight={true}
              />
            </tbody>
          </table>
        </div>
        </div>

        <div className="border border-border rounded-lg overflow-hidden mt-5">
          <div className="px-4 py-2.5 bg-[#F9FAFD] border-b border-border">
            <p className="text-xs font-bold text-navy">Table B · Confusion matrix counts</p>
            <p className="text-[11px] text-muted">TP = Fake caught · TN = Real passed · FP = Real flagged as Fake · FN = Fake missed</p>
          </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-xs font-bold text-navy">Model</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">TP</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">TN</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">FP</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">FN</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border last:border-0 bg-white">
                <td className="py-3 px-4 text-xs text-navy">Baseline EfficientNetB7</td>
                <td className="py-3 px-4 text-xs text-navy text-center">5,340</td>
                <td className="py-3 px-4 text-xs text-navy text-center">4,113</td>
                <td className="py-3 px-4 text-xs text-navy text-center">1,300</td>
                <td className="py-3 px-4 text-xs text-navy text-center">152</td>
              </tr>
              <tr className="border-b border-border last:border-0 bg-[#EAF3FE] border-l-4 border-brand font-semibold">
                <td className="py-3 px-4 text-xs text-navy">VERITAS (Final)</td>
                <td className="py-3 px-4 text-xs text-navy text-center">5,321</td>
                <td className="py-3 px-4 text-xs text-navy text-center">4,446</td>
                <td className="py-3 px-4 text-xs text-navy text-center">967</td>
                <td className="py-3 px-4 text-xs text-navy text-center">171</td>
              </tr>
            </tbody>
          </table>
        </div>
        </div>

        <div className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-[#DADADA] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Note:</strong> VERITAS (Final) is 2.87 percentage points more accurate than
            Baseline EfficientNetB7 on the same 10,905 Test images (5,413 Real, 5,492 Fake), ROC-AUC 0.9733. Fake is the positive class.
          </p>
        </div>
      </div>

      {/* SOP 2: Segmentation Performance */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">SOP 2: Localization & Segmentation</h2>
        <p className="text-xs text-muted mb-4">
          VERITAS masks compared with the generated masks (Baseline EfficientNetB7 has no segmentation head)
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <SegmentationMetric
            metric="Mean Dice (Fake)"
            value="0.9464"
            description="Agreement with generated masks on 5,492 Fake Test images (median 0.9763)"
          />
          <SegmentationMetric
            metric="Mean mIoU (Fake)"
            value="0.9225"
            description="Intersection over union on the same 5,492 Fake Test images (median 0.9537)"
          />
          <SegmentationMetric
            metric="Real: positive-pixel fraction"
            value="9.63%"
            description="Average share of a Real face's pixels predicted as manipulated"
          />
          <SegmentationMetric
            metric="Real: nonempty-mask rate"
            value="21.76%"
            description="Share of Real Test images with any predicted positive pixel"
          />
        </div>

        <div className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-[#DADADA] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Note:</strong> Dice and mIoU measure agreement with pre-generated masks,
            not hand-annotated manipulation boundaries. Letterbox padding is excluded.
          </p>
        </div>
      </div>

      {/* SOP 3: Computational Analysis */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">SOP 3: Computational Trade-offs</h2>
        <p className="text-xs text-muted mb-4">
          Descriptive model-only benchmark on a Kaggle Tesla T4 (batch size 4; Baseline FP32, VERITAS (Final) mixed precision FP16)
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-[#ECF4FE] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/testing-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Inference Latency (per image)</p>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Baseline EfficientNetB7</p>
                <p className="text-xs font-semibold text-navy text-right">63.29 ms</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">VERITAS</p>
                <p className="text-xs font-semibold text-navy text-right">42.73 ms</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Difference</p>
                <p className="text-xs font-semibold text-navy text-right">−20.56 ms</p>
              </div>
            </div>
          </div>

          <div className="bg-[#E9F5F1] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/box-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Peak Allocated VRAM</p>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Baseline EfficientNetB7</p>
                <p className="text-xs font-semibold text-navy text-right">859 MB</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">VERITAS</p>
                <p className="text-xs font-semibold text-navy text-right">936 MB</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Difference</p>
                <p className="text-xs font-semibold text-navy text-right">+77 MB</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-[#DADADA] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Note:</strong> Per-image time is the batch-of-4 latency divided by 4. The latency
            difference reflects mixed-precision FP16 (VERITAS) versus FP32 (Baseline) execution. At batch 1,
            VERITAS (Final) recorded 54.47 ms and 535 MB. It excludes preprocessing, so it is not the app's end-to-end time.
          </p>
        </div>
      </div>

    </div>
  );
}
