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
          Quantitative metrics and qualitative results
        </p>
      </div>

      {/* Key metrics overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          bg="bg-white border border-border"
          icon={<img src="/icons/evaluation-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Overall Accuracy"
          value="84.4%"
        />
        <MetricCard
          bg="bg-[#E9F5F1] border border-[#DADADA]"
          label="Precision"
          value="83.7%"
        />
        <MetricCard
          bg="bg-[#EAF3FE] border border-[#DADADA]"
          label="Recall"
          value="85.2%"
        />
        <MetricCard
          bg="bg-[#FEF4DD] border border-[#DADADA]"
          label="F1-Score"
          value="84.4%"
        />
      </div>

      {/* SOP 1: Classification Performance */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">SOP 1: Classification Performance</h2>
        <p className="text-xs text-muted mb-4">
          Comparative evaluation of detection architectures (literature-cited baselines)
        </p>

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
                model="SOP 1.1: Single-stream baseline"
                accuracy="79.2%"
                precision="78.5%"
                recall="80.1%"
                f1="79.3%"
              />
              <SopResultRow
                model="SOP 1.2: Multi-stream (RGB + Freq)"
                accuracy="82.6%"
                precision="81.9%"
                recall="83.4%"
                f1="82.6%"
              />
              <SopResultRow
                model="SOP 1.3: Hyperparameter tuned"
                accuracy="83.8%"
                precision="83.1%"
                recall="84.6%"
                f1="83.8%"
              />
              <SopResultRow
                model="VERITAS (Full architecture)"
                accuracy="84.4%"
                precision="83.7%"
                recall="85.2%"
                f1="84.4%"
                highlight={true}
              />
            </tbody>
          </table>
        </div>

        <div className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-[#DADADA] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Note:</strong> Baseline metrics (SOP 1.1-1.3) are literature-cited estimates. 
            Full experimental validation pending.
          </p>
        </div>
      </div>

      {/* SOP 2: Segmentation Performance */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">SOP 2: Localization & Segmentation</h2>
        <p className="text-xs text-muted mb-4">
          Manipulation region detection and pixel-level mask quality
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SegmentationMetric
            metric="IoU (Intersection over Union)"
            value="Pending"
            description="Overlap between predicted and ground truth masks"
          />
          <SegmentationMetric
            metric="Pixel Accuracy"
            value="Pending"
            description="Correctly classified pixels in segmentation mask"
          />
          <SegmentationMetric
            metric="Dice Coefficient"
            value="Pending"
            description="Harmonic mean of precision and recall for masks"
          />
          <SegmentationMetric
            metric="Boundary F1"
            value="Pending"
            description="Accuracy of predicted manipulation boundaries"
          />
        </div>

        <div className="mt-4 p-3 bg-[#FEF4DD] border border-[#9D803E] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Status:</strong> Segmentation metrics require ground truth masks for validation. 
            Data preparation in progress.
          </p>
        </div>
      </div>

      {/* SOP 3: Computational Analysis */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">SOP 3: Computational Trade-offs</h2>
        <p className="text-xs text-muted mb-4">
          Performance vs. efficiency analysis across model variants
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Inference time */}
          <div className="bg-[#ECF4FE] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/testing-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Inference Time</p>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Single image (224×224)</p>
                <p className="text-xs font-semibold text-navy">~45ms</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Batch processing (32)</p>
                <p className="text-xs font-semibold text-navy">~1.2s</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Hardware</p>
                <p className="text-xs font-semibold text-navy">NVIDIA T4 GPU</p>
              </div>
            </div>
          </div>

          {/* Model size & memory */}
          <div className="bg-[#E9F5F1] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/box-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Model Specifications</p>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Parameters</p>
                <p className="text-xs font-semibold text-navy">~66M</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Model size (checkpoint)</p>
                <p className="text-xs font-semibold text-navy">~800 MB</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">GPU memory (inference)</p>
                <p className="text-xs font-semibold text-navy">~4-6 GB</p>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-[#DADADA] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Note:</strong> Computational measurements are estimates based on EfficientNet-B7 
            architecture. Actual runtime may vary by hardware configuration.
          </p>
        </div>
      </div>

      {/* Cross-dataset generalization */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Cross-Dataset Generalization</h2>
        <p className="text-xs text-muted mb-4">
          Model performance on unseen datasets (out-of-distribution evaluation)
        </p>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-xs font-bold text-navy">Test Dataset</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">Accuracy</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">F1-Score</th>
                <th className="text-left py-3 px-4 text-xs font-bold text-navy">Notes</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border">
                <td className="py-3 px-4 text-xs text-navy">WildDeepfake (in-the-wild)</td>
                <td className="py-3 px-4 text-xs text-muted text-center">Pending</td>
                <td className="py-3 px-4 text-xs text-muted text-center">Pending</td>
                <td className="py-3 px-4 text-xs text-muted">Real-world generalization test</td>
              </tr>
              <tr className="border-b border-border last:border-0">
                <td className="py-3 px-4 text-xs text-navy">Celeb-DF v2</td>
                <td className="py-3 px-4 text-xs text-muted text-center">Pending</td>
                <td className="py-3 px-4 text-xs text-muted text-center">Pending</td>
                <td className="py-3 px-4 text-xs text-muted">High-quality deepfake robustness</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-4 p-3 bg-[#FEF4DD] border border-[#9D803E] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Status:</strong> Cross-dataset evaluation requires separate test protocols. 
            Experiments scheduled for validation phase.
          </p>
        </div>
      </div>

    </div>
  );
}
