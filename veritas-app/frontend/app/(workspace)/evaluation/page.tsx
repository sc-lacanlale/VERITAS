interface ClassificationResult {
  model: string;
  accuracy: string;
  precision: string;
  recall: string;
  f1: string;
  deltaAccuracy: string;
  tp: string;
  tn: string;
  fp: string;
  fn: string;
  highlight?: boolean;
}

interface InferenceResult {
  model: string;
  images: string;
  latency: string;
  deltaLatency: string;
  vram: string;
  highlight?: boolean;
}

const CLASSIFICATION_RESULTS: ClassificationResult[] = [
  {
    model: "Baseline EfficientNetB7",
    accuracy: "86.80%",
    precision: "0.815",
    recall: "0.955",
    f1: "0.879",
    deltaAccuracy: "+0.00 pp",
    tp: "5,244",
    tn: "4,221",
    fp: "1,192",
    fn: "248",
  },
  {
    model: "Multi-stream (no segmentation)",
    accuracy: "86.82%",
    precision: "0.806",
    recall: "0.972",
    f1: "0.881",
    deltaAccuracy: "+0.03 pp",
    tp: "5,340",
    tn: "4,128",
    fp: "1,285",
    fn: "152",
  },
  {
    model: "Segmentation (no multi-stream)",
    accuracy: "86.88%",
    precision: "0.806",
    recall: "0.973",
    f1: "0.882",
    deltaAccuracy: "+0.08 pp",
    tp: "5,345",
    tn: "4,129",
    fp: "1,284",
    fn: "147",
  },
  {
    model: "VERITAS",
    accuracy: "87.55%",
    precision: "0.816",
    recall: "0.972",
    f1: "0.887",
    deltaAccuracy: "+0.75 pp",
    tp: "5,336",
    tn: "4,211",
    fp: "1,202",
    fn: "156",
    highlight: true,
  },
];

const SEGMENTATION_RESULTS = [
  { model: "Baseline EfficientNetB7", dice: "—", miou: "—" },
  { model: "Multi-stream (no segmentation)", dice: "—", miou: "—" },
  { model: "Segmentation (no multi-stream)", dice: "0.9780", miou: "0.9570" },
  { model: "VERITAS", dice: "0.9780", miou: "0.9570", highlight: true },
];

const INFERENCE_RESULTS: InferenceResult[] = [
  {
    model: "Baseline EfficientNetB7",
    images: "4,000",
    latency: "63.29 ms/image",
    deltaLatency: "+0.00 ms",
    vram: "859 MB",
  },
  {
    model: "Multi-stream (no segmentation)",
    images: "4,000",
    latency: "68.99 ms/image",
    deltaLatency: "+5.71 ms",
    vram: "916 MB",
  },
  {
    model: "Segmentation (no multi-stream)",
    images: "4,000",
    latency: "70.79 ms/image",
    deltaLatency: "+7.50 ms",
    vram: "862 MB",
  },
  {
    model: "VERITAS",
    images: "4,000",
    latency: "79.69 ms/image",
    deltaLatency: "+16.41 ms",
    vram: "919 MB",
    highlight: true,
  },
];

const MCNEMAR_RESULTS = [
  { comparison: "Baseline EfficientNetB7 vs VERITAS", discordant: "464 (191 / 273)", pValue: "1.637e-04", result: "Significant; favors VERITAS" },
  { comparison: "Baseline EfficientNetB7 vs Multi-stream (no segmentation)", discordant: "467 (232 / 235)", pValue: "0.926", result: "Not significant" },
  { comparison: "Baseline EfficientNetB7 vs Segmentation (no multi-stream)", discordant: "469 (230 / 239)", pValue: "0.712", result: "Not significant" },
  { comparison: "Multi-stream vs VERITAS", discordant: "271 (96 / 175)", pValue: "1.839e-06", result: "Significant; favors VERITAS" },
  { comparison: "Segmentation vs VERITAS", discordant: "263 (95 / 168)", pValue: "7.941e-06", result: "Significant; favors VERITAS" },
];

function SectionTitle({ title, description }: { title: string; description?: string }): React.ReactElement {
  return (
    <div className="mb-4">
      <h2 className="text-base font-bold text-navy mb-1">{title}</h2>
      {description && <p className="text-xs text-muted">{description}</p>}
    </div>
  );
}

function MetricCard({ label, value, featured = false }: { label: string; value: string; featured?: boolean }): React.ReactElement {
  return (
    <div className={`${featured ? "bg-[#EAF3FE]" : "bg-white"} border border-border rounded-xl p-4 flex flex-col gap-2`}>
      <p className="text-xs font-semibold text-navy">{label}</p>
      <p className="text-2xl font-extrabold text-navy tracking-tight">{value}</p>
    </div>
  );
}

function TableShell({ children, minWidth }: { children: React.ReactNode; minWidth: string }): React.ReactElement {
  return (
    <div className="overflow-x-auto border border-border rounded-lg">
      <table className={`w-full ${minWidth}`}>
        {children}
      </table>
    </div>
  );
}

function TableHead({ children }: { children: React.ReactNode }): React.ReactElement {
  return (
    <thead>
      <tr className="bg-[#F9FAFD] border-b border-border text-left">
        {children}
      </tr>
    </thead>
  );
}

function HeaderCell({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "center" }): React.ReactElement {
  return <th className={`py-3 px-3 text-xs font-bold text-navy ${align === "center" ? "text-center" : "text-left"}`}>{children}</th>;
}

function DataCell({ children, align = "left", className = "" }: { children: React.ReactNode; align?: "left" | "center"; className?: string }): React.ReactElement {
  return <td className={`py-3 px-3 text-xs text-navy ${align === "center" ? "text-center" : "text-left"} ${className}`}>{children}</td>;
}

function ResultRow({ children, highlight = false }: { children: React.ReactNode; highlight?: boolean }): React.ReactElement {
  return <tr className={`border-b border-border last:border-0 ${highlight ? "bg-[#EAF3FE] font-semibold" : "bg-white"}`}>{children}</tr>;
}

export default function EvaluationPage(): React.ReactElement {
  return (
    <div className="flex flex-col gap-6 font-inter w-full">
      <header>
        <h1 className="text-3xl font-extrabold text-navy m-0 tracking-[-0.002rem]">Evaluation</h1>
        <p className="text-sm text-muted mt-1">
          Research benchmark results for four model variants on the official OpenForensics Test set.
          The deployed app can run VERITAS and the baseline checkpoint when available.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="VERITAS Test Accuracy" value="87.55%" featured />
        <MetricCard label="Precision" value="0.816" />
        <MetricCard label="Recall" value="0.972" />
        <MetricCard label="F1-Score" value="0.887" />
      </div>

      <section className="bg-white border border-border rounded-xl p-6">
        <SectionTitle
          title="SOP 1: Classification Performance"
          description="OpenForensics Test · 10,905 images · 5,413 Real and 5,492 Fake · Fake is the positive class"
        />

        <p className="text-xs font-bold text-navy mb-2">Classification metrics</p>
        <TableShell minWidth="min-w-[760px]">
          <TableHead>
            <HeaderCell>Model</HeaderCell>
            <HeaderCell align="center">Accuracy</HeaderCell>
            <HeaderCell align="center">Precision</HeaderCell>
            <HeaderCell align="center">Recall</HeaderCell>
            <HeaderCell align="center">F1-Score</HeaderCell>
            <HeaderCell align="center">Δ Accuracy</HeaderCell>
          </TableHead>
          <tbody>
            {CLASSIFICATION_RESULTS.map((row) => (
              <ResultRow key={row.model} highlight={row.highlight}>
                <DataCell>{row.model}</DataCell>
                <DataCell align="center">{row.accuracy}</DataCell>
                <DataCell align="center">{row.precision}</DataCell>
                <DataCell align="center">{row.recall}</DataCell>
                <DataCell align="center">{row.f1}</DataCell>
                <DataCell align="center">{row.deltaAccuracy}</DataCell>
              </ResultRow>
            ))}
          </tbody>
        </TableShell>

        <p className="text-xs font-bold text-navy mt-5 mb-2">Confusion matrix counts</p>
        <TableShell minWidth="min-w-[680px]">
          <TableHead>
            <HeaderCell>Model</HeaderCell>
            <HeaderCell align="center">TP</HeaderCell>
            <HeaderCell align="center">TN</HeaderCell>
            <HeaderCell align="center">FP</HeaderCell>
            <HeaderCell align="center">FN</HeaderCell>
          </TableHead>
          <tbody>
            {CLASSIFICATION_RESULTS.map((row) => (
              <ResultRow key={row.model} highlight={row.highlight}>
                <DataCell>{row.model}</DataCell>
                <DataCell align="center">{row.tp}</DataCell>
                <DataCell align="center">{row.tn}</DataCell>
                <DataCell align="center">{row.fp}</DataCell>
                <DataCell align="center">{row.fn}</DataCell>
              </ResultRow>
            ))}
          </tbody>
        </TableShell>

        <p className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-border rounded-lg text-xs text-muted">
          VERITAS is 0.75 percentage points more accurate than the matched Baseline EfficientNetB7 on this Test set.
          TP = Fake correctly flagged; TN = Real correctly passed; FP = Real incorrectly flagged; FN = Fake missed.
        </p>
      </section>

      <section className="bg-white border border-border rounded-xl p-6">
        <SectionTitle
          title="SOP 2: Localization and Segmentation"
          description="Median agreement with generated Fake masks on the OpenForensics Test set (n = 5,492)"
        />
        <TableShell minWidth="min-w-[560px]">
          <TableHead>
            <HeaderCell>Model</HeaderCell>
            <HeaderCell align="center">Median Dice (Fake)</HeaderCell>
            <HeaderCell align="center">Median mIoU (Fake)</HeaderCell>
            <HeaderCell align="center">Fake masks evaluated</HeaderCell>
          </TableHead>
          <tbody>
            {SEGMENTATION_RESULTS.map((row) => (
              <ResultRow key={row.model} highlight={row.highlight}>
                <DataCell>{row.model}</DataCell>
                <DataCell align="center">{row.dice}</DataCell>
                <DataCell align="center">{row.miou}</DataCell>
                <DataCell align="center">5,492</DataCell>
              </ResultRow>
            ))}
          </tbody>
        </TableShell>
        <p className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-border rounded-lg text-xs text-muted">
          These scores measure agreement with pre-generated Fake masks, not hand-drawn manipulation boundaries.
          The displayed VERITAS medians follow Chapter IV Table 4.4, as selected for this screen.
        </p>
      </section>

      <section className="bg-white border border-border rounded-xl p-6">
        <SectionTitle
          title="SOP 3: Computational Trade-offs"
          description="Model-only benchmark · Kaggle Tesla T4 · batch size 4 · 4,000 images per model"
        />
        <TableShell minWidth="min-w-[680px]">
          <TableHead>
            <HeaderCell>Model</HeaderCell>
            <HeaderCell align="center">Images</HeaderCell>
            <HeaderCell align="center">Mean latency</HeaderCell>
            <HeaderCell align="center">Δ Latency</HeaderCell>
            <HeaderCell align="center">Peak allocated VRAM</HeaderCell>
          </TableHead>
          <tbody>
            {INFERENCE_RESULTS.map((row) => (
              <ResultRow key={row.model} highlight={row.highlight}>
                <DataCell>{row.model}</DataCell>
                <DataCell align="center">{row.images}</DataCell>
                <DataCell align="center">{row.latency}</DataCell>
                <DataCell align="center">{row.deltaLatency}</DataCell>
                <DataCell align="center">{row.vram}</DataCell>
              </ResultRow>
            ))}
          </tbody>
        </TableShell>
        <p className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-border rounded-lg text-xs text-muted">
          Descriptive batch-4 Kaggle Tesla T4 measurements. Latency excludes preprocessing, upload, face detection, and
          drawing; it is not the live end-to-end time shown during app analysis. This table is not a statistical test of
          latency.
        </p>
      </section>

      <details className="bg-white border border-border rounded-xl group">
        <summary className="cursor-pointer list-none p-6 flex items-center justify-between gap-4">
          <span>
            <span className="block text-base font-bold text-navy">Statistical analysis</span>
            <span className="block text-xs text-muted mt-1">Paired McNemar and Wilcoxon results from the draft</span>
          </span>
          <span aria-hidden="true" className="text-brand text-xl group-open:rotate-180 transition-transform">⌄</span>
        </summary>
        <div className="px-6 pb-6">
          <p className="text-xs font-bold text-navy mb-2">McNemar’s exact tests · two-sided · α = .05</p>
          <TableShell minWidth="min-w-[720px]">
            <TableHead>
              <HeaderCell>Model comparison</HeaderCell>
              <HeaderCell align="center">Discordant pairs</HeaderCell>
              <HeaderCell align="center">p-value</HeaderCell>
              <HeaderCell>Interpretation</HeaderCell>
            </TableHead>
            <tbody>
              {MCNEMAR_RESULTS.map((row) => (
                <tr key={row.comparison} className="border-b border-border last:border-0 bg-white">
                  <DataCell>{row.comparison}</DataCell>
                  <DataCell align="center">{row.discordant}</DataCell>
                  <DataCell align="center">{row.pValue}</DataCell>
                  <DataCell>{row.result}</DataCell>
                </tr>
              ))}
            </tbody>
          </TableShell>
          <p className="text-xs font-bold text-navy mt-5 mb-2">Wilcoxon signed-rank tests · 5,492 paired Fake masks</p>
          <TableShell minWidth="min-w-[620px]">
            <TableHead>
              <HeaderCell>Metric</HeaderCell>
              <HeaderCell align="center">Segmentation (no multi-stream) median</HeaderCell>
              <HeaderCell align="center">VERITAS median</HeaderCell>
              <HeaderCell align="center">p-value</HeaderCell>
            </TableHead>
            <tbody>
              <tr className="border-b border-border bg-white">
                <DataCell>Dice</DataCell>
                <DataCell align="center">0.9780</DataCell>
                <DataCell align="center">0.9779</DataCell>
                <DataCell align="center">5.529e-06</DataCell>
              </tr>
              <tr className="border-b border-border last:border-0 bg-white">
                <DataCell>mIoU</DataCell>
                <DataCell align="center">0.9570</DataCell>
                <DataCell align="center">0.9567</DataCell>
                <DataCell align="center">6.456e-06</DataCell>
              </tr>
            </tbody>
          </TableShell>
          <p className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-border rounded-lg text-xs text-muted">
            The paired-test table reports a very small advantage for Segmentation-only. The summary medians above follow
            Table 4.4 as selected, so they are displayed as equal after rounding; the test p-values use paired per-image
            scores and should not be inferred from the rounded summary values.
          </p>
        </div>
      </details>
    </div>
  );
}
