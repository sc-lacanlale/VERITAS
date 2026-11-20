import { SampleGallery } from "@/components/dataset/SampleGallery";

// ── Types ────────────────────────────────────────────────────────────────────
interface DatasetSourceProps {
  name: string;
  description: string;
  images: string;
  faces: string;
  manipulation: string;
}

interface StatCardProps {
  label: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
  bg: string;
}

interface DonutProps {
  percentage: number;
  color: string;
  size?: number;
}

// ── Donut chart ───────────────────────────────────────────────────────────────
function DonutChart({ percentage, color, size = 52 }: DonutProps): React.ReactElement {
  const r = 16;
  const circ = 2 * Math.PI * r;
  const dash = (percentage / 100) * circ;
  return (
    <svg width={size} height={size} viewBox="0 0 56 56">
      <circle cx="28" cy="28" r={r} fill="none" stroke="#E5E7EB" strokeWidth="9" />
      <circle
        cx="28"
        cy="28"
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="10"
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        transform="rotate(-90 28 28)"
      />
    </svg>
  );
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ label, value, subtitle, icon, bg }: StatCardProps): React.ReactElement {
  return (
    <div className={`${bg} rounded-xl p-4 flex flex-col gap-3`}>
      <div className="flex items-center gap-2">
        <span className="flex-shrink-0">{icon}</span>
        <p className="text-xs font-semibold text-navy">{label}</p>
      </div>
      <p className="text-2xl font-extrabold text-navy tracking-tight">{value}</p>
      <p className="text-[11px] text-muted">{subtitle}</p>
    </div>
  );
}

// ── Dataset source row ────────────────────────────────────────────────────────
function DatasetSourceRow({ name, description, images, faces, manipulation }: DatasetSourceProps): React.ReactElement {
  return (
    <tr className="border-b border-border last:border-0">
      <td className="py-3 px-3">
        <p className="text-xs font-semibold text-navy">{name}</p>
        <p className="text-[11px] text-muted mt-0.5">{description}</p>
      </td>
      <td className="py-3 px-3 text-xs text-navy text-center">{images}</td>
      <td className="py-3 px-3 text-xs text-navy text-center">{faces}</td>
      <td className="py-3 px-3 text-xs text-muted">{manipulation}</td>
    </tr>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function DatasetPage(): React.ReactElement {
  return (
    <div className="flex flex-col gap-6 font-inter w-full">

      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-navy m-0 tracking-[-0.002rem]">
          Dataset
        </h1>
        <p className="text-sm text-muted mt-1">
          OpenForensics labels, generated masks, and the stored-crop pipeline used for training and evaluation
        </p>
      </div>

      {/* Overview stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          bg="bg-white border border-border"
          icon={<img src="/icons/blue-dataset-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Total face crops"
          value="190,335"
          subtitle="One stored face crop per image"
        />
        <StatCard
          bg="bg-white border border-border"
          icon={<img src="/icons/file-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Generated masks"
          value="190,335"
          subtitle="Pre-generated 600×600 binary localization targets"
        />
        <StatCard
          bg="bg-[#EAF3FE] border border-[#DADADA]"
          icon={<img src="/icons/check-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Real (Test set)"
          value="5,413"
          subtitle="49.6% of the 10,905 Test images"
        />
        <StatCard
          bg="bg-[#FEEBEA] border border-[#DADADA]"
          icon={<img src="/icons/red-image-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Fake (Test set)"
          value="5,492"
          subtitle="50.4% of the 10,905 Test images"
        />
      </div>

      {/* Split distribution */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Data Split Distribution</h2>
        <p className="text-xs text-muted mb-5">Official OpenForensics Train, Validation, and Test folders (no random re-split)</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#ECF4FE] rounded-xl p-5 flex flex-col items-center text-center gap-3">
            <DonutChart percentage={73.6} color="#3333FF" size={64} />
            <div>
              <p className="text-xl font-bold text-navy">140,002</p>
              <p className="text-xs font-semibold text-navy mt-1">Training</p>
            </div>
            <div className="w-full border-t border-[#DADADA] pt-3 mt-1">
              <p className="text-xs text-navy font-semibold">73.6% of 190,335 face crops</p>
              <p className="text-[11px] text-muted mt-0.5">Model learning (weight updates)</p>
            </div>
          </div>
          <div className="bg-[#FEF4DD] rounded-xl p-5 flex flex-col items-center text-center gap-3">
            <DonutChart percentage={20.7} color="#9D803E" size={64} />
            <div>
              <p className="text-xl font-bold text-navy">39,428</p>
              <p className="text-xs font-semibold text-navy mt-1">Validation</p>
            </div>
            <div className="w-full border-t border-[#DADADA] pt-3 mt-1">
              <p className="text-xs text-navy font-semibold">20.7% of 190,335 face crops</p>
              <p className="text-[11px] text-muted mt-0.5">Checkpoint selection uses the best Validation accuracy</p>
            </div>
          </div>
          <div className="bg-[#E9F5F1] rounded-xl p-5 flex flex-col items-center text-center gap-3">
            <DonutChart percentage={5.7} color="#2C8833" size={64} />
            <div>
              <p className="text-xl font-bold text-navy">10,905</p>
              <p className="text-xs font-semibold text-navy mt-1">Testing</p>
            </div>
            <div className="w-full border-t border-[#DADADA] pt-3 mt-1">
              <p className="text-xs text-navy font-semibold">Official OpenForensics Test split</p>
              <p className="text-[11px] text-muted mt-0.5">Used for the matched four-model comparison</p>
            </div>
          </div>
        </div>
      </div>

      {/* Source dataset */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Data Source</h2>
        <p className="text-xs text-muted mb-4">
          OpenForensics supplies the Real and Fake labels and generated masks for all four model variants
        </p>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-3 text-xs font-bold text-navy">Dataset</th>
                <th className="text-center py-3 px-3 text-xs font-bold text-navy">Images</th>
                <th className="text-center py-3 px-3 text-xs font-bold text-navy">Face crops</th>
                <th className="text-left py-3 px-3 text-xs font-bold text-navy">Manipulation type</th>
              </tr>
            </thead>
            <tbody>
              <DatasetSourceRow
                name="OpenForensics"
                description="Official Train, Validation, and Test folders; Real = 0 and Fake = 1"
                images="190,335"
                faces="190,335"
                manipulation="Binary Real/Fake labels; no method-specific manipulation labels"
              />
            </tbody>
          </table>
        </div>
      </div>

      {/* Characteristics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Class balance */}
        <div className="bg-white border border-border rounded-xl p-6">
          <h2 className="text-base font-bold text-navy mb-1">Test Set Class Balance</h2>
          <p className="text-xs text-muted mb-4">No samples were discarded to force class balance</p>

          <div className="flex items-center gap-1 mb-4">
            <div className="h-8 bg-[#EAF3FE] rounded-lg flex items-center justify-center" style={{ flex: 49.6 }}>
              <p className="text-xs font-bold text-brand">49.6%</p>
            </div>
            <div className="h-8 bg-[#FEEBEA] rounded-lg flex items-center justify-center" style={{ flex: 50.4 }}>
              <p className="text-xs font-bold text-danger">50.4%</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-[#3333FF] rounded-sm flex-shrink-0"></div>
              <p className="text-xs text-navy">5,413 Real</p>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-[#C05852] rounded-sm flex-shrink-0"></div>
              <p className="text-xs text-navy">5,492 Fake</p>
            </div>
          </div>
        </div>

        {/* Quality & preprocessing */}
        <div className="bg-white border border-border rounded-xl p-6">
          <h2 className="text-base font-bold text-navy mb-1">Quality & Preprocessing</h2>
          <p className="text-xs text-muted mb-4">Standardized pipeline for model input</p>

          <div className="flex flex-col gap-3">
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Face crop</p>
                <p className="text-[11px] text-muted">Stored crop box from the generated-mask index (one crop per image)</p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Letterbox & normalization</p>
                <p className="text-[11px] text-muted">Stored crop letterboxed to 600×600 and ImageNet-normalized; the generated 600×600 mask is not letterboxed again. Letterbox padding is excluded by the valid-canvas mask.</p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Generated masks</p>
                <p className="text-[11px] text-muted">Train/Validation masks audited; the Test set was not used for mask-quality decisions</p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Augmentation (Train only)</p>
                <p className="text-[11px] text-muted">50% brightness/contrast on the RGB crop; 50% horizontal flip of RGB, mask, and valid-canvas mask together. Validation and Test are not augmented.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sample gallery */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Sample Gallery</h2>
        <p className="text-xs text-muted mb-4">
          Representative OpenForensics samples with generated localization masks
        </p>

        <SampleGallery />
      </div>

    </div>
  );
}
