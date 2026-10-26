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
          Curated training data for facial manipulation detection
        </p>
      </div>

      {/* Overview stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          bg="bg-white border border-border"
          icon={<img src="/icons/blue-dataset-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Total images"
          value="115,081"
          subtitle="Across all source datasets"
        />
        <StatCard
          bg="bg-white border border-border"
          icon={<img src="/icons/file-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Total faces"
          value="333,072"
          subtitle="Detected and aligned"
        />
        <StatCard
          bg="bg-[#EAF3FE] border border-[#DADADA]"
          icon={<img src="/icons/check-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Real faces"
          value="166,536"
          subtitle="50% of total dataset"
        />
        <StatCard
          bg="bg-[#FEEBEA] border border-[#DADADA]"
          icon={<img src="/icons/red-image-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Manipulated faces"
          value="166,536"
          subtitle="50% of total dataset"
        />
      </div>

      {/* Split distribution */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Data Split Distribution</h2>
        <p className="text-xs text-muted mb-5">70% training, 15% validation, 15% testing</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Training */}
          <div className="bg-[#ECF4FE] rounded-xl p-5 flex flex-col items-center text-center gap-3">
            <DonutChart percentage={70} color="#3333FF" size={64} />
            <div>
              <p className="text-xl font-bold text-navy">70%</p>
              <p className="text-xs font-semibold text-navy mt-1">Training</p>
            </div>
            <div className="w-full border-t border-[#DADADA] pt-3 mt-1">
              <p className="text-xs text-navy font-semibold">49,056 images</p>
              <p className="text-[11px] text-muted mt-0.5">150,784 faces detected</p>
            </div>
          </div>

          {/* Validation */}
          <div className="bg-[#FEF4DD] rounded-xl p-5 flex flex-col items-center text-center gap-3">
            <DonutChart percentage={15} color="#9D803E" size={64} />
            <div>
              <p className="text-xl font-bold text-navy">15%</p>
              <p className="text-xs font-semibold text-navy mt-1">Validation</p>
            </div>
            <div className="w-full border-t border-[#DADADA] pt-3 mt-1">
              <p className="text-xs text-navy font-semibold">10,512 images</p>
              <p className="text-[11px] text-muted mt-0.5">32,407 faces detected</p>
            </div>
          </div>

          {/* Testing */}
          <div className="bg-[#E9F5F1] rounded-xl p-5 flex flex-col items-center text-center gap-3">
            <DonutChart percentage={15} color="#2C8833" size={64} />
            <div>
              <p className="text-xl font-bold text-navy">15%</p>
              <p className="text-xs font-semibold text-navy mt-1">Testing</p>
            </div>
            <div className="w-full border-t border-[#DADADA] pt-3 mt-1">
              <p className="text-xs text-navy font-semibold">10,513 images</p>
              <p className="text-[11px] text-muted mt-0.5">32,286 faces detected</p>
            </div>
          </div>
        </div>
      </div>

      {/* Source datasets */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Source Datasets</h2>
        <p className="text-xs text-muted mb-4">
          Aggregated from established facial manipulation benchmarks
        </p>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-3 text-xs font-bold text-navy">Dataset</th>
                <th className="text-center py-3 px-3 text-xs font-bold text-navy">Images</th>
                <th className="text-center py-3 px-3 text-xs font-bold text-navy">Faces</th>
                <th className="text-left py-3 px-3 text-xs font-bold text-navy">Manipulation type</th>
              </tr>
            </thead>
            <tbody>
              <DatasetSourceRow
                name="OpenForensics"
                description="Multi-ethnic deepfake dataset"
                images="~38,000"
                faces="~110,000"
                manipulation="DeepFake, Face2Face, FaceSwap, NeuralTextures"
              />
              <DatasetSourceRow
                name="FaceForensics++"
                description="High-quality manipulation benchmark"
                images="~50,000"
                faces="~145,000"
                manipulation="DeepFakes, Face2Face, FaceSwap, NeuralTextures"
              />
              <DatasetSourceRow
                name="WildDeepfake"
                description="In-the-wild collected videos"
                images="~15,000"
                faces="~45,000"
                manipulation="Real-world deepfakes (various methods)"
              />
              <DatasetSourceRow
                name="Celeb-DF"
                description="Celebrity impersonation dataset"
                images="~10,000"
                faces="~28,000"
                manipulation="High-quality celebrity deepfakes"
              />
              <DatasetSourceRow
                name="DFDC Preview"
                description="Deepfake Detection Challenge subset"
                images="~2,081"
                faces="~5,072"
                manipulation="Mixed deepfake techniques"
              />
            </tbody>
          </table>
        </div>
      </div>

      {/* Characteristics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Class balance */}
        <div className="bg-white border border-border rounded-xl p-6">
          <h2 className="text-base font-bold text-navy mb-1">Class Balance</h2>
          <p className="text-xs text-muted mb-4">Equal representation of real and manipulated samples</p>

          <div className="flex items-center gap-4 mb-4">
            <div className="flex-1 h-8 bg-[#EAF3FE] rounded-lg flex items-center justify-center">
              <p className="text-xs font-bold text-brand">50%</p>
            </div>
            <div className="flex-1 h-8 bg-[#FEEBEA] rounded-lg flex items-center justify-center">
              <p className="text-xs font-bold text-danger">50%</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-[#3333FF] rounded-sm flex-shrink-0"></div>
              <p className="text-xs text-navy">166,536 real faces</p>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-[#C05852] rounded-sm flex-shrink-0"></div>
              <p className="text-xs text-navy">166,536 fake faces</p>
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
                <p className="text-xs font-semibold text-navy">Face detection</p>
                <p className="text-[11px] text-muted">MTCNN with 224×224 letterboxed output</p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Alignment & normalization</p>
                <p className="text-[11px] text-muted">Facial landmarks for consistent orientation</p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Augmentation</p>
                <p className="text-[11px] text-muted">Rotation, flips, color jitter for robustness</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sample gallery placeholder */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Sample Gallery</h2>
        <p className="text-xs text-muted mb-4">
          Representative examples from the aggregated dataset
        </p>

        <div className="bg-[#F9FAFD] border border-dashed border-[#DADADA] rounded-xl p-12 flex flex-col items-center justify-center text-center gap-3">
          <div className="w-16 h-16 rounded-full bg-white border border-[#DADADA] flex items-center justify-center">
            <img src="/icons/blue-dataset-icon.svg" alt="" width={32} height={32} className="object-contain" />
          </div>
          <p className="text-sm font-semibold text-navy">Sample visualization pending</p>
          <p className="text-xs text-muted max-w-md">
            Gallery view will display representative real and manipulated face samples from the training dataset.
          </p>
        </div>
      </div>

    </div>
  );
}
