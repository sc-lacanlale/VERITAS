// ── Types ────────────────────────────────────────────────────────────────────
interface PipelineStepProps {
  number: string;
  title: string;
  description: string;
  details: string[];
  bg: string;
  icon: React.ReactNode;
}

interface StatsCardProps {
  label: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
}

// ── Pipeline step ─────────────────────────────────────────────────────────────
function PipelineStep({ number, title, description, details, bg, icon }: PipelineStepProps): React.ReactElement {
  return (
    <div className={`${bg} rounded-xl p-5 flex flex-col gap-3`}>
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-lg bg-white border border-[#DADADA] flex items-center justify-center flex-shrink-0">
          {icon}
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-bold text-brand bg-white px-2 py-0.5 rounded">{number}</span>
            <p className="text-sm font-bold text-navy">{title}</p>
          </div>
          <p className="text-xs text-muted">{description}</p>
        </div>
      </div>
      
      <div className="flex flex-col gap-1.5 mt-1">
        {details.map((detail, idx) => (
          <div key={idx} className="flex items-start gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
            <p className="text-xs text-navy">{detail}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Stats card ────────────────────────────────────────────────────────────────
function StatsCard({ label, value, subtitle, icon }: StatsCardProps): React.ReactElement {
  return (
    <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="flex-shrink-0">{icon}</span>
        <p className="text-xs font-semibold text-navy">{label}</p>
      </div>
      <p className="text-2xl font-extrabold text-navy tracking-tight">{value}</p>
      <p className="text-[11px] text-muted">{subtitle}</p>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function PreprocessingPage(): React.ReactElement {
  return (
    <div className="flex flex-col gap-6 font-inter w-full">

      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-navy m-0 tracking-[-0.002rem]">
          Preprocessing
        </h1>
        <p className="text-sm text-muted mt-1">
          Face detection, alignment, and augmentation pipeline
        </p>
      </div>

      {/* Pipeline statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          icon={<img src="/icons/blue-dataset-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Input images"
          value="115,081"
          subtitle="Total images processed"
        />
        <StatsCard
          icon={<img src="/icons/file-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Faces detected"
          value="333,072"
          subtitle="MTCNN detection output"
        />
        <StatsCard
          icon={<img src="/icons/check-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Success rate"
          value="98.7%"
          subtitle="Valid face crops extracted"
        />
        <StatsCard
          icon={<img src="/icons/settings-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Output size"
          value="600×600"
          subtitle="Letterboxed RGB images"
        />
      </div>

      {/* Pipeline overview */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Preprocessing Pipeline</h2>
        <p className="text-xs text-muted mb-5">
          End-to-end transformation from raw images to model-ready inputs
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <PipelineStep
            number="Step 1"
            title="Face Detection"
            description="Locate and extract facial regions using MTCNN"
            bg="bg-[#ECF4FE]"
            icon={<img src="/icons/image-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "Multi-task Cascaded Convolutional Networks (MTCNN)",
              "Detects faces with bounding boxes and landmarks",
              "Confidence threshold: 0.9 for high precision"
            ]}
          />

          <PipelineStep
            number="Step 2"
            title="Alignment & Cropping"
            description="Normalize face orientation using detected landmarks"
            bg="bg-[#E9F5F1]"
            icon={<img src="/icons/settings-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "5-point facial landmarks (eyes, nose, mouth corners)",
              "Affine transformation for rotation and scale",
              "Consistent face positioning across all samples"
            ]}
          />

          <PipelineStep
            number="Step 3"
            title="Resize & Letterboxing"
            description="Standardize image dimensions while preserving aspect ratio"
            bg="bg-[#FEF4DD]"
            icon={<img src="/icons/box-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "Target size: 600×600 pixels",
              "Letterboxing with black padding to avoid distortion",
              "Maintains facial proportions and features"
            ]}
          />

          <PipelineStep
            number="Step 4"
            title="Data Augmentation"
            description="Training-time transformations for model robustness"
            bg="bg-[#F1EFFD]"
            icon={<img src="/icons/yellow-file-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "Random horizontal flips (50% probability)",
              "Rotation: ±15° random angle",
              "Color jitter: brightness, contrast, saturation"
            ]}
          />
        </div>
      </div>

      {/* MTCNN Configuration */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">MTCNN Configuration</h2>
        <p className="text-xs text-muted mb-4">
          Face detector parameters and thresholds
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#F9FAFD] rounded-lg p-4">
            <p className="text-xs font-bold text-navy mb-2">Detection Thresholds</p>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Stage 1 (P-Net)</p>
                <p className="text-xs font-semibold text-navy">0.6</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Stage 2 (R-Net)</p>
                <p className="text-xs font-semibold text-navy">0.7</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Stage 3 (O-Net)</p>
                <p className="text-xs font-semibold text-navy">0.9</p>
              </div>
            </div>
          </div>

          <div className="bg-[#F9FAFD] rounded-lg p-4">
            <p className="text-xs font-bold text-navy mb-2">Scale Parameters</p>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Min face size</p>
                <p className="text-xs font-semibold text-navy">40 pixels</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Scale factor</p>
                <p className="text-xs font-semibold text-navy">0.709</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Max detections</p>
                <p className="text-xs font-semibold text-navy">10 per image</p>
              </div>
            </div>
          </div>

          <div className="bg-[#F9FAFD] rounded-lg p-4">
            <p className="text-xs font-bold text-navy mb-2">Post-processing</p>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">NMS threshold</p>
                <p className="text-xs font-semibold text-navy">0.7</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Margin expansion</p>
                <p className="text-xs font-semibold text-navy">10%</p>
              </div>
              <div className="flex justify-between items-center">
                <p className="text-xs text-muted">Device</p>
                <p className="text-xs font-semibold text-navy">CUDA</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Augmentation examples */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Data Augmentation Strategy</h2>
        <p className="text-xs text-muted mb-4">
          Training-time transformations to improve generalization
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Geometric */}
          <div className="bg-[#ECF4FE] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/settings-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Geometric Transforms</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Horizontal flip</p>
                  <p className="text-[11px] text-muted">50% probability, mirror faces</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Random rotation</p>
                  <p className="text-[11px] text-muted">±15° range, simulate head tilt</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Slight zoom</p>
                  <p className="text-[11px] text-muted">90-110% scale variation</p>
                </div>
              </div>
            </div>
          </div>

          {/* Photometric */}
          <div className="bg-[#E9F5F1] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/yellow-file-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Photometric Transforms</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-positive mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Brightness jitter</p>
                  <p className="text-[11px] text-muted">±20% intensity variation</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-positive mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Contrast adjustment</p>
                  <p className="text-[11px] text-muted">±15% dynamic range</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-positive mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Saturation shift</p>
                  <p className="text-[11px] text-muted">±10% color intensity</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-[#DADADA] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Note:</strong> Augmentation is applied only during training. 
            Validation and test sets use original preprocessed images without augmentation.
          </p>
        </div>
      </div>

      {/* Quality control */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Quality Control</h2>
        <p className="text-xs text-muted mb-4">
          Automated filtering for preprocessing pipeline integrity
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex items-start gap-2 p-3 bg-[#E9F5F1] rounded-lg">
            <div className="w-6 h-6 rounded-lg bg-white border border-[#C4F0E3] flex items-center justify-center flex-shrink-0 mt-0.5">
              <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
            </div>
            <div>
              <p className="text-xs font-semibold text-navy">Face size validation</p>
              <p className="text-[11px] text-muted mt-0.5">Reject faces smaller than 40×40 pixels</p>
            </div>
          </div>

          <div className="flex items-start gap-2 p-3 bg-[#E9F5F1] rounded-lg">
            <div className="w-6 h-6 rounded-lg bg-white border border-[#C4F0E3] flex items-center justify-center flex-shrink-0 mt-0.5">
              <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
            </div>
            <div>
              <p className="text-xs font-semibold text-navy">Confidence filtering</p>
              <p className="text-[11px] text-muted mt-0.5">Keep only high-confidence detections (≥0.9)</p>
            </div>
          </div>

          <div className="flex items-start gap-2 p-3 bg-[#E9F5F1] rounded-lg">
            <div className="w-6 h-6 rounded-lg bg-white border border-[#C4F0E3] flex items-center justify-center flex-shrink-0 mt-0.5">
              <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
            </div>
            <div>
              <p className="text-xs font-semibold text-navy">Duplicate removal</p>
              <p className="text-[11px] text-muted mt-0.5">Filter near-identical frames from videos</p>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
