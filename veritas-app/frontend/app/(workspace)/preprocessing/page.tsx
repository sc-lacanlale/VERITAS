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
          Training preprocessing in the notebooks and live preprocessing in the app
        </p>
      </div>

      {/* Pipeline statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          icon={<img src="/icons/blue-dataset-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Input face crops"
          value="190,335"
          subtitle="One stored crop per image"
        />
        <StatsCard
          icon={<img src="/icons/file-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Generated masks"
          value="190,335"
          subtitle="600×600 binary localization targets"
        />
        <StatsCard
          icon={<img src="/icons/navy-settings-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Output size"
          value="600×600"
          subtitle="Letterboxed RGB, ImageNet-normalized"
        />
        <StatsCard
          icon={<img src="/icons/navy-check-icon.svg" alt="" width={24} height={24} className="object-contain" />}
          label="Augmentation"
          value="Train only"
          subtitle="50% brightness/contrast · 50% flip"
        />
      </div>

      {/* Pipeline overview */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Training Preprocessing Pipeline</h2>
        <p className="text-xs text-muted mb-5">
          From stored face crops and generated masks to model-ready inputs
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <PipelineStep
            number="Step 1"
            title="Stored Face Crop"
            description="Crop box taken from the generated-mask index"
            bg="bg-[#ECF4FE]"
            icon={<img src="/icons/image-icon-dark.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "One crop per image (no face detector at training time)",
              "Paired with its pre-generated 600×600 binary mask",
              "Label from the folder: Real = 0, Fake = 1"
            ]}
          />

          <PipelineStep
            number="Step 2"
            title="Letterbox & Valid-Canvas Mask"
            description="Standardize size while preserving aspect ratio"
            bg="bg-[#E9F5F1]"
            icon={<img src="/icons/settings-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "600×600 letterbox with black padding (no stretching)",
              "Valid-canvas mask marks the real image pixels",
              "Padding is excluded from segmentation loss and metrics"
            ]}
          />

          <PipelineStep
            number="Step 3"
            title="Normalization & Targets"
            description="Prepare the RGB tensor and localization target"
            bg="bg-[#FEF4DD]"
            icon={<img src="/icons/box-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225)",
              "Mask binarized (pixel value above 127 = manipulated)",
              "Real images use an all-zero localization target"
            ]}
          />

          <PipelineStep
            number="Step 4"
            title="Data Augmentation (Train only)"
            description="Light augmentation that keeps forensic traces intact"
            bg="bg-[#F1EFFD]"
            icon={<img src="/icons/yellow-file-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "50% chance: brightness/contrast on the RGB crop only",
              "Then 50% chance: horizontal flip of crop, mask, and valid-canvas together",
              "Validation and Test images are not augmented"
            ]}
          />

        </div>
      </div>

      {/* Live app face detection */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Live App Face Detection</h2>
        <p className="text-xs text-muted mb-4">
          Used only in the deployed system; training and Test scoring use the stored crops
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-[#F9FAFD] rounded-lg p-4">
            <p className="text-xs font-bold text-navy mb-2">Detection</p>
            <div className="space-y-2">
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Detector</p>
                <p className="text-xs font-semibold text-navy text-right">MediaPipe BlazeFace</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Detection threshold</p>
                <p className="text-xs font-semibold text-navy text-right">0.5</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Faces analyzed</p>
                <p className="text-xs font-semibold text-navy text-right">Every detected face</p>
              </div>
            </div>
          </div>
          <div className="bg-[#F9FAFD] rounded-lg p-4">
            <p className="text-xs font-bold text-navy mb-2">Face-quality check</p>
            <div className="space-y-2">
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Min. detector confidence</p>
                <p className="text-xs font-semibold text-navy text-right">0.60</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Min. face size</p>
                <p className="text-xs font-semibold text-navy text-right">8% of image's larger side</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">If no face is clear</p>
                <p className="text-xs font-semibold text-navy text-right">Insufficient facial exposure</p>
              </div>
            </div>
          </div>
          <div className="bg-[#F9FAFD] rounded-lg p-4">
            <p className="text-xs font-bold text-navy mb-2">Cropping</p>
            <div className="space-y-2">
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Margin expansion</p>
                <p className="text-xs font-semibold text-navy text-right">10% per side</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Letterbox</p>
                <p className="text-xs font-semibold text-navy text-right">600×600</p>
              </div>
              <div className="flex justify-between items-center gap-3">
                <p className="text-xs text-muted">Device</p>
                <p className="text-xs font-semibold text-navy text-right">GPU if CUDA is available, else CPU</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Augmentation */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Data Augmentation Strategy</h2>
        <p className="text-xs text-muted mb-4">
          Training-time transformations used by all four models
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Geometric */}
          <div className="bg-[#ECF4FE] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/settings-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Geometric Transform</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Horizontal flip</p>
                  <p className="text-[11px] text-muted">50% probability; crop, mask, and valid-canvas flipped together</p>
                </div>
              </div>
            </div>
          </div>

          {/* Photometric */}
          <div className="bg-[#E9F5F1] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/yellow-file-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Photometric Transform</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-positive mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Brightness/contrast</p>
                  <p className="text-[11px] text-muted">50% probability; gain ±15% and offset ±12 intensity levels, RGB crop only</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-[#F9FAFD] border border-dashed border-[#DADADA] rounded-lg">
          <p className="text-xs text-muted">
            <strong className="text-navy">Note:</strong> Augmentation is applied only during training.
            Validation and test sets use the preprocessed images without augmentation.
          </p>
        </div>
      </div>

      {/* Quality control */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Quality Control</h2>
        <p className="text-xs text-muted mb-4">
          Checks applied to the data and generated masks
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex items-start gap-2 p-3 bg-[#E9F5F1] rounded-lg">
            <div className="w-6 h-6 rounded-lg bg-white border border-[#C4F0E3] flex items-center justify-center flex-shrink-0 mt-0.5">
              <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
            </div>
            <div>
              <p className="text-xs font-semibold text-navy">Mask audit (Train/Validation)</p>
              <p className="text-[11px] text-muted mt-0.5">Coverage and sample plausibility checked; Test not used for mask decisions</p>
            </div>
          </div>
          <div className="flex items-start gap-2 p-3 bg-[#E9F5F1] rounded-lg">
            <div className="w-6 h-6 rounded-lg bg-white border border-[#C4F0E3] flex items-center justify-center flex-shrink-0 mt-0.5">
              <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
            </div>
            <div>
              <p className="text-xs font-semibold text-navy">Empty Fake masks</p>
              <p className="text-[11px] text-muted mt-0.5">Kept for classification, skipped in the segmentation loss</p>
            </div>
          </div>
          <div className="flex items-start gap-2 p-3 bg-[#E9F5F1] rounded-lg">
            <div className="w-6 h-6 rounded-lg bg-white border border-[#C4F0E3] flex items-center justify-center flex-shrink-0 mt-0.5">
              <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
            </div>
            <div>
              <p className="text-xs font-semibold text-navy">No class-balance discarding</p>
              <p className="text-[11px] text-muted mt-0.5">All provided image entries are used</p>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
