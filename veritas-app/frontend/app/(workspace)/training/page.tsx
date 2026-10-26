// ── Types ────────────────────────────────────────────────────────────────────
interface ArchitectureBlockProps {
  title: string;
  description: string;
  details: string[];
  bg: string;
  icon: React.ReactNode;
}

interface HyperparameterRowProps {
  parameter: string;
  value: string;
  description: string;
}

interface ConfigCardProps {
  title: string;
  items: { label: string; value: string }[];
  bg: string;
}

// ── Architecture block ────────────────────────────────────────────────────────
function ArchitectureBlock({ title, description, details, bg, icon }: ArchitectureBlockProps): React.ReactElement {
  return (
    <div className={`${bg} rounded-xl p-5 flex flex-col gap-3`}>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-white border border-[#DADADA] flex items-center justify-center flex-shrink-0">
          {icon}
        </div>
        <div>
          <p className="text-sm font-bold text-navy">{title}</p>
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

// ── Hyperparameter row ────────────────────────────────────────────────────────
function HyperparameterRow({ parameter, value, description }: HyperparameterRowProps): React.ReactElement {
  return (
    <tr className="border-b border-border last:border-0">
      <td className="py-3 px-4 text-xs font-semibold text-navy">{parameter}</td>
      <td className="py-3 px-4 text-xs text-brand text-center font-mono">{value}</td>
      <td className="py-3 px-4 text-xs text-muted">{description}</td>
    </tr>
  );
}

// ── Config card ───────────────────────────────────────────────────────────────
function ConfigCard({ title, items, bg }: ConfigCardProps): React.ReactElement {
  return (
    <div className={`${bg} rounded-xl p-5`}>
      <p className="text-sm font-bold text-navy mb-3">{title}</p>
      <div className="space-y-2">
        {items.map((item, idx) => (
          <div key={idx} className="flex justify-between items-center">
            <p className="text-xs text-muted">{item.label}</p>
            <p className="text-xs font-semibold text-navy">{item.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function TrainingPage(): React.ReactElement {
  return (
    <div className="flex flex-col gap-6 font-inter w-full">

      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-navy m-0 tracking-[-0.002rem]">
          Training
        </h1>
        <p className="text-sm text-muted mt-1">
          Model configuration, hyperparameters, and optimization strategy
        </p>
      </div>

      {/* Training summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <img src="/icons/training-icon.svg" alt="" width={24} height={24} className="object-contain" />
            <p className="text-xs font-semibold text-navy">Total epochs</p>
          </div>
          <p className="text-2xl font-extrabold text-navy tracking-tight">3</p>
          <p className="text-[11px] text-muted">Incremental training protocol</p>
        </div>

        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <img src="/icons/file-icon.svg" alt="" width={24} height={24} className="object-contain" />
            <p className="text-xs font-semibold text-navy">Batch size</p>
          </div>
          <p className="text-2xl font-extrabold text-navy tracking-tight">4</p>
          <p className="text-[11px] text-muted">Per GPU allocation</p>
        </div>

        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <img src="/icons/testing-icon.svg" alt="" width={24} height={24} className="object-contain" />
            <p className="text-xs font-semibold text-navy">Learning rate</p>
          </div>
          <p className="text-2xl font-extrabold text-navy tracking-tight">1e-4</p>
          <p className="text-[11px] text-muted">With cosine annealing</p>
        </div>

        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <img src="/icons/box-icon.svg" alt="" width={24} height={24} className="object-contain" />
            <p className="text-xs font-semibold text-navy">Hardware</p>
          </div>
          <p className="text-2xl font-extrabold text-navy tracking-tight">T4</p>
          <p className="text-[11px] text-muted">NVIDIA Tesla GPU</p>
        </div>
      </div>

      {/* Multi-stream architecture */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Multi-Stream Architecture</h2>
        <p className="text-xs text-muted mb-5">
          Dual-pathway feature extraction (RGB + Frequency domain)
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <ArchitectureBlock
            title="RGB Stream"
            description="Spatial feature extraction"
            bg="bg-[#ECF4FE]"
            icon={<img src="/icons/image-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "Backbone: EfficientNet-B7 (ImageNet pretrained)",
              "Input: 600×600 letterboxed RGB images",
              "Output: 2560-dimensional feature vector",
              "Captures spatial patterns and textures"
            ]}
          />

          <ArchitectureBlock
            title="Frequency Stream"
            description="Spectral artifact detection"
            bg="bg-[#F1EFFD]"
            icon={<img src="/icons/sop1.2-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "Transform: Discrete Cosine Transform (DCT)",
              "Backbone: EfficientNet-B7 (shared weights)",
              "Input: 600×600 DCT coefficients",
              "Detects compression and synthesis artifacts"
            ]}
          />

          <ArchitectureBlock
            title="Fusion & Output"
            description="Multi-task prediction head"
            bg="bg-[#E9F5F1]"
            icon={<img src="/icons/box-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "Concatenate RGB + Freq features (5120-dim)",
              "Classification head: Binary real/fake",
              "Segmentation head: Pixel-wise manipulation mask (DeepLabV3+)",
              "Loss: Weighted BCE + Dice for segmentation"
            ]}
          />
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Hyperparameter Configuration</h2>
        <p className="text-xs text-muted mb-4">
          Optimized settings for VERITAS training
        </p>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-xs font-bold text-navy">Parameter</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">Value</th>
                <th className="text-left py-3 px-4 text-xs font-bold text-navy">Description</th>
              </tr>
            </thead>
            <tbody>
              <HyperparameterRow
                parameter="Optimizer"
                value="AdamW"
                description="Adam with weight decay for regularization"
              />
              <HyperparameterRow
                parameter="Learning rate"
                value="1e-4"
                description="Initial LR with cosine annealing schedule"
              />
              <HyperparameterRow
                parameter="Weight decay"
                value="1e-4"
                description="L2 regularization coefficient"
              />
              <HyperparameterRow
                parameter="Batch size"
                value="4"
                description="Training samples per iteration"
              />
              <HyperparameterRow
                parameter="Epochs"
                value="3"
                description="Total training epochs (incremental protocol)"
              />
              <HyperparameterRow
                parameter="Warmup epochs"
                value="None"
                description="Direct cosine annealing (no warmup)"
              />
              <HyperparameterRow
                parameter="Patience"
                value="N/A"
                description="Manual epoch-by-epoch training"
              />
              <HyperparameterRow
                parameter="Gradient clip"
                value="None"
                description="No explicit gradient clipping"
              />
            </tbody>
          </table>
        </div>
      </div>

      {/* Loss function configuration */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Loss Function Configuration</h2>
        <p className="text-xs text-muted mb-4">
          Multi-task learning with weighted objectives
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Classification loss */}
          <div className="bg-[#ECF4FE] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/sop1.1-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Classification Loss</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Binary Cross-Entropy (BCE)</p>
                  <p className="text-[11px] text-muted">Real vs. fake prediction</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Weight: 0.5</p>
                  <p className="text-[11px] text-muted">Equal to segmentation task</p>
                </div>
              </div>
            </div>
          </div>

          {/* Segmentation loss */}
          <div className="bg-[#FEF4DD] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/sop2-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Segmentation Loss</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-warning mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Dice Coefficient Loss</p>
                  <p className="text-[11px] text-muted">Pixel-wise mask overlap optimization</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-warning mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Weight: 0.5</p>
                  <p className="text-[11px] text-muted">Balanced multi-task objective</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-[#F9FAFD] rounded-lg">
          <p className="text-xs font-semibold text-navy mb-2">Combined Loss Function</p>
          <p className="text-xs font-mono text-muted bg-white px-3 py-2 rounded border border-[#DADADA]">
            L_total = 0.5 × L_BCE(classification) + 0.5 × L_Dice(segmentation)
          </p>
        </div>
      </div>

      {/* Training strategy & hardware */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Training strategy */}
        <div className="bg-white border border-border rounded-xl p-6">
          <h2 className="text-base font-bold text-navy mb-1">Training Strategy</h2>
          <p className="text-xs text-muted mb-4">
            Optimization techniques and regularization
          </p>

          <div className="flex flex-col gap-3">
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Transfer learning</p>
                <p className="text-[11px] text-muted">EfficientNet-B4 pretrained on ImageNet</p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Learning rate scheduling</p>
                <p className="text-[11px] text-muted">Cosine annealing with warmup (5 epochs)</p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Early stopping</p>
                <p className="text-[11px] text-muted">Patience of 10 epochs on validation loss</p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
                <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
              </div>
              <div>
                <p className="text-xs font-semibold text-navy">Gradient clipping</p>
                <p className="text-[11px] text-muted">Max norm 1.0 for training stability</p>
              </div>
            </div>
          </div>
        </div>

        {/* Hardware specs */}
        <div className="bg-white border border-border rounded-xl p-6">
          <h2 className="text-base font-bold text-navy mb-1">Hardware Configuration</h2>
          <p className="text-xs text-muted mb-4">
            Training infrastructure and resource allocation
          </p>

          <ConfigCard
            title="GPU Specifications"
            bg="bg-[#F9FAFD]"
            items={[
              { label: "Model", value: "NVIDIA Tesla T4" },
              { label: "Memory", value: "16 GB GDDR6" },
              { label: "CUDA cores", value: "2,560" },
              { label: "Tensor cores", value: "320" }
            ]}
          />

          <div className="mt-3">
            <ConfigCard
              title="System Resources"
              bg="bg-[#F9FAFD]"
              items={[
                { label: "CPU", value: "Intel Xeon (8 cores)" },
                { label: "RAM", value: "32 GB DDR4" },
                { label: "Storage", value: "SSD (NVMe)" },
                { label: "Framework", value: "PyTorch 2.0+" }
              ]}
            />
          </div>
        </div>
      </div>

      {/* Training time estimate */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Training Time Estimate</h2>
        <p className="text-xs text-muted mb-4">
          Approximate duration per training phase
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <ConfigCard
            title="Per Epoch"
            bg="bg-[#ECF4FE]"
            items={[
              { label: "Training", value: "~45-60 min" },
              { label: "Validation", value: "~8-10 min" },
              { label: "Total", value: "~55-70 min" }
            ]}
          />

          <ConfigCard
            title="Full Training (3 epochs)"
            bg="bg-[#FEF4DD]"
            items={[
              { label: "Estimated", value: "~3-4 hours" },
              { label: "Protocol", value: "1 epoch per run" },
              { label: "Checkpoints", value: "After each epoch" }
            ]}
          />

          <ConfigCard
            title="Resource Usage"
            bg="bg-[#E9F5F1]"
            items={[
              { label: "GPU utilization", value: "~80-90%" },
              { label: "Memory peak", value: "~14 GB" },
              { label: "Power consumption", value: "~70W avg" }
            ]}
          />
        </div>
      </div>

    </div>
  );
}
