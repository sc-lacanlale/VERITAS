// ── Types ─────────────────────────────────────────────────────────────────────
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
  finalValue: string;
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
function HyperparameterRow({ parameter, value, finalValue, description }: HyperparameterRowProps): React.ReactElement {
  return (
    <tr className="border-b border-border last:border-0">
      <td className="py-3 px-4 text-xs font-semibold text-navy">{parameter}</td>
      <td className="py-3 px-4 text-xs text-brand text-center font-mono">{value}</td>
      <td className="py-3 px-4 text-xs text-brand text-center font-mono">{finalValue}</td>
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
          <div key={idx} className="flex justify-between items-center gap-3">
            <p className="text-xs text-muted">{item.label}</p>
            <p className="text-xs font-semibold text-navy text-right">{item.value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Strategy item ─────────────────────────────────────────────────────────────
function StrategyItem({ title, text }: { title: string; text: string }): React.ReactElement {
  return (
    <div className="flex items-start gap-2">
      <div className="w-6 h-6 rounded-lg bg-[#E9F5F1] flex items-center justify-center flex-shrink-0 mt-0.5">
        <img src="/icons/check-icon.svg" alt="" width={16} height={16} className="object-contain" />
      </div>
      <div>
        <p className="text-xs font-semibold text-navy">{title}</p>
        <p className="text-[11px] text-muted">{text}</p>
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
          VERITAS (Final) architecture and the training configuration of the two deployed models
        </p>
      </div>

      {/* Training summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <img src="/icons/navy-training-icon.svg" alt="" width={24} height={24} className="object-contain" />
            <p className="text-xs font-semibold text-navy">Total epochs</p>
          </div>
          <p className="text-2xl font-extrabold text-navy tracking-tight">3</p>
          <p className="text-[11px] text-muted">VERITAS (Final): 1 warmup + 2 fine-tuning epochs</p>
        </div>

        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <img src="/icons/file-icon.svg" alt="" width={24} height={24} className="object-contain" />
            <p className="text-xs font-semibold text-navy">Batch size</p>
          </div>
          <p className="text-2xl font-extrabold text-navy tracking-tight">16</p>
          <p className="text-[11px] text-muted">Effective batch: 4 per GPU × 2 GPUs × 2 accumulation steps</p>
        </div>

        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <img src="/icons/navy-chart-icon.svg" alt="" width={24} height={24} className="object-contain" />
            <p className="text-xs font-semibold text-navy">Learning rate</p>
          </div>
          <p className="text-2xl font-extrabold text-navy tracking-tight">3e-4 → 1.5e-4</p>
          <p className="text-[11px] text-muted">Stage 1 warmup, then Stage 2 fine-tuning (backbone × 0.1)</p>
        </div>

        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <img src="/icons/navy-box-icon.svg" alt="" width={24} height={24} className="object-contain" />
            <p className="text-xs font-semibold text-navy">Hardware</p>
          </div>
          <p className="text-2xl font-extrabold text-navy tracking-tight">T4 × 2</p>
          <p className="text-[11px] text-muted">Kaggle dual Tesla T4 (16 GB each), DDP</p>
        </div>
      </div>

      {/* Architecture */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">VERITAS Architecture</h2>
        <p className="text-xs text-muted mb-5">
          VERITAS (Final): RGB EfficientNet-B7 expert and a 9-channel color SRM forensic expert joined by bounded dual-expert fusion, with a multi-scale segmentation head
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <ArchitectureBlock
            title="RGB Backbone"
            description="Main spatial pathway"
            bg="bg-[#ECF4FE]"
            icon={<img src="/icons/image-icon-dark.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "EfficientNet-B7, ImageNet pretrained",
              "Input: 600×600 letterboxed RGB face crop",
              "Final features (2,560 channels) feed the pure RGB expert (no mask input)",
              "Stage 2, 5 and 7 features feed the segmentation head"
            ]}
          />

          <ArchitectureBlock
            title="Color SRM Forensic Stream"
            description="Computed from the same RGB crop"
            bg="bg-[#F1EFFD]"
            icon={<img src="/icons/sop1.2-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "9-channel color SRM: fixed high-pass residual filters on each RGB channel",
              "Spatial forensic encoder (38×38 and 19×19 feature maps)",
              "Not a separate EfficientNet backbone"
            ]}
          />

          <ArchitectureBlock
            title="Fusion & Heads"
            description="Classification and localization"
            bg="bg-[#E9F5F1]"
            icon={<img src="/icons/box-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            details={[
              "Forensic expert: SRM features + mask-guided pooling of the predicted mask",
              "Bounded dual-expert logit fusion, α in [0.35, 0.65] (learned ≈ 0.46 RGB / 0.54 SRM)",
              "Classification: one Fake probability per face",
              "Segmentation: multi-scale separable ASPP head with SRM features"
            ]}
          />
        </div>

        <div className="mt-4 p-4 bg-[#F9FAFD] rounded-lg">
          <p className="text-xs font-semibold text-navy mb-2">Deployed models</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {[
              ["Baseline EfficientNetB7", "RGB only, average pooling, no segmentation"],
              ["VERITAS (Final)", "RGB + color SRM dual-expert fusion + multi-scale segmentation"],
            ].map(([name, desc]) => (
              <div key={name} className="bg-white border border-[#DADADA] rounded-lg px-3 py-2">
                <p className="text-xs font-semibold text-navy">{name}</p>
                <p className="text-[11px] text-muted">{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Hyperparameter Configuration</h2>
        <p className="text-xs text-muted mb-4">
          Training configuration of the two deployed models
        </p>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-border">
                <th className="text-left py-3 px-4 text-xs font-bold text-navy">Parameter</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">Baseline EfficientNetB7</th>
                <th className="text-center py-3 px-4 text-xs font-bold text-navy">VERITAS (Final)</th>
                <th className="text-left py-3 px-4 text-xs font-bold text-navy">Description</th>
              </tr>
            </thead>
            <tbody>
              <HyperparameterRow parameter="Optimizer" value="AdamW" finalValue="AdamW" description="Adam with decoupled weight decay" />
              <HyperparameterRow parameter="Learning rate" value="1e-4" finalValue="3e-4 → 1.5e-4" description="New layers; VERITAS: Stage 1 warmup, then Stage 2 fine-tuning" />
              <HyperparameterRow parameter="Backbone learning rate" value="1e-5" finalValue="Frozen → 1.5e-5" description="EfficientNet-B7 encoder (0.1 × learning rate); frozen in VERITAS Stage 1" />
              <HyperparameterRow parameter="Weight decay" value="1e-4" finalValue="1e-4" description="Applied through AdamW" />
              <HyperparameterRow parameter="LR schedule" value="5% warmup + cosine" finalValue="Cosine; 5% warmup + cosine" description="VERITAS: cosine in Stage 1, 5% warmup then cosine in Stage 2" />
              <HyperparameterRow parameter="Batch size" value="4" finalValue="16 (effective)" description="VERITAS: 4 per GPU × 2 GPUs × 2 gradient-accumulation steps" />
              <HyperparameterRow parameter="Epochs" value="3" finalValue="1 + 2" description="VERITAS: 1 warmup epoch + 2 end-to-end fine-tuning epochs" />
              <HyperparameterRow parameter="Training images" value="OpenForensics Train" finalValue="140,002 (full Train)" description="Official OpenForensics Train split" />
              <HyperparameterRow parameter="Backbone BatchNorm" value="Frozen statistics" finalValue="Frozen statistics" description="Running means/variances of the B7 encoder are not updated" />
              <HyperparameterRow parameter="Mixed precision" value="On (CUDA)" finalValue="On (CUDA)" description="Automatic mixed precision" />
              <HyperparameterRow parameter="Thresholds" value="0.5 / 0.5" finalValue="0.5 / 0.5" description="Classification and mask thresholds" />
              <HyperparameterRow parameter="Random seed" value="42" finalValue="42" description="Same seed for both models" />
            </tbody>
          </table>
        </div>
      </div>

      {/* Loss function configuration */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Loss Function Configuration</h2>
        <p className="text-xs text-muted mb-4">
          Classification loss for both models; VERITAS adds expert and segmentation losses
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                  <p className="text-[11px] text-muted">Real = 0, Fake = 1</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Weight λcls = 1.0</p>
                  <p className="text-[11px] text-muted">Used by both models</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-brand mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">VERITAS (Final): expert supervision</p>
                  <p className="text-[11px] text-muted">Extra BCE on the RGB expert (λrgb = 0.50) and the SRM forensic expert (λaux = 0.50)</p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#FEF4DD] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <img src="/icons/sop2-icon.svg" alt="" width={22} height={22} className="object-contain" />
              <p className="text-sm font-bold text-navy">Segmentation Loss</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-warning mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Fake (nonempty mask): (BCE + Dice) / 2</p>
                  <p className="text-[11px] text-muted">Against the generated mask; Fake images with empty masks are skipped</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-warning mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Real: pixel BCE against an all-zero mask</p>
                  <p className="text-[11px] text-muted">Letterbox padding is excluded by the valid-canvas mask</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 rounded-full bg-warning mt-1.5 flex-shrink-0"></div>
                <div>
                  <p className="text-xs font-semibold text-navy">Weight λseg = 0.50 (VERITAS only)</p>
                  <p className="text-[11px] text-muted">Dice weight 1.0</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 p-4 bg-[#F9FAFD] rounded-lg">
          <p className="text-xs font-semibold text-navy mb-2">Combined Loss Function</p>
          <p className="text-xs font-mono text-muted bg-white px-3 py-2 rounded border border-[#DADADA]">
            Baseline EfficientNetB7: L_total = 1.0 × L_cls(BCE)
          </p>
          <p className="text-xs font-mono text-muted bg-white px-3 py-2 rounded border border-[#DADADA] mt-2">
            VERITAS (Final): L_total = 1.0 × L_cls + 0.50 × L_rgb + 0.50 × L_aux + 0.50 × L_seg
          </p>
        </div>
      </div>

      {/* Training strategy & hardware */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-border rounded-xl p-6">
          <h2 className="text-base font-bold text-navy mb-1">Training Strategy</h2>
          <p className="text-xs text-muted mb-4">
            How each model is trained and its checkpoint chosen
          </p>

          <div className="flex flex-col gap-3">
            <StrategyItem title="Transfer learning" text="EfficientNet-B7 pretrained on ImageNet, fine-tuned at a 10× lower learning rate (frozen during VERITAS Stage 1)" />
            <StrategyItem title="Training runs" text="Baseline: one epoch per Kaggle session, resumed from checkpoint. VERITAS (Final): one dual-T4 DDP run (1 warmup + 2 fine-tuning epochs)" />
            <StrategyItem title="Checkpoint selection" text="The epoch with the highest Validation accuracy is kept (VERITAS: 8,000 stratified Validation images)" />
            <StrategyItem title="No early stopping" text="All 3 epochs are always completed" />
            <StrategyItem title="Test set held out" text="Not used in training or checkpoint selection" />
          </div>
        </div>

        <div className="bg-white border border-border rounded-xl p-6">
          <h2 className="text-base font-bold text-navy mb-1">Hardware Configuration</h2>
          <p className="text-xs text-muted mb-4">
            Training and benchmarking environment
          </p>

          <ConfigCard
            title="Training Environment"
            bg="bg-[#F9FAFD]"
            items={[
              { label: "Platform", value: "Kaggle Notebooks" },
              { label: "GPU", value: "Tesla T4 (Baseline: 1; VERITAS: 2, DDP)" },
              { label: "GPU memory", value: "16 GB" },
              { label: "Framework", value: "PyTorch + torchvision" }
            ]}
          />

          <div className="mt-3">
            <ConfigCard
              title="Session Settings (Baseline)"
              bg="bg-[#F9FAFD]"
              items={[
                { label: "Epochs per session", value: "1" },
                { label: "Session time budget", value: "11 hours" },
                { label: "Mid-epoch checkpoint", value: "Every 30 minutes" }
              ]}
            />
          </div>
        </div>
      </div>

    </div>
  );
}
