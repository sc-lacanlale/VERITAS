import Link from "next/link";

// ── Types ────────────────────────────────────────────────────────────────────
interface DonutProps {
  percentage: number;
  color: string;
}

interface WorkflowStepProps {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  bg: string;
}

interface OutcomeCardProps {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  bg: string;
}

interface SopCardProps {
  id: string;
  label: string;
  icon: React.ReactNode;
}

// ── Donut chart ───────────────────────────────────────────────────────────────
function DonutChart({ percentage, color }: DonutProps): React.ReactElement {
  const r = 16;
  const circ = 2 * Math.PI * r;
  const dash = (percentage / 100) * circ;
  return (
    <svg width="52" height="52" viewBox="0 0 56 56">
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

// ── Workflow step ─────────────────────────────────────────────────────────────
function WorkflowStep({ icon, title, subtitle, bg }: WorkflowStepProps): React.ReactElement {
  return (
    <div className={`flex flex-col items-center text-center gap-2 rounded-xl p-4 flex-shrink-0 min-w-[120px] md:flex-1 ${bg}`}>
      <div className="flex items-center justify-center">{icon}</div>
      <p className="text-xs font-semibold text-navy leading-tight">{title}</p>
      <p className="text-[11px] text-muted leading-snug">{subtitle}</p>
    </div>
  );
}

// ── Outcome card ──────────────────────────────────────────────────────────────
function OutcomeCard({ icon, title, subtitle, bg }: OutcomeCardProps): React.ReactElement {
  return (
    <div className={`flex items-start gap-2 rounded-xl p-3 ${bg}`}>
      <span className="mt-0.5 flex-shrink-0">{icon}</span>
      <div>
        <p className="text-xs font-semibold text-navy">{title}</p>
        <p className="text-[11px] text-muted leading-snug">{subtitle}</p>
      </div>
    </div>
  );
}

// ── SOP card ──────────────────────────────────────────────────────────────────
function SopCard({ id, label, icon }: SopCardProps): React.ReactElement {
  return (
    <div className="border border-[#3333FF] rounded-xl p-3 flex items-start gap-2 bg-white">
      <span className="flex items-center pt-0.5 flex-shrink-0">{icon}</span>
      <div>
        <p className="text-[11px] font-bold text-brand">{id}</p>
        <p className="text-[11px] text-muted leading-snug">{label}</p>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function OverviewPage(): React.ReactElement {
  return (
    <div className="flex flex-col gap-6 font-inter w-full">

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-navy m-0 tracking-[-0.002rem]">
            VERITAS Workspace
          </h1>
          <p className="text-sm text-muted mt-1">
            Facial manipulation detection and localization
          </p>
        </div>
        <Link
          href="/analyze"
          className="flex items-center gap-2 bg-brand text-white px-5 py-2.5 rounded-lg font-semibold text-sm whitespace-nowrap hover:bg-[#2222cc] transition-colors">
          <img src="/icons/image-icon.svg" alt="" width={22} height={22} className="object-contain"/>
          Analyze an image →
        </Link>
      </div>

      {/* Split cards row */}
      <div className="grid grid-cols-4 gap-4">

        {/* Dataset */}
        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-9 h-9 rounded-lg bg-[#EAF3FE] flex items-center justify-center flex-shrink-0">
                <img src="/icons/blue-dataset-icon.svg" alt="" width={25} height={25} className="object-contain" />
              </div>
              <div className="mt-1.5">
                <p className="title font-semibold text-navy">Dataset</p>
                <p className="text-sm font-semibold text-positive">Verified</p>
              </div>
            </div>
            <div className="w-7 h-7 rounded-full bg-[#C4F0E3] flex items-center justify-center flex-shrink-0">
              <img src="/icons/check-icon.svg" alt="verified" width={20} height={20} className="object-contain" />
            </div>
          </div>
          <p className="text-[11px] text-muted mt-auto">115,081 images · 333,072 faces</p>
        </div>

        {/* Training */}
        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-9 h-9 rounded-lg bg-[#EAF3FE] flex items-center justify-center flex-shrink-0">
                <img src="/icons/training-icon.svg" alt="" width={25} height={25} className="object-contain" />
              </div>
              <span className="title font-semibold text-navy">Training</span>
            </div>
            <DonutChart percentage={70} color="#3333FF" />
          </div>
          <p className="split-percentage font-bold text-navy">70%</p>
          <p className="text-[11px] text-muted">49,056 images · 150,784 faces</p>
        </div>

        {/* Validation */}
        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-9 h-9 rounded-lg bg-[#EAF3FE] flex items-center justify-center flex-shrink-0">
                <img src="/icons/file-icon.svg" alt="" width={25} height={25} className="object-contain" />
              </div>
              <span className="title font-semibold text-navy">Validation</span>
            </div>
            <DonutChart percentage={15} color="#9D803E" />
          </div>
          <p className="split-percentage font-bold text-navy">15%</p>
          <p className="text-[11px] text-muted">10,512 images · 32,407 faces</p>
        </div>

        {/* Testing */}
        <div className="bg-white border border-border rounded-xl p-4 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-9 h-9 rounded-lg bg-[#EAF3FE] flex items-center justify-center flex-shrink-0">
                <img src="/icons/testing-icon.svg" alt="" width={25} height={25} className="object-contain" />
              </div>
              <span className="title font-semibold text-navy">Testing</span>
            </div>
            <DonutChart percentage={15} color="#2C8833" />
          </div>
          <p className="split-percentage font-bold text-navy">15%</p>
          <p className="text-[11px] text-muted">10,513 images · 32,286 faces</p>
        </div>
      </div>

      {/* Architecture Workflow */}
      <div className="bg-white border border-border rounded-xl p-6">
        <h2 className="text-base font-bold text-navy mb-1">Architecture Workflow</h2>
        <p className="text-xs text-muted mb-5">End-to-end pipeline data to analysis</p>

        <div className="flex items-center gap-2 overflow-x-auto pb-2">
          <WorkflowStep
            bg="bg-[#ECF4FE]"
            icon={
              <img src="/icons/database-icon.svg" alt="" width={22} height={22} className="object-contain" />
            }
            title="Dataset"
            subtitle="Collection and verification"
          />
          <span className="text-muted text-lg flex-shrink-0">→</span>
          <WorkflowStep
            bg="bg-[#E9F5F1]"
            icon={
              <img src="/icons/settings-icon.svg" alt="" width={22} height={22} className="object-contain" />
            }
            title="Preprocessing"
            subtitle="Face detection, alignment, and augmentation"
          />
          <span className="text-muted text-lg flex-shrink-0">→</span>
          <WorkflowStep
            bg="bg-[#F1EFFD]"
            icon={
              <img src="/icons/box-icon.svg" alt="" width={22} height={22} className="object-contain" />
            }
            title="Configuration & Training"
            subtitle="Model setup and optimization"
          />
          <span className="text-muted text-lg flex-shrink-0">→</span>
          <WorkflowStep
            bg="bg-[#FEF4DD]"
            icon={
              <img src="/icons/yellow-file-icon.svg" alt="" width={22} height={22} className="object-contain" />
            }
            title="Trained Model"
            subtitle="Saved checkpoint and inference"
          />
          <span className="text-muted text-lg flex-shrink-0">→</span>
          <div className="flex flex-col gap-2 flex-shrink-0 min-w-[150px] md:flex-1">
            <OutcomeCard
              bg="bg-[#EAF3FE]"
              icon={
                <img src="/icons/evaluation-icon.svg" alt="" width={22} height={22} className="object-contain" />
              }
              title="Evaluation"
              subtitle="Quantitative metrics and qualitative results"
            />
            <OutcomeCard
              bg="bg-[#FEEBEA]"
              icon={
                <img src="/icons/red-image-icon.svg" alt="" width={22} height={22} className="object-contain" />
              }
              title="Image Analysis"
              subtitle="Forgery detection and localization"
            />
          </div>
        </div>
      </div>

      {/* Bottom row */}
      <div className="flex gap-4 items-start w-full">

        {/* Research objectives */}
        <div className="bg-[#f1f2fd] border border-border rounded-xl p-6 flex-[2]">
          <h2 className="text-base font-bold text-navy mb-1">Research objectives</h2>
          <p className="text-xs text-muted mb-4">Key scientific objectives explored in VERITAS</p>
          <div className="grid grid-cols-3 gap-3">
            <SopCard
              id="SOP 1.1"
              label="Multi-task learning"
              icon={<img src="/icons/sop1.1-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            />
            <SopCard
              id="SOP 1.2"
              label="Multi-stream extraction"
              icon={<img src="/icons/sop1.2-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            />
            <SopCard
              id="SOP 1.3"
              label="Hyperparameter optimization"
              icon={<img src="/icons/sop1.3-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            />
            <SopCard
              id="SOP 1.4"
              label="Baseline comparison"
              icon={<img src="/icons/sop1.4-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            />
            <SopCard
              id="SOP 2"
              label="Localization and segmentation"
              icon={<img src="/icons/sop2-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            />
            <SopCard
              id="SOP 3"
              label="Computational trade-offs"
              icon={<img src="/icons/sop3-icon.svg" alt="" width={22} height={22} className="object-contain" />}
            />
          </div>
        </div>

        {/* Active model */}
        <div className="bg-white border border-border rounded-xl p-6 flex-1 min-w-[220px] w-full">
          <h2 className="text-base font-bold text-navy mb-1 text-center md:text-left">Active Trained Model</h2>
          <p className="text-xs text-muted mb-4 text-center md:text-left">VERITAS model loaded for image analysis</p>
          <div className="bg-[#E9F5F1] border border-[#C4F0E3] rounded-xl p-5 flex flex-col items-center text-center gap-3">
            <div className="w-12 h-12 rounded-full bg-white border-2 border-[#C4F0E3] flex items-center justify-center">
              <img src="/icons/check-icon.svg" alt="" width={22} height={22} className="object-contain" />
            </div>
            <p className="text-sm font-bold text-navy">Ready for analysis</p>
            <p className="text-xs text-muted leading-snug">
              Upload a facial image to detect and localize manipulation.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}
