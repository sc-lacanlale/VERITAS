export default function PreprocessingPage(): React.ReactElement {
  return (
    <div className="flex flex-col gap-6 font-inter w-full">
      <div className="text-left">
        <h1 className="text-3xl font-extrabold text-navy m-0 tracking-[-0.002rem] text-left">Preprocessing</h1>
        <p className="text-sm text-muted mt-1 text-left">Face detection, alignment, and augmentation</p>
      </div>
      <div className="bg-white border border-border rounded-xl p-10 flex flex-col items-center text-center gap-3">
        <div className="w-14 h-14 rounded-xl bg-[#ECF4FE] flex items-center justify-center">
          <img src="/icons/icon photo.svg" alt="" width={28} height={28} className="object-contain" />
        </div>
        <p className="text-sm font-semibold text-navy">Preprocessing</p>
        <p className="text-sm text-muted">Coming soon — preprocessing pipeline and statistics.</p>
      </div>
    </div>
  );
}
