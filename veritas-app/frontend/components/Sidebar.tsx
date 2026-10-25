"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

interface NavItem {
  label: string;
  href: string;
  icon: React.ReactNode;
}

const NAV: NavItem[] = [
  {
    label: "Overview",
    href: "/overview",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
        <polyline points="9 22 9 12 15 12 15 22" />
      </svg>
    ),
  },
  {
    label: "Analyze Image",
    href: "/analyze",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <circle cx="8.5" cy="8.5" r="1.5" />
        <polyline points="21 15 16 10 5 21" />
      </svg>
    ),
  },
  {
    label: "Dataset",
    href: "/dataset",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      </svg>
    ),
  },
  {
    label: "Preprocessing",
    href: "/preprocessing",
    icon: (
      // Photo-scan: an image framed by scan corner brackets
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 8V6a2 2 0 0 1 2-2h2" />
        <path d="M16 4h2a2 2 0 0 1 2 2v2" />
        <path d="M20 16v2a2 2 0 0 1-2 2h-2" />
        <path d="M8 20H6a2 2 0 0 1-2-2v-2" />
        <circle cx="9.5" cy="10" r="1.2" />
        <path d="M7.5 15.5 10 13l2 2 2.5-2.5 2 2" />
      </svg>
    ),
  },
  {
    label: "Training",
    href: "/training",
    icon: (
      // Adjustments-horizontal: two sliders
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="4" y1="8" x2="20" y2="8" />
        <line x1="4" y1="16" x2="20" y2="16" />
        <circle cx="9" cy="8" r="2.2" />
        <circle cx="15" cy="16" r="2.2" />
      </svg>
    ),
  },
  {
    label: "Evaluation",
    href: "/evaluation",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </svg>
    ),
  },
];

function NavLinks({ onClose }: { onClose?: () => void }) {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-1">
      {NAV.map((item) => {
        const active =
          pathname === item.href ||
          (item.href !== "/" && pathname.startsWith(item.href));
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onClose}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors duration-150
              ${active
                ? "bg-white/10 text-white"
                : "text-sidebar hover:bg-white/10 hover:text-white"
              }`}
          >
            <span className="flex items-center flex-shrink-0">{item.icon}</span>
            <span>{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

export function Sidebar(): React.ReactElement {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();

  // Close the mobile menu whenever the route changes, so it never lingers
  // open after navigating to another page.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <>
      {/* ── Desktop sidebar (lg and above) ── */}
      <aside className="hidden lg:flex w-[220px] min-h-screen bg-navy flex-col py-6 fixed top-0 left-0 z-50">
        <div className="px-3 pb-8">
          <img src="/logo.svg" alt="VERITAS" className="h-10 w-auto" />
        </div>
        <div className="px-3">
          <NavLinks />
        </div>
      </aside>

      {/* ── Mobile top bar (below lg) ── */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-navy">
        <div className="flex items-center justify-between px-4 py-3">
          <img src="/logo.svg" alt="VERITAS" className="h-9 w-auto" />
          <button
            onClick={() => setOpen(!open)}
            className="text-white p-1"
            aria-label="Toggle menu"
          >
            {open ? (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            ) : (
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            )}
          </button>
        </div>

        {/* Dropdown menu — drops down from the top bar */}
        {open && (
          <div className="bg-navy border-t border-white/10 px-3 pb-4">
            <NavLinks onClose={() => setOpen(false)} />
          </div>
        )}
      </div>
    </>
  );
}
