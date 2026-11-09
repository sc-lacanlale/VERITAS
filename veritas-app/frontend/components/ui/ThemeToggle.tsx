"use client";

import { useEffect, useState } from "react";

const KEY = "veritas-theme";

/** Light / dark switch for the sidebar. Choice is saved per browser. */
export function ThemeToggle({ compact = false }: { compact?: boolean }): React.ReactElement {
  const [dark, setDark] = useState(false);

  // Sync with the class the layout script applied before paint.
  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle(): void {
    const next = !dark;
    const root = document.documentElement;
    root.classList.add("theme-anim");
    root.classList.toggle("dark", next);
    window.setTimeout(() => root.classList.remove("theme-anim"), 300);
    try {
      localStorage.setItem(KEY, next ? "dark" : "light");
    } catch {
      /* storage blocked: theme still switches for this visit */
    }
    setDark(next);
  }

  const icon = dark ? (
    // Moon
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  ) : (
    // Sun
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </svg>
  );

  if (compact) {
    return (
      <button
        type="button"
        onClick={toggle}
        aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
        aria-pressed={dark}
        className="text-white p-1.5 rounded-lg hover:bg-white/10 transition-colors"
      >
        {icon}
      </button>
    );
  }

  return (
    <button
      type="button"
      role="switch"
      aria-checked={dark}
      onClick={toggle}
      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-sidebar hover:bg-white/10 hover:text-white transition-colors"
    >
      <span className="flex items-center flex-shrink-0">{icon}</span>
      <span className="flex-1 text-left">Dark mode</span>
      {/* Switch track */}
      <span className={`relative w-9 h-5 rounded-full transition-colors ${dark ? "bg-brand" : "bg-white/20"}`}>
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200 ${dark ? "translate-x-4" : ""}`}
        />
      </span>
    </button>
  );
}
