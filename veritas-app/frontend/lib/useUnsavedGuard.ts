"use client";

import { useEffect } from "react";

/**
 * Guards against leaving the page when `active` is true.
 *
 * 1. Browser refresh / tab close → native beforeunload prompt.
 * 2. In-app navigation (sidebar <a> links) → intercepted; calls `onNavigate(href)`
 *    so the page can show a custom confirm dialog instead of navigating immediately.
 *
 * This attaches a capturing click listener on document, so it does NOT require
 * editing the Sidebar component.
 */
export function useUnsavedGuard(
  active: boolean,
  onNavigate: (href: string) => void
): void {
  useEffect(() => {
    if (!active) return;

    function onBeforeUnload(e: BeforeUnloadEvent): void {
      e.preventDefault();
      e.returnValue = "";
    }

    function onClickCapture(e: MouseEvent): void {
      const target = e.target as HTMLElement | null;
      const anchor = target?.closest("a");
      if (!anchor) return;

      const href = anchor.getAttribute("href");
      if (!href || href.startsWith("http") || href.startsWith("#")) return;

      // Don't intercept navigation to the current analyze route
      if (href === "/analyze" || href.startsWith("/analyze")) return;

      e.preventDefault();
      e.stopPropagation();
      onNavigate(href);
    }

    window.addEventListener("beforeunload", onBeforeUnload);
    // capture phase so we run before Next.js Link's own handler
    document.addEventListener("click", onClickCapture, true);

    return () => {
      window.removeEventListener("beforeunload", onBeforeUnload);
      document.removeEventListener("click", onClickCapture, true);
    };
  }, [active, onNavigate]);
}
