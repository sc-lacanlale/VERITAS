"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import { LoadingOverlay } from "./LoadingOverlay";

/**
 * Shows a brief loading overlay on every in-app route change.
 * Mount once in the workspace layout. Does not alter page layout.
 */
export function RouteLoader(): React.ReactElement | null {
  const pathname = usePathname();
  const [loading, setLoading] = useState(false);
  const first = useRef(true);

  useEffect(() => {
    // Skip the very first render (initial page load)
    if (first.current) {
      first.current = false;
      return;
    }
    setLoading(true);
    const t = window.setTimeout(() => setLoading(false), 550);
    return () => window.clearTimeout(t);
  }, [pathname]);

  // Same-page sidebar click (or a confirmed reset): brief loader + back to top.
  useEffect(() => {
    let t: number | undefined;
    function onShow(): void {
      setLoading(true);
      window.scrollTo({ top: 0 });
      window.clearTimeout(t);
      t = window.setTimeout(() => setLoading(false), 550);
    }
    window.addEventListener("veritas:show-loader", onShow);
    return () => {
      window.removeEventListener("veritas:show-loader", onShow);
      window.clearTimeout(t);
    };
  }, []);

  if (!loading) return null;
  return <LoadingOverlay variant="overlay" label="Loading..." />;
}
