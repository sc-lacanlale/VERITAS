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

  if (!loading) return null;
  return <LoadingOverlay variant="overlay" label="Loading..." />;
}
