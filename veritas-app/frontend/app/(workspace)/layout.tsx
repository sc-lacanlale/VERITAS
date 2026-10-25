import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Sidebar } from "@/components/Sidebar";
import { RouteLoader } from "@/components/ui/RouteLoader";
import "./workspace.css";
import "./print.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VERITAS",
  description: "Facial manipulation detection and localization",
};

export default function WorkspaceLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div
      className={`${inter.variable} font-inter`}
      style={{ minHeight: "100vh", backgroundColor: "#F9FAFD" }}
    >
      <RouteLoader />
      <Sidebar />

      <main
        className="workspace-main"
        style={{
          maxWidth: "none",
          margin: 0,
          padding: 0,
          minHeight: "100vh",
          backgroundColor: "#F9FAFD",
          boxSizing: "border-box",
        }}
      >
        {/* Top spacing — on mobile accounts for fixed top bar height */}
        <div className="h-6 lg:h-6 block lg:hidden" style={{ height: "60px" }} />
        <div className="hidden lg:block" style={{ height: "50px" }} />

        {/* Gray divider — full width */}
        <div style={{ width: "100%", height: "1px", backgroundColor: "#DADADA" }} />

        {/* Page content */}
        <div
          className="workspace-content"
          style={{
            maxWidth: "1400px",
            marginInline: "auto",
            padding: "32px 48px",
          }}
        >
          {children}
        </div>
      </main>
    </div>
  );
}
