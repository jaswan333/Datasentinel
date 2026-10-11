import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { Menu, Activity, User, Shield } from "lucide-react";
import { Badge } from "@/components/ui/Badge";

interface TopbarProps {
  onToggleSidebar: () => void;
}

interface PageMeta {
  title: string;
  description: string;
}

const pageTitles: Record<string, PageMeta> = {
  "/": {
    title: "Overview",
    description: "Dataset Trust, Poisoning Defense & Backdoor Detection Engine",
  },
  "/machine-learning": {
    title: "Tabular Machine Learning",
    description: "Feature Anomalies, Confident Learning & Label Disagreement Audits",
  },
  "/deep-learning": {
    title: "Computer Vision Deep Learning",
    description: "Deep Latent Probes, Trojan Backdoors & Spectral Attenuation Defense",
  },
  "/help": {
    title: "Help & About",
    description: "Threat Models, Evidence Signals & Detector Interpretability Guide",
  },
  "/settings": {
    title: "Settings & System",
    description: "Platform Configuration, Policy Thresholds & Operator Diagnostics",
  },
};

export const Topbar: React.FC<TopbarProps> = ({ onToggleSidebar }) => {
  const location = useLocation();
  const currentMeta = pageTitles[location.pathname] || {
    title: "DataSentinel",
    description: "Dataset Trust & Poisoning Defense Platform",
  };

  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [modelName, setModelName] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function checkBackend() {
      try {
        const res = await fetch("/api/health");
        if (!res.ok) throw new Error("Health check non-200");
        const data = await res.json();
        if (isMounted) {
          if (data.status === "ok") {
            setBackendStatus("online");
            if (data.groq_model) setModelName(data.groq_model);
          } else {
            setBackendStatus("offline");
          }
        }
      } catch {
        if (isMounted) {
          setBackendStatus("offline");
        }
      }
    }

    checkBackend();
    const interval = setInterval(checkBackend, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="h-16 bg-white border-b border-slate-200/80 px-4 lg:px-8 flex items-center justify-between sticky top-0 z-30">
      {/* Left side: hamburger + titles */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg text-slate-600 hover:text-slate-900 hover:bg-slate-100 lg:hidden focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          aria-label="Open sidebar navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div>
          <h1 className="text-base font-bold text-slate-900 leading-tight">
            {currentMeta.title}
          </h1>
          <p className="hidden sm:block text-xs text-slate-500 leading-tight">
            {currentMeta.description}
          </p>
        </div>
      </div>

      {/* Right side: status + operator avatar */}
      <div className="flex items-center gap-3">
        {/* Real-time backend status pill */}
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-slate-50 border-slate-200">
          <Activity
            className={`w-3.5 h-3.5 ${
              backendStatus === "online"
                ? "text-emerald-500 animate-pulse"
                : backendStatus === "checking"
                ? "text-amber-500"
                : "text-slate-400"
            }`}
          />
          <span className="text-slate-700">
            {backendStatus === "online"
              ? `API Online ${modelName ? `(${modelName})` : ""}`
              : backendStatus === "checking"
              ? "Checking API..."
              : "Backend Offline"}
          </span>
        </div>

        {/* Multi-modal badge */}
        <Badge variant="default" size="sm" className="hidden sm:inline-flex gap-1">
          <Shield className="w-3 h-3" />
          <span>Milestone 8</span>
        </Badge>

        {/* Operator profile badge */}
        <div
          className="flex items-center gap-2 pl-2 border-l border-slate-200"
          title="Security Operations Console"
        >
          <div className="w-8 h-8 rounded-full bg-slate-100 border border-slate-300 flex items-center justify-center text-slate-700 font-medium text-xs">
            <User className="w-4 h-4" />
          </div>
          <div className="hidden lg:block text-left">
            <div className="text-xs font-semibold text-slate-800 leading-none">
              SecOps Operator
            </div>
            <div className="text-[10px] text-slate-500 leading-none mt-0.5 font-mono">
              Role: Reviewer
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
