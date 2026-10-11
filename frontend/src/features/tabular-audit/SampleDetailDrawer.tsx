import React from "react";
import { X, AlertCircle, FileSearch } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import type { SuspiciousSample } from "@/types/audit";

interface SampleDetailDrawerProps {
  sample: SuspiciousSample | null;
  onClose: () => void;
}

export const SampleDetailDrawer: React.FC<SampleDetailDrawerProps> = ({
  sample,
  onClose,
}) => {
  if (!sample) return null;

  const getTierBadge = (tier: string) => {
    switch (tier) {
      case "HIGH":
        return <Badge variant="danger">HIGH RISK</Badge>;
      case "MEDIUM":
        return <Badge variant="warning">MEDIUM RISK</Badge>;
      case "LOW":
        return <Badge variant="success">LOW RISK</Badge>;
      case "INSUFFICIENT_EVIDENCE":
      default:
        return <Badge variant="neutral">INSUFFICIENT EVIDENCE</Badge>;
    }
  };

  const signalEntries = Object.entries(sample.contributing_signals || {});

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="drawer-sample-title"
      className="fixed inset-0 z-50 flex justify-end bg-slate-900/30 backdrop-blur-xs"
    >
      <div className="bg-white w-full max-w-md h-full shadow-2xl border-l border-slate-200 flex flex-col justify-between overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
          <div className="flex items-center gap-2">
            <FileSearch className="w-4 h-4 text-indigo-700" />
            <h2 id="drawer-sample-title" className="text-sm font-bold text-slate-900">
              Sample #{sample.sample_id} Evidence
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 p-1 rounded hover:bg-slate-200/50"
            aria-label="Close sample detail drawer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto space-y-4 flex-1 text-xs">
          {/* Risk Metrics */}
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 font-medium">Risk Score</span>
              <span className="font-mono font-bold text-slate-900 text-sm">
                {sample.risk_score.toFixed(2)} / 100
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500 font-medium">Assigned Risk Tier</span>
              {getTierBadge(sample.risk_tier)}
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500 font-medium">Recommended Action</span>
              <span className="font-mono text-slate-700 font-medium">
                {sample.recommended_action || "HUMAN_REVIEW"}
              </span>
            </div>
          </div>

          {/* Contributing Signals Attribution */}
          <div className="space-y-2">
            <h3 className="font-semibold text-slate-900 flex items-center gap-1.5">
              <span>Contributing Evidence Signals</span>
              <span className="text-[10px] text-slate-400">({signalEntries.length})</span>
            </h3>

            {signalEntries.length === 0 ? (
              <p className="text-slate-400 italic py-2">
                No signal-specific attribution scores available for this sample.
              </p>
            ) : (
              <div className="space-y-1.5">
                {signalEntries.map(([sigKey, sigVal]) => (
                  <div
                    key={sigKey}
                    className="flex items-center justify-between p-2 rounded bg-slate-50 border border-slate-100"
                  >
                    <span className="font-mono text-slate-700 text-[11px]">{sigKey}</span>
                    <span className="font-mono font-semibold text-slate-900 text-[11px]">
                      {typeof sigVal === "number" ? sigVal.toFixed(3) : String(sigVal)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Metadata if present */}
          {sample.metadata && Object.keys(sample.metadata).length > 0 && (
            <div className="space-y-2 pt-2 border-t border-slate-100">
              <h3 className="font-semibold text-slate-900">Sample Context & Metadata</h3>
              <div className="bg-slate-900 text-slate-100 p-2.5 rounded font-mono text-[10px] overflow-x-auto">
                <pre>{JSON.stringify(sample.metadata, null, 2)}</pre>
              </div>
            </div>
          )}

          {/* Non-conclusive warning */}
          <div className="p-3 bg-amber-50 rounded border border-amber-200 text-amber-900 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <p className="text-[11px] leading-relaxed">
              <strong>Caution:</strong> Elevated risk score indicates statistical anomalies or disagreement with model boundaries. It is not conclusive proof that this record is an adversarial backdoor.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-200 bg-slate-50 flex justify-end">
          <Button variant="outline" size="sm" onClick={onClose}>
            Done
          </Button>
        </div>
      </div>
    </div>
  );
};
