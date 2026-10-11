import React from "react";
import { X, AlertCircle, ScanEye, Tag, Database } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { getCifarClassName } from "@/lib/cifar";
import type { SuspiciousSample } from "@/types/audit";

interface ImageSampleDetailDrawerProps {
  sample: SuspiciousSample | null;
  onClose: () => void;
}

export const ImageSampleDetailDrawer: React.FC<ImageSampleDetailDrawerProps> = ({
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
  const assignedLabel = sample.metadata?.label;
  const className = getCifarClassName(assignedLabel);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="vision-drawer-sample-title"
      className="fixed inset-0 z-50 flex justify-end bg-slate-900/30 backdrop-blur-xs"
    >
      <div className="bg-white w-full max-w-md h-full shadow-2xl border-l border-slate-200 flex flex-col justify-between overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 flex items-center justify-between bg-purple-50/40">
          <div className="flex items-center gap-2">
            <ScanEye className="w-4 h-4 text-purple-700" />
            <h2 id="vision-drawer-sample-title" className="text-sm font-bold text-slate-900">
              Image Sample #{sample.sample_id} Evidence
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
          {/* Visual Sample Card */}
          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 flex items-center gap-3">
            <div className="w-14 h-14 rounded-md bg-slate-800 text-white flex flex-col items-center justify-center font-mono text-[10px] shrink-0 border border-slate-700">
              <span className="text-xs font-bold">#{sample.sample_id}</span>
              <span className="text-[9px] text-slate-300 capitalize">{className}</span>
            </div>

            <div className="space-y-1 min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-800 text-xs">
                  Class {assignedLabel !== undefined ? assignedLabel : "?"}: {className}
                </span>
                {getTierBadge(sample.risk_tier)}
              </div>
              <p className="text-[11px] text-slate-500 font-mono">
                Risk Score: <strong>{sample.risk_score.toFixed(2)}</strong> / 100
              </p>
              {sample.metadata?.partition && (
                <span className="inline-block text-[10px] bg-slate-200/80 text-slate-700 px-1.5 py-0.5 rounded font-mono">
                  Partition: {sample.metadata.partition}
                </span>
              )}
            </div>
          </div>

          {/* Action Recommendation */}
          <div className="p-2.5 bg-slate-50 rounded border border-slate-200 flex items-center justify-between">
            <span className="text-slate-500 font-medium text-[11px]">Recommended Review Action</span>
            <span className="font-mono font-bold text-slate-800 text-xs">
              {sample.recommended_action || "HUMAN_REVIEW"}
            </span>
          </div>

          {/* Contributing Vision Signals Attribution */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-900 flex items-center gap-1.5">
                <Tag className="w-3.5 h-3.5 text-indigo-600" />
                <span>Contributing Vision Signals</span>
              </h3>
              <span className="text-[10px] text-slate-400">({signalEntries.length} evaluated)</span>
            </div>

            {signalEntries.length === 0 ? (
              <p className="text-slate-400 italic py-2">
                No signal-specific attribution scores recorded for this image.
              </p>
            ) : (
              <div className="space-y-1.5 max-h-60 overflow-y-auto pr-1">
                {signalEntries.map(([sigKey, sigVal]) => (
                  <div
                    key={sigKey}
                    className="flex items-center justify-between p-2 rounded bg-slate-50 border border-slate-100"
                  >
                    <span className="font-mono text-slate-700 text-[11px] truncate max-w-[240px]">
                      {sigKey}
                    </span>
                    <span
                      className={`font-mono font-semibold text-[11px] ${
                        typeof sigVal === "number" && sigVal >= 0.70
                          ? "text-rose-700"
                          : typeof sigVal === "number" && sigVal >= 0.40
                          ? "text-amber-700"
                          : "text-slate-700"
                      }`}
                    >
                      {typeof sigVal === "number" ? sigVal.toFixed(3) : String(sigVal)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Sample Metadata if present */}
          {sample.metadata && Object.keys(sample.metadata).length > 0 && (
            <div className="space-y-1.5 pt-2 border-t border-slate-100">
              <h3 className="font-semibold text-slate-900 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-slate-500" />
                <span>Image Diagnostics Metadata</span>
              </h3>
              <div className="bg-slate-900 text-slate-100 p-2.5 rounded font-mono text-[10px] overflow-x-auto">
                <pre>{JSON.stringify(sample.metadata, null, 2)}</pre>
              </div>
            </div>
          )}

          {/* Non-conclusive warning */}
          <div className="p-3 bg-amber-50 rounded border border-amber-200 text-amber-900 flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <p className="text-[11px] leading-relaxed">
              <strong>Caution:</strong> Elevated risk score indicates statistical latent outlier distance or prediction margin anomaly. It does not constitute proof of an intentional adversarial trigger.
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
