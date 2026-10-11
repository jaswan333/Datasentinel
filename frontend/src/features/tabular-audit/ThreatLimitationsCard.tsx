import React from "react";
import { ShieldX, Info } from "lucide-react";
import type { ThreatLimitation } from "@/types/audit";

interface ThreatLimitationsCardProps {
  limitations: ThreatLimitation[];
}

export const ThreatLimitationsCard: React.FC<ThreatLimitationsCardProps> = ({
  limitations,
}) => {
  // If no explicit limitations in response, provide standard boundary disclosure for tabular audits
  const displayLimitations =
    Array.isArray(limitations) && limitations.length > 0
      ? limitations
      : [
          {
            threat_family: "clean_label_feature_perturbations",
            status: "UNRESOLVED_LIMITATION",
            observed_recall: "Boundary dependent",
            operating_fpr: "<= 5.0% FPR",
            statement:
              "Stealthy clean-label attacks without label flips or extreme feature density anomalies may evade distance-based and isolation forest filters under conservative false-positive constraints.",
            mitigation_recommendation:
              "Deploy differential victim vs reference loss tracking and Mixup defensive retraining for high-stakes models.",
          },
        ];

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
      <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/40">
        <div className="flex items-center gap-2">
          <ShieldX className="w-4 h-4 text-amber-700" />
          <h3 className="text-sm font-semibold text-slate-900">
            Threat Limitations & Defense Boundaries
          </h3>
        </div>
        <span className="text-[11px] font-medium text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
          Scientific Transparency Disclosure
        </span>
      </div>

      <div className="p-4 space-y-3 text-xs">
        <p className="text-slate-600 leading-relaxed text-[11px]">
          No defense platform guarantees 100% detection against all poisoning paradigms. DataSentinel enforces transparent boundaries: a <strong>LOW_RISK</strong> rating reflects evidence coverage across evaluated detectors, not an absolute guarantee of dataset purity.
        </p>

        <div className="space-y-2.5">
          {displayLimitations.map((lim, idx) => (
            <div
              key={idx}
              className="p-3 bg-slate-50 rounded-md border border-slate-200/90 space-y-1.5"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="font-mono font-semibold text-slate-800 text-xs">
                  {lim.threat_family}
                </span>
                <span className="font-mono text-[10px] text-amber-800 bg-amber-50 px-1.5 py-0.5 rounded border border-amber-200 font-medium">
                  {lim.status}
                </span>
              </div>

              <p className="text-[11px] text-slate-600 leading-relaxed">
                {lim.statement || lim.limitation_statement}
              </p>

              {(lim.observed_recall || lim.operating_fpr) && (
                <div className="flex flex-wrap items-center gap-3 text-[10px] text-slate-500 pt-0.5">
                  {lim.observed_recall && (
                    <span>
                      <strong>Observed Test Recall:</strong> {lim.observed_recall}
                    </span>
                  )}
                  {lim.operating_fpr && (
                    <span>
                      <strong>Operating Constraint:</strong> {lim.operating_fpr}
                    </span>
                  )}
                </div>
              )}

              {lim.mitigation_recommendation && (
                <div className="pt-1 text-[11px] text-slate-500 flex items-start gap-1.5">
                  <Info className="w-3.5 h-3.5 text-indigo-600 shrink-0 mt-0.5" />
                  <span>
                    <strong>Mitigation Guidance:</strong> {lim.mitigation_recommendation}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
