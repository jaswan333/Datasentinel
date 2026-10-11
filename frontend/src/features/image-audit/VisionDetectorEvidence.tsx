import React, { useState } from "react";
import {
  ScanEye,
  CheckCircle,
  XCircle,
  AlertCircle,
  HelpCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { formatPercent, cn } from "@/lib/utils";
import type { SignalDetail } from "@/types/audit";

interface VisionDetectorEvidenceProps {
  signals: Record<string, SignalDetail>;
}

export const VisionDetectorEvidence: React.FC<VisionDetectorEvidenceProps> = ({
  signals,
}) => {
  const [expandedSignal, setExpandedSignal] = useState<string | null>(null);

  const signalList = Object.entries(signals || {});

  const renderStatusBadge = (status: string) => {
    switch (status) {
      case "AVAILABLE":
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
            <CheckCircle className="w-3 h-3 text-emerald-600" />
            AVAILABLE
          </span>
        );
      case "UNAVAILABLE":
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
            <XCircle className="w-3 h-3 text-slate-400" />
            UNAVAILABLE
          </span>
        );
      case "ERROR":
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-medium text-rose-800 bg-rose-50 px-2 py-0.5 rounded border border-rose-200">
            <AlertCircle className="w-3 h-3 text-rose-600" />
            ERROR
          </span>
        );
    }
  };

  const formatFraction = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return "N/A";
    if (typeof val === "object") return "N/A";
    return formatPercent(val, 1);
  };

  const formatMeanRisk = (val: number | null | undefined): string => {
    if (val === null || val === undefined) return "N/A";
    if (typeof val === "object") return "N/A";
    return Number(val).toFixed(3);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
      <div className="p-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ScanEye className="w-4 h-4 text-slate-700" />
          <h3 className="text-sm font-semibold text-slate-900">
            Vision Detector Signals & Latent Evidence
          </h3>
          <span className="text-xs text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
            {signalList.length} detectors active
          </span>
        </div>
        <p className="text-[11px] text-slate-400 hidden sm:block">
          14 ResNet18 representations across label consistency, latent geometry, image statistics, and patch sensitivity
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50/70 text-slate-600 font-semibold text-[11px]">
              <th className="py-2.5 px-3.5">Signal Identifier</th>
              <th className="py-2.5 px-3.5">Diagnostic Domain</th>
              <th className="py-2.5 px-3.5">Status</th>
              <th className="py-2.5 px-3.5">Policy Weight</th>
              <th className="py-2.5 px-3.5">High-Risk Fraction</th>
              <th className="py-2.5 px-3.5">Mean Risk</th>
              <th className="py-2.5 px-3.5 text-right">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-700">
            {signalList.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-slate-400">
                  No vision detector signals available. Load a sample benchmark or submit image data.
                </td>
              </tr>
            ) : (
              signalList.map(([key, signal]) => {
                const isExpanded = expandedSignal === key;
                const isAvailable = signal.status === "AVAILABLE";

                return (
                  <React.Fragment key={key}>
                    <tr
                      className={cn(
                        "hover:bg-slate-50/60 transition-colors",
                        isExpanded && "bg-slate-50/80"
                      )}
                    >
                      <td className="py-2.5 px-3.5 font-mono font-medium text-slate-900 text-xs">
                        {signal.name || key}
                      </td>

                      <td className="py-2.5 px-3.5">
                        <Badge variant="neutral" size="sm">
                          {signal.category || "Vision"}
                        </Badge>
                      </td>

                      <td className="py-2.5 px-3.5">
                        {renderStatusBadge(signal.status)}
                      </td>

                      <td className="py-2.5 px-3.5 font-mono text-slate-600">
                        {typeof signal.weight === "number" ? signal.weight.toFixed(2) : "N/A"}
                      </td>

                      <td className="py-2.5 px-3.5 font-mono">
                        {isAvailable ? (
                          <span
                            className={cn(
                              signal.high_risk_fraction && signal.high_risk_fraction > 0.05
                                ? "text-amber-700 font-semibold"
                                : "text-slate-700"
                            )}
                          >
                            {formatFraction(signal.high_risk_fraction)}
                          </span>
                        ) : (
                          <span className="text-slate-400">N/A</span>
                        )}
                      </td>

                      <td className="py-2.5 px-3.5 font-mono">
                        {isAvailable ? (
                          <span>{formatMeanRisk(signal.mean_risk)}</span>
                        ) : (
                          <span className="text-slate-400">N/A</span>
                        )}
                      </td>

                      <td className="py-2.5 px-3.5 text-right">
                        <button
                          type="button"
                          onClick={() => setExpandedSignal(isExpanded ? null : key)}
                          className="text-slate-400 hover:text-slate-700 p-1 rounded inline-flex items-center gap-1 text-[11px]"
                          aria-label={`Toggle details for ${signal.name || key}`}
                        >
                          <HelpCircle className="w-3.5 h-3.5" />
                          {isExpanded ? (
                            <ChevronUp className="w-3 h-3" />
                          ) : (
                            <ChevronDown className="w-3 h-3" />
                          )}
                        </button>
                      </td>
                    </tr>

                    {/* Expandable Technical Details */}
                    {isExpanded && (
                      <tr className="bg-slate-50/90 border-b border-slate-200">
                        <td colSpan={7} className="p-3.5 pl-6 text-xs text-slate-600">
                          <div className="max-w-2xl space-y-1">
                            <p className="font-semibold text-slate-800 text-[11px]">
                              Vision Detector Mechanics & Diagnostic Rationale:
                            </p>
                            <p className="text-[11px] leading-relaxed">
                              {signal.description ||
                                "Extracts deep latent anomalies, activation clustering silhouettes, or high-frequency spectral signatures."}
                            </p>
                            <div className="flex items-center gap-4 text-[10px] text-slate-400 pt-1">
                              <span>Raw Status: {signal.status}</span>
                              <span>Policy Weight: {signal.weight}</span>
                              {signal.mean_risk !== null && (
                                <span>Normalized Mean: {signal.mean_risk}</span>
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
