import React from "react";
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Info,
  Clock,
  Sparkles,
} from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { formatPercent } from "@/lib/utils";
import type { UnifiedAuditResult } from "@/types/audit";

interface VisionAuditSummaryProps {
  audit: UnifiedAuditResult;
  isDemo?: boolean;
}

export const VisionAuditSummary: React.FC<VisionAuditSummaryProps> = ({
  audit,
  isDemo = false,
}) => {
  const hasLimitations =
    Array.isArray(audit.unresolved_threat_limitations) &&
    audit.unresolved_threat_limitations.length > 0;

  const getRiskBadge = (level: string) => {
    switch (level) {
      case "CRITICAL_RISK":
        return <Badge variant="danger">CRITICAL RISK</Badge>;
      case "ELEVATED_RISK":
      case "HIGH_RISK":
        return <Badge variant="danger">HIGH RISK</Badge>;
      case "MODERATE_RISK":
        return <Badge variant="warning">MODERATE RISK</Badge>;
      case "LOW_RISK":
      default:
        return <Badge variant="success">LOW RISK</Badge>;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "SEVERELY_POISONED":
      case "POTENTIALLY_COMPROMISED":
        return <Badge variant="danger">{status}</Badge>;
      case "SUSPICIOUS":
        return <Badge variant="warning">{status}</Badge>;
      case "QUARANTINE_CONFIRMED":
        return <Badge variant="info">QUARANTINE CONFIRMED</Badge>;
      case "AUDITED_SECURE":
      case "TRUSTED":
      default:
        return (
          <Badge variant="neutral">
            {status}
            {hasLimitations && " (BOUNDED)"}
          </Badge>
        );
    }
  };

  return (
    <div className="space-y-3">
      {/* Demo Banner */}
      {isDemo && (
        <div className="bg-amber-50/90 border border-amber-200 text-amber-900 rounded-md px-3.5 py-2 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-600 shrink-0" />
            <span>
              <strong>DEMO — Benchmark Result:</strong> Displaying pre-computed deep learning audit evidence for <em>{audit.dataset_name}</em>.
            </span>
          </div>
          <span className="text-[10px] font-mono bg-white text-amber-800 px-2 py-0.5 rounded border border-amber-200 shrink-0">
            Scenario ID: {audit.audit_id}
          </span>
        </div>
      )}

      {/* Main Analytics Summary Strip */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono text-xs font-bold text-slate-900 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                {audit.audit_id}
              </span>
              <h2 className="text-sm font-bold text-slate-900">
                {audit.dataset_name}
              </h2>
              <Badge variant="info" size="sm">
                {audit.modality}
              </Badge>
              {getStatusBadge(audit.overall_status)}
            </div>

            <div className="flex items-center gap-4 text-[11px] text-slate-500 pt-0.5">
              <span>
                <strong>Population:</strong> {audit.n_rows.toLocaleString()} evaluation images × 32×32×3
              </span>
              {audit.created_at && (
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3 text-slate-400" />
                  {new Date(audit.created_at).toLocaleDateString()} {new Date(audit.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              )}
              {audit.dataset_fingerprint && (
                <span className="font-mono text-[10px] text-slate-400 truncate max-w-xs" title={audit.dataset_fingerprint}>
                  Fingerprint: {audit.dataset_fingerprint.slice(0, 12)}…
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-3">
            {getRiskBadge(audit.risk_level)}
          </div>
        </div>

        {/* Dense KPI Grid with Subtle Pastel Tinting */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4">
          {/* Risk Index */}
          <div className="p-3 bg-purple-50/40 rounded border border-purple-200/80">
            <div className="flex items-center justify-between text-purple-900 text-[11px]">
              <span className="font-medium">Evidence Risk Score</span>
              <ShieldAlert className="w-3.5 h-3.5 text-purple-600" />
            </div>
            <div className="text-xl font-bold font-mono text-slate-900 mt-1">
              {audit.risk_score.toFixed(2)}
              <span className="text-xs text-slate-400 font-normal"> / 100</span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 leading-tight">
              Evidence-weighted index (not a safety probability)
            </p>
          </div>

          {/* Trust Index */}
          <div className="p-3 bg-teal-50/40 rounded border border-teal-200/80">
            <div className="flex items-center justify-between text-teal-900 text-[11px]">
              <span className="font-medium">Data Trust Score</span>
              <ShieldCheck className="w-3.5 h-3.5 text-teal-600" />
            </div>
            <div className="text-xl font-bold font-mono text-slate-900 mt-1">
              {audit.trust_score.toFixed(2)}
              <span className="text-xs text-slate-400 font-normal"> / 100</span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 leading-tight">
              Calculated as 100.0 − Risk Index
            </p>
          </div>

          {/* Evidence Coverage */}
          <div className="p-3 bg-blue-50/40 rounded border border-blue-200/80">
            <div className="flex items-center justify-between text-blue-900 text-[11px]">
              <span className="font-medium">Evidence Coverage</span>
              <Info className="w-3.5 h-3.5 text-blue-600" />
            </div>
            <div className="text-xl font-bold font-mono text-slate-900 mt-1">
              {formatPercent(audit.evidence_coverage)}
            </div>
            <p className="text-[10px] text-slate-500 mt-1 leading-tight">
              {audit.available_signals.length} of {audit.available_signals.length + audit.missing_signals.length} signals active (14 total)
            </p>
          </div>

          {/* Staged Quarantine Count */}
          <div className="p-3 bg-amber-50/40 rounded border border-amber-200/80">
            <div className="flex items-center justify-between text-amber-900 text-[11px]">
              <span className="font-medium">Quarantine Candidates</span>
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
            </div>
            <div className="text-xl font-bold font-mono text-slate-900 mt-1">
              {audit.quarantine_manifest?.quarantined_count ?? 0}
              <span className="text-xs text-slate-400 font-normal"> images</span>
            </div>
            <p className="text-[10px] text-slate-500 mt-1 leading-tight">
              {audit.quarantine_manifest?.confirmation_status === "CONFIRMED"
                ? "Quarantine confirmed by operator"
                : "Staged for non-destructive review"}
            </p>
          </div>
        </div>

        {/* Policy Guidance Callout */}
        <div className="mt-4 pt-3 border-t border-slate-100 flex items-start gap-2 text-xs text-slate-600 bg-slate-50/50 p-2.5 rounded">
          <Info className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
          <div className="text-[11px] leading-relaxed">
            <strong>Recommended Policy Action:</strong> {audit.recommended_action}
            {hasLimitations && (
              <span className="block text-amber-800 font-medium mt-0.5">
                Note: A low risk score does not prove the absence of clean-label centroid blends. Refer to the Threat Limitations disclosure below.
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
