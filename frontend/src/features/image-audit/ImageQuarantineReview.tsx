import React, { useState } from "react";
import {
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Lock,
  UserCheck,
  Loader2,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import type { QuarantineManifest } from "@/types/audit";

interface ImageQuarantineReviewProps {
  manifest: QuarantineManifest;
  datasetId: string;
  onConfirmQuarantine: (notes: string) => Promise<void>;
  isConfirming: boolean;
}

export const ImageQuarantineReview: React.FC<ImageQuarantineReviewProps> = ({
  manifest,
  datasetId,
  onConfirmQuarantine,
  isConfirming,
}) => {
  const [operatorNotes, setOperatorNotes] = useState("");
  const [showSignOffModal, setShowSignOffModal] = useState(false);

  const sampleIdCount = Array.isArray(manifest.quarantined_sample_ids)
    ? manifest.quarantined_sample_ids.length
    : 0;
  const declaredCount = manifest.quarantined_count ?? 0;

  // Data Integrity Disagreement Check: count vs ID list mismatch
  const hasCountMismatch = declaredCount > 0 && sampleIdCount !== declaredCount;
  const isConfirmed = manifest.confirmation_status === "CONFIRMED";

  const handleConfirmSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onConfirmQuarantine(operatorNotes || "Vision operator sign-off verified via DataSentinel");
    setShowSignOffModal(false);
  };

  return (
    <>
      <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-amber-50/20">
          <div className="flex items-center gap-2">
            <Lock className="w-4 h-4 text-slate-700" />
            <h3 className="text-sm font-semibold text-slate-900">
              Vision Quarantine Manifest & Staged Isolation
            </h3>
            {isConfirmed ? (
              <Badge variant="success" size="sm">
                CONFIRMED BY OPERATOR
              </Badge>
            ) : (
              <Badge variant="warning" size="sm">
                PENDING OPERATOR REVIEW
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-2">
            {!isConfirmed && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setShowSignOffModal(true)}
                disabled={isConfirming || declaredCount === 0}
                className="text-xs h-7 text-indigo-700 border-indigo-200 hover:bg-indigo-50"
                icon={<UserCheck className="w-3.5 h-3.5" />}
              >
                Sign Off & Authorize Screened Retraining
              </Button>
            )}
          </div>
        </div>

        <div className="p-4 space-y-4 text-xs">
          {/* Data Integrity Warning Banner if Mismatch Exists */}
          {hasCountMismatch && (
            <div className="p-3 bg-amber-50 border border-amber-300 rounded-md text-amber-900 flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="font-semibold text-xs text-amber-950">
                  Data-Integrity Notice: Candidate Count & ID Discrepancy
                </p>
                <p className="text-[11px] leading-relaxed">
                  The manifest reports <strong>{declaredCount}</strong> quarantine candidates, but the associated candidate identifier list contains <strong>{sampleIdCount}</strong> sample IDs. The platform does not silently reconcile these values. Full individual identifiers require sample evidence export before destructive removal.
                </p>
              </div>
            </div>
          )}

          {/* Policy & Screening Specs Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 block">Candidate Count</span>
              <span className="font-mono font-bold text-slate-900 text-sm">
                {declaredCount}
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                of {manifest.total_candidate_samples} images
              </span>
            </div>

            <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 block">Operating Threshold</span>
              <span className="font-mono font-bold text-slate-900 text-sm">
                {typeof manifest.operating_threshold === "number"
                  ? manifest.operating_threshold.toFixed(4)
                  : String(manifest.operating_threshold)}
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                Decision boundary
              </span>
            </div>

            <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 block">Quarantine Rate</span>
              <span className="font-mono font-bold text-slate-900 text-sm">
                {(manifest.quarantine_rate * 100).toFixed(2)}%
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                Pool proportion
              </span>
            </div>

            <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
              <span className="text-[10px] text-slate-500 block">Human Review Requirement</span>
              <span className="font-semibold text-slate-900 text-xs flex items-center gap-1 mt-0.5">
                <ShieldAlert className="w-3.5 h-3.5 text-indigo-600" />
                {manifest.requires_human_review ? "Mandatory" : "Optional"}
              </span>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                Non-destructive guarantee
              </span>
            </div>
          </div>

          {/* Threshold Policy Description */}
          <div className="p-3 bg-slate-50/70 rounded border border-slate-100 flex items-start gap-2">
            <FileText className="w-3.5 h-3.5 text-slate-500 shrink-0 mt-0.5" />
            <div className="text-[11px] text-slate-600">
              <span className="font-semibold text-slate-800">Calibrated Policy Rule:</span>{" "}
              {manifest.threshold_policy || "maximize recall subject to validation FPR <= 0.05"}
            </div>
          </div>

          {/* Staged Candidate IDs List */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-800 text-[11px]">
                Staged Candidate Image Manifest ({sampleIdCount} IDs)
              </span>
              <span className="text-[10px] text-slate-400">
                Non-destructive staging area
              </span>
            </div>

            {sampleIdCount === 0 ? (
              <p className="text-[11px] text-slate-400 italic p-2.5 bg-slate-50 rounded border border-slate-100">
                No individual sample IDs populated in this manifest partition. Refer to the suspicious image explorer above.
              </p>
            ) : (
              <div className="p-2.5 bg-slate-50 rounded border border-slate-200 max-h-24 overflow-y-auto flex flex-wrap gap-1">
                {manifest.quarantined_sample_ids.map((id) => (
                  <span
                    key={String(id)}
                    className="font-mono text-[10px] px-1.5 py-0.5 bg-white border border-slate-300 rounded text-slate-800"
                  >
                    Image #{id}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Confirmed Notice */}
          {isConfirmed && (
            <div className="p-3 bg-emerald-50 border border-emerald-200 rounded text-emerald-900 flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <div className="text-[11px]">
                <p className="font-semibold">Operator Sign-Off Confirmed</p>
                <p className="mt-0.5">
                  Notes: {manifest.operator_notes || "Verified by vision operator"}
                </p>
                {manifest.confirmed_at && (
                  <p className="text-[10px] text-emerald-700 mt-0.5 font-mono">
                    Confirmed at: {manifest.confirmed_at}
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Operator Sign-Off Modal */}
      {showSignOffModal && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs"
        >
          <div className="bg-white rounded-lg border border-slate-200 max-w-md w-full shadow-lg p-5 space-y-4">
            <div className="border-b border-slate-100 pb-3">
              <h3 className="text-sm font-semibold text-slate-900">
                Operator Sign-Off: Authorize Screened Retraining
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Confirm non-destructive isolation of {declaredCount} candidate images for dataset {datasetId}.
              </p>
            </div>

            <form onSubmit={handleConfirmSubmit} className="space-y-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Operator Review Notes
                </label>
                <textarea
                  rows={3}
                  value={operatorNotes}
                  onChange={(e) => setOperatorNotes(e.target.value)}
                  placeholder="e.g., Reviewed high-suspicion backdoor patches; approved for non-destructive exclusion before retraining ResNet18."
                  className="w-full text-xs p-2.5 border border-slate-300 rounded focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              </div>

              <div className="p-2.5 bg-slate-50 rounded border border-slate-200 text-[11px] text-slate-600">
                DataSentinel strictly adheres to non-destructive screening. Original image collections remain uncorrupted; screened partitions are generated separately.
              </div>

              <div className="flex justify-end gap-2 pt-2 border-t border-slate-100">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setShowSignOffModal(false)}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="primary"
                  size="sm"
                  disabled={isConfirming}
                  className="bg-indigo-700 hover:bg-indigo-800 text-white"
                  icon={
                    isConfirming ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : undefined
                  }
                >
                  {isConfirming ? "Confirming..." : "Confirm Quarantine"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};
