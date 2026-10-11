import React from "react";
import { TrendingUp, BarChart2, Info, ArrowUpRight, ArrowDownRight, ShieldCheck } from "lucide-react";
import { formatAccuracy } from "@/lib/utils";
import type { ModelUtilityComparison } from "@/types/audit";

interface VisionModelUtilityProps {
  utility: ModelUtilityComparison | null | undefined;
}

export const VisionModelUtility: React.FC<VisionModelUtilityProps> = ({ utility }) => {
  if (!utility) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs text-xs text-slate-500">
        <div className="flex items-center gap-2 mb-1">
          <BarChart2 className="w-4 h-4 text-slate-400" />
          <h3 className="font-semibold text-slate-700">Downstream Vision Model Utility</h3>
        </div>
        <p className="text-[11px] text-slate-400 italic">
          Model utility benchmarks were not evaluated for this image partition.
        </p>
      </div>
    );
  }

  // Pre & post clean accuracy
  const preAcc = utility.clean_accuracy_pre ?? utility.victim_clean_accuracy;
  const postAcc = utility.clean_accuracy_post ?? utility.screened_clean_accuracy;
  const refAcc = utility.clean_reference_accuracy;

  // Percentage point delta
  const accDeltaPp =
    utility.clean_accuracy_delta_pp ??
    (postAcc !== null && preAcc !== null && postAcc !== undefined && preAcc !== undefined
      ? postAcc - preAcc
      : null);

  // Pre & post ASR
  const preAsr = utility.asr_pre ?? utility.victim_asr;
  const postAsr = utility.asr_post ?? utility.screened_asr;
  const hasAsr = preAsr !== null && preAsr !== undefined;
  const asrDeltaPp =
    utility.asr_reduction_pp ??
    (preAsr !== null && postAsr !== null && preAsr !== undefined && postAsr !== undefined
      ? preAsr - postAsr
      : null);

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
      <div className="p-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-indigo-700" />
          <h3 className="text-sm font-semibold text-slate-900">
            Downstream ResNet18 Model Utility & ASR Mitigation
          </h3>
        </div>
        <span className="text-[11px] text-slate-500">
          Evaluated on {utility.evaluation_dataset}
        </span>
      </div>

      <div className="p-4 space-y-4 text-xs">
        {/* Metric Comparison Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* Pre-Screening Accuracy */}
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <span className="text-[10px] text-slate-500 block">Victim Clean Accuracy (Pre)</span>
            <span className="font-mono font-bold text-slate-900 text-base">
              {formatAccuracy(preAcc)}
            </span>
            <span className="text-[10px] text-slate-400 block mt-0.5">
              Trained on contaminated data
            </span>
          </div>

          {/* Post-Screening Accuracy */}
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <span className="text-[10px] text-slate-500 block">Screened Clean Accuracy (Post)</span>
            <span className="font-mono font-bold text-slate-900 text-base">
              {formatAccuracy(postAcc)}
            </span>
            <span className="text-[10px] text-slate-400 block mt-0.5">
              Trained on screened dataset
            </span>
          </div>

          {/* Clean Reference Accuracy */}
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <span className="text-[10px] text-slate-500 block">Clean Reference Baseline</span>
            <span className="font-mono font-bold text-slate-900 text-base">
              {formatAccuracy(refAcc)}
            </span>
            <span className="text-[10px] text-slate-400 block mt-0.5">
              Pristine unpoisoned model
            </span>
          </div>
        </div>

        {/* Attack Success Rate (ASR) Section (If Evaluated) */}
        {hasAsr ? (
          <div className="p-3.5 bg-indigo-50/40 rounded-lg border border-indigo-200 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-indigo-950 text-xs flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-indigo-700" />
                Attack Success Rate (ASR) Mitigation Impact
              </span>
              {asrDeltaPp !== null && (
                <span className="font-mono font-bold text-xs text-emerald-800 bg-emerald-100/80 px-2 py-0.5 rounded border border-emerald-300">
                  ASR Reduced by {asrDeltaPp.toFixed(2)} pp
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 pt-1">
              <div className="p-2 bg-white rounded border border-indigo-100 text-slate-700">
                <span className="text-[10px] text-slate-500 block">Pre-Screening Victim ASR</span>
                <span className="font-mono font-bold text-rose-700 text-sm">
                  {formatAccuracy(preAsr)}
                </span>
                <span className="text-[10px] text-slate-400 block mt-0.5">Vulnerable trigger rate</span>
              </div>

              <div className="p-2 bg-white rounded border border-indigo-100 text-slate-700">
                <span className="text-[10px] text-slate-500 block">Post-Screening Retrained ASR</span>
                <span className="font-mono font-bold text-emerald-700 text-sm">
                  {formatAccuracy(postAsr)}
                </span>
                <span className="text-[10px] text-slate-400 block mt-0.5">Screened model resistance</span>
              </div>

              <div className="p-2 bg-white rounded border border-indigo-100 text-slate-700 sm:col-span-1 col-span-2">
                <span className="text-[10px] text-slate-500 block">Total ASR Reduction</span>
                <span className="font-mono font-bold text-emerald-800 text-sm flex items-center">
                  <ArrowDownRight className="w-3.5 h-3.5 mr-0.5" />
                  -{asrDeltaPp?.toFixed(2)} pp
                </span>
                <span className="text-[10px] text-slate-400 block mt-0.5">Percentage-point drop</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-3 bg-slate-50 rounded border border-slate-200 text-slate-500 flex items-center justify-between">
            <span className="font-medium text-[11px]">Attack Success Rate (ASR):</span>
            <span className="font-mono text-[11px] bg-slate-100 px-2 py-0.5 rounded text-slate-600">
              Not evaluated (Clean-label scenario without patch trigger)
            </span>
          </div>
        )}

        {/* Accuracy Delta Callout */}
        {accDeltaPp !== null && (
          <div className="flex items-center gap-2 p-2.5 bg-slate-50/80 rounded border border-slate-100 text-slate-700">
            <span className="font-medium text-[11px] text-slate-600">Clean Accuracy Impact:</span>
            <span
              className={`font-mono font-semibold text-xs flex items-center ${
                accDeltaPp >= 0 ? "text-emerald-700" : "text-slate-700"
              }`}
            >
              {accDeltaPp >= 0 ? (
                <ArrowUpRight className="w-3.5 h-3.5 inline mr-0.5" />
              ) : (
                <ArrowDownRight className="w-3.5 h-3.5 inline mr-0.5" />
              )}
              {accDeltaPp >= 0 ? `+${accDeltaPp.toFixed(2)}` : accDeltaPp.toFixed(2)} pp
            </span>
            <span className="text-[10px] text-slate-400">(percentage points)</span>
          </div>
        )}

        {/* Evaluation Notes */}
        {utility.evaluation_notes && (
          <div className="flex items-start gap-2 text-[11px] text-slate-500 pt-1">
            <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
            <span>
              <strong>Evaluation Framework:</strong> {utility.evaluation_notes}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
