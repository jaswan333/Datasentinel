import React from "react";
import { TrendingUp, BarChart2, Info, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { formatAccuracy } from "@/lib/utils";
import type { ModelUtilityComparison } from "@/types/audit";

interface ModelUtilityCardProps {
  utility: ModelUtilityComparison | null | undefined;
}

export const ModelUtilityCard: React.FC<ModelUtilityCardProps> = ({ utility }) => {
  if (!utility) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs text-xs text-slate-500">
        <div className="flex items-center gap-2 mb-1">
          <BarChart2 className="w-4 h-4 text-slate-400" />
          <h3 className="font-semibold text-slate-700">Downstream Model Utility</h3>
        </div>
        <p className="text-[11px] text-slate-400 italic">
          Model utility benchmarks were not evaluated for this audit partition.
        </p>
      </div>
    );
  }

  // Pre & post accuracy
  const preAcc = utility.clean_accuracy_pre ?? utility.victim_clean_accuracy;
  const postAcc = utility.clean_accuracy_post ?? utility.screened_clean_accuracy;
  const refAcc = utility.clean_reference_accuracy;

  // Percentage point delta (NOT relative percentage change)
  const accDeltaPp =
    utility.clean_accuracy_delta_pp ??
    (postAcc !== null && preAcc !== null && postAcc !== undefined && preAcc !== undefined
      ? postAcc - preAcc
      : null);

  // Pre & post ASR
  const preAsr = utility.asr_pre ?? utility.victim_asr;
  const postAsr = utility.asr_post ?? utility.screened_asr;
  const asrDeltaPp =
    utility.asr_reduction_pp ??
    (preAsr !== null && postAsr !== null && preAsr !== undefined && postAsr !== undefined
      ? preAsr - postAsr
      : null);

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
      <div className="p-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-slate-700" />
          <h3 className="text-sm font-semibold text-slate-900">
            Downstream Model Utility & Screening Impact
          </h3>
        </div>
        <span className="text-[11px] text-slate-500">
          Evaluated on {utility.evaluation_dataset}
        </span>
      </div>

      <div className="p-4 space-y-4 text-xs">
        {/* Metric Comparison Grid */}
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
              Trained on quarantined data
            </span>
          </div>

          {/* Clean Reference Accuracy */}
          <div className="p-3 bg-slate-50 rounded border border-slate-200">
            <span className="text-[10px] text-slate-500 block">Clean Reference Baseline</span>
            <span className="font-mono font-bold text-slate-900 text-base">
              {formatAccuracy(refAcc)}
            </span>
            <span className="text-[10px] text-slate-400 block mt-0.5">
              Ideal pristine reference
            </span>
          </div>
        </div>

        {/* Delta Callout */}
        <div className="flex flex-wrap items-center gap-4 p-3 bg-slate-50/70 rounded border border-slate-100 text-slate-700">
          {accDeltaPp !== null && (
            <div className="flex items-center gap-1.5">
              <span className="font-medium text-[11px] text-slate-600">Clean Accuracy Impact:</span>
              <span
                className={`font-mono font-semibold text-xs flex items-center ${
                  accDeltaPp >= 0 ? "text-emerald-700" : "text-rose-700"
                }`}
              >
                {accDeltaPp >= 0 ? (
                  <ArrowUpRight className="w-3.5 h-3.5 inline" />
                ) : (
                  <ArrowDownRight className="w-3.5 h-3.5 inline" />
                )}
                {accDeltaPp >= 0 ? `+${accDeltaPp.toFixed(2)}` : accDeltaPp.toFixed(2)} pp
              </span>
              <span className="text-[10px] text-slate-400">(percentage points)</span>
            </div>
          )}

          {asrDeltaPp !== null && (
            <div className="flex items-center gap-1.5 border-l border-slate-200 pl-4">
              <span className="font-medium text-[11px] text-slate-600">Attack Success Rate (ASR) Reduction:</span>
              <span className="font-mono font-semibold text-xs text-emerald-700 flex items-center">
                <ArrowDownRight className="w-3.5 h-3.5 inline" />
                -{asrDeltaPp.toFixed(2)} pp
              </span>
            </div>
          )}
        </div>

        {/* Evaluation Notes */}
        {utility.evaluation_notes && (
          <div className="flex items-start gap-2 text-[11px] text-slate-500">
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
