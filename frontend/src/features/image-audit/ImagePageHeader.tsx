import React, { useState } from "react";
import { Sparkles, HelpCircle, X, Shield, Info, AlertTriangle, ScanEye } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface ImagePageHeaderProps {
  onLoadSampleBenchmark: () => void;
  isLoadingBenchmark: boolean;
  currentDatasetLabel?: string;
}

export const ImagePageHeader: React.FC<ImagePageHeaderProps> = ({
  onLoadSampleBenchmark,
  isLoadingBenchmark,
  currentDatasetLabel,
}) => {
  const [showHelpModal, setShowHelpModal] = useState(false);

  return (
    <>
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-1 border-b border-slate-200">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-slate-900">
              Deep Learning
            </h1>
            <span className="text-xs px-2 py-0.5 font-medium rounded bg-indigo-50 text-indigo-800 border border-indigo-200">
              Computer Vision Workspace
            </span>
            {currentDatasetLabel && (
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200 truncate max-w-xs">
                {currentDatasetLabel}
              </span>
            )}
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Image dataset security, backdoor screening, and vision evidence analysis
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onLoadSampleBenchmark}
            disabled={isLoadingBenchmark}
            className="text-xs h-8 border-slate-300 hover:bg-slate-50"
            icon={<Sparkles className="w-3.5 h-3.5 text-indigo-600" />}
          >
            {isLoadingBenchmark ? "Loading Benchmark..." : "Load Sample Audit"}
          </Button>

          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setShowHelpModal(true)}
            className="text-xs h-8 text-slate-600 hover:text-slate-900 px-2"
            title="Vision Signals & Methodology"
          >
            <HelpCircle className="w-4 h-4 mr-1 text-slate-400" />
            <span>About Vision Signals</span>
          </Button>
        </div>
      </div>

      {/* Vision Methodology Help Modal */}
      {showHelpModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="vision-help-title"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs"
        >
          <div className="bg-white rounded-lg border border-slate-200 max-w-xl w-full shadow-lg p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <ScanEye className="w-4 h-4 text-indigo-600" />
                <h2 id="vision-help-title" className="text-sm font-semibold text-slate-900">
                  Vision Deep Learning Audit Methodology
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setShowHelpModal(false)}
                className="text-slate-400 hover:text-slate-600 p-1 rounded"
                aria-label="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="text-xs text-slate-600 space-y-3 leading-relaxed max-h-[60vh] overflow-y-auto pr-1">
              <div className="p-3 bg-slate-50 rounded border border-slate-200">
                <h3 className="font-semibold text-slate-800 flex items-center gap-1.5 mb-1">
                  <Shield className="w-3.5 h-3.5 text-indigo-600" />
                  14 Vision Detector Signals
                </h3>
                <p>
                  Evaluates four diagnostic groups: Label Consistency (out-of-fold loss, entropy, margin), Deep Representations (512D ResNet18 embeddings, LOF, activation clustering, spectral signature), Image Integrity (exact MD5, 64-bit dHash perceptual collision, RGB moments), and Model Behavior (patch occlusion sensitivity).
                </p>
              </div>

              <div>
                <h3 className="font-semibold text-slate-800 mb-1">
                  Decoupled Model Utility vs ASR
                </h3>
                <p>
                  DataSentinel strictly separates <strong>Clean Accuracy</strong> (model utility on pristine data) from <strong>Attack Success Rate (ASR)</strong> (backdoor trigger execution). Staged quarantine screening aims to collapse ASR without sacrificing clean test accuracy.
                </p>
              </div>

              <div className="p-3 bg-amber-50 rounded border border-amber-200 text-amber-900">
                <h3 className="font-semibold flex items-center gap-1.5 mb-1">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-700" />
                  Clean-Label Defense Boundary
                </h3>
                <p>
                  Stealthy clean-label centroid blending exhibits high-frequency spatial attenuation under 2D FFT, but remains an unresolved detector limitation at operational false-positive ceilings (0/12 recall under FPR ≤ 5%). Transparent disclosure is enforced.
                </p>
              </div>

              <div className="p-3 bg-slate-50 rounded border border-slate-200 flex items-start gap-1.5 text-slate-600">
                <Info className="w-3.5 h-3.5 text-indigo-600 shrink-0 mt-0.5" />
                <p>
                  The risk score is a composite evidence index (0–100), not a probability of purity.
                </p>
              </div>
            </div>

            <div className="flex justify-end pt-2 border-t border-slate-100">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowHelpModal(false)}
              >
                Close
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
