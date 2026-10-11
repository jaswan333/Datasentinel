import React, { useState } from "react";
import { Sparkles, HelpCircle, X, Shield, Info, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface PageHeaderProps {
  onLoadSampleBenchmark: () => void;
  isLoadingBenchmark: boolean;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  onLoadSampleBenchmark,
  isLoadingBenchmark,
}) => {
  const [showHelpModal, setShowHelpModal] = useState(false);

  return (
    <>
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-1 border-b border-slate-200">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-slate-900">
              Machine Learning
            </h1>
            <span className="text-xs px-2 py-0.5 font-medium rounded bg-slate-100 text-slate-700 border border-slate-200">
              Tabular Workspace
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Tabular dataset security, poisoning screening, and evidence-weighted risk analysis
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
            title="Audit Methodology & Help"
          >
            <HelpCircle className="w-4 h-4 mr-1 text-slate-400" />
            <span>Methodology</span>
          </Button>
        </div>
      </div>

      {/* Contextual Help Modal */}
      {showHelpModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="help-title"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs"
        >
          <div className="bg-white rounded-lg border border-slate-200 max-w-xl w-full shadow-lg p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-indigo-600" />
                <h2 id="help-title" className="text-sm font-semibold text-slate-900">
                  Tabular Audit Methodology & Interpretation
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
                  <Info className="w-3.5 h-3.5 text-indigo-600" />
                  Evidence Index vs Safety Probability
                </h3>
                <p>
                  The risk score (0–100) is a normalized, evidence-weighted composite index derived from available detector signals. It is <strong>not</strong> a calibrated probability of poisoning. A low risk score does not prove the absence of sophisticated attacks (such as clean-label feature blending).
                </p>
              </div>

              <div>
                <h3 className="font-semibold text-slate-800 mb-1">
                  Signal Extraction & Coverage
                </h3>
                <p>
                  Tabular audits evaluate up to 10 statistical and algorithmic signals across label consistency (Cleanlab, Out-of-fold disagreement), feature density (Isolation Forest, LOF), duplicate proximity, and trigger rarity. Evidence coverage indicates what fraction of configured detector weight successfully executed.
                </p>
              </div>

              <div>
                <h3 className="font-semibold text-slate-800 mb-1">
                  Non-Destructive Quarantine
                </h3>
                <p>
                  Flagged samples enter a candidate quarantine pool. Records are never deleted or permanently modified without explicit operator sign-off and review of the quarantine manifest.
                </p>
              </div>

              <div className="p-3 bg-amber-50 rounded border border-amber-200 text-amber-900">
                <h3 className="font-semibold flex items-center gap-1.5 mb-1">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-700" />
                  Threat Limitations Disclosure
                </h3>
                <p>
                  Defenses have verified boundary conditions. All unresolved threat limitations remain visible regardless of the overall risk score.
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
