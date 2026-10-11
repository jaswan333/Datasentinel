import React from "react";
import { Sparkles, ShieldAlert, Cpu, CheckCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";

export interface BenchmarkScenario {
  id: string;
  name: string;
  threatType: string;
  description: string;
  keyMetric: string;
  badgeText: string;
  badgeVariant: "info" | "warning";
}

export const VISION_BENCHMARKS: BenchmarkScenario[] = [
  {
    id: "image_backdoor",
    name: "CIFAR-10 3% Trojan Backdoor (Patch Trigger)",
    threatType: "Trojan Backdoor with 4x4 Corner Patch Trigger",
    description:
      "Evaluates a compromised ResNet18 model on CIFAR-10 with a 3% synthetic checkerboard patch. Detectors screen high-loss anomalies and collapse ASR from 99.64% to 6.18%.",
    keyMetric: "93.45 pp ASR collapse via screened retraining",
    badgeText: "M6 Backdoor Benchmark",
    badgeVariant: "info",
  },
  {
    id: "image_clean_label",
    name: "CIFAR-10 5% Clean-Label Manipulation (Centroid Blend)",
    threatType: "Clean-Label Feature Blending (Correctly Labeled)",
    description:
      "Evaluates clean-label poisoned samples without label corruption. Evaluates 2D FFT spectral attenuation and transparently discloses 0/12 held-out recall under FPR ≤ 5%.",
    keyMetric: "Transparent 0/12 recall boundary disclosure",
    badgeText: "M6.3 Research Boundary",
    badgeVariant: "warning",
  },
];

interface ImageBenchmarkSelectorProps {
  selectedBenchmarkId: string | null;
  onSelectBenchmark: (benchmarkId: string) => void;
  isLoading: boolean;
}

export const ImageBenchmarkSelector: React.FC<ImageBenchmarkSelectorProps> = ({
  selectedBenchmarkId,
  onSelectBenchmark,
  isLoading,
}) => {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-xs space-y-3">
      <div className="flex items-center justify-between pb-2 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <h2 className="text-sm font-semibold text-slate-900">
            Pre-Computed Vision Benchmark Scenarios
          </h2>
        </div>
        <span className="text-[11px] text-slate-500">
          Validated CIFAR-10 Deep Learning benchmarks (ResNet18, Seed 42)
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {VISION_BENCHMARKS.map((b) => {
          const isSelected = selectedBenchmarkId === b.id;

          return (
            <div
              key={b.id}
              className={cn(
                "p-3.5 rounded-lg border transition-all flex flex-col justify-between space-y-3 text-xs",
                isSelected
                  ? "bg-indigo-50/50 border-indigo-300 ring-1 ring-indigo-200"
                  : "bg-slate-50/50 border-slate-200 hover:border-slate-300 hover:bg-slate-50"
              )}
            >
              <div className="space-y-1.5">
                <div className="flex items-start justify-between gap-2">
                  <Badge variant={b.badgeVariant} size="sm">
                    {b.badgeText}
                  </Badge>
                  {isSelected && (
                    <span className="flex items-center gap-1 text-[11px] font-medium text-indigo-700">
                      <CheckCircle className="w-3.5 h-3.5 text-indigo-600" />
                      Active
                    </span>
                  )}
                </div>

                <h3 className="font-bold text-slate-900 text-xs">{b.name}</h3>

                <p className="text-[11px] text-slate-600 leading-relaxed">
                  {b.description}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-200/60 flex items-center justify-between">
                <div className="flex items-center gap-1 text-[10px] text-slate-500 font-medium">
                  {b.id === "image_backdoor" ? (
                    <Cpu className="w-3 h-3 text-indigo-600" />
                  ) : (
                    <ShieldAlert className="w-3 h-3 text-amber-600" />
                  )}
                  <span>{b.keyMetric}</span>
                </div>

                <Button
                  type="button"
                  variant={isSelected ? "secondary" : "outline"}
                  size="sm"
                  onClick={() => onSelectBenchmark(b.id)}
                  disabled={isLoading}
                  className="text-xs h-7 px-3"
                  aria-label={`Load benchmark ${b.name}`}
                >
                  {isLoading && isSelected
                    ? "Loading..."
                    : isSelected
                    ? "Active Scenario"
                    : "Load Benchmark"}
                </Button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
