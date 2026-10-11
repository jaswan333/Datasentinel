import React from "react";
import { Check, Database, Search, ShieldCheck, FileText } from "lucide-react";
import { cn } from "@/lib/utils";

export type AuditWorkflowStep = 1 | 2 | 3 | 4;

interface WorkflowIndicatorProps {
  currentStep: AuditWorkflowStep;
  completedSteps: AuditWorkflowStep[];
  onStepClick?: (step: AuditWorkflowStep) => void;
}

const STEPS = [
  { step: 1 as AuditWorkflowStep, label: "Select Dataset", icon: Database },
  { step: 2 as AuditWorkflowStep, label: "Inspect Data", icon: Search },
  { step: 3 as AuditWorkflowStep, label: "Run Audit", icon: ShieldCheck },
  { step: 4 as AuditWorkflowStep, label: "Review Evidence", icon: FileText },
];

export const WorkflowIndicator: React.FC<WorkflowIndicatorProps> = ({
  currentStep,
  completedSteps,
  onStepClick,
}) => {
  return (
    <nav
      aria-label="Audit Progress"
      className="bg-white border border-slate-200 rounded-lg p-2.5 shadow-sm"
    >
      <ol className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {STEPS.map(({ step, label, icon: Icon }) => {
          const isCurrent = currentStep === step;
          const isCompleted = completedSteps.includes(step);
          const isClickable = Boolean(onStepClick && (isCompleted || isCurrent));

          return (
            <li key={step}>
              <button
                type="button"
                disabled={!isClickable}
                onClick={() => isClickable && onStepClick?.(step)}
                className={cn(
                  "w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-all text-left",
                  isCurrent &&
                    "bg-slate-100 text-slate-900 border border-slate-300 shadow-xs",
                  isCompleted &&
                    !isCurrent &&
                    "text-slate-700 hover:bg-slate-50 cursor-pointer",
                  !isCompleted && !isCurrent && "text-slate-400 cursor-not-allowed"
                )}
              >
                <span
                  className={cn(
                    "w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-semibold shrink-0 transition-colors",
                    isCurrent && "bg-slate-900 text-white",
                    isCompleted && !isCurrent && "bg-emerald-100 text-emerald-800",
                    !isCompleted && !isCurrent && "bg-slate-100 text-slate-400"
                  )}
                >
                  {isCompleted && !isCurrent ? (
                    <Check className="w-3 h-3 text-emerald-700" />
                  ) : (
                    <span>{step}</span>
                  )}
                </span>

                <div className="flex items-center gap-1.5 min-w-0">
                  <Icon className="w-3.5 h-3.5 shrink-0 opacity-70" />
                  <span className="truncate">{label}</span>
                </div>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
};
