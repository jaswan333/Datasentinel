import React from "react";
import { Check, Image, Search, ShieldCheck, FileText } from "lucide-react";
import { cn } from "@/lib/utils";

export type ImageWorkflowStep = 1 | 2 | 3 | 4;

interface ImageWorkflowIndicatorProps {
  currentStep: ImageWorkflowStep;
  completedSteps: ImageWorkflowStep[];
  onStepClick?: (step: ImageWorkflowStep) => void;
}

const STEPS = [
  {
    step: 1 as ImageWorkflowStep,
    label: "Select Dataset",
    icon: Image,
    colorStyle: {
      active: "bg-blue-50/90 border-blue-400 text-blue-950 shadow-xs",
      completed: "bg-blue-50/40 text-blue-900 border-blue-200/80 hover:bg-blue-50",
      iconColor: "text-blue-600",
      badgeColor: "bg-blue-700 text-white",
    },
  },
  {
    step: 2 as ImageWorkflowStep,
    label: "Inspect Images",
    icon: Search,
    colorStyle: {
      active: "bg-teal-50/90 border-teal-400 text-teal-950 shadow-xs",
      completed: "bg-teal-50/40 text-teal-900 border-teal-200/80 hover:bg-teal-50",
      iconColor: "text-teal-600",
      badgeColor: "bg-teal-700 text-white",
    },
  },
  {
    step: 3 as ImageWorkflowStep,
    label: "Run Audit",
    icon: ShieldCheck,
    colorStyle: {
      active: "bg-purple-50/90 border-purple-400 text-purple-950 shadow-xs",
      completed: "bg-purple-50/40 text-purple-900 border-purple-200/80 hover:bg-purple-50",
      iconColor: "text-purple-600",
      badgeColor: "bg-purple-700 text-white",
    },
  },
  {
    step: 4 as ImageWorkflowStep,
    label: "Review Evidence",
    icon: FileText,
    colorStyle: {
      active: "bg-amber-50/90 border-amber-400 text-amber-950 shadow-xs",
      completed: "bg-amber-50/40 text-amber-900 border-amber-200/80 hover:bg-amber-50",
      iconColor: "text-amber-600",
      badgeColor: "bg-amber-700 text-white",
    },
  },
];

export const ImageWorkflowIndicator: React.FC<ImageWorkflowIndicatorProps> = ({
  currentStep,
  completedSteps,
  onStepClick,
}) => {
  return (
    <nav
      aria-label="Vision Audit Progress"
      className="bg-white border border-slate-200 rounded-lg p-2.5 shadow-xs"
    >
      <ol className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {STEPS.map(({ step, label, icon: Icon, colorStyle }) => {
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
                  "w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-all text-left border",
                  isCurrent && colorStyle.active,
                  isCompleted && !isCurrent && colorStyle.completed,
                  !isCompleted &&
                    !isCurrent &&
                    "bg-slate-50/50 border-slate-200 text-slate-400 cursor-not-allowed"
                )}
              >
                <span
                  className={cn(
                    "w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-semibold shrink-0 transition-colors",
                    isCurrent && colorStyle.badgeColor,
                    isCompleted && !isCurrent && "bg-emerald-100 text-emerald-800",
                    !isCompleted && !isCurrent && "bg-slate-200/70 text-slate-500"
                  )}
                >
                  {isCompleted && !isCurrent ? (
                    <Check className="w-3 h-3 text-emerald-700" />
                  ) : (
                    <span>{step}</span>
                  )}
                </span>

                <div className="flex items-center gap-1.5 min-w-0">
                  <Icon
                    className={cn(
                      "w-3.5 h-3.5 shrink-0",
                      isCurrent || isCompleted ? colorStyle.iconColor : "text-slate-400"
                    )}
                  />
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
