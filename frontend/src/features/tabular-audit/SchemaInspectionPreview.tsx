import React, { useState } from "react";
import { ChevronDown, ChevronUp, Table, CheckCircle2 } from "lucide-react";
import type { InspectColumnsResponse } from "@/types/audit";

interface SchemaInspectionPreviewProps {
  inspection: InspectColumnsResponse;
}

export const SchemaInspectionPreview: React.FC<SchemaInspectionPreviewProps> = ({
  inspection,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
      <div className="p-3.5 flex items-center justify-between border-b border-slate-100 bg-slate-50/50">
        <div className="flex items-center gap-2">
          <Table className="w-4 h-4 text-slate-600" />
          <h3 className="text-xs font-semibold text-slate-800">
            Detected Schema & Column Types
          </h3>
          <span className="text-[11px] font-mono text-slate-500 bg-white px-2 py-0.5 rounded border border-slate-200">
            {inspection.columns.length} columns detected
          </span>
        </div>

        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-indigo-700 hover:text-indigo-900 font-medium flex items-center gap-1 cursor-pointer"
        >
          <span>{isExpanded ? "Collapse Schema" : "Inspect Columns"}</span>
          {isExpanded ? (
            <ChevronUp className="w-3.5 h-3.5" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5" />
          )}
        </button>
      </div>

      {isExpanded && (
        <div className="p-3.5 overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/70 text-slate-600 font-semibold text-[11px]">
                <th className="py-2 px-3">#</th>
                <th className="py-2 px-3">Column Name</th>
                <th className="py-2 px-3">Inferred Type</th>
                <th className="py-2 px-3">Unique Values</th>
                <th className="py-2 px-3">Suggested Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700 text-xs">
              {inspection.column_stats.map((stat, idx) => (
                <tr key={stat.name} className="hover:bg-slate-50/50">
                  <td className="py-1.5 px-3 font-mono text-slate-400 text-[11px]">
                    {idx + 1}
                  </td>
                  <td className="py-1.5 px-3 font-medium text-slate-900 font-mono text-[11px]">
                    {stat.name}
                  </td>
                  <td className="py-1.5 px-3">
                    <span className="font-mono text-[11px] bg-slate-100 px-1.5 py-0.5 rounded text-slate-700">
                      {stat.dtype}
                    </span>
                  </td>
                  <td className="py-1.5 px-3 font-mono text-[11px]">
                    {stat.unique_count}
                  </td>
                  <td className="py-1.5 px-3">
                    {stat.is_suggested_target ? (
                      <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded font-medium bg-amber-50 text-amber-800 border border-amber-200">
                        <CheckCircle2 className="w-3 h-3 text-amber-600" />
                        Target Label
                      </span>
                    ) : stat.is_suggested_cat ? (
                      <span className="inline-flex items-center text-[10px] px-2 py-0.5 rounded font-medium bg-indigo-50 text-indigo-700 border border-indigo-200">
                        Categorical Feature
                      </span>
                    ) : (
                      <span className="text-slate-400 text-[11px]">Numeric Feature</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
