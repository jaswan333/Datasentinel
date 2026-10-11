import React, { useState } from "react";
import { AlertCircle, FileSearch, Filter } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { SampleDetailDrawer } from "./SampleDetailDrawer";
import type { SuspiciousSample, RiskTier } from "@/types/audit";

interface SuspiciousSamplesTableProps {
  samples: SuspiciousSample[];
}

export const SuspiciousSamplesTable: React.FC<SuspiciousSamplesTableProps> = ({
  samples,
}) => {
  const [selectedSample, setSelectedSample] = useState<SuspiciousSample | null>(null);
  const [tierFilter, setTierFilter] = useState<string>("ALL");
  const [page, setPage] = useState(1);
  const pageSize = 15;

  const renderTierBadge = (tier: RiskTier | string) => {
    switch (tier) {
      case "HIGH":
        return <Badge variant="danger" size="sm">HIGH</Badge>;
      case "MEDIUM":
        return <Badge variant="warning" size="sm">MEDIUM</Badge>;
      case "LOW":
        return <Badge variant="success" size="sm">LOW</Badge>;
      case "INSUFFICIENT_EVIDENCE":
      default:
        return <Badge variant="neutral" size="sm">{tier || "INSUFFICIENT"}</Badge>;
    }
  };

  const filteredSamples = samples.filter((sample) => {
    if (tierFilter === "ALL") return true;
    return sample.risk_tier === tierFilter;
  });

  const totalPages = Math.ceil(filteredSamples.length / pageSize) || 1;
  const paginatedSamples = filteredSamples.slice((page - 1) * pageSize, page * pageSize);

  return (
    <>
      <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-slate-700" />
            <h3 className="text-sm font-semibold text-slate-900">
              Prioritized Suspicious Records
            </h3>
            <span className="text-xs text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {samples.length} records evaluated
            </span>
          </div>

          {/* Filter Chips */}
          <div className="flex items-center gap-1.5 text-xs">
            <Filter className="w-3.5 h-3.5 text-slate-400 mr-1" />
            {["ALL", "HIGH", "MEDIUM", "LOW"].map((tier) => (
              <button
                key={tier}
                type="button"
                onClick={() => {
                  setTierFilter(tier);
                  setPage(1);
                }}
                className={`text-[11px] px-2 py-0.5 rounded border transition-colors cursor-pointer ${
                  tierFilter === tier
                    ? "bg-slate-900 text-white border-slate-900"
                    : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                }`}
              >
                {tier}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/70 text-slate-600 font-semibold text-[11px]">
                <th className="py-2.5 px-3.5">Sample ID</th>
                <th className="py-2.5 px-3.5">Risk Score</th>
                <th className="py-2.5 px-3.5">Risk Tier</th>
                <th className="py-2.5 px-3.5">Active Signals</th>
                <th className="py-2.5 px-3.5">Recommended Action</th>
                <th className="py-2.5 px-3.5 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {paginatedSamples.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-400">
                    No sample records match the current filter.
                  </td>
                </tr>
              ) : (
                paginatedSamples.map((sample) => (
                  <tr
                    key={String(sample.sample_id)}
                    className="hover:bg-slate-50/60 transition-colors"
                  >
                    <td className="py-2 px-3.5 font-mono font-medium text-slate-900">
                      #{sample.sample_id}
                    </td>

                    <td className="py-2 px-3.5 font-mono">
                      <span
                        className={
                          sample.risk_score >= 70
                            ? "font-semibold text-rose-700"
                            : sample.risk_score >= 40
                            ? "font-semibold text-amber-700"
                            : "text-slate-700"
                        }
                      >
                        {sample.risk_score.toFixed(2)}
                      </span>
                    </td>

                    <td className="py-2 px-3.5">
                      {renderTierBadge(sample.risk_tier)}
                    </td>

                    <td className="py-2 px-3.5 font-mono text-slate-500 text-[11px]">
                      {Object.keys(sample.contributing_signals || {}).length} signals
                    </td>

                    <td className="py-2 px-3.5 font-mono text-[11px] text-slate-700">
                      {sample.recommended_action || "HUMAN_REVIEW"}
                    </td>

                    <td className="py-2 px-3.5 text-right">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedSample(sample)}
                        className="text-xs h-6 px-2 text-indigo-700 hover:text-indigo-900 hover:bg-indigo-50"
                        icon={<FileSearch className="w-3 h-3" />}
                      >
                        Inspect
                      </Button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination bar if multiple pages */}
        {totalPages > 1 && (
          <div className="p-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 bg-slate-50/50">
            <span>
              Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, filteredSamples.length)} of {filteredSamples.length}
            </span>
            <div className="flex items-center gap-1">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="h-6 px-2 text-[11px]"
              >
                Prev
              </Button>
              <span className="px-2 text-[11px]">
                {page} / {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="h-6 px-2 text-[11px]"
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Drawer */}
      <SampleDetailDrawer
        sample={selectedSample}
        onClose={() => setSelectedSample(null)}
      />
    </>
  );
};
