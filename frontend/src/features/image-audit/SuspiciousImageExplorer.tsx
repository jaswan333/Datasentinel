import React, { useState } from "react";
import { AlertCircle, FileSearch, Filter, Search } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { getCifarClassName } from "@/lib/cifar";
import { ImageSampleDetailDrawer } from "./ImageSampleDetailDrawer";
import type { SuspiciousSample, RiskTier } from "@/types/audit";

interface SuspiciousImageExplorerProps {
  samples: SuspiciousSample[];
}

export const SuspiciousImageExplorer: React.FC<SuspiciousImageExplorerProps> = ({
  samples,
}) => {
  const [selectedSample, setSelectedSample] = useState<SuspiciousSample | null>(null);
  const [tierFilter, setTierFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
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
    if (tierFilter !== "ALL" && sample.risk_tier !== tierFilter) {
      return false;
    }
    if (searchQuery.trim().length > 0) {
      const q = searchQuery.toLowerCase().trim();
      const idMatch = String(sample.sample_id).toLowerCase().includes(q);
      const label = sample.metadata?.label;
      const classMatch = getCifarClassName(label).toLowerCase().includes(q);
      return idMatch || classMatch;
    }
    return true;
  });

  const totalPages = Math.ceil(filteredSamples.length / pageSize) || 1;
  const paginatedSamples = filteredSamples.slice((page - 1) * pageSize, page * pageSize);

  return (
    <>
      <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-purple-700" />
            <h3 className="text-sm font-semibold text-slate-900">
              Prioritized Suspicious Image Records
            </h3>
            <span className="text-xs text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
              {samples.length} image records evaluated
            </span>
          </div>

          {/* Search & Filter Controls */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {/* Search Input */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2 top-2" />
              <input
                type="text"
                placeholder="Search ID or class..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setPage(1);
                }}
                className="pl-7 pr-2.5 py-1 text-xs bg-slate-50 border border-slate-200 rounded text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white"
              />
            </div>

            {/* Filter Chips */}
            <div className="flex items-center gap-1">
              <Filter className="w-3.5 h-3.5 text-slate-400 mr-0.5" />
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
                      ? "bg-slate-900 text-white border-slate-900 font-medium"
                      : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                  }`}
                >
                  {tier}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50/70 text-slate-600 font-semibold text-[11px]">
                <th className="py-2.5 px-3.5">Image / Sample ID</th>
                <th className="py-2.5 px-3.5">Class Label</th>
                <th className="py-2.5 px-3.5">Risk Score</th>
                <th className="py-2.5 px-3.5">Risk Tier</th>
                <th className="py-2.5 px-3.5">Partition</th>
                <th className="py-2.5 px-3.5">Recommended Action</th>
                <th className="py-2.5 px-3.5 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {paginatedSamples.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400">
                    No image records match the current filter or search criteria.
                  </td>
                </tr>
              ) : (
                paginatedSamples.map((sample) => {
                  const assignedLabel = sample.metadata?.label;
                  const className = getCifarClassName(assignedLabel);

                  return (
                    <tr
                      key={String(sample.sample_id)}
                      className="hover:bg-slate-50/60 transition-colors"
                    >
                      <td className="py-2 px-3.5 font-mono font-medium text-slate-900">
                        <div className="flex items-center gap-2">
                          <span className="w-5 h-5 rounded bg-slate-800 text-white flex items-center justify-center text-[10px]">
                            #{sample.sample_id}
                          </span>
                          <span>Image #{sample.sample_id}</span>
                        </div>
                      </td>

                      <td className="py-2 px-3.5">
                        <span className="font-medium text-slate-800 capitalize">
                          {assignedLabel !== undefined ? `${className} (${assignedLabel})` : "Unlabeled"}
                        </span>
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
                        {sample.metadata?.partition || "eval"}
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
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <div className="p-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 bg-slate-50/50">
            <span>
              Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, filteredSamples.length)} of {filteredSamples.length} images
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

      {/* Detail Drawer */}
      <ImageSampleDetailDrawer
        sample={selectedSample}
        onClose={() => setSelectedSample(null)}
      />
    </>
  );
};
