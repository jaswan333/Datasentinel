import React, { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Layers,
  Grid,
  Cpu,
  CheckCircle,
  Image as ImageIcon,
  Tag,
  AlertTriangle,
  ZoomIn,
} from "lucide-react";
import { CIFAR10_CLASSES } from "@/lib/cifar";
import { Badge } from "@/components/ui/Badge";
import type { ImageDatasetInspectionResponse, ImageSamplePreview } from "@/types/audit";

interface ImageInspectionPanelProps {
  datasetName: string;
  totalImages?: number;
  datasetId?: string;
  inspection?: ImageDatasetInspectionResponse | null;
}

export const ImageInspectionPanel: React.FC<ImageInspectionPanelProps> = ({
  datasetName,
  totalImages = 1200,
  datasetId = "m6_cifar10",
  inspection = null,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activePreviewSample, setActivePreviewSample] = useState<ImageSamplePreview | null>(null);

  const displayCount = inspection ? inspection.total_images : totalImages;

  return (
    <div className="bg-teal-50/30 border border-teal-200/80 rounded-lg shadow-xs overflow-hidden">
      {/* Top Header */}
      <div className="p-3.5 flex items-center justify-between border-b border-teal-100 bg-white">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-teal-700" />
          <h3 className="text-xs font-semibold text-slate-800">
            Image Dataset Inspection & Representation Specs
          </h3>
          <span className="text-[11px] font-mono text-teal-800 bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
            {displayCount.toLocaleString()} images •{" "}
            {inspection ? inspection.dimensions_summary.common : "32×32×3 RGB"}
          </span>
          {inspection && (
            <Badge variant="success" size="sm">
              Live Ingested
            </Badge>
          )}
        </div>

        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-teal-800 hover:text-teal-950 font-medium flex items-center gap-1 cursor-pointer"
        >
          <span>{isExpanded ? "Collapse Specs" : "Inspect Specs & Thumbnails"}</span>
          {isExpanded ? (
            <ChevronUp className="w-3.5 h-3.5" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5" />
          )}
        </button>
      </div>

      {/* Metadata Overview Strip */}
      <div className="p-3.5 grid grid-cols-2 sm:grid-cols-4 gap-3 bg-white/50 text-xs">
        <div className="p-2.5 bg-white rounded border border-teal-100">
          <span className="text-[10px] text-slate-400 block">Dataset Modality</span>
          <span className="font-semibold text-slate-900 mt-0.5 block flex items-center gap-1">
            <Grid className="w-3.5 h-3.5 text-teal-600" />
            Computer Vision
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">
            {inspection
              ? Object.entries(inspection.formats)
                  .map(([fmt, cnt]) => `${fmt} (${cnt})`)
                  .join(", ")
              : "3 channels, normalized"}
          </span>
        </div>

        <div className="p-2.5 bg-white rounded border border-teal-100">
          <span className="text-[10px] text-slate-400 block">
            {inspection ? "Verified Population" : "Evaluation Population"}
          </span>
          <span className="font-mono font-bold text-slate-900 text-sm mt-0.5 block">
            {displayCount.toLocaleString()}
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">
            {inspection ? `${inspection.uncompressed_size_mb} MB uncompressed` : "Stratified partitions"}
          </span>
        </div>

        <div className="p-2.5 bg-white rounded border border-teal-100">
          <span className="text-[10px] text-slate-400 block">Feature Extractor</span>
          <span className="font-semibold text-slate-900 text-xs mt-0.5 block flex items-center gap-1">
            <Cpu className="w-3.5 h-3.5 text-teal-600" />
            ResNet18 (512D)
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">
            Penultimate latent vectors
          </span>
        </div>

        <div className="p-2.5 bg-white rounded border border-teal-100">
          <span className="text-[10px] text-slate-400 block">Class Labels</span>
          <span className="font-semibold text-slate-900 text-xs mt-0.5 block truncate">
            {inspection
              ? inspection.has_labels
                ? `${Object.keys(inspection.labels_distribution).length} Inferred Classes`
                : "Unlabeled Collection"
              : "10 Balanced Classes"}
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">
            {inspection ? "Inferred from folder names" : "CIFAR-10 canonical split"}
          </span>
        </div>
      </div>

      {/* Expanded Inspection & Preview Thumbnails */}
      {isExpanded && (
        <div className="p-4 border-t border-teal-100 bg-white space-y-4">
          {inspection ? (
            <div className="space-y-4">
              {/* Labels Distribution */}
              {inspection.has_labels && (
                <div>
                  <h4 className="text-xs font-semibold text-slate-800 mb-1.5 flex items-center gap-1.5">
                    <Tag className="w-3.5 h-3.5 text-teal-700" />
                    <span>Inferred Label Distribution:</span>
                  </h4>
                  <div className="flex flex-wrap gap-1.5">
                    {Object.entries(inspection.labels_distribution).map(([label, count]) => (
                      <span
                        key={label}
                        className="inline-flex items-center gap-1 text-[11px] px-2.5 py-1 rounded bg-teal-50 border border-teal-200 text-teal-900 font-mono"
                      >
                        <span className="font-bold capitalize">{label}:</span>
                        <span>{count}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Sample Previews Grid */}
              {inspection.sample_previews && inspection.sample_previews.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
                      <ImageIcon className="w-3.5 h-3.5 text-teal-700" />
                      <span>Decoded Image Previews ({inspection.sample_previews.length} displayed):</span>
                    </h4>
                    <span className="text-[10px] text-slate-400">
                      Dimensions: {inspection.dimensions_summary.min} to {inspection.dimensions_summary.max}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 sm:grid-cols-6 md:grid-cols-8 gap-2.5">
                    {inspection.sample_previews.map((sample) => (
                      <div
                        key={sample.sample_id}
                        className="border border-slate-200 rounded p-1.5 bg-slate-50 hover:bg-white hover:border-teal-400 transition-all flex flex-col items-center space-y-1 cursor-pointer group"
                        onClick={() => setActivePreviewSample(sample)}
                      >
                        <div className="w-16 h-16 rounded overflow-hidden bg-slate-200 flex items-center justify-center relative">
                          <img
                            src={sample.preview_url}
                            alt={sample.filename}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                            loading="lazy"
                            onError={(e) => {
                              // Fallback on broken image
                              (e.target as HTMLElement).style.display = "none";
                            }}
                          />
                          <div className="absolute inset-0 bg-slate-900/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white">
                            <ZoomIn className="w-4 h-4" />
                          </div>
                        </div>
                        <span className="text-[10px] font-mono text-slate-700 truncate w-full text-center">
                          #{sample.sample_id}
                        </span>
                        {sample.label && sample.label !== "Unlabeled" && (
                          <span className="text-[9px] px-1 bg-teal-100/80 text-teal-800 rounded font-medium truncate max-w-full">
                            {sample.label}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Rejected Files Warnings */}
              {inspection.rejected_count > 0 && (
                <div className="p-2.5 bg-amber-50 rounded border border-amber-200 text-xs text-amber-900 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                  <div className="text-[11px] space-y-1">
                    <p className="font-semibold">
                      {inspection.rejected_count} Files Skipped or Rejected During Validation:
                    </p>
                    <ul className="list-disc pl-4 space-y-0.5">
                      {inspection.rejected_files.map((rej, i) => (
                        <li key={i}>
                          <code>{rej.filename}</code>: {rej.reason}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Benchmark CIFAR-10 Classes View */
            <div>
              <h4 className="text-xs font-semibold text-slate-800 mb-1.5 flex items-center gap-1.5">
                <span>Canonical Class Taxonomy (CIFAR-10)</span>
                <Badge variant="info" size="sm">
                  10 Classes
                </Badge>
              </h4>
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(CIFAR10_CLASSES).map(([idx, name]) => (
                  <span
                    key={idx}
                    className="inline-flex items-center gap-1 text-[11px] px-2.5 py-1 rounded bg-teal-50/60 border border-teal-200 text-teal-900 font-mono"
                  >
                    <CheckCircle className="w-3 h-3 text-teal-600" />
                    <span className="font-bold">{idx}:</span>
                    <span>{name}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="pt-2 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
            <span>
              Dataset Name: <strong>{datasetName}</strong>
            </span>
            <span className="font-mono text-[10px] text-slate-400">
              Identifier: {datasetId}
            </span>
          </div>
        </div>
      )}

      {/* Lightbox Preview Modal */}
      {activePreviewSample && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs"
        >
          <div className="bg-white rounded-lg border border-slate-200 max-w-sm w-full shadow-lg p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <span className="font-mono text-xs font-semibold text-slate-900">
                Sample #{activePreviewSample.sample_id}
              </span>
              <button
                type="button"
                onClick={() => setActivePreviewSample(null)}
                className="text-slate-400 hover:text-slate-700 text-sm font-bold px-1"
              >
                ✕
              </button>
            </div>

            <div className="flex justify-center bg-slate-100 rounded p-2 border border-slate-200">
              <img
                src={activePreviewSample.preview_url}
                alt={activePreviewSample.filename}
                className="max-h-48 rounded object-contain"
              />
            </div>

            <div className="text-xs space-y-1 text-slate-600 font-mono text-[11px]">
              <div><strong>Filename:</strong> {activePreviewSample.filename}</div>
              <div><strong>Dimensions:</strong> {activePreviewSample.width} × {activePreviewSample.height}</div>
              <div><strong>Channels:</strong> {activePreviewSample.channels}</div>
              <div><strong>Inferred Label:</strong> {activePreviewSample.label}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
