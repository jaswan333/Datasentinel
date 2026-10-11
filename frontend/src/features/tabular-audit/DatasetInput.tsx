import React, { useRef, useState } from "react";
import {
  UploadCloud,
  FileSpreadsheet,
  X,
  Play,
  Loader2,
  AlertCircle,
  Tag,
  Target,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { formatFileSize, cn } from "@/lib/utils";
import type { InspectColumnsResponse } from "@/types/audit";

interface DatasetInputProps {
  selectedFile: File | null;
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
  inspection: InspectColumnsResponse | null;
  isInspecting: boolean;
  targetColumn: string;
  onTargetColumnChange: (col: string) => void;
  categoricalColumns: string[];
  onCategoricalToggle: (col: string) => void;
  onRunAudit: () => void;
  isAuditing: boolean;
  error: string | null;
}

const ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export const DatasetInput: React.FC<DatasetInputProps> = ({
  selectedFile,
  onFileSelect,
  onFileRemove,
  inspection,
  isInspecting,
  targetColumn,
  onTargetColumnChange,
  categoricalColumns,
  onCategoricalToggle,
  onRunAudit,
  isAuditing,
  error,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileValidationError, setFileValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndHandleFile = (file: File) => {
    setFileValidationError(null);
    const ext = `.${file.name.split(".").pop()?.toLowerCase()}`;

    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setFileValidationError(
        `Unsupported file type '${ext}'. Only CSV, XLSX, and XLS tabular files are supported.`
      );
      return;
    }

    if (file.size === 0) {
      setFileValidationError("The selected file is empty (0 bytes).");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setFileValidationError(
        `File size (${formatFileSize(file.size)}) exceeds the maximum limit of 50 MB.`
      );
      return;
    }

    onFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      validateAndHandleFile(files[0]);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      validateAndHandleFile(files[0]);
    }
    // Reset input so same file can be re-selected if removed
    e.target.value = "";
  };

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-xs space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-slate-100">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            Dataset Ingestion & Configuration
          </h2>
          <p className="text-xs text-slate-500">
            Upload a structured tabular dataset (.csv, .xlsx, .xls) for multi-detector poisoning screening
          </p>
        </div>
      </div>

      {/* Upload Zone or File Card */}
      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleBrowseClick}
          className={cn(
            "border border-dashed rounded-lg p-6 text-center cursor-pointer transition-all",
            isDragging
              ? "border-indigo-500 bg-indigo-50/50"
              : "border-slate-300 hover:border-slate-400 bg-slate-50/40"
          )}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
            className="hidden"
            data-testid="tabular-file-input"
          />

          <div className="flex flex-col items-center justify-center space-y-2">
            <div className="w-10 h-10 rounded-full bg-slate-100 border border-slate-200 flex items-center justify-center text-slate-600">
              <UploadCloud className="w-5 h-5 text-slate-600" />
            </div>

            <div className="space-y-0.5">
              <p className="text-xs font-medium text-slate-800">
                <span className="text-indigo-600 font-semibold hover:underline">
                  Click to browse
                </span>{" "}
                or drag and drop tabular file here
              </p>
              <p className="text-[11px] text-slate-400">
                Supported formats: CSV, XLSX, XLS • Up to 50 MB
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="border border-slate-200 bg-slate-50/60 rounded-md p-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-8 h-8 rounded bg-indigo-50 border border-indigo-200 text-indigo-700 flex items-center justify-center shrink-0">
              <FileSpreadsheet className="w-4 h-4" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-slate-800 truncate">
                  {selectedFile.name}
                </span>
                <span className="text-[10px] text-slate-500 font-mono bg-white px-1.5 py-0.5 rounded border border-slate-200">
                  {formatFileSize(selectedFile.size)}
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-0.5">
                {isInspecting ? (
                  <span className="flex items-center gap-1 text-indigo-600">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Inspecting columns & schema...
                  </span>
                ) : inspection ? (
                  `${inspection.columns.length} columns detected • Ready for audit`
                ) : (
                  "Ready"
                )}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleBrowseClick}
              disabled={isAuditing || isInspecting}
              className="text-xs h-7 px-2"
            >
              Replace
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              type="button"
              onClick={onFileRemove}
              disabled={isAuditing || isInspecting}
              className="text-slate-400 hover:text-slate-600 p-1 rounded hover:bg-slate-200/60 transition-colors disabled:opacity-50"
              title="Remove file"
              aria-label="Remove file"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Validation Feedback */}
      {(fileValidationError || error) && (
        <div className="flex items-start gap-2 p-3 bg-rose-50 border border-rose-200 rounded-md text-xs text-rose-800">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="font-semibold">Dataset Ingestion Notice</p>
            <p className="text-[11px]">{fileValidationError || error}</p>
          </div>
        </div>
      )}

      {/* Target & Categorical Configuration (if columns inspected) */}
      {inspection && inspection.columns.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-slate-100">
          {/* Target Column Selection */}
          <div className="space-y-1.5">
            <label
              htmlFor="target-column-select"
              className="text-xs font-semibold text-slate-700 flex items-center gap-1.5"
            >
              <Target className="w-3.5 h-3.5 text-indigo-600" />
              Target / Label Column
            </label>
            <div className="flex items-center gap-2">
              <select
                id="target-column-select"
                value={targetColumn}
                onChange={(e) => onTargetColumnChange(e.target.value)}
                disabled={isAuditing}
                className="w-full text-xs bg-white border border-slate-300 rounded px-2.5 py-1.5 text-slate-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              >
                <option value="">-- None (Unsupervised mode) --</option>
                {inspection.columns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                    {col === inspection.suggested_target ? " (Suggested Target)" : ""}
                  </option>
                ))}
              </select>
            </div>
            <p className="text-[11px] text-slate-500">
              Supervised mode enables Cleanlab label consistency and out-of-fold disagreement analysis.
            </p>
          </div>

          {/* Categorical Features */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5 text-slate-500" />
              Categorical Features ({categoricalColumns.length} selected)
            </label>
            <div className="max-h-24 overflow-y-auto p-2 bg-slate-50 border border-slate-200 rounded flex flex-wrap gap-1.5">
              {inspection.columns
                .filter((c) => c !== targetColumn)
                .map((col) => {
                  const isSelected = categoricalColumns.includes(col);
                  return (
                    <button
                      key={col}
                      type="button"
                      onClick={() => onCategoricalToggle(col)}
                      disabled={isAuditing}
                      className={cn(
                        "text-[10px] px-2 py-0.5 rounded border transition-colors cursor-pointer",
                        isSelected
                          ? "bg-indigo-100 text-indigo-800 border-indigo-300 font-medium"
                          : "bg-white text-slate-600 border-slate-300 hover:bg-slate-100"
                      )}
                    >
                      {col}
                    </button>
                  );
                })}
            </div>
            <p className="text-[11px] text-slate-500">
              Categoricals are one-hot or frequency encoded before distance and density checks.
            </p>
          </div>
        </div>
      )}

      {/* Run Audit Action */}
      <div className="flex items-center justify-between pt-2 border-t border-slate-100">
        <div className="text-[11px] text-slate-500">
          {!selectedFile ? (
            "Select a file to begin"
          ) : isInspecting ? (
            "Extracting column headers..."
          ) : isAuditing ? (
            "Running tabular detector suite (M1–M4.5)..."
          ) : (
            "Dataset ready for multi-detector screening"
          )}
        </div>

        <Button
          type="button"
          variant="primary"
          size="sm"
          onClick={onRunAudit}
          disabled={!selectedFile || isInspecting || isAuditing}
          className="bg-indigo-700 hover:bg-indigo-800 text-white text-xs h-8 px-4"
          icon={
            isAuditing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5" />
            )
          }
        >
          {isAuditing ? "Auditing Dataset..." : "Run Audit"}
        </Button>
      </div>
    </div>
  );
};
