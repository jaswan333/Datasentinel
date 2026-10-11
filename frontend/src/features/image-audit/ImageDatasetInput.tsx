import React, { useRef, useState } from "react";
import {
  UploadCloud,
  FileArchive,
  Image as ImageIcon,
  X,
  Play,
  Loader2,
  AlertCircle,
  Info,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { formatFileSize, cn } from "@/lib/utils";

interface ImageDatasetInputProps {
  selectedFile: File | null;
  selectedFilesCount?: number;
  onFileSelect: (file: File) => void;
  onFilesSelect?: (files: File[]) => void;
  onFileRemove: () => void;
  onRunAudit: () => void;
  isAuditing: boolean;
  isInspecting?: boolean;
  error: string | null;
}

const ALLOWED_EXTENSIONS = [".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".webp"];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB per individual archive/file

export const ImageDatasetInput: React.FC<ImageDatasetInputProps> = ({
  selectedFile,
  selectedFilesCount = 1,
  onFileSelect,
  onFilesSelect,
  onFileRemove,
  onRunAudit,
  isAuditing,
  isInspecting = false,
  error,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileValidationError, setFileValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndHandleFiles = (filesList: FileList | File[]) => {
    setFileValidationError(null);
    const filesArray = Array.from(filesList);

    if (filesArray.length === 0) return;

    // Check each file
    for (const f of filesArray) {
      const fileName = f.name.toLowerCase();
      const isSupported = ALLOWED_EXTENSIONS.some((ext) => fileName.endsWith(ext));

      if (!isSupported) {
        setFileValidationError(
          `Unsupported file '${f.name}'. Deep Learning accepts ZIP archives or image files (.png, .jpg, .jpeg, .webp).`
        );
        return;
      }

      if (f.size === 0) {
        setFileValidationError(`File '${f.name}' is empty (0 bytes).`);
        return;
      }

      if (f.size > MAX_FILE_SIZE) {
        setFileValidationError(
          `File '${f.name}' (${formatFileSize(f.size)}) exceeds the maximum allowed limit of 50 MB.`
        );
        return;
      }
    }

    if (onFilesSelect && filesArray.length > 1) {
      onFilesSelect(filesArray);
    } else {
      onFileSelect(filesArray[0]);
    }
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
      validateAndHandleFiles(files);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      validateAndHandleFiles(files);
    }
    e.target.value = "";
  };

  const isZip = selectedFile?.name.toLowerCase().endsWith(".zip");

  return (
    <div className="bg-blue-50/30 border border-blue-200/80 rounded-lg p-5 shadow-xs space-y-4">
      <div className="flex items-center justify-between pb-2 border-b border-blue-100">
        <div>
          <h2 className="text-sm font-semibold text-slate-900">
            Image Dataset Ingestion & Configuration
          </h2>
          <p className="text-xs text-slate-500">
            Upload images (.jpg, .png, .webp) or a ZIP archive (supports labeled subfolders like <code>cats/img1.jpg</code>)
          </p>
        </div>
      </div>

      {/* Upload Zone or Selected File View */}
      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleBrowseClick}
          className={cn(
            "border border-dashed rounded-lg p-6 text-center cursor-pointer transition-all",
            isDragging
              ? "border-blue-500 bg-blue-100/40"
              : "border-slate-300 hover:border-slate-400 bg-white"
          )}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".zip,.tar,.gz,.png,.jpg,.jpeg,.webp"
            onChange={handleFileChange}
            className="hidden"
            data-testid="vision-file-input"
          />

          <div className="flex flex-col items-center justify-center space-y-2">
            <div className="w-10 h-10 rounded-full bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600">
              <UploadCloud className="w-5 h-5 text-blue-600" />
            </div>

            <div className="space-y-0.5">
              <p className="text-xs font-medium text-slate-800">
                <span className="text-blue-700 font-semibold hover:underline">
                  Click to browse
                </span>{" "}
                or drag and drop image dataset / archive here
              </p>
              <p className="text-[11px] text-slate-400">
                Supports individual JPG, PNG, WEBP files or ZIP archives with folders • Up to 50 MB
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="border border-blue-200 bg-white rounded-md p-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-8 h-8 rounded bg-blue-50 border border-blue-200 text-blue-700 flex items-center justify-center shrink-0">
              {isZip ? <FileArchive className="w-4 h-4" /> : <ImageIcon className="w-4 h-4" />}
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-slate-800 truncate">
                  {selectedFile.name}
                  {selectedFilesCount > 1 && ` (+${selectedFilesCount - 1} other files)`}
                </span>
                <span className="text-[10px] text-slate-500 font-mono bg-slate-50 px-1.5 py-0.5 rounded border border-slate-200">
                  {formatFileSize(selectedFile.size)}
                </span>
                {isInspecting && (
                  <span className="text-[10px] text-indigo-700 font-medium flex items-center gap-1">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Inspecting...
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-500 mt-0.5 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                Validated & staged for vision detector execution
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
              multiple
              accept=".zip,.tar,.gz,.png,.jpg,.jpeg,.webp"
              onChange={handleFileChange}
              className="hidden"
            />
            <button
              type="button"
              onClick={onFileRemove}
              disabled={isAuditing || isInspecting}
              className="text-slate-400 hover:text-slate-600 p-1 rounded hover:bg-slate-100 transition-colors disabled:opacity-50"
              title="Remove file"
              aria-label="Remove file"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Real Ingestion Capability Note */}
      <div className="p-3 bg-white rounded border border-blue-200 text-xs text-slate-600 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
        <div className="space-y-0.5 text-[11px] leading-relaxed">
          <p className="font-semibold text-slate-800">
            Real Image Ingestion & Anomaly Screening Pipeline:
          </p>
          <p>
            Uploaded image datasets are decompressed with path-traversal protection, decoded via PIL, and evaluated across image-level integrity detectors (exact MD5 hashes, 64-bit dHash perceptual near-duplicates, gradient statistics) and ResNet18 latent embedding anomalies. For pre-computed Trojan and clean-label benchmarks, use the <strong>Benchmark Selector</strong> above.
          </p>
        </div>
      </div>

      {/* Error / Validation Feedback */}
      {(fileValidationError || error) && (
        <div className="flex items-start gap-2 p-3 bg-rose-50 border border-rose-200 rounded-md text-xs text-rose-800">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <p className="font-semibold">Dataset Ingestion Notice</p>
            <p className="text-[11px]">{fileValidationError || error}</p>
          </div>
        </div>
      )}

      {/* Action Row */}
      <div className="flex items-center justify-between pt-2 border-t border-blue-100">
        <div className="text-[11px] text-slate-500">
          {!selectedFile
            ? "Upload image files or choose a pre-computed benchmark scenario above"
            : isInspecting
            ? "Decoding and verifying image files..."
            : isAuditing
            ? "Executing vision detector suite (Integrity hashes, ResNet18 embeddings)..."
            : "Dataset inspected and ready for vision security audit"}
        </div>

        <Button
          type="button"
          variant="primary"
          size="sm"
          onClick={onRunAudit}
          disabled={!selectedFile || isAuditing || isInspecting}
          className="bg-indigo-700 hover:bg-indigo-800 text-white text-xs h-8 px-4"
          icon={
            isAuditing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5" />
            )
          }
        >
          {isAuditing ? "Auditing Vision Data..." : "Run Image Audit"}
        </Button>
      </div>
    </div>
  );
};
