import React, { useState, useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowLeft, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";

// Feature Components
import {
  ImageWorkflowIndicator,
  type ImageWorkflowStep,
} from "@/features/image-audit/ImageWorkflowIndicator";
import { ImagePageHeader } from "@/features/image-audit/ImagePageHeader";
import { ImageBenchmarkSelector } from "@/features/image-audit/ImageBenchmarkSelector";
import { ImageDatasetInput } from "@/features/image-audit/ImageDatasetInput";
import { ImageInspectionPanel } from "@/features/image-audit/ImageInspectionPanel";
import { VisionAuditSummary } from "@/features/image-audit/VisionAuditSummary";
import { VisionDetectorEvidence } from "@/features/image-audit/VisionDetectorEvidence";
import { SuspiciousImageExplorer } from "@/features/image-audit/SuspiciousImageExplorer";
import { VisionModelUtility } from "@/features/image-audit/VisionModelUtility";
import { VisionThreatLimitations } from "@/features/image-audit/VisionThreatLimitations";
import { ImageQuarantineReview } from "@/features/image-audit/ImageQuarantineReview";
import { ImageAgentInvestigation } from "@/features/image-audit/ImageAgentInvestigation";

// Services and Types
import { auditService } from "@/services/auditService";
import type { AuditResponse, UnifiedAuditResult } from "@/types/audit";

export const DeepLearningPage: React.FC = () => {
  // Workflow step state
  const [currentStep, setCurrentStep] = useState<ImageWorkflowStep>(1);
  const [completedSteps, setCompletedSteps] = useState<ImageWorkflowStep[]>([]);

  // Dataset ingestion state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string | null>(null);
  const [inspection, setInspection] = useState<import("@/types/audit").ImageDatasetInspectionResponse | null>(null);
  const [isInspecting, setIsInspecting] = useState(false);

  // Audit execution state
  const [isAuditing, setIsAuditing] = useState(false);
  const [isLoadingBenchmark, setIsLoadingBenchmark] = useState(false);
  const [auditResponse, setAuditResponse] = useState<AuditResponse | null>(null);
  const [auditError, setAuditError] = useState<string | null>(null);

  // Quarantine confirmation state
  const [isConfirmingQuarantine, setIsConfirmingQuarantine] = useState(false);

  // Single file or multiple files selection
  const handleFileSelect = async (file: File) => {
    handleFilesSelect([file]);
  };

  const handleFilesSelect = async (files: File[]) => {
    if (files.length === 0) return;
    const primaryFile = files[0];
    setSelectedFile(primaryFile);
    setSelectedFiles(files);
    setSelectedBenchmarkId(null);
    setAuditError(null);
    setInspection(null);
    setIsInspecting(true);
    setCurrentStep(2);
    setCompletedSteps((prev) => Array.from(new Set([...prev, 1])));

    try {
      const insp = await auditService.inspectImageDataset(files, primaryFile.name);
      setInspection(insp);
    } catch (err: any) {
      setAuditError(err?.message || "Failed to inspect uploaded image files.");
    } finally {
      setIsInspecting(false);
    }
  };

  const handleFileRemove = () => {
    setSelectedFile(null);
    setSelectedFiles([]);
    setSelectedBenchmarkId(null);
    setInspection(null);
    setAuditResponse(null);
    setAuditError(null);
    setCurrentStep(1);
    setCompletedSteps([]);
  };

  // Run audit on uploaded dataset (via dedicated image audit API)
  const handleRunAudit = async () => {
    if (inspection?.dataset_id) {
      setIsAuditing(true);
      setAuditError(null);
      setCurrentStep(3);

      try {
        const res = await auditService.runUploadedImageAudit(inspection.dataset_id);
        setAuditResponse(res);
        setCurrentStep(4);
        setCompletedSteps([1, 2, 3, 4]);
      } catch (err: any) {
        setAuditError(
          err?.message ||
            "Vision security audit execution error on uploaded dataset."
        );
        setCurrentStep(2);
      } finally {
        setIsAuditing(false);
      }
    } else if (selectedFile) {
      // Legacy fallback
      setIsAuditing(true);
      setAuditError(null);
      setCurrentStep(3);

      try {
        const res = await auditService.runAudit(selectedFile);
        setAuditResponse(res);
        setCurrentStep(4);
        setCompletedSteps([1, 2, 3, 4]);
      } catch (err: any) {
        setAuditError(
          err?.message ||
            "Vision audit execution error. Note: The backend detector engine operates on calibrated CIFAR-10 benchmark datasets. Use the Benchmark Selector above for live evaluation results."
        );
        setCurrentStep(2);
      } finally {
        setIsAuditing(false);
      }
    }
  };

  // Benchmark loading (image_backdoor or image_clean_label)
  const handleSelectBenchmark = async (benchmarkId: string) => {
    setIsLoadingBenchmark(true);
    setSelectedBenchmarkId(benchmarkId);
    setAuditError(null);
    setSelectedFile(null);

    try {
      const benchmarkData = await auditService.loadBenchmark(benchmarkId);
      setAuditResponse(benchmarkData);
      setCurrentStep(4);
      setCompletedSteps([1, 2, 3, 4]);
    } catch (err: any) {
      setAuditError(
        `Failed to load vision benchmark scenario '${benchmarkId}': ${
          err?.message || "Backend error"
        }`
      );
    } finally {
      setIsLoadingBenchmark(false);
    }
  };

  const [searchParams] = useSearchParams();

  useEffect(() => {
    const bench = searchParams.get("benchmark");
    if (bench && (bench === "image_backdoor" || bench === "image_clean_label")) {
      handleSelectBenchmark(bench);
    }
  }, [searchParams]);

  // Quarantine Confirmation Operator Sign-off
  const handleConfirmQuarantine = async (notes: string) => {
    if (!auditResponse?.dataset_id) return;

    setIsConfirmingQuarantine(true);
    try {
      await auditService.confirmQuarantine(auditResponse.dataset_id, notes);
      if (auditResponse.audit?.quarantine_manifest) {
        const updatedAudit: UnifiedAuditResult = {
          ...auditResponse.audit,
          overall_status: "QUARANTINE_CONFIRMED",
          quarantine_manifest: {
            ...auditResponse.audit.quarantine_manifest,
            confirmation_status: "CONFIRMED",
            operator_notes: notes,
            confirmed_at: new Date().toISOString(),
          },
        };
        setAuditResponse({
          ...auditResponse,
          audit: updatedAudit,
        });
      }
    } catch (err: any) {
      alert(`Vision quarantine confirmation error: ${err?.message || "Failed to confirm"}`);
    } finally {
      setIsConfirmingQuarantine(false);
    }
  };

  const audit = auditResponse?.audit;

  return (
    <div className="space-y-5 max-w-7xl mx-auto pb-12">
      {/* Breadcrumb Navigation */}
      <nav aria-label="Breadcrumb" className="flex items-center gap-2 text-xs text-slate-500">
        <Link
          to="/"
          className="hover:text-indigo-600 flex items-center gap-1 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Home</span>
        </Link>
        <span>/</span>
        <span className="text-slate-900 font-medium">Deep Learning Workspace</span>
      </nav>

      {/* Page Header (No dark hero banner!) */}
      <ImagePageHeader
        onLoadSampleBenchmark={() => handleSelectBenchmark("image_backdoor")}
        isLoadingBenchmark={isLoadingBenchmark}
        currentDatasetLabel={audit?.dataset_name}
      />

      {/* 4-Stage Pastel Workflow Indicator */}
      <ImageWorkflowIndicator
        currentStep={currentStep}
        completedSteps={completedSteps}
        onStepClick={(step) => setCurrentStep(step)}
      />

      {/* Error Notice */}
      {auditError && (
        <div className="p-3.5 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start justify-between gap-3 shadow-xs">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-rose-900">Vision Audit Notice</p>
              <p className="text-[11px] mt-0.5">{auditError}</p>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAuditError(null)}
            className="text-xs h-6 px-2 text-rose-800 border-rose-300 hover:bg-rose-100"
          >
            Dismiss
          </Button>
        </div>
      )}

      {/* Stage 1: Select Dataset (Benchmarks + Upload) */}
      <section aria-labelledby="benchmark-selection-heading" className="space-y-4">
        <ImageBenchmarkSelector
          selectedBenchmarkId={selectedBenchmarkId}
          onSelectBenchmark={handleSelectBenchmark}
          isLoading={isLoadingBenchmark}
        />

        <ImageDatasetInput
          selectedFile={selectedFile}
          selectedFilesCount={selectedFiles.length}
          onFileSelect={handleFileSelect}
          onFilesSelect={handleFilesSelect}
          onFileRemove={handleFileRemove}
          onRunAudit={handleRunAudit}
          isAuditing={isAuditing}
          isInspecting={isInspecting}
          error={auditError}
        />
      </section>

      {/* Stage 2: Inspect Images Specs (Rendered when file selected, inspection ready, or benchmark loaded) */}
      {(selectedFile || inspection || audit) && (
        <section aria-labelledby="image-inspection-heading">
          <ImageInspectionPanel
            datasetName={
              inspection?.dataset_name ||
              audit?.dataset_name ||
              selectedFile?.name ||
              "CIFAR-10 Partition"
            }
            totalImages={inspection?.total_images || audit?.n_rows || 1200}
            datasetId={inspection?.dataset_id || audit?.audit_id || "m6_cifar10"}
            inspection={inspection}
          />
        </section>
      )}

      {/* Stage 4: Review Evidence Suite (Rendered when audit result exists) */}
      {audit && (
        <section aria-labelledby="vision-results-heading" className="space-y-5 pt-2">
          {/* Vision Audit Summary Strip */}
          <VisionAuditSummary
            audit={audit}
            isDemo={Boolean(auditResponse.is_demo)}
          />

          {/* Vision Detector Signals Table (14 Detectors) */}
          <div id="detector-evidence-section">
            <VisionDetectorEvidence signals={audit.signal_breakdown} />
          </div>

          {/* Suspicious Image Explorer & Detail Drawer */}
          {audit.sample_results && audit.sample_results.length > 0 && (
            <div id="suspicious-samples-section">
              <SuspiciousImageExplorer samples={audit.sample_results} />
            </div>
          )}

          {/* Model Utility & ASR Mitigation Card */}
          {audit.model_utility && (
            <div id="model-utility-section">
              <VisionModelUtility utility={audit.model_utility} />
            </div>
          )}

          {/* Threat Limitations & Clean-Label Boundary */}
          <div id="threat-limitations-section">
            <VisionThreatLimitations
              limitations={audit.unresolved_threat_limitations}
            />
          </div>

          {/* Quarantine Manifest Card */}
          {audit.quarantine_manifest && (
            <div id="quarantine-manifest-section">
              <ImageQuarantineReview
                manifest={audit.quarantine_manifest}
                datasetId={auditResponse.dataset_id}
                onConfirmQuarantine={handleConfirmQuarantine}
                isConfirming={isConfirmingQuarantine}
              />
            </div>
          )}

          {/* Agentic Vision Analyst Q&A Console */}
          <ImageAgentInvestigation datasetId={auditResponse.dataset_id} />
        </section>
      )}
    </div>
  );
};
