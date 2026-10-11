import React, { useState, useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowLeft, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/Button";

// Feature Components
import { WorkflowIndicator, type AuditWorkflowStep } from "@/features/tabular-audit/WorkflowIndicator";
import { PageHeader } from "@/features/tabular-audit/PageHeader";
import { DatasetInput } from "@/features/tabular-audit/DatasetInput";
import { SchemaInspectionPreview } from "@/features/tabular-audit/SchemaInspectionPreview";
import { AuditSummaryHeader } from "@/features/tabular-audit/AuditSummaryHeader";
import { DetectorEvidenceTable } from "@/features/tabular-audit/DetectorEvidenceTable";
import { SuspiciousSamplesTable } from "@/features/tabular-audit/SuspiciousSamplesTable";
import { QuarantineManifestCard } from "@/features/tabular-audit/QuarantineManifestCard";
import { ModelUtilityCard } from "@/features/tabular-audit/ModelUtilityCard";
import { ThreatLimitationsCard } from "@/features/tabular-audit/ThreatLimitationsCard";
import { AgentInvestigationConsole } from "@/features/tabular-audit/AgentInvestigationConsole";

// Services and Types
import { auditService } from "@/services/auditService";
import type {
  InspectColumnsResponse,
  AuditResponse,
  UnifiedAuditResult,
} from "@/types/audit";

export const MachineLearningPage: React.FC = () => {
  // Workflow step state
  const [currentStep, setCurrentStep] = useState<AuditWorkflowStep>(1);
  const [completedSteps, setCompletedSteps] = useState<AuditWorkflowStep[]>([]);

  // Dataset ingestion state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [inspection, setInspection] = useState<InspectColumnsResponse | null>(null);
  const [isInspecting, setIsInspecting] = useState(false);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [categoricalColumns, setCategoricalColumns] = useState<string[]>([]);

  // Audit execution state
  const [isAuditing, setIsAuditing] = useState(false);
  const [auditResponse, setAuditResponse] = useState<AuditResponse | null>(null);
  const [auditError, setAuditError] = useState<string | null>(null);

  // Benchmark loading state
  const [isLoadingBenchmark, setIsLoadingBenchmark] = useState(false);

  // Quarantine confirmation state
  const [isConfirmingQuarantine, setIsConfirmingQuarantine] = useState(false);

  // Step 1: File selection & Schema Inspection
  const handleFileSelect = async (file: File) => {
    setSelectedFile(file);
    setAuditError(null);
    setIsInspecting(true);

    try {
      const data = await auditService.inspectColumns(file);
      setInspection(data);
      if (data.suggested_target) {
        setTargetColumn(data.suggested_target);
      }
      if (Array.isArray(data.suggested_categoricals)) {
        setCategoricalColumns(data.suggested_categoricals);
      }
      setCurrentStep(2);
      setCompletedSteps((prev) => Array.from(new Set([...prev, 1])));
    } catch (err: any) {
      setAuditError(
        err?.message || "Failed to inspect file schema. The file may be corrupt or formatted improperly."
      );
      setInspection(null);
    } finally {
      setIsInspecting(false);
    }
  };

  const handleFileRemove = () => {
    setSelectedFile(null);
    setInspection(null);
    setTargetColumn("");
    setCategoricalColumns([]);
    setAuditResponse(null);
    setAuditError(null);
    setCurrentStep(1);
    setCompletedSteps([]);
  };

  const handleCategoricalToggle = (col: string) => {
    setCategoricalColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  // Step 3: Run Audit on Uploaded Dataset
  const handleRunAudit = async () => {
    if (!selectedFile) return;

    setIsAuditing(true);
    setAuditError(null);
    setCurrentStep(3);

    try {
      const res = await auditService.runAudit(
        selectedFile,
        targetColumn || null,
        categoricalColumns
      );
      setAuditResponse(res);
      setCurrentStep(4);
      setCompletedSteps([1, 2, 3, 4]);
    } catch (err: any) {
      setAuditError(
        err?.message || "Audit execution encountered an error in the backend detector suite."
      );
      // Stay on step 2/3 so user can inspect or reconfigure
      setCurrentStep(2);
    } finally {
      setIsAuditing(false);
    }
  };

  // Step 4: Load Sample Benchmark Audit
  const handleLoadSampleBenchmark = async () => {
    setIsLoadingBenchmark(true);
    setAuditError(null);

    try {
      const benchmarkData = await auditService.loadBenchmark("tabular_credit");
      setAuditResponse(benchmarkData);
      setSelectedFile(null);
      setInspection(null);
      setCurrentStep(4);
      setCompletedSteps([1, 2, 3, 4]);
    } catch (err: any) {
      setAuditError(
        `Failed to load sample benchmark scenario: ${err?.message || "Backend error"}`
      );
    } finally {
      setIsLoadingBenchmark(false);
    }
  };

  const [searchParams] = useSearchParams();

  useEffect(() => {
    const bench = searchParams.get("benchmark");
    if (bench && bench === "tabular_credit") {
      handleLoadSampleBenchmark();
    }
  }, [searchParams]);

  // Quarantine Confirmation Operator Sign-off
  const handleConfirmQuarantine = async (notes: string) => {
    if (!auditResponse?.dataset_id) return;

    setIsConfirmingQuarantine(true);
    try {
      await auditService.confirmQuarantine(auditResponse.dataset_id, notes);
      // Update local audit manifest state
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
      alert(`Quarantine confirmation error: ${err?.message || "Failed to confirm"}`);
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
        <span className="text-slate-900 font-medium">Machine Learning Workspace</span>
      </nav>

      {/* Page Header (No dark hero banner!) */}
      <PageHeader
        onLoadSampleBenchmark={handleLoadSampleBenchmark}
        isLoadingBenchmark={isLoadingBenchmark}
      />

      {/* 4-Step Understated Workflow Indicator */}
      <WorkflowIndicator
        currentStep={currentStep}
        completedSteps={completedSteps}
        onStepClick={(step) => setCurrentStep(step)}
      />

      {/* Top Error Notice if API fails */}
      {auditError && (
        <div className="p-3.5 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start justify-between gap-3 shadow-xs">
          <div className="flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-rose-900">Audit Operation Failed</p>
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

      {/* Section 1: Ingestion & Schema Inspection */}
      <section aria-labelledby="ingestion-heading" className="space-y-4">
        <DatasetInput
          selectedFile={selectedFile}
          onFileSelect={handleFileSelect}
          onFileRemove={handleFileRemove}
          inspection={inspection}
          isInspecting={isInspecting}
          targetColumn={targetColumn}
          onTargetColumnChange={setTargetColumn}
          categoricalColumns={categoricalColumns}
          onCategoricalToggle={handleCategoricalToggle}
          onRunAudit={handleRunAudit}
          isAuditing={isAuditing}
          error={auditError}
        />

        {inspection && <SchemaInspectionPreview inspection={inspection} />}
      </section>

      {/* Section 2: Audit Results Workspace (Rendered after successful audit or benchmark load) */}
      {audit && (
        <section aria-labelledby="results-heading" className="space-y-5 pt-2">
          {/* Audit Summary Ribbon */}
          <AuditSummaryHeader
            audit={audit}
            isDemo={Boolean(auditResponse.is_demo)}
          />

          {/* Detector Evidence Table */}
          <div id="detector-evidence-section">
            <DetectorEvidenceTable signals={audit.signal_breakdown} />
          </div>

          {/* Prioritized Suspicious Samples Table & Detail Drawer */}
          {audit.sample_results && audit.sample_results.length > 0 && (
            <div id="suspicious-samples-section">
              <SuspiciousSamplesTable samples={audit.sample_results} />
            </div>
          )}

          {/* Quarantine Manifest Card with Discrepancy Integrity Check */}
          {audit.quarantine_manifest && (
            <div id="quarantine-manifest-section">
              <QuarantineManifestCard
                manifest={audit.quarantine_manifest}
                datasetId={auditResponse.dataset_id}
                onConfirmQuarantine={handleConfirmQuarantine}
                isConfirming={isConfirmingQuarantine}
              />
            </div>
          )}

          {/* Model Utility Comparison (if available) */}
          {audit.model_utility && (
            <div id="model-utility-section">
              <ModelUtilityCard utility={audit.model_utility} />
            </div>
          )}

          {/* Explicit Threat Limitations (Always visible) */}
          <div id="threat-limitations-section">
            <ThreatLimitationsCard
              limitations={audit.unresolved_threat_limitations}
            />
          </div>

          {/* Agentic AI Investigation Console */}
          <AgentInvestigationConsole datasetId={auditResponse.dataset_id} />
        </section>
      )}
    </div>
  );
};
