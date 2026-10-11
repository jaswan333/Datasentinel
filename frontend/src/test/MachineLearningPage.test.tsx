import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { MachineLearningPage } from "@/pages/MachineLearningPage";
import { auditService } from "@/services/auditService";

// Mock auditService
vi.mock("@/services/auditService", () => ({
  auditService: {
    inspectColumns: vi.fn(),
    runAudit: vi.fn(),
    loadBenchmark: vi.fn(),
    confirmQuarantine: vi.fn(),
    getQuarantineDetails: vi.fn(),
    getLimitations: vi.fn(),
  },
}));

describe("MachineLearningPage Workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders page header and workflow indicator without a dark hero banner", () => {
    render(
      <BrowserRouter>
        <MachineLearningPage />
      </BrowserRouter>
    );

    // Title and Subtitle
    expect(
      screen.getByRole("heading", { name: /machine learning/i, level: 1 })
    ).toBeInTheDocument();
    expect(
      screen.getByText(/tabular dataset security, poisoning screening/i)
    ).toBeInTheDocument();

    // Verify 4-Step workflow is rendered
    const workflowNav = screen.getByRole("navigation", { name: /audit progress/i });
    expect(within(workflowNav).getByText("Select Dataset")).toBeInTheDocument();
    expect(within(workflowNav).getByText("Inspect Data")).toBeInTheDocument();
    expect(within(workflowNav).getByText("Run Audit")).toBeInTheDocument();
    expect(within(workflowNav).getByText("Review Evidence")).toBeInTheDocument();

    // Verify "Load Sample Audit" action exists
    expect(
      screen.getByRole("button", { name: /load sample audit/i })
    ).toBeInTheDocument();

    // Verify there is NO oversized hero banner
    expect(screen.queryByText(/Phase 2 Workflow/i)).toBeNull();
  });

  it("loads and displays sample benchmark audit when 'Load Sample Audit' is clicked", async () => {
    const mockBenchmarkResponse = {
      dataset_id: "m4_credit_sample_1000",
      filename: "UCI Credit Default (5% Label Flip Benchmark)",
      modality: "TABULAR",
      is_demo: true,
      audit: {
        audit_id: "m4_credit_sample_1000",
        dataset_name: "UCI Credit Default (5% Label Flip Benchmark)",
        modality: "TABULAR" as const,
        created_at: "2026-10-11T12:00:00Z",
        dataset_fingerprint: "abc123def456",
        n_rows: 1000,
        n_columns: 25,
        risk_score: 20.34,
        trust_score: 79.66,
        risk_level: "LOW_RISK" as const,
        overall_status: "AUDITED_SECURE" as const,
        evidence_coverage: 0.5,
        available_signals: ["cleanlab_risk", "lof_risk"],
        missing_signals: ["prediction_entropy"],
        signal_breakdown: {
          cleanlab_risk: {
            name: "cleanlab_risk",
            category: "label_consistency",
            weight: 0.2,
            status: "AVAILABLE" as const,
            mean_risk: 0.12,
            high_risk_fraction: 0.05,
            description: "Confident learning out-of-fold label inconsistency",
          },
        },
        quarantine_manifest: {
          total_candidate_samples: 1000,
          quarantined_count: 10,
          quarantine_rate: 0.01,
          operating_threshold: 0.7,
          threshold_policy: "confidence-weighted risk >= 0.70",
          requires_human_review: true,
          confirmation_status: "PENDING_OPERATOR_REVIEW" as const,
          quarantined_sample_ids: [],
        },
        model_utility: {
          evaluation_dataset: "UCI Credit Test Partition",
          victim_clean_accuracy: 36.0,
          screened_clean_accuracy: 38.0,
          clean_reference_accuracy: 42.0,
          clean_accuracy_delta_pp: 2.0,
        },
        unresolved_threat_limitations: [
          {
            threat_family: "clean_label_feature_perturbations",
            status: "UNRESOLVED_LIMITATION",
            statement: "Stealthy perturbations may evade distance filters.",
          },
        ],
        recommended_action: "Review 10 flagged samples before retraining.",
        sample_results: [
          {
            sample_id: "sample_001",
            risk_score: 74.2,
            risk_tier: "HIGH" as const,
            contributing_signals: { cleanlab_risk: 0.8 },
            recommended_action: "QUARANTINE",
          },
        ],
      },
    };

    vi.mocked(auditService.loadBenchmark).mockResolvedValueOnce(mockBenchmarkResponse as any);

    render(
      <BrowserRouter>
        <MachineLearningPage />
      </BrowserRouter>
    );

    const loadBenchmarkBtn = screen.getByRole("button", { name: /load sample audit/i });
    fireEvent.click(loadBenchmarkBtn);

    await waitFor(() => {
      expect(auditService.loadBenchmark).toHaveBeenCalledWith("tabular_credit");
    });

    // Check DEMO badge
    expect(await screen.findByText(/DEMO — Sample Benchmark Dataset:/i)).toBeInTheDocument();

    // Check risk score
    expect(screen.getByText("20.34")).toBeInTheDocument();

    // Check clean accuracy (36.00%, 38.00%, NOT 3600%)
    expect(screen.getByText("36.00%")).toBeInTheDocument();
    expect(screen.getByText("38.00%")).toBeInTheDocument();

    // Check sample table displays HIGH risk tier
    expect(screen.getByText("#sample_001")).toBeInTheDocument();
    expect(screen.getAllByText("HIGH").length).toBeGreaterThan(0);

    // Check quarantine mismatch alert appears
    expect(screen.getByText(/Data-Integrity Notice: Candidate Count & ID Discrepancy/i)).toBeInTheDocument();
  });

  it("handles unsupported file extensions gracefully", async () => {
    render(
      <BrowserRouter>
        <MachineLearningPage />
      </BrowserRouter>
    );

    const fileInput = screen.getByTestId("tabular-file-input");
    const badFile = new File(["test data"], "invalid.png", { type: "image/png" });

    fireEvent.change(fileInput, { target: { files: [badFile] } });

    expect(
      await screen.findByText(/Unsupported file type '\.png'/i)
    ).toBeInTheDocument();
  });
});
