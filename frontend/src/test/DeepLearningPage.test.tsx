import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { DeepLearningPage } from "@/pages/DeepLearningPage";
import { auditService } from "@/services/auditService";

// Mock auditService
vi.mock("@/services/auditService", () => ({
  auditService: {
    runAudit: vi.fn(),
    loadBenchmark: vi.fn(),
    confirmQuarantine: vi.fn(),
    getQuarantineDetails: vi.fn(),
    getLimitations: vi.fn(),
  },
}));

// Mock agentService
vi.mock("@/services/agentService", () => ({
  agentService: {
    askQuestion: vi.fn(),
    startInvestigation: vi.fn(),
    generateReport: vi.fn(),
  },
}));

const mockBackdoorBenchmark = {
  dataset_id: "m6_cifar10_backdoor",
  filename: "CIFAR-10 3% Trojan Backdoor (Patch Trigger)",
  modality: "IMAGE",
  is_demo: true,
  audit: {
    audit_id: "m6_cifar10_backdoor",
    dataset_name: "CIFAR-10 3% Trojan Backdoor (Patch Trigger)",
    modality: "IMAGE" as const,
    created_at: "2026-10-11T12:00:00Z",
    dataset_fingerprint: "img_fingerprint_cifar_10_backdoor",
    n_rows: 1200,
    n_columns: 3,
    risk_score: 87.45,
    trust_score: 12.55,
    risk_level: "CRITICAL_RISK" as const,
    overall_status: "POISONING_SUSPECTED" as const,
    evidence_coverage: 1.0,
    available_signals: [
      "training_loss_risk",
      "prediction_entropy_risk",
      "prediction_margin_risk",
      "assigned_confidence_risk",
      "oof_label_disagreement",
      "knn_embedding_risk",
      "lof_embedding_risk",
      "centroid_distance_risk",
      "activation_cluster_risk",
      "spectral_signature_risk",
      "exact_duplicate_risk",
      "perceptual_duplicate_risk",
      "image_statistics_risk",
      "local_patch_sensitivity_risk",
    ],
    missing_signals: [],
    signal_breakdown: {
      spectral_signature_risk: {
        name: "spectral_signature_risk",
        category: "spectral_signature",
        weight: 0.15,
        status: "AVAILABLE" as const,
        mean_risk: 0.88,
        high_risk_fraction: 0.035,
        description: "Top singular vector projection outlier score",
      },
      local_patch_sensitivity_risk: {
        name: "local_patch_sensitivity_risk",
        category: "patch_sensitivity",
        weight: 0.15,
        status: "AVAILABLE" as const,
        mean_risk: 0.92,
        high_risk_fraction: 0.03,
        description: "High-frequency localized trigger patch sensitivity",
      },
    },
    quarantine_manifest: {
      total_candidate_samples: 1200,
      quarantined_count: 36,
      quarantine_rate: 0.03,
      operating_threshold: 0.75,
      threshold_policy: "evidence-fusion risk >= 0.75",
      requires_human_review: true,
      confirmation_status: "PENDING_OPERATOR_REVIEW" as const,
      quarantined_sample_ids: ["img_0042", "img_0088", "img_0105"],
    },
    model_utility: {
      evaluation_dataset: "CIFAR-10 Test Partition (10k)",
      clean_reference_accuracy: 36.33,
      victim_clean_accuracy: 39.67,
      screened_clean_accuracy: 28.67,
      clean_accuracy_delta_pp: -11.0,
      victim_asr: 99.64,
      screened_asr: 6.18,
      asr_reduction_pp: 93.45,
    },
    unresolved_threat_limitations: [
      {
        threat_family: "clean_label_feature_collision",
        status: "UNRESOLVED_LIMITATION",
        statement: "Clean-label centroid manipulation evades simple patch triggers (0/12 recall).",
      },
    ],
    recommended_action: "Quarantine 36 flagged samples and retrain ResNet18 model.",
    sample_results: [
      {
        sample_id: "img_0042",
        risk_score: 94.6,
        risk_tier: "CRITICAL" as const,
        contributing_signals: {
          spectral_signature_risk: 0.95,
          local_patch_sensitivity_risk: 0.98,
        },
        recommended_action: "QUARANTINE",
      },
    ],
  },
};

const mockCleanLabelBenchmark = {
  dataset_id: "m6_cifar10_clean_label",
  filename: "CIFAR-10 5% Clean-Label Manipulation (Centroid Blend)",
  modality: "IMAGE",
  is_demo: true,
  audit: {
    audit_id: "m6_cifar10_clean_label",
    dataset_name: "CIFAR-10 5% Clean-Label Manipulation (Centroid Blend)",
    modality: "IMAGE" as const,
    created_at: "2026-10-11T12:00:00Z",
    dataset_fingerprint: "img_fingerprint_cifar_10_clean_label",
    n_rows: 1200,
    n_columns: 3,
    risk_score: 24.18,
    trust_score: 75.82,
    risk_level: "LOW_RISK" as const,
    overall_status: "AUDITED_SECURE" as const,
    evidence_coverage: 1.0,
    available_signals: [
      "training_loss_risk",
      "spectral_signature_risk",
    ],
    missing_signals: [],
    signal_breakdown: {
      spectral_signature_risk: {
        name: "spectral_signature_risk",
        category: "spectral_signature",
        weight: 0.15,
        status: "AVAILABLE" as const,
        mean_risk: 0.22,
        high_risk_fraction: 0.015,
        description: "Top singular vector projection outlier score",
      },
    },
    quarantine_manifest: {
      total_candidate_samples: 1200,
      quarantined_count: 8,
      quarantine_rate: 0.0067,
      operating_threshold: 0.75,
      threshold_policy: "evidence-fusion risk >= 0.75",
      requires_human_review: true,
      confirmation_status: "PENDING_OPERATOR_REVIEW" as const,
      quarantined_sample_ids: ["img_0012"],
    },
    model_utility: {
      evaluation_dataset: "CIFAR-10 Test Partition (10k)",
      clean_reference_accuracy: 39.67,
      victim_clean_accuracy: 36.0,
      screened_clean_accuracy: 38.0,
      clean_accuracy_delta_pp: 2.0,
      victim_asr: null,
      screened_asr: null,
      asr_reduction_pp: null,
    },
    unresolved_threat_limitations: [
      {
        threat_family: "clean_label_centroid_manipulation",
        status: "UNRESOLVED_LIMITATION",
        statement:
          "Zero of 12 held-out clean-label poisoned samples were recalled under strict FPR constraint. Low risk score does NOT guarantee absence of poisoning.",
      },
    ],
    recommended_action: "Inspect flagged centroid outliers and consider Mixup training.",
    sample_results: [],
  },
};

describe("DeepLearningPage Workspace", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("renders page header and pastel workflow indicator without an oversized hero banner", () => {
    render(
      <BrowserRouter>
        <DeepLearningPage />
      </BrowserRouter>
    );

    // Title and Subtitle
    expect(
      screen.getByRole("heading", { name: /deep learning/i, level: 1 })
    ).toBeInTheDocument();
    expect(
      screen.getByText(/image dataset security, backdoor screening, and vision evidence analysis/i)
    ).toBeInTheDocument();

    // Verify 4-Stage pastel workflow navigation
    const workflowNav = screen.getByRole("navigation", { name: /vision audit progress/i });
    expect(within(workflowNav).getByText("Select Dataset")).toBeInTheDocument();
    expect(within(workflowNav).getByText("Inspect Images")).toBeInTheDocument();
    expect(within(workflowNav).getByText("Run Audit")).toBeInTheDocument();
    expect(within(workflowNav).getByText("Review Evidence")).toBeInTheDocument();

    // Verify "Load Sample Audit" action exists in header
    expect(
      screen.getByRole("button", { name: /load sample audit/i })
    ).toBeInTheDocument();

    // Verify "About Vision Signals" methodology trigger exists
    expect(
      screen.getByRole("button", { name: /about vision signals/i })
    ).toBeInTheDocument();

    // Verify both benchmark buttons are present in BenchmarkSelector
    expect(
      screen.getByRole("button", { name: /load benchmark cifar-10 3% trojan backdoor/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /load benchmark cifar-10 5% clean-label/i })
    ).toBeInTheDocument();
  });

  it("loads and displays CIFAR-10 Trojan Backdoor benchmark with accurate ASR collapse and ResNet18 metrics", async () => {
    vi.mocked(auditService.loadBenchmark).mockResolvedValueOnce(mockBackdoorBenchmark as any);

    render(
      <BrowserRouter>
        <DeepLearningPage />
      </BrowserRouter>
    );

    const loadBenchmarkBtn = screen.getByRole("button", { name: /load sample audit/i });
    fireEvent.click(loadBenchmarkBtn);

    await waitFor(() => {
      expect(auditService.loadBenchmark).toHaveBeenCalledWith("image_backdoor");
    });

    // Check DEMO badge with dataset label
    expect(
      await screen.findByText(/DEMO — Benchmark Result/i)
    ).toBeInTheDocument();

    // Check Risk Score (87.45)
    expect(screen.getByText("87.45")).toBeInTheDocument();

    // Check Accuracy values (36.33%, 39.67%, 28.67% — NOT multiplied by 100 again)
    expect(screen.getByText("36.33%")).toBeInTheDocument();
    expect(screen.getByText("39.67%")).toBeInTheDocument();
    expect(screen.getByText("28.67%")).toBeInTheDocument();

    // Check ASR metric values (99.64%, 6.18%, -93.45 pp)
    expect(screen.getByText("99.64%")).toBeInTheDocument();
    expect(screen.getByText("6.18%")).toBeInTheDocument();
    expect(screen.getByText("-93.45 pp")).toBeInTheDocument();

    // Check sample table displays img_0042 and CRITICAL tier
    expect(screen.getAllByText(/img_0042/).length).toBeGreaterThan(0);
    expect(screen.getByText("CRITICAL")).toBeInTheDocument();
  });

  it("loads CIFAR-10 Clean-Label Benchmark and renders 'Not evaluated' for null ASR and persistent limitation", async () => {
    vi.mocked(auditService.loadBenchmark).mockResolvedValueOnce(mockCleanLabelBenchmark as any);

    render(
      <BrowserRouter>
        <DeepLearningPage />
      </BrowserRouter>
    );

    const cleanLabelBtn = screen.getByRole("button", { name: /load benchmark cifar-10 5% clean-label/i });
    fireEvent.click(cleanLabelBtn);

    await waitFor(() => {
      expect(auditService.loadBenchmark).toHaveBeenCalledWith("image_clean_label");
    });

    // Check Accuracy values
    expect(await screen.findByText("39.67%")).toBeInTheDocument();
    expect(screen.getByText("36.00%")).toBeInTheDocument();
    expect(screen.getByText("38.00%")).toBeInTheDocument();
    expect(screen.getByText("+2.00 pp")).toBeInTheDocument();

    // Check ASR is displayed as 'Not evaluated'
    expect(screen.getByText(/Not evaluated/i)).toBeInTheDocument();

    // Verify Clean-Label Centroid Manipulation limitation panel remains visible despite LOW_RISK score
    expect(screen.getByText("clean_label_centroid_manipulation")).toBeInTheDocument();
    expect(
      screen.getByText(/zero of 12 held-out clean-label poisoned samples were recalled/i)
    ).toBeInTheDocument();
  });

  it("handles operator quarantine confirmation workflow", async () => {
    vi.mocked(auditService.loadBenchmark).mockResolvedValueOnce(mockBackdoorBenchmark as any);
    vi.mocked(auditService.confirmQuarantine).mockResolvedValueOnce({
      success: true,
      dataset_id: "m6_cifar10_backdoor",
      status: "CONFIRMED",
    } as any);

    render(
      <BrowserRouter>
        <DeepLearningPage />
      </BrowserRouter>
    );

    const loadBenchmarkBtn = screen.getByRole("button", { name: /load sample audit/i });
    fireEvent.click(loadBenchmarkBtn);

    await screen.findByText(/DEMO — Benchmark Result/i);

    // Click "Sign Off & Authorize Screened Retraining" button
    const signOffBtn = screen.getByRole("button", { name: /sign off & authorize/i });
    fireEvent.click(signOffBtn);

    // Modal opens
    expect(
      screen.getByText(/operator sign-off: authorize screened retraining/i)
    ).toBeInTheDocument();
    const notesInput = screen.getByPlaceholderText(/reviewed high-suspicion backdoor patches/i);
    fireEvent.change(notesInput, { target: { value: "Verified spectral outlier clustering." } });

    // Submit button inside modal
    const confirmSubmitBtn = screen.getByRole("button", { name: /confirm quarantine/i });
    fireEvent.click(confirmSubmitBtn);

    await waitFor(() => {
      expect(auditService.confirmQuarantine).toHaveBeenCalledWith(
        "m6_cifar10_backdoor",
        "Verified spectral outlier clustering."
      );
    });
  });

  it("displays honest error notification when benchmark loading fails", async () => {
    vi.mocked(auditService.loadBenchmark).mockRejectedValueOnce(
      new Error("Network connection refused on port 8000")
    );

    render(
      <BrowserRouter>
        <DeepLearningPage />
      </BrowserRouter>
    );

    const loadBenchmarkBtn = screen.getByRole("button", { name: /load sample audit/i });
    fireEvent.click(loadBenchmarkBtn);

    await waitFor(() => {
      expect(auditService.loadBenchmark).toHaveBeenCalledWith("image_backdoor");
    });

    const errorElements = await screen.findAllByText(
      /failed to load vision benchmark scenario 'image_backdoor'/i
    );
    expect(errorElements.length).toBeGreaterThan(0);
  });
});
