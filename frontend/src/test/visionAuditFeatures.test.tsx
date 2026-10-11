import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { VisionDetectorEvidence } from "@/features/image-audit/VisionDetectorEvidence";
import { VisionAuditSummary } from "@/features/image-audit/VisionAuditSummary";
import { SuspiciousImageExplorer } from "@/features/image-audit/SuspiciousImageExplorer";
import { VisionModelUtility } from "@/features/image-audit/VisionModelUtility";
import { VisionThreatLimitations } from "@/features/image-audit/VisionThreatLimitations";
import { ImageQuarantineReview } from "@/features/image-audit/ImageQuarantineReview";
import { ImageAgentInvestigation } from "@/features/image-audit/ImageAgentInvestigation";
import { getCifarClassName, CIFAR10_CLASSES } from "@/lib/cifar";

import type {
  UnifiedAuditResult,
  SignalDetail,
  SuspiciousSample,
  QuarantineManifest,
  ModelUtilityComparison,
  ThreatLimitation,
} from "@/types/audit";

describe("CIFAR-10 Class Taxonomy Helper", () => {
  it("resolves class indices to correct names", () => {
    expect(getCifarClassName(0)).toBe("airplane");
    expect(getCifarClassName("2")).toBe("bird");
    expect(getCifarClassName(5)).toBe("dog");
    expect(getCifarClassName(9)).toBe("truck");
    expect(Object.keys(CIFAR10_CLASSES).length).toBe(10);
  });
});

describe("VisionDetectorEvidence Component", () => {
  const mockVisionSignals: Record<string, SignalDetail> = {
    training_loss_risk: {
      name: "training_loss_risk",
      category: "label_consistency",
      weight: 0.08,
      status: "AVAILABLE",
      mean_risk: 0.25,
      high_risk_fraction: 0.12,
      description: "Penultimate cross-entropy training loss percentile",
    },
    knn_embedding_risk: {
      name: "knn_embedding_risk",
      category: "deep_representation",
      weight: 0.08,
      status: "AVAILABLE",
      mean_risk: 0.45,
      high_risk_fraction: 0.22,
      description: "K-nearest neighbor distance in 512D ResNet18 latent space",
    },
    local_patch_sensitivity_risk: {
      name: "local_patch_sensitivity_risk",
      category: "model_behavior",
      weight: 0.07,
      status: "UNAVAILABLE",
      mean_risk: null,
      high_risk_fraction: null,
      description: "Spatial prediction sensitivity across occlusion masks",
    },
  };

  it("renders vision detector rows and formats fractions accurately without [object Object]", () => {
    render(<VisionDetectorEvidence signals={mockVisionSignals} />);

    expect(screen.getByText("training_loss_risk")).toBeInTheDocument();
    expect(screen.getByText("knn_embedding_risk")).toBeInTheDocument();
    expect(screen.getByText("local_patch_sensitivity_risk")).toBeInTheDocument();

    // Check fractions
    expect(screen.getByText("12.0%")).toBeInTheDocument();
    expect(screen.getByText("22.0%")).toBeInTheDocument();
    expect(screen.queryByText(/\[object Object\]/)).toBeNull();

    // Check statuses
    expect(screen.getAllByText("AVAILABLE").length).toBeGreaterThan(0);
    expect(screen.getByText("UNAVAILABLE")).toBeInTheDocument();
  });
});

describe("VisionAuditSummary Component", () => {
  const mockAudit: UnifiedAuditResult = {
    audit_id: "m6_cifar10_backdoor",
    dataset_name: "CIFAR-10 3% Trojan Backdoor (Patch Trigger)",
    modality: "IMAGE",
    created_at: "2026-10-11T14:00:00Z",
    dataset_fingerprint: "cifar10_fingerprint_123",
    n_rows: 1200,
    n_columns: 3,
    risk_score: 20.0,
    trust_score: 80.0,
    risk_level: "LOW_RISK",
    overall_status: "AUDITED_SECURE",
    evidence_coverage: 1.0,
    available_signals: ["training_loss_risk", "knn_embedding_risk"],
    missing_signals: [],
    signal_breakdown: {},
    quarantine_manifest: {
      total_candidate_samples: 1200,
      quarantined_count: 89,
      quarantine_rate: 0.0742,
      operating_threshold: 0.86,
      threshold_policy: "maximize recall subject to FPR <= 0.05",
      requires_human_review: true,
      confirmation_status: "PENDING_OPERATOR_REVIEW",
      quarantined_sample_ids: [],
    },
    unresolved_threat_limitations: [
      {
        threat_family: "clean_label_manipulation",
        status: "UNRESOLVED_LIMITATION",
        statement: "Clean-label centroid manipulation evades under FPR <= 5%.",
      },
    ],
    recommended_action: "Quarantine 89 candidates and retrain screened ResNet18 model.",
    sample_results: [],
  };

  it("renders vision audit metrics and demo benchmark banner", () => {
    render(<VisionAuditSummary audit={mockAudit} isDemo={true} />);

    expect(screen.getByText("m6_cifar10_backdoor")).toBeInTheDocument();
    expect(screen.getAllByText("CIFAR-10 3% Trojan Backdoor (Patch Trigger)").length).toBeGreaterThan(0);
    expect(screen.getByText("20.00")).toBeInTheDocument();
    expect(screen.getByText("80.00")).toBeInTheDocument();
    expect(screen.getByText("100.0%")).toBeInTheDocument();
    expect(screen.getByText(/DEMO — Benchmark Result/i)).toBeInTheDocument();
    expect(screen.getByText(/Evidence-weighted index \(not a safety probability\)/)).toBeInTheDocument();
  });
});

describe("SuspiciousImageExplorer Component", () => {
  const mockImageSamples: SuspiciousSample[] = [
    {
      sample_id: "img_162",
      risk_score: 88.5,
      risk_tier: "HIGH",
      contributing_signals: { knn_embedding_risk: 0.95, spectral_signature_risk: 0.88 },
      recommended_action: "QUARANTINE",
      metadata: { label: 2, group_id: 162, partition: "test" },
    },
    {
      sample_id: "img_494",
      risk_score: 42.0,
      risk_tier: "MEDIUM",
      contributing_signals: { activation_cluster_risk: 0.45 },
      recommended_action: "FLAG_SECONDARY",
      metadata: { label: 7, group_id: 494, partition: "test" },
    },
  ];

  it("binds sample risk tier and displays human-readable CIFAR class names", () => {
    render(<SuspiciousImageExplorer samples={mockImageSamples} />);

    expect(screen.getByText("Image #img_162")).toBeInTheDocument();
    expect(screen.getByText("bird (2)")).toBeInTheDocument();
    expect(screen.getByText("horse (7)")).toBeInTheDocument();

    expect(screen.getAllByText("HIGH").length).toBeGreaterThan(0);
    expect(screen.getAllByText("MEDIUM").length).toBeGreaterThan(0);
  });

  it("opens image sample detail drawer when Inspect is clicked", () => {
    render(<SuspiciousImageExplorer samples={mockImageSamples} />);

    const inspectBtns = screen.getAllByRole("button", { name: /inspect/i });
    fireEvent.click(inspectBtns[0]);

    expect(screen.getByText("Image Sample #img_162 Evidence")).toBeInTheDocument();
    expect(screen.getByText("knn_embedding_risk")).toBeInTheDocument();
  });
});

describe("VisionModelUtility Component", () => {
  it("displays pre/post ASR collapse and accuracy delta in percentage points (pp)", () => {
    const backdoorUtility: ModelUtilityComparison = {
      evaluation_dataset: "Independent Pristine CIFAR-10 Test Benchmark",
      clean_reference_accuracy: 36.33,
      victim_clean_accuracy: 39.67,
      screened_clean_accuracy: 28.67,
      victim_asr: 99.64,
      screened_asr: 6.18,
      asr_reduction_pp: 93.45,
      evaluation_notes: "ResNet18 Adam Seed 42",
    };

    render(<VisionModelUtility utility={backdoorUtility} />);

    expect(screen.getByText("36.33%")).toBeInTheDocument();
    expect(screen.getByText("39.67%")).toBeInTheDocument();
    expect(screen.getByText("28.67%")).toBeInTheDocument();

    // ASR metrics
    expect(screen.getByText("99.64%")).toBeInTheDocument();
    expect(screen.getByText("6.18%")).toBeInTheDocument();
    expect(screen.getByText("-93.45 pp")).toBeInTheDocument();
    expect(screen.queryByText("9964%")).toBeNull();
  });

  it("handles null ASR gracefully for clean-label benchmark", () => {
    const cleanLabelUtility: ModelUtilityComparison = {
      evaluation_dataset: "Pristine CIFAR-10 Partition",
      clean_reference_accuracy: 39.67,
      victim_clean_accuracy: 36.0,
      screened_clean_accuracy: 38.0,
      clean_accuracy_delta_pp: 2.0,
      victim_asr: null,
      screened_asr: null,
      asr_reduction_pp: null,
    };

    render(<VisionModelUtility utility={cleanLabelUtility} />);

    expect(screen.getByText("36.00%")).toBeInTheDocument();
    expect(screen.getByText("38.00%")).toBeInTheDocument();
    expect(screen.getByText("+2.00 pp")).toBeInTheDocument();
    expect(screen.getByText(/Not evaluated/i)).toBeInTheDocument();
  });
});

describe("VisionThreatLimitations Component", () => {
  it("displays empirical clean-label centroid boundary disclosure", () => {
    const limitations: ThreatLimitation[] = [
      {
        threat_family: "clean_label_manipulation",
        status: "UNRESOLVED_LIMITATION",
        observed_recall: "0 / 12 (0.0% on held-out test split under FPR <= 0.05)",
        operating_fpr: "3.07% test FPR (calibrated on Val <= 5%)",
        statement: "Clean-label centroid manipulation remains an unresolved detector limitation under FPR <= 5%.",
        mitigation_recommendation: "Apply defensive training (Mixup alpha=0.20) as empirical mitigation.",
      },
    ];

    render(<VisionThreatLimitations limitations={limitations} />);

    expect(screen.getByText("Vision Threat Limitations & Clean-Label Defense Boundary")).toBeInTheDocument();
    expect(screen.getByText("clean_label_manipulation")).toBeInTheDocument();
    expect(screen.getByText("0 / 12 (0.0% on held-out test split under FPR <= 0.05)")).toBeInTheDocument();
    expect(screen.getByText("3.07% test FPR (calibrated on Val <= 5%)")).toBeInTheDocument();
  });
});

describe("ImageQuarantineReview Component", () => {
  it("displays data-integrity warning if count and candidate ID list mismatch", () => {
    const mismatchManifest: QuarantineManifest = {
      total_candidate_samples: 1200,
      quarantined_count: 89,
      quarantine_rate: 0.0742,
      operating_threshold: 0.86,
      threshold_policy: "maximize recall subject to FPR <= 0.05",
      requires_human_review: true,
      confirmation_status: "PENDING_OPERATOR_REVIEW",
      quarantined_sample_ids: [],
    };

    render(
      <ImageQuarantineReview
        manifest={mismatchManifest}
        datasetId="m6_cifar10_backdoor"
        onConfirmQuarantine={vi.fn()}
        isConfirming={false}
      />
    );

    expect(
      screen.getByText(/Data-Integrity Notice: Candidate Count & ID Discrepancy/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Full individual identifiers require sample evidence export before destructive removal/i)
    ).toBeInTheDocument();
  });
});

describe("ImageAgentInvestigation Component", () => {
  it("renders vision investigation preset prompt chips", () => {
    render(<ImageAgentInvestigation datasetId="m6_cifar10_backdoor" />);

    expect(screen.getByText("Why was this image dataset flagged?")).toBeInTheDocument();
    expect(screen.getByText("Which images should I review first?")).toBeInTheDocument();
    expect(screen.getByText(/Did the evaluation measure attack success rate/i)).toBeInTheDocument();
    expect(screen.getByText("What limitations remain unresolved?")).toBeInTheDocument();
  });
});
