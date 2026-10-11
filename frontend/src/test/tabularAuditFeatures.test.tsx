import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// Components to test
import { DetectorEvidenceTable } from "@/features/tabular-audit/DetectorEvidenceTable";
import { AuditSummaryHeader } from "@/features/tabular-audit/AuditSummaryHeader";
import { SuspiciousSamplesTable } from "@/features/tabular-audit/SuspiciousSamplesTable";
import { QuarantineManifestCard } from "@/features/tabular-audit/QuarantineManifestCard";
import { ModelUtilityCard } from "@/features/tabular-audit/ModelUtilityCard";
import { ThreatLimitationsCard } from "@/features/tabular-audit/ThreatLimitationsCard";
import { AgentInvestigationConsole } from "@/features/tabular-audit/AgentInvestigationConsole";
import { formatPercent, formatAccuracy } from "@/lib/utils";

import type {
  UnifiedAuditResult,
  SignalDetail,
  SuspiciousSample,
  QuarantineManifest,
  ModelUtilityComparison,
  ThreatLimitation,
} from "@/types/audit";

describe("Formatting Utilities (Bug Fix Verification)", () => {
  it("formats high_risk_fraction correctly and never produces [object Object]%", () => {
    expect(formatPercent(0.05)).toBe("5.0%");
    expect(formatPercent(0.1234, 2)).toBe("12.34%");
    expect(formatPercent(0)).toBe("0.0%");
    expect(formatPercent(null)).toBe("N/A");
    expect(formatPercent(undefined)).toBe("N/A");
    expect(formatPercent({} as any)).toBe("N/A");
  });

  it("formats model accuracy without double-multiplying by 100", () => {
    expect(formatAccuracy(36.0)).toBe("36.00%");
    expect(formatAccuracy(38.25)).toBe("38.25%");
    expect(formatAccuracy(null)).toBe("N/A");
  });
});

describe("DetectorEvidenceTable", () => {
  const mockSignals: Record<string, SignalDetail> = {
    cleanlab_risk: {
      name: "cleanlab_risk",
      category: "label_consistency",
      weight: 0.2,
      status: "AVAILABLE",
      mean_risk: 0.15,
      high_risk_fraction: 0.08,
      description: "Confident learning out-of-fold label inconsistency",
    },
    trigger_rarity_risk: {
      name: "trigger_rarity_risk",
      category: "trigger_anomaly",
      weight: 0.15,
      status: "UNAVAILABLE",
      mean_risk: null,
      high_risk_fraction: null,
      description: "Low-frequency trigger pattern anomaly",
    },
  };

  it("renders signal rows and correctly formats high_risk_fraction as 8.0%", () => {
    render(<DetectorEvidenceTable signals={mockSignals} />);

    expect(screen.getByText("cleanlab_risk")).toBeInTheDocument();
    expect(screen.getByText("trigger_rarity_risk")).toBeInTheDocument();

    // High risk fraction should be 8.0%, NOT [object Object]%
    expect(screen.getByText("8.0%")).toBeInTheDocument();
    expect(screen.queryByText(/\[object Object\]/)).toBeNull();

    // Status badges
    expect(screen.getByText("AVAILABLE")).toBeInTheDocument();
    expect(screen.getByText("UNAVAILABLE")).toBeInTheDocument();
  });
});

describe("AuditSummaryHeader", () => {
  const mockAudit: UnifiedAuditResult = {
    audit_id: "test_tabular_audit_01",
    dataset_name: "UCI_Credit_Poisoned.csv",
    modality: "TABULAR",
    created_at: "2026-10-11T12:00:00Z",
    dataset_fingerprint: "a1b2c3d4e5f67890",
    n_rows: 1000,
    n_columns: 24,
    risk_score: 18.5,
    trust_score: 81.5,
    risk_level: "LOW_RISK",
    overall_status: "AUDITED_SECURE",
    evidence_coverage: 0.8,
    available_signals: ["cleanlab_risk", "lof_risk"],
    missing_signals: ["trigger_rarity_risk"],
    signal_breakdown: {},
    quarantine_manifest: {
      total_candidate_samples: 1000,
      quarantined_count: 5,
      quarantine_rate: 0.005,
      operating_threshold: 0.7,
      threshold_policy: "confidence >= 0.70",
      requires_human_review: true,
      confirmation_status: "PENDING_OPERATOR_REVIEW",
      quarantined_sample_ids: [],
    },
    unresolved_threat_limitations: [],
    recommended_action: "Dataset approved for monitored training.",
    sample_results: [],
  };

  it("renders audit summary metrics and explicit non-probability disclaimer", () => {
    render(<AuditSummaryHeader audit={mockAudit} isDemo={true} />);

    expect(screen.getByText("test_tabular_audit_01")).toBeInTheDocument();
    expect(screen.getByText("UCI_Credit_Poisoned.csv")).toBeInTheDocument();
    expect(screen.getByText("18.50")).toBeInTheDocument();
    expect(screen.getByText("81.50")).toBeInTheDocument();
    expect(screen.getByText("80.0%")).toBeInTheDocument();
    expect(screen.getByText(/DEMO — Sample Benchmark Dataset/)).toBeInTheDocument();
    expect(screen.getByText(/Evidence-weighted index \(not a safety probability\)/)).toBeInTheDocument();
  });
});

describe("SuspiciousSamplesTable", () => {
  const mockSamples: SuspiciousSample[] = [
    {
      sample_id: "rec_101",
      risk_score: 85.5,
      risk_tier: "HIGH",
      contributing_signals: { cleanlab_risk: 0.92, lof_risk: 0.78 },
      recommended_action: "ISOLATE",
    },
    {
      sample_id: "rec_202",
      risk_score: 45.0,
      risk_tier: "MEDIUM",
      contributing_signals: { isolation_forest_risk: 0.45 },
      recommended_action: "HUMAN_REVIEW",
    },
  ];

  it("binds sample risk tier directly and never renders UNKNOWN", () => {
    render(<SuspiciousSamplesTable samples={mockSamples} />);

    expect(screen.getByText("#rec_101")).toBeInTheDocument();
    expect(screen.getAllByText("HIGH").length).toBeGreaterThan(0);
    expect(screen.getAllByText("MEDIUM").length).toBeGreaterThan(0);
    expect(screen.queryByText("UNKNOWN")).toBeNull();
  });

  it("opens the detail drawer when Inspect is clicked", () => {
    render(<SuspiciousSamplesTable samples={mockSamples} />);

    const inspectBtns = screen.getAllByRole("button", { name: /inspect/i });
    fireEvent.click(inspectBtns[0]);

    expect(screen.getByText("Sample #rec_101 Evidence")).toBeInTheDocument();
    expect(screen.getByText("cleanlab_risk")).toBeInTheDocument();
  });
});

describe("QuarantineManifestCard (Data Integrity Check)", () => {
  it("renders explicit warning if candidate count and sample ID list disagree", () => {
    const mismatchManifest: QuarantineManifest = {
      total_candidate_samples: 1000,
      quarantined_count: 7,
      quarantine_rate: 0.007,
      operating_threshold: 0.7,
      threshold_policy: "confidence >= 0.70",
      requires_human_review: true,
      confirmation_status: "PENDING_OPERATOR_REVIEW",
      quarantined_sample_ids: [], // Disagreement: count 7 vs 0 IDs
    };

    render(
      <QuarantineManifestCard
        manifest={mismatchManifest}
        datasetId="ds_123"
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

describe("ModelUtilityCard", () => {
  it("displays accuracy in correct units (36.00%, 38.00%) without double-multiplying", () => {
    const utility: ModelUtilityComparison = {
      evaluation_dataset: "Credit Default Test Split",
      clean_reference_accuracy: 42.0,
      victim_clean_accuracy: 36.0,
      screened_clean_accuracy: 38.0,
      clean_accuracy_delta_pp: 2.0,
      evaluation_notes: "Evaluated with Random Forest (300 estimators)",
    };

    render(<ModelUtilityCard utility={utility} />);

    expect(screen.getByText("36.00%")).toBeInTheDocument();
    expect(screen.getByText("38.00%")).toBeInTheDocument();
    expect(screen.getByText("42.00%")).toBeInTheDocument();
    expect(screen.getByText("+2.00 pp")).toBeInTheDocument();
    expect(screen.queryByText("3600%")).toBeNull();
  });
});

describe("ThreatLimitationsCard", () => {
  it("remains visible even with zero risk score", () => {
    const limitations: ThreatLimitation[] = [
      {
        threat_family: "clean_label_centroid_blend",
        status: "UNRESOLVED_LIMITATION",
        observed_recall: "0/12 (0.0%)",
        operating_fpr: "3.07% FPR",
        statement: "Stealthy centroid blending evades distance filters.",
        mitigation_recommendation: "Deploy Mixup defensive training.",
      },
    ];

    render(<ThreatLimitationsCard limitations={limitations} />);

    expect(screen.getByText("Threat Limitations & Defense Boundaries")).toBeInTheDocument();
    expect(screen.getByText("clean_label_centroid_blend")).toBeInTheDocument();
    expect(screen.getByText(/Stealthy centroid blending evades distance filters/)).toBeInTheDocument();
  });
});

describe("AgentInvestigationConsole", () => {
  it("renders suggested investigation prompt chips", () => {
    render(<AgentInvestigationConsole datasetId="test_dataset_99" />);

    expect(screen.getByText("Why was this dataset flagged?")).toBeInTheDocument();
    expect(screen.getByText("Which records should I review first?")).toBeInTheDocument();
    expect(screen.getByText("What are the limitations of this audit?")).toBeInTheDocument();
  });
});
