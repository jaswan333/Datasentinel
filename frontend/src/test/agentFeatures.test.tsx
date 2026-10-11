import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MarkdownRenderer } from "@/components/ui/MarkdownRenderer";
import { AgentInvestigationConsole } from "@/features/tabular-audit/AgentInvestigationConsole";
import { ImageAgentInvestigation } from "@/features/image-audit/ImageAgentInvestigation";
import { agentService } from "@/services/agentService";
import { extractEvidenceLinks } from "@/features/agent/agentHelpers";

// Mock agentService
vi.mock("@/services/agentService", () => ({
  agentService: {
    askQuestion: vi.fn(),
  },
}));

describe("Phase 4: MarkdownRenderer Component", () => {
  it("renders headings, bold text, lists, and tables without raw Markdown characters", () => {
    const markdownSample = `
### Key Security Findings
The dataset contains **36 quarantined candidates**.

* First item with \`cleanlab_risk\`
* Second item with \`spectral_signature_risk\`

| Detector | Status | Risk |
| --- | --- | --- |
| cleanlab | AVAILABLE | 0.85 |
`;

    render(<MarkdownRenderer content={markdownSample} />);

    // Verify h3 heading is rendered as heading element, not raw '###'
    expect(screen.getByRole("heading", { level: 3, name: "Key Security Findings" })).toBeInTheDocument();
    expect(screen.queryByText(/### Key Security Findings/)).toBeNull();

    // Verify bold text
    expect(screen.getByText("36 quarantined candidates")).toBeInTheDocument();

    // Verify lists
    expect(screen.getByText(/First item with/)).toBeInTheDocument();

    // Verify table structure
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("Detector")).toBeInTheDocument();
    expect(screen.getByText("cleanlab")).toBeInTheDocument();
  });
});

describe("Phase 4: Traceable Evidence Extraction", () => {
  it("extracts grounded evidence links for tabular and vision signals", () => {
    const tabularReport = "High risk detected by cleanlab_risk on sample_001. A total of 10 samples in quarantine.";
    const tabularLinks = extractEvidenceLinks(tabularReport, "TABULAR");

    expect(tabularLinks.some((l) => l.targetElementId === "detector-evidence-section")).toBe(true);
    expect(tabularLinks.some((l) => l.targetElementId === "suspicious-samples-section")).toBe(true);
    expect(tabularLinks.some((l) => l.targetElementId === "quarantine-manifest-section")).toBe(true);

    const visionReport = "Spectral signature outlier detected on img_0042. ResNet18 ASR collapsed from 99.64% to 6.18%. Clean-label recall limitation 0/12.";
    const visionLinks = extractEvidenceLinks(visionReport, "IMAGE");

    expect(visionLinks.some((l) => l.targetElementId === "detector-evidence-section")).toBe(true);
    expect(visionLinks.some((l) => l.targetElementId === "suspicious-samples-section")).toBe(true);
    expect(visionLinks.some((l) => l.targetElementId === "model-utility-section")).toBe(true);
    expect(visionLinks.some((l) => l.targetElementId === "threat-limitations-section")).toBe(true);
  });
});

describe("Phase 4: Tabular AgentInvestigationConsole", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("submits question, renders markdown answer, and extracts evidence links", async () => {
    vi.mocked(agentService.askQuestion).mockResolvedValueOnce({
      dataset_id: "m4_credit",
      question: "Why was this dataset flagged?",
      answer: "### Finding\nFlagged due to **cleanlab_risk** on Sample #sample_001. Quarantine manifest contains 10 records.",
    });

    render(<AgentInvestigationConsole datasetId="m4_credit" />);

    const chip = screen.getByRole("button", { name: "Why was this dataset flagged?" });
    fireEvent.click(chip);

    await waitFor(() => {
      expect(agentService.askQuestion).toHaveBeenCalledWith("m4_credit", "Why was this dataset flagged?");
    });

    // Verify response is rendered with Markdown heading
    expect(await screen.findByRole("heading", { level: 3, name: "Finding" })).toBeInTheDocument();
    expect(screen.getByText("cleanlab_risk")).toBeInTheDocument();

    // Verify evidence references
    expect(screen.getByText("Detector Evidence Breakdown")).toBeInTheDocument();
    expect(screen.getByText("Prioritized Records Table")).toBeInTheDocument();
  });

  it("supports conversational follow-up questions while retaining dataset ID", async () => {
    vi.mocked(agentService.askQuestion)
      .mockResolvedValueOnce({
        dataset_id: "m4_credit",
        question: "Why was this dataset flagged?",
        answer: "Initial analysis: Found out-of-fold label inconsistency.",
      })
      .mockResolvedValueOnce({
        dataset_id: "m4_credit",
        question: "What about model utility?",
        answer: "Accuracy shifted by +2.00 pp after screening.",
      });

    render(<AgentInvestigationConsole datasetId="m4_credit" />);

    // First question
    const chip = screen.getByRole("button", { name: "Why was this dataset flagged?" });
    fireEvent.click(chip);

    expect(await screen.findByText("Initial analysis: Found out-of-fold label inconsistency.")).toBeInTheDocument();

    // Follow-up question via textarea
    const textarea = screen.getByPlaceholderText(/Ask an evidence-grounded question/i);
    fireEvent.change(textarea, { target: { value: "What about model utility?" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });

    await waitFor(() => {
      expect(agentService.askQuestion).toHaveBeenCalledWith("m4_credit", "What about model utility?");
    });

    // Both messages exist in conversation history
    expect(screen.getByText("Initial analysis: Found out-of-fold label inconsistency.")).toBeInTheDocument();
    expect(await screen.findByText("Accuracy shifted by +2.00 pp after screening.")).toBeInTheDocument();
  });

  it("clears conversation history when Clear Session is clicked", async () => {
    vi.mocked(agentService.askQuestion).mockResolvedValueOnce({
      dataset_id: "m4_credit",
      question: "Why was this dataset flagged?",
      answer: "Dataset was flagged for label anomalies.",
    });

    render(<AgentInvestigationConsole datasetId="m4_credit" />);

    const chip = screen.getByRole("button", { name: "Why was this dataset flagged?" });
    fireEvent.click(chip);

    expect(await screen.findByText("Dataset was flagged for label anomalies.")).toBeInTheDocument();

    // Click Clear Session
    const clearBtn = screen.getByRole("button", { name: /clear session/i });
    fireEvent.click(clearBtn);

    expect(screen.queryByText("Dataset was flagged for label anomalies.")).toBeNull();
  });
});

describe("Phase 4: Vision ImageAgentInvestigation", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("handles vision-specific investigation prompt with ResNet18 and ASR references", async () => {
    vi.mocked(agentService.askQuestion).mockResolvedValueOnce({
      dataset_id: "m6_cifar10_backdoor",
      question: "Did the evaluation measure attack success rate?",
      answer: "### Evaluation Metrics\nVictim model ASR was **99.64%**, which collapsed to **6.18%** under screened retraining (93.45 pp drop).",
    });

    render(<ImageAgentInvestigation datasetId="m6_cifar10_backdoor" />);

    const chip = screen.getByRole("button", { name: "Did the evaluation measure attack success rate?" });
    fireEvent.click(chip);

    await waitFor(() => {
      expect(agentService.askQuestion).toHaveBeenCalledWith(
        "m6_cifar10_backdoor",
        "Did the evaluation measure attack success rate?"
      );
    });

    expect(await screen.findByRole("heading", { level: 3, name: "Evaluation Metrics" })).toBeInTheDocument();
    expect(screen.getByText("99.64%")).toBeInTheDocument();
    expect(screen.getByText("ResNet18 Utility & ASR Mitigation")).toBeInTheDocument();
  });
});
