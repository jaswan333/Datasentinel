import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { HomePage } from "@/pages/HomePage";
import { HelpAboutPage } from "@/pages/HelpAboutPage";
import { SettingsPage } from "@/pages/SettingsPage";
import { auditService } from "@/services/auditService";
import * as apiClient from "@/services/apiClient";

vi.mock("@/services/auditService", () => ({
  auditService: {
    getAuditHistory: vi.fn(),
  },
}));

vi.mock("@/services/apiClient", () => ({
  apiRequest: vi.fn(),
}));

describe("Phase 6 Remaining Pages Verification", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  describe("HomePage Recent Audits Integration", () => {
    it("renders recent audits from backend with correct modality and risk badges", async () => {
      vi.mocked(auditService.getAuditHistory).mockResolvedValueOnce([
        {
          dataset_id: "m6_cifar10_backdoor",
          dataset_name: "CIFAR-10 Trojan Backdoor",
          modality: "IMAGE",
          created_at: "2026-10-11T12:00:00Z",
          risk_score: 20.0,
          trust_score: 80.0,
          risk_level: "LOW_RISK",
          overall_status: "AUDITED_SECURE",
          evidence_coverage: 1.0,
          quarantined_count: 36,
          total_candidate_samples: 1200,
          confirmation_status: "PENDING_OPERATOR_REVIEW",
          has_unresolved_limitations: false,
        },
      ]);

      render(
        <BrowserRouter>
          <HomePage />
        </BrowserRouter>
      );

      expect(await screen.findByText("CIFAR-10 Trojan Backdoor")).toBeInTheDocument();
      expect(screen.getByText("IMAGE")).toBeInTheDocument();
      expect(screen.getByText("LOW (20.0)")).toBeInTheDocument();
      expect(screen.getByText("36 flagged")).toBeInTheDocument();
    });

    it("renders truthful empty state when no audits are stored", async () => {
      vi.mocked(auditService.getAuditHistory).mockResolvedValueOnce([]);

      render(
        <BrowserRouter>
          <HomePage />
        </BrowserRouter>
      );

      expect(await screen.findByText("No Audits Recorded Yet")).toBeInTheDocument();
      expect(screen.getByText("Start Tabular Audit")).toBeInTheDocument();
      expect(screen.getByText("Start Image Audit")).toBeInTheDocument();
    });
  });

  describe("HelpAboutPage System Documentation & Limitations", () => {
    it("renders the 4-step auditing workflow and upload specifications", () => {
      render(
        <BrowserRouter>
          <HelpAboutPage />
        </BrowserRouter>
      );

      expect(screen.getByText("The 4-Step Auditing Workflow")).toBeInTheDocument();
      expect(screen.getByText("Tabular ML Ingestion Specifications")).toBeInTheDocument();
      expect(screen.getByText("Vision DL Ingestion Specifications")).toBeInTheDocument();
      expect(screen.getByText(".csv, .xlsx, .xls")).toBeInTheDocument();
      expect(screen.getByText("150 MB uncompressed ceiling")).toBeInTheDocument();
    });

    it("renders scientific limitation of clean-label centroid manipulation", () => {
      render(
        <BrowserRouter>
          <HelpAboutPage />
        </BrowserRouter>
      );

      expect(
        screen.getByText(/What is the 0\/12 held-out recall clean-label limitation\?/i)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/How does the evidence-weighted Risk Index differ from a safety probability\?/i)
      ).toBeInTheDocument();
    });
  });

  describe("SettingsPage Preferences & Persistence", () => {
    it("toggles table density, saves to localStorage, and shows feedback", async () => {
      vi.mocked(apiClient.apiRequest).mockResolvedValueOnce({
        status: "healthy",
        version: "0.8.0",
        policy_version: "1.0.0-evidence-weighted",
      });

      render(
        <BrowserRouter>
          <SettingsPage />
        </BrowserRouter>
      );

      expect(await screen.findByText("ONLINE")).toBeInTheDocument();

      const compactBtn = screen.getByRole("button", { name: /compact/i });
      fireEvent.click(compactBtn);

      const saveBtn = screen.getByRole("button", { name: /save preferences/i });
      fireEvent.click(saveBtn);

      expect(await screen.findByText("Preferences saved successfully to local storage.")).toBeInTheDocument();
      const stored = JSON.parse(localStorage.getItem("datasentinel_user_preferences") || "{}");
      expect(stored.tableDensity).toBe("compact");
    });

    it("resets preferences to default when reset button is clicked", async () => {
      localStorage.setItem(
        "datasentinel_user_preferences",
        JSON.stringify({ tableDensity: "compact", defaultWorkspace: "deep-learning" })
      );

      render(
        <BrowserRouter>
          <SettingsPage />
        </BrowserRouter>
      );

      const resetBtn = screen.getByRole("button", { name: /reset defaults/i });
      fireEvent.click(resetBtn);

      expect(await screen.findByText("Preferences saved successfully to local storage.")).toBeInTheDocument();
      const stored = JSON.parse(localStorage.getItem("datasentinel_user_preferences") || "{}");
      expect(stored.tableDensity).toBe("comfortable");
    });
  });
});
