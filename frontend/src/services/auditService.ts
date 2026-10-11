/**
 * DataSentinel Audit Service
 * Typed interface for dataset schema inspection, tabular audit execution,
 * pre-computed benchmarks, quarantine manifests, and operator sign-offs.
 */

import { apiRequest } from "./apiClient";
import type {
  InspectColumnsResponse,
  AuditResponse,
  QuarantineDetailsResponse,
  QuarantineConfirmResponse,
  ThreatLimitation,
} from "@/types/audit";

export const auditService = {
  /**
   * Inspect column names, data types, and suggested targets/categoricals
   * without running the full detector pipeline.
   */
  async inspectColumns(file: File): Promise<InspectColumnsResponse> {
    const formData = new FormData();
    formData.append("file", file);

    return apiRequest<InspectColumnsResponse>("/api/inspect-columns", {
      method: "POST",
      body: formData,
    });
  },

  /**
   * Run the full multi-signal tabular audit on an uploaded dataset.
   */
  async runAudit(
    file: File,
    targetColumn?: string | null,
    categoricalColumns?: string[]
  ): Promise<AuditResponse> {
    const formData = new FormData();
    formData.append("file", file);

    if (targetColumn && targetColumn.trim().length > 0) {
      formData.append("target_column", targetColumn.trim());
    }

    if (categoricalColumns && categoricalColumns.length > 0) {
      formData.append("categorical_columns", categoricalColumns.join(","));
    }

    return apiRequest<AuditResponse>("/api/audit", {
      method: "POST",
      body: formData,
    });
  },

  /**
   * Load a verified benchmark scenario from the backend (e.g. tabular_credit).
   * Result is clearly tagged with is_demo = true.
   */
  async loadBenchmark(benchmarkId: string = "tabular_credit"): Promise<AuditResponse> {
    const response = await apiRequest<AuditResponse>(
      `/api/audit/benchmark/${encodeURIComponent(benchmarkId)}`,
      {
        method: "POST",
      }
    );

    return {
      ...response,
      is_demo: true,
    };
  },

  /**
   * Retrieve quarantine manifest and candidate sample records for review.
   */
  async getQuarantineDetails(datasetId: string): Promise<QuarantineDetailsResponse> {
    return apiRequest<QuarantineDetailsResponse>(
      `/api/audit/${encodeURIComponent(datasetId)}/quarantine`,
      {
        method: "GET",
      }
    );
  },

  /**
   * Operator sign-off approving non-destructive quarantine screening.
   */
  async confirmQuarantine(
    datasetId: string,
    notes: string = "Operator sign-off verified via DataSentinel"
  ): Promise<QuarantineConfirmResponse> {
    return apiRequest<QuarantineConfirmResponse>(
      `/api/audit/${encodeURIComponent(datasetId)}/quarantine/confirm`,
      {
        method: "POST",
        body: JSON.stringify({ notes }),
      }
    );
  },

  /**
   * Fetch explicit threat limitation disclosures for a dataset.
   */
  async getLimitations(
    datasetId: string
  ): Promise<{ dataset_id: string; unresolved_threat_limitations: ThreatLimitation[] }> {
    return apiRequest<{
      dataset_id: string;
      unresolved_threat_limitations: ThreatLimitation[];
    }>(`/api/audit/${encodeURIComponent(datasetId)}/limitations`, {
      method: "GET",
    });
  },

  /**
   * Upload and inspect image dataset files (JPG, PNG, WEBP, ZIP).
   */
  async inspectImageDataset(
    files: File[],
    datasetName?: string
  ): Promise<import("@/types/audit").ImageDatasetInspectionResponse> {
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    if (datasetName && datasetName.trim().length > 0) {
      formData.append("dataset_name", datasetName.trim());
    }

    return apiRequest<import("@/types/audit").ImageDatasetInspectionResponse>(
      "/api/image-datasets/inspect",
      {
        method: "POST",
        body: formData,
      }
    );
  },

  /**
   * Get staged image dataset inspection details.
   */
  async getImageDatasetInspection(
    datasetId: string
  ): Promise<import("@/types/audit").ImageDatasetInspectionResponse> {
    return apiRequest<import("@/types/audit").ImageDatasetInspectionResponse>(
      `/api/image-datasets/${encodeURIComponent(datasetId)}`,
      {
        method: "GET",
      }
    );
  },

  /**
   * Run the vision security detectors on an uploaded staged image dataset.
   */
  async runUploadedImageAudit(datasetId: string): Promise<AuditResponse> {
    return apiRequest<AuditResponse>("/api/image-datasets/audit", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId }),
    });
  },

  /**
   * Fetch recent real audit history from backend runtime/results.
   */
  async getAuditHistory(limit: number = 10): Promise<import("@/types/audit").AuditHistoryItem[]> {
    return apiRequest<import("@/types/audit").AuditHistoryItem[]>(
      `/api/audit/history?limit=${limit}`,
      {
        method: "GET",
      }
    );
  },
};
