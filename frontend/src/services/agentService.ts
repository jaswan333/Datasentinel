/**
 * DataSentinel Agent Service
 * Typed interface for agentic multi-agent investigations (Audit, Investigation, Reporter).
 */

import { apiRequest } from "./apiClient";
import type { AgentAskResponse } from "@/types/audit";

export const agentService = {
  /**
   * Submit an analyst investigation prompt grounded in the current dataset's audit evidence.
   */
  async askQuestion(datasetId: string, question: string): Promise<AgentAskResponse> {
    return apiRequest<AgentAskResponse>("/api/agent/ask", {
      method: "POST",
      body: JSON.stringify({
        dataset_id: datasetId,
        question: question.trim(),
      }),
    });
  },
};
