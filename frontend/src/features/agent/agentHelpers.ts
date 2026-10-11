/**
 * Helper types and utilities for DataSentinel Interactive Agent Investigations
 */

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  datasetId: string;
  evidenceLinks?: TraceableEvidenceLink[];
}

export interface TraceableEvidenceLink {
  id: string;
  label: string;
  category: "signal" | "sample" | "utility" | "limitation" | "quarantine";
  targetElementId: string;
}

/**
 * Parses markdown/text content to extract grounded evidence links to the current audit dashboard sections.
 */
export function extractEvidenceLinks(content: string, modality: "TABULAR" | "IMAGE"): TraceableEvidenceLink[] {
  const links: TraceableEvidenceLink[] = [];
  const lower = content.toLowerCase();

  // 1. Detectors / Signals
  const detectorPatterns = [
    "cleanlab_risk",
    "cleanlab",
    "lof_risk",
    "isolation_forest_risk",
    "spectral_signature_risk",
    "spectral",
    "local_patch_sensitivity_risk",
    "patch",
    "knn_embedding_risk",
    "activation_cluster_risk",
    "oof_label_disagreement",
    "prediction_entropy",
    "training_loss",
    "detector",
    "signals",
  ];
  if (detectorPatterns.some((pattern) => lower.includes(pattern))) {
    links.push({
      id: "evidence-detectors",
      label: modality === "IMAGE" ? "Vision Detectors (14 Signals)" : "Detector Evidence Breakdown",
      category: "signal",
      targetElementId: "detector-evidence-section",
    });
  }

  // 2. Suspicious Samples
  if (
    lower.includes("sample #") ||
    lower.includes("record #") ||
    lower.includes("img_") ||
    lower.includes("sample_") ||
    lower.includes("suspicious records") ||
    lower.includes("suspicious samples")
  ) {
    links.push({
      id: "evidence-samples",
      label: modality === "IMAGE" ? "Suspicious Image Samples" : "Prioritized Records Table",
      category: "sample",
      targetElementId: "suspicious-samples-section",
    });
  }

  // 3. Model Utility & Downstream Accuracy / ASR
  if (
    lower.includes("accuracy") ||
    lower.includes("asr") ||
    lower.includes("utility") ||
    lower.includes("attack success rate") ||
    lower.includes("delta") ||
    lower.includes("resnet18")
  ) {
    links.push({
      id: "evidence-utility",
      label: modality === "IMAGE" ? "ResNet18 Utility & ASR Mitigation" : "Downstream Model Utility",
      category: "utility",
      targetElementId: "model-utility-section",
    });
  }

  // 4. Quarantine Manifest
  if (
    lower.includes("quarantine") ||
    lower.includes("isolated") ||
    lower.includes("operator sign-off") ||
    lower.includes("staged")
  ) {
    links.push({
      id: "evidence-quarantine",
      label: "Quarantine Staging Manifest",
      category: "quarantine",
      targetElementId: "quarantine-manifest-section",
    });
  }

  // 5. Threat Limitations & Clean-Label Boundary
  if (
    lower.includes("limitation") ||
    lower.includes("clean-label") ||
    lower.includes("0/12") ||
    lower.includes("recall") ||
    lower.includes("boundary")
  ) {
    links.push({
      id: "evidence-limitations",
      label: "Documented Threat Boundaries",
      category: "limitation",
      targetElementId: "threat-limitations-section",
    });
  }

  return links;
}
