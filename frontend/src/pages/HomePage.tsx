import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  ShieldCheck,
  Database,
  ScanEye,
  Brain,
  Filter,
  ArrowRight,
  ShieldAlert,
  SlidersHorizontal,
  FileCheck2,
  AlertTriangle,
  FileSearch,
  Clock,
  ExternalLink,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { auditService } from "@/services/auditService";
import type { AuditHistoryItem } from "@/types/audit";

export const HomePage: React.FC = () => {
  const [history, setHistory] = useState<AuditHistoryItem[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setIsLoadingHistory(true);
    setHistoryError(null);
    try {
      const data = await auditService.getAuditHistory(10);
      setHistory(data);
    } catch (err: any) {
      setHistoryError(err.message || "Failed to load audit history from backend.");
    } finally {
      setIsLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const formatDate = (isoStr: string) => {
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return isoStr;
    }
  };

  const getRiskBadge = (level: string, score: number) => {
    switch (level) {
      case "CRITICAL_RISK":
        return <Badge variant="danger" size="sm">CRITICAL ({score.toFixed(1)})</Badge>;
      case "ELEVATED_RISK":
        return <Badge variant="warning" size="sm">ELEVATED ({score.toFixed(1)})</Badge>;
      case "MODERATE_RISK":
        return <Badge variant="warning" size="sm">MODERATE ({score.toFixed(1)})</Badge>;
      default:
        return <Badge variant="success" size="sm">LOW ({score.toFixed(1)})</Badge>;
    }
  };

  return (
    <div className="space-y-12 pb-8">
      {/* 1. HERO SECTION */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950 text-white p-8 sm:p-12 lg:p-16 border border-slate-800 shadow-xl">
        <div
          className="absolute inset-0 opacity-10 bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:16px_16px] pointer-events-none"
          aria-hidden="true"
        />

        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center gap-2">
            <Badge
              variant="info"
              size="md"
              className="bg-blue-900/50 text-blue-300 border-blue-700/60 font-mono tracking-wide"
            >
              <ShieldCheck className="w-3.5 h-3.5 mr-1" />
              Dataset Security & Trust Platform
            </Badge>
            <span className="text-xs text-slate-400 font-mono hidden sm:inline">
              Milestone 8 Unified Architecture
            </span>
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Trust Your <span className="text-blue-400">Training Data.</span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl">
            Investigate suspicious training data with evidence-based machine-learning
            and deep-learning signals, supported by agentic AI explanations.
          </p>

          {/* Action buttons */}
          <div className="flex flex-wrap items-center gap-4 pt-2">
            <Link to="/machine-learning">
              <Button
                variant="primary"
                size="lg"
                icon={<Database className="w-4 h-4" />}
                className="shadow-md shadow-blue-600/30"
              >
                Explore Tabular Audits
              </Button>
            </Link>

            <Link to="/deep-learning">
              <Button
                variant="outline"
                size="lg"
                icon={<ScanEye className="w-4 h-4 text-slate-400" />}
                className="bg-slate-800/80 hover:bg-slate-800 text-slate-100 border-slate-700 hover:border-slate-600"
              >
                Explore Image Audits
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* 2. RECENT AUDITS SECTION (REAL BACKEND AUDIT HISTORY) */}
      <section className="space-y-4" aria-labelledby="history-heading">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2
              id="history-heading"
              className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight"
            >
              Recent Security Audits
            </h2>
            <p className="text-sm text-slate-500">
              Live audit history persisted in the backend runtime storage.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchHistory}
            disabled={isLoadingHistory}
            icon={<RefreshCw className={`w-3.5 h-3.5 ${isLoadingHistory ? "animate-spin" : ""}`} />}
            className="self-start sm:self-auto text-xs"
          >
            Refresh History
          </Button>
        </div>

        <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
          {isLoadingHistory ? (
            <div className="p-8 text-center space-y-3">
              <div className="inline-block animate-spin rounded-full h-6 w-6 border-2 border-indigo-600 border-t-transparent" />
              <p className="text-xs text-slate-500">Loading audit history from runtime store...</p>
            </div>
          ) : historyError ? (
            <div className="p-6 text-center space-y-3 bg-red-50/50">
              <AlertTriangle className="w-6 h-6 text-amber-600 mx-auto" />
              <p className="text-xs text-slate-700 font-medium">{historyError}</p>
              <Button variant="outline" size="sm" onClick={fetchHistory} className="text-xs">
                Retry Connection
              </Button>
            </div>
          ) : history.length === 0 ? (
            <div className="p-10 text-center space-y-3">
              <Clock className="w-8 h-8 text-slate-400 mx-auto" />
              <div className="space-y-1">
                <p className="text-sm font-semibold text-slate-800">No Audits Recorded Yet</p>
                <p className="text-xs text-slate-500 max-w-md mx-auto">
                  Run a Tabular audit or Image inspection to generate persistent evidence records.
                </p>
              </div>
              <div className="flex justify-center gap-3 pt-2">
                <Link to="/machine-learning">
                  <Button variant="outline" size="sm">Start Tabular Audit</Button>
                </Link>
                <Link to="/deep-learning">
                  <Button variant="outline" size="sm">Start Image Audit</Button>
                </Link>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50/80 border-b border-slate-200 text-slate-600 font-medium">
                    <th className="py-3 px-4">Dataset Name / Identifier</th>
                    <th className="py-3 px-3">Modality</th>
                    <th className="py-3 px-3">Recorded</th>
                    <th className="py-3 px-3">Evidence Risk Index</th>
                    <th className="py-3 px-3">Trust Score</th>
                    <th className="py-3 px-3">Coverage</th>
                    <th className="py-3 px-3">Quarantine</th>
                    <th className="py-3 px-3">Status</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {history.map((item) => {
                    const isTabular = item.modality === "TABULAR";
                    const targetLink = isTabular ? "/machine-learning" : "/deep-learning";

                    return (
                      <tr key={item.dataset_id} className="hover:bg-slate-50/60 transition-colors">
                        <td className="py-3 px-4">
                          <div className="font-semibold text-slate-900 max-w-[220px] truncate" title={item.dataset_name}>
                            {item.dataset_name}
                          </div>
                          <div className="text-[10px] text-slate-400 font-mono truncate max-w-[200px]">
                            {item.dataset_id}
                          </div>
                        </td>
                        <td className="py-3 px-3">
                          <Badge
                            variant={isTabular ? "default" : "info"}
                            size="sm"
                            className="font-mono text-[10px]"
                          >
                            {item.modality}
                          </Badge>
                        </td>
                        <td className="py-3 px-3 text-slate-500 whitespace-nowrap">
                          {formatDate(item.created_at)}
                        </td>
                        <td className="py-3 px-3">
                          {getRiskBadge(item.risk_level, item.risk_score)}
                        </td>
                        <td className="py-3 px-3 font-semibold text-slate-700">
                          {item.trust_score.toFixed(1)}
                        </td>
                        <td className="py-3 px-3 text-slate-600 font-mono">
                          {(item.evidence_coverage * 100).toFixed(0)}%
                        </td>
                        <td className="py-3 px-3">
                          {item.quarantined_count > 0 ? (
                            <span className="text-amber-800 font-semibold bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                              {item.quarantined_count} flagged
                            </span>
                          ) : (
                            <span className="text-slate-400">0 staged</span>
                          )}
                        </td>
                        <td className="py-3 px-3">
                          <span className="text-[11px] font-medium text-slate-700">
                            {item.overall_status}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right whitespace-nowrap">
                          <Link to={targetLink}>
                            <Button variant="ghost" size="sm" className="h-7 text-xs text-indigo-700 hover:text-indigo-800 gap-1 px-2">
                              Inspect <ExternalLink className="w-3 h-3" />
                            </Button>
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      {/* 3. CORE FEATURES SECTION */}
      <section className="space-y-6" aria-labelledby="features-heading">
        <div className="space-y-1">
          <h2
            id="features-heading"
            className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight"
          >
            Core Security Capabilities
          </h2>
          <p className="text-sm text-slate-500">
            Engineered for high fidelity detection without heuristic hand-waving or ungrounded probabilities.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
          {/* Feature 1 */}
          <Card hoverEffect className="border-slate-200">
            <CardContent className="p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center border border-blue-100">
                <ShieldAlert className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-slate-900">
                Threat Detection
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Analyzes tabular or image datasets using the specialized detectors available
                in the platform, covering label flips, perturbations, duplicate injections,
                and Trojan backdoors.
              </p>
            </CardContent>
          </Card>

          {/* Feature 2 */}
          <Card hoverEffect className="border-slate-200">
            <CardContent className="p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center border border-teal-100">
                <SlidersHorizontal className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-slate-900">
                Evidence-Based Risk
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Combines available detector signals into an interpretable risk assessment.
                The score represents an evidence-weighted index of suspicion, not an
                uncalibrated safety probability.
              </p>
            </CardContent>
          </Card>

          {/* Feature 3 */}
          <Card hoverEffect className="border-slate-200">
            <CardContent className="p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-purple-50 text-purple-600 flex items-center justify-center border border-purple-100">
                <Brain className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-slate-900">
                Agentic Investigation
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                A multi-agent security council (Audit, Investigation, Reporting) summarizes
                findings, inspects individual sample evidence, and answers questions grounded
                strictly in recorded detector facts.
              </p>
            </CardContent>
          </Card>

          {/* Feature 4 */}
          <Card hoverEffect className="border-slate-200">
            <CardContent className="p-6 space-y-3">
              <div className="w-10 h-10 rounded-xl bg-amber-50 text-amber-600 flex items-center justify-center border border-amber-100">
                <Filter className="w-5 h-5" />
              </div>
              <h3 className="text-base font-semibold text-slate-900">
                Controlled Review
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                Suspicious samples are staged in an auditable quarantine manifest for human
                operator review before any destructive removal or retraining action is confirmed.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 4. HOW IT WORKS (3-STEP WORKFLOW) */}
      <section className="space-y-6" aria-labelledby="workflow-heading">
        <div className="space-y-1">
          <h2
            id="workflow-heading"
            className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight"
          >
            How DataSentinel Operates
          </h2>
          <p className="text-sm text-slate-500">
            A structured three-step auditing lifecycle designed for security operations and ML engineering teams.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative">
          {/* Step 1 */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-sm relative space-y-3">
            <div className="flex items-center justify-between">
              <div className="w-8 h-8 rounded-lg bg-blue-600 text-white font-mono font-bold text-xs flex items-center justify-center shadow-sm">
                01
              </div>
              <FileSearch className="w-5 h-5 text-slate-400" />
            </div>
            <h3 className="font-semibold text-slate-900 text-base">
              Select Dataset or Demonstration
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Ingest training data (CSV / Excel files for tabular, image archives for vision)
              or select a pre-calibrated baseline benchmark to examine threats.
            </p>
          </div>

          {/* Step 2 */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-sm relative space-y-3">
            <div className="flex items-center justify-between">
              <div className="w-8 h-8 rounded-lg bg-teal-600 text-white font-mono font-bold text-xs flex items-center justify-center shadow-sm">
                02
              </div>
              <SlidersHorizontal className="w-5 h-5 text-slate-400" />
            </div>
            <h3 className="font-semibold text-slate-900 text-base">
              Review Detector Signals & Evidence
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Inspect multi-signal telemetry: confident learning out-of-fold disagreement,
              isolation forests, deep latent clusters, and frequency distributions.
            </p>
          </div>

          {/* Step 3 */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-sm relative space-y-3">
            <div className="flex items-center justify-between">
              <div className="w-8 h-8 rounded-lg bg-slate-900 text-white font-mono font-bold text-xs flex items-center justify-center shadow-sm">
                03
              </div>
              <FileCheck2 className="w-5 h-5 text-slate-400" />
            </div>
            <h3 className="font-semibold text-slate-900 text-base">
              Investigate Findings & Review Actions
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Examine high-risk samples with agentic explanations, review staged quarantine
              candidates, and compare model accuracy before confirming screening.
            </p>
          </div>
        </div>
      </section>

      {/* 5. ANALYSIS MODES (TABULAR VS VISION) */}
      <section className="space-y-6" aria-labelledby="modes-heading">
        <div className="space-y-1">
          <h2
            id="modes-heading"
            className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight"
          >
            Audit Modalities
          </h2>
          <p className="text-sm text-slate-500">
            Select the pipeline matching your dataset modality.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Mode Card 1: Tabular */}
          <Card hoverEffect className="border-slate-200/90 flex flex-col justify-between">
            <CardContent className="p-8 space-y-6">
              <div className="flex items-center justify-between">
                <div className="w-12 h-12 rounded-xl bg-blue-50 border border-blue-200 text-blue-600 flex items-center justify-center shadow-sm">
                  <Database className="w-6 h-6" />
                </div>
                <Badge variant="default" size="sm">
                  10 Signals • Supervised & Unsupervised
                </Badge>
              </div>

              <div className="space-y-2">
                <h3 className="text-xl font-bold text-slate-900">
                  Tabular Machine Learning
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Investigate supported structured datasets such as CSV or Excel files where
                  the backend supports them. Evaluates label noise, tree isolation paths,
                  near-duplicate records, and trigger rarity signatures.
                </p>
              </div>

              <div className="pt-2">
                <Link to="/machine-learning">
                  <Button
                    variant="primary"
                    size="md"
                    icon={<ArrowRight className="w-4 h-4" />}
                    className="w-full sm:w-auto"
                  >
                    Open Machine Learning
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>

          {/* Mode Card 2: Deep Learning */}
          <Card hoverEffect className="border-slate-200/90 flex flex-col justify-between">
            <CardContent className="p-8 space-y-6">
              <div className="flex items-center justify-between">
                <div className="w-12 h-12 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-600 flex items-center justify-center shadow-sm">
                  <ScanEye className="w-6 h-6" />
                </div>
                <Badge variant="info" size="sm">
                  14 Signals • ResNet18 Embeddings
                </Badge>
              </div>

              <div className="space-y-2">
                <h3 className="text-xl font-bold text-slate-900">
                  Image Deep Learning
                </h3>
                <p className="text-sm text-slate-600 leading-relaxed">
                  Investigate supported image datasets with the existing vision detectors.
                  Extracts 512-dimensional latent feature shifts, out-of-fold loss probes,
                  dHash perceptual collisions, and spatial frequency attenuation.
                </p>
              </div>

              <div className="pt-2">
                <Link to="/deep-learning">
                  <Button
                    variant="secondary"
                    size="md"
                    icon={<ArrowRight className="w-4 h-4" />}
                    className="w-full sm:w-auto"
                  >
                    Open Deep Learning
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 6. SAMPLE DEMONSTRATION ENTRY POINTS */}
      <section className="space-y-6" aria-labelledby="demos-heading">
        <div className="space-y-1">
          <h2
            id="demos-heading"
            className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight"
          >
            Validated Benchmark Scenarios
          </h2>
          <p className="text-sm text-slate-500">
            Pre-computed reference evaluations across established threat scenarios.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Demo 1 */}
          <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200/80 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-blue-600" />
                <span className="font-semibold text-slate-900 text-sm">
                  Tabular Demonstration
                </span>
              </div>
              <Badge variant="neutral" size="sm">
                Reference Scenario
              </Badge>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              UCI Credit Card Default benchmark with label-flipping simulation. Demonstrates
              out-of-fold label inconsistency detection and feature space isolation.
            </p>
            <div className="p-3 bg-white rounded-lg border border-slate-200/70 text-xs text-slate-500 flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-slate-700 font-medium">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                Scenario Ready
              </span>
              <Link to="/machine-learning?benchmark=tabular_credit">
                <Button variant="outline" size="sm" className="h-7 text-xs">
                  Run in ML Workspace
                </Button>
              </Link>
            </div>
          </div>

          {/* Demo 2 */}
          <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200/80 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <ScanEye className="w-4 h-4 text-indigo-600" />
                <span className="font-semibold text-slate-900 text-sm">
                  Image Deep Learning Demonstration
                </span>
              </div>
              <Badge variant="neutral" size="sm">
                Reference Scenario
              </Badge>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              CIFAR-10 Trojan Backdoor and Clean-Label Centroid Manipulation benchmark.
              Demonstrates 14-signal fusion, ASR suppression, and explicit boundary disclosure.
            </p>
            <div className="p-3 bg-white rounded-lg border border-slate-200/70 text-xs text-slate-500 flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-slate-700 font-medium">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                Scenario Ready
              </span>
              <Link to="/deep-learning?benchmark=image_backdoor">
                <Button variant="outline" size="sm" className="h-7 text-xs">
                  Run in DL Workspace
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* 7. TRUST & LIMITATIONS MANDATORY CALLOUT */}
      <section className="bg-amber-50/70 border border-amber-200 rounded-2xl p-6 sm:p-8" aria-label="Security Notice">
        <div className="flex items-start gap-4">
          <div className="p-2 rounded-xl bg-amber-100 text-amber-800 shrink-0 mt-0.5">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div className="space-y-1.5">
            <h3 className="text-sm font-bold text-amber-900 uppercase tracking-wider font-mono">
              Scientific Integrity & Threat Limitations Note
            </h3>
            <p className="text-xs sm:text-sm text-amber-900/90 leading-relaxed">
              Detection depends on the threat type, dataset, and available evidence. A low risk
              score does not guarantee that a dataset is free of poisoning. Review unresolved
              threat limitations before making security decisions.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
