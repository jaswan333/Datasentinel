import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  ShieldCheck,
  AlertTriangle,
  Database,
  ScanEye,
  SlidersHorizontal,
  Bot,
  Layers,
  HelpCircle,
  ChevronDown,
  ChevronRight,
  HardDriveDownload,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

interface FAQItem {
  id: string;
  question: string;
  answer: React.ReactNode;
}

export const HelpAboutPage: React.FC = () => {
  const [openSection, setOpenSection] = useState<string | null>("workflow");
  const [openFAQ, setOpenFAQ] = useState<string | null>("risk-vs-prob");

  const toggleSection = (section: string) => {
    setOpenSection((prev) => (prev === section ? null : section));
  };

  const toggleFAQ = (faqId: string) => {
    setOpenFAQ((prev) => (prev === faqId ? null : faqId));
  };

  const faqs: FAQItem[] = [
    {
      id: "risk-vs-prob",
      question: "How does the evidence-weighted Risk Index differ from a safety probability?",
      answer: (
        <div className="space-y-2 text-xs text-slate-600 leading-relaxed">
          <p>
            The DataSentinel Risk Score <span className="font-mono font-semibold text-slate-800">R ∈ [0, 100]</span> is an
            <strong className="text-slate-900"> evidence-weighted composite index of diagnostic suspicion</strong>, calculated across
            all currently available and executed detector signals. It is <strong>not</strong> a calibrated probability of dataset safety.
          </p>
          <p>
            Labeling heuristic anomaly scores as &ldquo;99% safe&rdquo; or calibrated probabilities is methodologically invalid because
            real-world adversaries continuously craft out-of-distribution triggers. A score of 15.0 means low measured evidence of known
            perturbation signatures under tested detectors—it does not prove the total absence of novel or unmeasured attack vectors.
          </p>
        </div>
      ),
    },
    {
      id: "clean-label-limitation",
      question: "What is the 0/12 held-out recall clean-label limitation?",
      answer: (
        <div className="space-y-2 text-xs text-slate-600 leading-relaxed">
          <p>
            Under our strict false-positive constraint (<span className="font-mono">FPR<sub>val</sub> ≤ 0.05</span>), the current
            spectral signature and activation-clustering vision detectors achieved an empirical recall of
            <strong className="text-amber-800"> 0/12 (0.0%)</strong> on the held-out clean-label centroid manipulation benchmark split.
          </p>
          <p>
            Because clean-label attacks perturb images in pixel space while constraining latent representations within the legitimate class convex hull,
            they successfully evade standalone representation clustering under conservative operational false positive rates.
            <strong>Scientific integrity rule:</strong> This benchmark limitation is explicitly displayed on all audits and must not be generalized
            to claim all clean-label attacks are undetectable; rather, operators should combine audit screening with defensive training (e.g. Mixup).
          </p>
        </div>
      ),
    },
    {
      id: "benchmark-vs-upload",
      question: "What is the difference between benchmark evaluations and user-uploaded dataset audits?",
      answer: (
        <div className="space-y-2 text-xs text-slate-600 leading-relaxed">
          <p>
            <strong>Benchmarks (CIFAR-10 Trojan / UCI Credit):</strong> Run against verified reference datasets where ground-truth poisoned labels,
            clean partitions, and victim models are pre-evaluated. These evaluate exact Attack Success Rate (ASR) collapse and post-screening retrained accuracy.
          </p>
          <p>
            <strong>User-Uploaded Datasets:</strong> Run all applicable detector signals (Group C duplicate & statistics signals, Group B latent KNN/LOF/Centroid
            anomalies when class subdirectories exist). Because user uploads arrive without pre-trained victim weights or poison ground-truth,
            uncalibrated signals remain marked <span className="font-mono bg-slate-100 px-1 py-0.5 rounded text-slate-700">UNAVAILABLE</span>, and ASR is
            reported as &ldquo;Not evaluated&rdquo; rather than fabricated.
          </p>
        </div>
      ),
    },
    {
      id: "quarantine-policy",
      question: "Why is quarantine strictly non-destructive and requires human sign-off?",
      answer: (
        <div className="space-y-2 text-xs text-slate-600 leading-relaxed">
          <p>
            DataSentinel adheres to a non-destructive screening policy. Flagged suspicious samples are isolated in an auditable quarantine
            manifest (<span className="font-mono">PENDING_OPERATOR_REVIEW</span>) without immediately deleting or modifying underlying storage.
          </p>
          <p>
            Destructive auto-purging can inadvertently purge rare legitimate edge-cases or minority class examples, harming model fairness and utility.
            An authorized operator must review high-risk sample attribution evidence before signing off on permanent screening or retraining.
          </p>
        </div>
      ),
    },
    {
      id: "agent-grounding",
      question: "How does the Agentic AI stay grounded without hallucinating findings?",
      answer: (
        <div className="space-y-2 text-xs text-slate-600 leading-relaxed">
          <p>
            The Agentic Triad (Audit Agent → Investigation Agent → Reporting Agent) takes the current Unified Audit Result as its strict source of truth.
            Dataset contents and user prompts are treated as untrusted data, never as system instructions.
          </p>
          <p>
            If a question asks about an unavailable signal, unmeasured ASR, or absent sample ID, the agent explicitly responds with &ldquo;Not available&rdquo;
            and describes the relevant prerequisite rather than hallucinating plausible figures.
          </p>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-8 pb-10">
      {/* Top Breadcrumb & Header */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Link to="/" className="hover:text-indigo-600 flex items-center gap-1 transition-colors">
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Home</span>
          </Link>
          <span>/</span>
          <span className="text-slate-800 font-medium">Help & System Documentation</span>
        </div>

        <div className="bg-gradient-to-r from-slate-900 via-slate-850 to-slate-900 rounded-2xl p-6 sm:p-8 text-white shadow-md border border-slate-800">
          <div className="max-w-3xl space-y-3">
            <div className="inline-flex items-center gap-2">
              <Badge variant="info" size="sm" className="bg-indigo-500/20 text-indigo-300 border-indigo-500/30 font-mono">
                Platform Specification
              </Badge>
              <span className="text-xs text-slate-400 font-mono">
                Unified Architecture v1.0.0
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
              DataSentinel Documentation & Threat Architecture
            </h1>
            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed max-w-2xl">
              Comprehensive reference guide for ML security operators, detailing multi-modal detector mechanics,
              evidence-weighted risk calculation, upload limits, quarantine policies, and empirical threat boundaries.
            </p>
          </div>
        </div>
      </div>

      {/* 4-Step Auditing Life Cycle */}
      <Card className="border-slate-200">
        <CardHeader
          className="cursor-pointer hover:bg-slate-50/50 transition-colors"
          onClick={() => toggleSection("workflow")}
        >
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-700 flex items-center justify-center border border-indigo-100">
                <Layers className="w-4 h-4" />
              </div>
              <div>
                <CardTitle className="text-base text-slate-900">The 4-Step Auditing Workflow</CardTitle>
                <p className="text-xs text-slate-500">End-to-end dataset screening lifecycle for Tabular ML and Vision DL</p>
              </div>
            </div>
            {openSection === "workflow" ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
          </div>
        </CardHeader>
        {openSection === "workflow" && (
          <CardContent className="pt-2 pb-6 px-6 space-y-4 text-xs border-t border-slate-100">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-indigo-600 font-bold">Step 01</span>
                  <HardDriveDownload className="w-4 h-4 text-slate-400" />
                </div>
                <h3 className="font-semibold text-slate-900 text-sm">Ingest & Stage</h3>
                <p className="text-slate-600 leading-relaxed">
                  Upload CSV/Excel tabular files or ZIP archives containing JPG/PNG images. Files are safely staged, decompressed with zip-slip validation, and parsed for columns or labels.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-teal-600 font-bold">Step 02</span>
                  <SlidersHorizontal className="w-4 h-4 text-slate-400" />
                </div>
                <h3 className="font-semibold text-slate-900 text-sm">Inspect & Configure</h3>
                <p className="text-slate-600 leading-relaxed">
                  Review column types, suggested target variables, image resolution distributions, channel formats, and class balance. Select appropriate detectors compatible with the data.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-blue-600 font-bold">Step 03</span>
                  <ShieldCheck className="w-4 h-4 text-slate-400" />
                </div>
                <h3 className="font-semibold text-slate-900 text-sm">Execute Detectors</h3>
                <p className="text-slate-600 leading-relaxed">
                  Run multi-signal detection suite. Group C integrity/duplicate tests, Group B deep latent representations (ResNet18 embeddings), and Group A loss/prediction disagreement probes.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-purple-600 font-bold">Step 04</span>
                  <Bot className="w-4 h-4 text-slate-400" />
                </div>
                <h3 className="font-semibold text-slate-900 text-sm">Investigate & Sign-Off</h3>
                <p className="text-slate-600 leading-relaxed">
                  Ask grounded questions in the Agent Console, review suspicious sample drawers with clickable evidence links, and sign off on non-destructive quarantine manifests.
                </p>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Supported Upload Formats & System Specifications */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card className="border-slate-200">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-blue-600" />
              <CardTitle className="text-sm">Tabular ML Ingestion Specifications</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-xs text-slate-600">
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Supported Formats:</span>
              <span className="font-mono font-medium text-slate-800">.csv, .xlsx, .xls</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Maximum File Size:</span>
              <span className="font-mono font-medium text-slate-800">50 MB per file</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Supervised Detection:</span>
              <span className="text-slate-800 font-medium">Confident learning, OOF loss, margin analysis</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Unsupervised Detection:</span>
              <span className="text-slate-800 font-medium">Isolation forest, KNN distance, exact duplicates</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-500">Quarantine Strategy:</span>
              <span className="text-slate-800 font-medium">High risk tier isolation (FPR_val ≤ 0.05)</span>
            </div>
          </CardContent>
        </Card>

        <Card className="border-slate-200">
          <CardHeader>
            <div className="flex items-center gap-2">
              <ScanEye className="w-4 h-4 text-indigo-600" />
              <CardTitle className="text-sm">Vision DL Ingestion Specifications</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 text-xs text-slate-600">
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Supported Formats:</span>
              <span className="font-mono font-medium text-slate-800">.jpg, .jpeg, .png, .webp, .zip</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Maximum Archive Size:</span>
              <span className="font-mono font-medium text-slate-800">150 MB uncompressed ceiling</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Max Image Count:</span>
              <span className="font-mono font-medium text-slate-800">5,000 images per staged dataset</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-slate-100">
              <span className="text-slate-500">Class Label Inference:</span>
              <span className="text-slate-800 font-medium">Supported via folder paths (e.g. &ldquo;cats/img1.jpg&rdquo;)</span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-slate-500">Feature Extractor:</span>
              <span className="font-mono font-medium text-slate-800">ResNet18 512-dim latent representation</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Mathematical & Scientific Deep-Dive FAQs */}
      <Card className="border-slate-200">
        <CardHeader>
          <div className="flex items-center gap-2">
            <HelpCircle className="w-4 h-4 text-purple-600" />
            <CardTitle className="text-base text-slate-900">Scientific Integrity & Methodology Reference</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {faqs.map((faq) => (
            <div
              key={faq.id}
              className="border border-slate-200/80 rounded-xl overflow-hidden transition-colors"
            >
              <button
                type="button"
                onClick={() => toggleFAQ(faq.id)}
                className="w-full text-left p-4 bg-slate-50/50 hover:bg-slate-50 flex items-center justify-between gap-3 text-xs font-semibold text-slate-900 transition-colors"
              >
                <span>{faq.question}</span>
                {openFAQ === faq.id ? (
                  <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-slate-400 shrink-0" />
                )}
              </button>
              {openFAQ === faq.id && (
                <div className="p-4 bg-white border-t border-slate-100">
                  {faq.answer}
                </div>
              )}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Scientific Limitation Mandatory Box */}
      <div className="bg-amber-50/70 border border-amber-200 rounded-xl p-5 flex items-start gap-3.5 text-xs text-amber-950">
        <AlertTriangle className="w-5 h-5 text-amber-700 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-bold uppercase tracking-wider text-[11px] text-amber-900 font-mono">
            Scientific Boundary Notice
          </p>
          <p className="leading-relaxed">
            DataSentinel is designed to eliminate speculative security claims. All findings trace back to reproducible
            empirical detector outputs. If your use case involves advanced clean-label gradient poisoning or zero-day triggers,
            combine pre-training dataset screening with certified adversarial training defenses.
          </p>
        </div>
      </div>
    </div>
  );
};
