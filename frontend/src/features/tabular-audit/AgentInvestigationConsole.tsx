import React, { useState, useEffect, useRef } from "react";
import {
  Bot,
  Send,
  Loader2,
  AlertCircle,
  Sparkles,
  HelpCircle,
  Copy,
  Check,
  RotateCcw,
  ExternalLink,
  User,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { MarkdownRenderer } from "@/components/ui/MarkdownRenderer";
import { agentService } from "@/services/agentService";
import {
  type ChatMessage,
  extractEvidenceLinks,
} from "@/features/agent/agentHelpers";

interface AgentInvestigationConsoleProps {
  datasetId: string;
}

const TABULAR_PRESET_PROMPTS = [
  "Why was this dataset flagged?",
  "Which records should I review first?",
  "What evidence supports this finding?",
  "Which detectors were unavailable?",
  "What are the limitations of this audit?",
  "How does quarantine affect downstream model accuracy?",
];

export const AgentInvestigationConsole: React.FC<AgentInvestigationConsoleProps> = ({
  datasetId,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Clear or scope conversation when datasetId changes
  useEffect(() => {
    setMessages([]);
    setError(null);
  }, [datasetId]);

  const scrollToBottom = () => {
    if (typeof chatEndRef.current?.scrollIntoView === "function") {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    if (messages.length > 0) {
      scrollToBottom();
    }
  }, [messages, isLoading]);

  const handleAsk = async (promptToAsk?: string) => {
    const q = (promptToAsk || question).trim();
    if (!q || !datasetId || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      datasetId,
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setIsLoading(true);
    setError(null);

    try {
      const res = await agentService.askQuestion(datasetId, q);
      const evidenceLinks = extractEvidenceLinks(res.answer, "TABULAR");

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: res.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        datasetId,
        evidenceLinks,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      setError(
        err?.message ||
          "Agent workflow communication error. Audit results remain intact."
      );
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  };

  const handleCopy = (messageId: string, content: string) => {
    navigator.clipboard.writeText(content);
    setCopiedMessageId(messageId);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const handleClearHistory = () => {
    setMessages([]);
    setError(null);
  };

  const handleScrollToSection = (targetId: string) => {
    const element = document.getElementById(targetId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-xs overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-indigo-700" />
          <h3 className="text-sm font-semibold text-slate-900">
            Agentic Investigation & Analyst Q&A
          </h3>
          <span className="text-[10px] text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200 font-mono">
            Audit ID: {datasetId}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {messages.length > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleClearHistory}
              className="text-xs h-7 px-2 text-slate-500 hover:text-slate-700"
              icon={<RotateCcw className="w-3 h-3" />}
            >
              Clear Session
            </Button>
          )}
          <span className="text-[11px] text-slate-400 hidden sm:block">
            Grounded triad: Audit → Investigation → Reporter
          </span>
        </div>
      </div>

      <div className="p-4 space-y-4 text-xs">
        {/* Preset Prompt Chips */}
        <div className="space-y-1.5">
          <span className="text-[11px] text-slate-500 font-medium block">
            Suggested Investigation Inquiries:
          </span>
          <div className="flex flex-wrap gap-1.5">
            {TABULAR_PRESET_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => handleAsk(prompt)}
                disabled={isLoading}
                className="text-[11px] px-2.5 py-1 rounded bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 transition-colors disabled:opacity-50 text-left cursor-pointer"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>

        {/* Conversation History */}
        {messages.length > 0 && (
          <div className="space-y-3.5 max-h-96 overflow-y-auto pr-1 border border-slate-100 rounded-lg p-3 bg-slate-50/30">
            {messages.map((msg) => {
              const isUser = msg.role === "user";

              return (
                <div
                  key={msg.id}
                  className={`flex flex-col space-y-1.5 ${
                    isUser ? "items-end" : "items-start"
                  }`}
                >
                  <div className="flex items-center gap-1.5 text-[10px] text-slate-400 px-1">
                    {isUser ? (
                      <>
                        <span>Analyst</span>
                        <User className="w-3 h-3 text-slate-500" />
                        <span>• {msg.timestamp}</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3 h-3 text-indigo-600" />
                        <span className="font-medium text-slate-700">DataSentinel Triad</span>
                        <span>• {msg.timestamp}</span>
                      </>
                    )}
                  </div>

                  <div
                    className={`max-w-2xl rounded-lg p-3 text-xs ${
                      isUser
                        ? "bg-indigo-600 text-white rounded-br-none"
                        : "bg-white border border-slate-200 text-slate-800 rounded-bl-none shadow-xs space-y-2 w-full"
                    }`}
                  >
                    {isUser ? (
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    ) : (
                      <>
                        <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
                          <span className="text-[10px] font-semibold text-indigo-900 flex items-center gap-1">
                            Evidence-Grounded Findings
                          </span>
                          <button
                            type="button"
                            onClick={() => handleCopy(msg.id, msg.content)}
                            title="Copy response"
                            className="text-slate-400 hover:text-slate-700 p-1 rounded inline-flex items-center gap-1 text-[10px]"
                          >
                            {copiedMessageId === msg.id ? (
                              <>
                                <Check className="w-3 h-3 text-emerald-600" />
                                <span className="text-emerald-700">Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy className="w-3 h-3" />
                                <span>Copy</span>
                              </>
                            )}
                          </button>
                        </div>

                        {/* Markdown Renderer without literal syntax */}
                        <MarkdownRenderer content={msg.content} />

                        {/* Traceable Evidence Links */}
                        {msg.evidenceLinks && msg.evidenceLinks.length > 0 && (
                          <div className="pt-2 border-t border-slate-100 space-y-1">
                            <span className="text-[10px] font-semibold text-slate-500 block">
                              Traceable Evidence References:
                            </span>
                            <div className="flex flex-wrap gap-1">
                              {msg.evidenceLinks.map((link) => (
                                <button
                                  key={link.id}
                                  type="button"
                                  onClick={() => handleScrollToSection(link.targetElementId)}
                                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border border-indigo-200 transition-colors"
                                >
                                  <span>{link.label}</span>
                                  <ExternalLink className="w-2.5 h-2.5 opacity-60" />
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              );
            })}

            {isLoading && (
              <div className="flex items-start gap-2 p-3 bg-white border border-slate-200 rounded-lg text-xs text-slate-600 animate-pulse max-w-md">
                <Loader2 className="w-4 h-4 animate-spin text-indigo-600 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <p className="font-semibold text-slate-800 text-[11px]">
                    Synthesizing Grounded Analyst Response...
                  </p>
                  <p className="text-[10px] text-slate-400">
                    Querying Audit Agent and validating evidence across detector findings.
                  </p>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        )}

        {/* Error Notice with Retry */}
        {error && (
          <div className="p-3 bg-rose-50 border border-rose-200 rounded text-rose-800 flex items-start justify-between gap-2">
            <div className="flex items-start gap-2 text-[11px]">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold">Investigation Workflow Notice:</span> {error}
                <p className="text-[10px] text-rose-700 mt-0.5">
                  Audit results and detector findings remain intact above.
                </p>
              </div>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => handleAsk()}
              className="text-xs h-6 px-2 text-rose-800 border-rose-300 hover:bg-rose-100"
            >
              Retry
            </Button>
          </div>
        )}

        {/* Question Input Form */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleAsk();
          }}
          className="space-y-2"
        >
          <div className="relative">
            <textarea
              ref={textareaRef}
              rows={2}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask an evidence-grounded question about this dataset audit (Press Enter to submit, Shift+Enter for new line)..."
              disabled={isLoading}
              className="w-full text-xs bg-slate-50/50 border border-slate-300 rounded px-3 py-2 text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white resize-none"
            />
          </div>

          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-400 flex items-center gap-1">
              <HelpCircle className="w-3 h-3" />
              Strictly grounded in current audit data. Dataset content treated as data, not instructions.
            </span>

            <Button
              type="submit"
              variant="primary"
              size="sm"
              disabled={!question.trim() || isLoading}
              className="bg-indigo-700 hover:bg-indigo-800 text-white h-7 px-3 text-xs"
              icon={
                isLoading ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <Send className="w-3 h-3" />
                )
              }
            >
              {isLoading ? "Analyzing..." : "Ask Agent"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
