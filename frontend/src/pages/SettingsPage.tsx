import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  Sliders,
  CheckCircle2,
  RotateCcw,
  Activity,
  HardDrive,
  Check,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { apiRequest } from "@/services/apiClient";

export interface UserPreferences {
  tableDensity: "comfortable" | "compact";
  defaultWorkspace: "machine-learning" | "deep-learning";
  numericPrecision: "1" | "2";
  timestampFormat: "relative" | "utc";
}

const DEFAULT_PREFERENCES: UserPreferences = {
  tableDensity: "comfortable",
  defaultWorkspace: "machine-learning",
  numericPrecision: "1",
  timestampFormat: "relative",
};

const PREFS_STORAGE_KEY = "datasentinel_user_preferences";

export const SettingsPage: React.FC = () => {
  const [preferences, setPreferences] = useState<UserPreferences>(() => {
    try {
      const stored = localStorage.getItem(PREFS_STORAGE_KEY);
      if (stored) {
        return { ...DEFAULT_PREFERENCES, ...JSON.parse(stored) };
      }
    } catch {
      // Fallback
    }
    return DEFAULT_PREFERENCES;
  });

  const [savedFeedback, setSavedFeedback] = useState(false);
  const [apiHealth, setApiHealth] = useState<{
    status: string;
    version: string;
    policy_version: string;
  } | null>(null);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);

  const checkHealth = async () => {
    setIsCheckingHealth(true);
    try {
      const data = await apiRequest<{
        status: string;
        version: string;
        policy_version: string;
      }>("/api/health");
      setApiHealth(data);
    } catch {
      setApiHealth({
        status: "offline",
        version: "unknown",
        policy_version: "unknown",
      });
    } finally {
      setIsCheckingHealth(false);
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const handleSave = () => {
    try {
      localStorage.setItem(PREFS_STORAGE_KEY, JSON.stringify(preferences));
      setSavedFeedback(true);
      setTimeout(() => setSavedFeedback(false), 3000);
    } catch {
      // Storage error
    }
  };

  const handleReset = () => {
    setPreferences(DEFAULT_PREFERENCES);
    try {
      localStorage.setItem(PREFS_STORAGE_KEY, JSON.stringify(DEFAULT_PREFERENCES));
      setSavedFeedback(true);
      setTimeout(() => setSavedFeedback(false), 3000);
    } catch {
      // Storage error
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Top Breadcrumb */}
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Link to="/" className="hover:text-indigo-600 flex items-center gap-1 transition-colors">
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Home</span>
          </Link>
          <span>/</span>
          <span className="text-slate-800 font-medium">Settings & Preferences</span>
        </div>

        <div className="bg-gradient-to-r from-slate-900 via-slate-850 to-slate-900 rounded-2xl p-6 sm:p-8 text-white shadow-md border border-slate-800">
          <div className="max-w-2xl space-y-2">
            <Badge variant="info" size="sm" className="bg-teal-500/20 text-teal-300 border-teal-500/30 font-mono">
              Workspace Preferences
            </Badge>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
              Platform Settings & Display Controls
            </h1>
            <p className="text-xs sm:text-sm text-slate-300 leading-relaxed">
              Configure local table density, default analysis workspaces, timestamp formatting, and view real-time API health.
            </p>
          </div>
        </div>
      </div>

      {/* Save / Reset Confirmation Feedback Banner */}
      {savedFeedback && (
        <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg flex items-center justify-between text-xs text-emerald-800 animate-fadeIn">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span className="font-semibold">Preferences saved successfully to local storage.</span>
          </div>
          <span className="text-[10px] text-emerald-600 font-mono">Updated</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Settings Left Column (Preferences Form) */}
        <div className="lg:col-span-2 space-y-6">
          {/* 1. Display & Density Preferences */}
          <Card className="border-slate-200">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-indigo-600" />
                <div>
                  <CardTitle className="text-base text-slate-900">Display & Table Density</CardTitle>
                  <CardDescription className="text-xs">
                    Controls row height and padding across sample inspection tables and evidence signals.
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-5 text-xs">
              <div className="space-y-2">
                <label className="font-semibold text-slate-800 block">Table Row Density</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setPreferences({ ...preferences, tableDensity: "comfortable" })}
                    className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                      preferences.tableDensity === "comfortable"
                        ? "border-indigo-600 bg-indigo-50/50 text-indigo-950 font-semibold ring-1 ring-indigo-600"
                        : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span>Comfortable</span>
                      {preferences.tableDensity === "comfortable" && <Check className="w-3.5 h-3.5 text-indigo-600" />}
                    </div>
                    <p className="text-[11px] text-slate-500 font-normal">Spacious row padding with ample whitespace</p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setPreferences({ ...preferences, tableDensity: "compact" })}
                    className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                      preferences.tableDensity === "compact"
                        ? "border-indigo-600 bg-indigo-50/50 text-indigo-950 font-semibold ring-1 ring-indigo-600"
                        : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span>Compact</span>
                      {preferences.tableDensity === "compact" && <Check className="w-3.5 h-3.5 text-indigo-600" />}
                    </div>
                    <p className="text-[11px] text-slate-500 font-normal">Dense data layout for high-volume analysis</p>
                  </button>
                </div>
              </div>

              {/* 2. Default Workspace Selection */}
              <div className="space-y-2 pt-2 border-t border-slate-100">
                <label className="font-semibold text-slate-800 block">Default Analysis Workspace</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setPreferences({ ...preferences, defaultWorkspace: "machine-learning" })}
                    className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                      preferences.defaultWorkspace === "machine-learning"
                        ? "border-indigo-600 bg-indigo-50/50 text-indigo-950 font-semibold ring-1 ring-indigo-600"
                        : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span>Tabular Machine Learning</span>
                      {preferences.defaultWorkspace === "machine-learning" && <Check className="w-3.5 h-3.5 text-indigo-600" />}
                    </div>
                    <p className="text-[11px] text-slate-500 font-normal">Default to tabular dataset security investigations</p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setPreferences({ ...preferences, defaultWorkspace: "deep-learning" })}
                    className={`p-3 rounded-lg border text-left transition-all cursor-pointer ${
                      preferences.defaultWorkspace === "deep-learning"
                        ? "border-indigo-600 bg-indigo-50/50 text-indigo-950 font-semibold ring-1 ring-indigo-600"
                        : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span>Deep Learning Vision</span>
                      {preferences.defaultWorkspace === "deep-learning" && <Check className="w-3.5 h-3.5 text-indigo-600" />}
                    </div>
                    <p className="text-[11px] text-slate-500 font-normal">Default to image security and Trojan backdoor reviews</p>
                  </button>
                </div>
              </div>

              {/* 3. Numeric Precision & Timestamp Format */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-slate-100">
                <div className="space-y-1.5">
                  <label className="font-semibold text-slate-800 block">Metric Precision</label>
                  <select
                    value={preferences.numericPrecision}
                    onChange={(e) => setPreferences({ ...preferences, numericPrecision: e.target.value as "1" | "2" })}
                    className="w-full bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 text-slate-800 focus:bg-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  >
                    <option value="1">Standard (1 decimal, e.g. 74.2%)</option>
                    <option value="2">High Precision (2 decimals, e.g. 74.23%)</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="font-semibold text-slate-800 block">Timestamp Display</label>
                  <select
                    value={preferences.timestampFormat}
                    onChange={(e) => setPreferences({ ...preferences, timestampFormat: e.target.value as "relative" | "utc" })}
                    className="w-full bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 text-slate-800 focus:bg-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  >
                    <option value="relative">Relative (e.g. 10m ago)</option>
                    <option value="utc">ISO UTC Format</option>
                  </select>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between pt-4 border-t border-slate-100">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  icon={<RotateCcw className="w-3.5 h-3.5" />}
                  className="text-xs text-slate-600"
                >
                  Reset Defaults
                </Button>

                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleSave}
                  icon={<CheckCircle2 className="w-3.5 h-3.5" />}
                  className="text-xs bg-indigo-700 hover:bg-indigo-800"
                >
                  Save Preferences
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Live Backend Health & Privacy Notice */}
        <div className="space-y-6">
          <Card className="border-slate-200">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-600" />
                <CardTitle className="text-sm">API Connectivity Status</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3 text-xs text-slate-600">
              <div className="flex items-center justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">FastAPI Service:</span>
                {apiHealth?.status === "healthy" || apiHealth?.status === "ok" ? (
                  <Badge variant="success" size="sm">ONLINE</Badge>
                ) : (
                  <Badge variant="danger" size="sm">OFFLINE</Badge>
                )}
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">API Version:</span>
                <span className="font-mono text-slate-800">{apiHealth?.version || "Checking..."}</span>
              </div>
              <div className="flex items-center justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Risk Policy:</span>
                <span className="font-mono text-slate-800">{apiHealth?.policy_version || "1.0.0-evidence-weighted"}</span>
              </div>
              <div className="pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={checkHealth}
                  disabled={isCheckingHealth}
                  className="w-full text-xs h-7"
                >
                  {isCheckingHealth ? "Pinging..." : "Check API Health"}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-200">
            <CardHeader>
              <div className="flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-slate-500" />
                <CardTitle className="text-sm">Data Privacy & Local Storage</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-2 text-xs text-slate-600 leading-relaxed">
              <p>
                DataSentinel respects enterprise privacy principles. Preferences configured on this page are stored strictly in your browser&apos;s
                <span className="font-mono font-medium text-slate-800"> localStorage</span>.
              </p>
              <p>
                No training dataset records, uploaded images, authentication tokens, or private secrets are ever stored in browser local storage.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
