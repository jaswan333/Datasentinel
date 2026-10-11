import React from "react";
import { Link } from "react-router-dom";
import { Shield } from "lucide-react";

export const Footer: React.FC = () => {
  return (
    <footer className="bg-white border-t border-slate-200/80 py-6 px-4 lg:px-8 mt-auto">
      <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
        <div className="flex items-center gap-2">
          <Shield className="w-4 h-4 text-blue-600" />
          <span className="font-semibold text-slate-700">DataSentinel</span>
          <span>— Agentic Training-Data Trust, Poisoning & Backdoor Defense Platform</span>
        </div>

        <div className="flex items-center gap-6">
          <Link to="/help" className="hover:text-blue-600 transition-colors">
            Help & Documentation
          </Link>
          <Link to="/settings" className="hover:text-blue-600 transition-colors">
            Settings
          </Link>
          <span className="text-slate-400 font-mono">v0.8.0</span>
          <span>© 2026 DataSentinel Project</span>
        </div>
      </div>
    </footer>
  );
};
