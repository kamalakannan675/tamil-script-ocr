import React from 'react';
import { ModelStatus } from '../types';
import { ModelStatusBadge } from './ModelStatusBadge';
import { HelpCircle, Info } from 'lucide-react';

interface Props {
  modelStatus: ModelStatus | null;
  loadingStatus: boolean;
  onRefreshStatus: () => void;
  onOpenHowItWorks: () => void;
  onOpenAbout: () => void;
}

export const Header: React.FC<Props> = ({
  modelStatus,
  loadingStatus,
  onRefreshStatus,
  onOpenHowItWorks,
  onOpenAbout,
}) => {
  return (
    <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-sm border-b border-slate-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-700 to-indigo-500 flex items-center justify-center text-white shadow-md shadow-indigo-100">
              <span className="font-tamil text-xl font-bold">த</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold tracking-tight text-slate-900">TamilScript OCR</span>
                <span className="hidden sm:inline-block text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 bg-indigo-50 text-indigo-700 rounded-md border border-indigo-200">
                  தமிழ்
                </span>
              </div>
              <p className="text-xs text-slate-500 hidden sm:block">
                Handwritten Full-Page Optical Character Recognition
              </p>
            </div>
          </div>

          {/* Navigation Links and Model Status */}
          <div className="flex items-center gap-2 sm:gap-4">
            <nav className="flex items-center gap-1 sm:gap-2 mr-1 sm:mr-3">
              <button
                onClick={onOpenHowItWorks}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-indigo-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <HelpCircle className="w-4 h-4 text-slate-400" />
                <span className="hidden md:inline">How It Works</span>
              </button>
              <button
                onClick={onOpenAbout}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-indigo-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <Info className="w-4 h-4 text-slate-400" />
                <span className="hidden md:inline">About</span>
              </button>
            </nav>

            <ModelStatusBadge
              status={modelStatus}
              loading={loadingStatus}
              onRefresh={onRefreshStatus}
            />
          </div>
        </div>
      </div>
    </header>
  );
};
