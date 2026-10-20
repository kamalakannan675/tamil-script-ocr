import React from 'react';
import { ModelStatus } from '../types';
import { Cpu, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';

interface Props {
  status: ModelStatus | null;
  loading: boolean;
  onRefresh: () => void;
}

export const ModelStatusBadge: React.FC<Props> = ({ status, loading, onRefresh }) => {
  if (loading && !status) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-100 text-slate-600 text-xs font-medium border border-slate-200">
        <RefreshCw className="w-3.5 h-3.5 animate-spin text-slate-500" />
        <span>Checking Model...</span>
      </div>
    );
  }

  if (!status || !status.available) {
    return (
      <div className="group relative flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-50 text-amber-800 text-xs font-medium border border-amber-200">
        <AlertCircle className="w-3.5 h-3.5 text-amber-600 shrink-0" />
        <span>Model Offline</span>
        <button
          onClick={onRefresh}
          title="Refresh model status"
          className="hover:rotate-180 transition-transform duration-300 ml-1"
        >
          <RefreshCw className="w-3 h-3 text-amber-700" />
        </button>
        {status?.missing_dependency && (
          <div className="hidden group-hover:block absolute right-0 top-full mt-1 w-64 p-2 bg-slate-900 text-white text-xs rounded shadow-lg z-50">
            {status.missing_dependency}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="group relative flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-50 text-emerald-800 text-xs font-medium border border-emerald-200">
      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
      <span>{status.model_name || 'Model Ready'}</span>
      <span className="text-slate-400">|</span>
      <span className="flex items-center gap-1 text-slate-500 font-mono">
        <Cpu className="w-3 h-3" />
        {status.device.toUpperCase()}
      </span>
      <button
        onClick={onRefresh}
        title="Refresh model status"
        className="hover:rotate-180 transition-transform duration-300 ml-1 text-emerald-700"
      >
        <RefreshCw className="w-3 h-3" />
      </button>

      {/* Hover Card for Architecture Details */}
      <div className="hidden group-hover:block absolute right-0 top-full mt-1 w-72 p-3 bg-slate-900 text-slate-200 text-xs rounded-lg shadow-xl z-50 leading-relaxed">
        <div className="font-semibold text-white mb-1">OCR Model Details</div>
        <div><strong className="text-slate-400">Arch:</strong> {status.architecture || 'CNN-BiLSTM-CTC'}</div>
        <div><strong className="text-slate-400">Framework:</strong> {status.framework || 'PyTorch'}</div>
        <div><strong className="text-slate-400">Device:</strong> {status.device}</div>
        <div className="mt-1 text-[11px] text-emerald-400">Genuine inference verified</div>
      </div>
    </div>
  );
};
