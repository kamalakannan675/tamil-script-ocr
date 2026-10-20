import React from 'react';
import { X, CheckCircle2 } from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const AboutModal: React.FC<Props> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6 border border-slate-200">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center font-tamil font-bold text-sm">
              த
            </div>
            <div>
              <h3 className="font-bold text-slate-900 text-sm">About TamilScript OCR</h3>
              <p className="text-[11px] text-slate-500">Tamil Handwritten Full-Page OCR Platform</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="py-4 space-y-3.5 text-xs text-slate-600 leading-relaxed">
          <p>
            <strong>TamilScript OCR</strong> is an optical character recognition system developed specifically for handwritten and historical Tamil script documents.
          </p>
          <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 space-y-1.5">
            <div className="font-semibold text-slate-800 text-xs">Core Engineering Principles:</div>
            <div className="flex items-center gap-2 text-slate-600">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span>Genuine neural network inference using PyTorch CRNN & CTC decoding</span>
            </div>
            <div className="flex items-center gap-2 text-slate-600">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span>Preservation of combined Tamil vowel signs, pulli marks, and Grantha characters</span>
            </div>
            <div className="flex items-center gap-2 text-slate-600">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span>Export fidelity in UTF-8 TXT, Microsoft Word (.docx), and embedded PDF</span>
            </div>
          </div>

          <p className="text-[11px] text-slate-500">
            <strong>Script Support:</strong> Tamil vowels (12), consonants (18), Aytham (ஃ), Grantha consonants (ஜ, ஷ, ஸ, ஹ, க்ஷ), Tamil numerals (௧–௰), and ASCII punctuation.
          </p>

          <p className="text-[11px] text-slate-500">
            <strong>Open-Source Font:</strong> Embedded with Google Noto Sans Tamil (OFL License) for cross-platform PDF and screen rendering.
          </p>
        </div>

        <div className="pt-3 border-t border-slate-100 flex items-center justify-between">
          <span className="text-[11px] text-slate-400">Version 1.0.0 • Production Ready</span>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold rounded-xl shadow-xs transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
