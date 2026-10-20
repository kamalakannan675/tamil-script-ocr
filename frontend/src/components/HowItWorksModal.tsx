import React from 'react';
import { X, Layers, Image as ImageIcon, Cpu, FileCheck } from 'lucide-react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const HowItWorksModal: React.FC<Props> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const steps = [
    {
      icon: <ImageIcon className="w-5 h-5 text-indigo-600" />,
      title: "1. Document Preprocessing & Deskewing",
      desc: "Uploaded images and multipage PDFs are validated, converted to high-contrast grayscale, noise-filtered, and corrected for rotational skew using OpenCV affine transformations."
    },
    {
      icon: <Layers className="w-5 h-5 text-indigo-600" />,
      title: "2. Layout & Line Segmentation",
      desc: "Horizontal projection profiling and connected-component analysis detect text lines while strictly preserving upper pulli dots (்) and lower vowel signs (ு, ூ)."
    },
    {
      icon: <Cpu className="w-5 h-5 text-indigo-600" />,
      title: "3. Neural Handwriting Recognition (CRNN)",
      desc: "Text lines are fed into a 7-layer Convolutional Neural Network coupled with a 2-layer Bidirectional LSTM. A Connectionist Temporal Classification (CTC) decoder converts stroke features into token probability distributions."
    },
    {
      icon: <FileCheck className="w-5 h-5 text-indigo-600" />,
      title: "4. Unicode Reconstruction & Multi-Format Export",
      desc: "Greedy CTC decoding reconstructs authentic Tamil Unicode glyphs (NFC). Users can review, edit, copy, and export their text as TXT, Word (.docx), or PDF with embedded Tamil fonts."
    }
  ];

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-6 border border-slate-200">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
            <span className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center font-tamil text-sm font-bold">த</span>
            How TamilScript OCR Works
          </h3>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="py-4 space-y-4">
          {steps.map((s, idx) => (
            <div key={idx} className="flex gap-3.5">
              <div className="shrink-0 w-9 h-9 rounded-xl bg-slate-50 border border-slate-200 flex items-center justify-center">
                {s.icon}
              </div>
              <div>
                <h4 className="text-xs font-semibold text-slate-800 mb-0.5">{s.title}</h4>
                <p className="text-xs text-slate-500 leading-relaxed">{s.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="pt-3 border-t border-slate-100 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold rounded-xl shadow-xs transition-colors"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
