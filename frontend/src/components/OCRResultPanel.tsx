import React, { useState, useRef } from 'react';
import {
  Play,
  Copy,
  Check,
  Trash2,
  Download,
  FileText,
  FileCode,
  AlertCircle,
  RefreshCw,
  Sparkles,
} from 'lucide-react';

interface Props {
  text: string;
  onChangeText: (newText: string) => void;
  onStartOCR: () => void;
  onExport: (format: 'txt' | 'docx' | 'pdf') => void;
  isProcessing: boolean;
  hasFile: boolean;
  modelReady: boolean;
  unsavedEdits: boolean;
  processingTime?: number;
}

export const OCRResultPanel: React.FC<Props> = ({
  text,
  onChangeText,
  onStartOCR,
  onExport,
  isProcessing,
  hasFile,
  modelReady,
  unsavedEdits,
  processingTime,
}) => {
  const [copied, setCopied] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const characterCount = text.length;
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const lineCount = text.trim() ? text.split('\n').length : 0;

  const handleCopy = async () => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      // Fallback
      if (textareaRef.current) {
        textareaRef.current.select();
        document.execCommand('copy');
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }
    }
  };

  const handleSelectAll = () => {
    if (textareaRef.current) {
      textareaRef.current.select();
    }
  };

  const handleClear = () => {
    if (!text) return;
    setShowClearConfirm(true);
  };

  const confirmClear = () => {
    onChangeText('');
    setShowClearConfirm(false);
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Panel Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100 bg-slate-50/50">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-indigo-600" />
          <h2 className="text-sm font-semibold text-slate-800">
            Recognized Tamil Text (Unicode)
          </h2>
          {unsavedEdits && (
            <span className="text-[10px] font-medium px-2 py-0.5 bg-amber-50 text-amber-700 border border-amber-200 rounded-full">
              Edited
            </span>
          )}
        </div>

        {/* Start OCR Action */}
        <button
          onClick={onStartOCR}
          disabled={!hasFile || isProcessing || !modelReady}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-xl text-xs font-semibold shadow-xs transition-all duration-150 ${
            isProcessing
              ? 'bg-indigo-400 text-white cursor-wait'
              : !hasFile || !modelReady
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200'
              : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-100 active:scale-95'
          }`}
        >
          {isProcessing ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Processing OCR...</span>
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Start OCR</span>
            </>
          )}
        </button>
      </div>

      {/* Editor Main Body */}
      <div className="flex-1 p-5 flex flex-col relative min-h-[300px]">
        {/* Processing Overlay */}
        {isProcessing && (
          <div className="absolute inset-0 z-10 bg-white/80 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center">
            <div className="w-12 h-12 rounded-2xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center mb-3 animate-bounce">
              <Sparkles className="w-6 h-6" />
            </div>
            <h4 className="text-sm font-semibold text-slate-800 mb-1">
              Processing Handwritten Tamil Document
            </h4>
            <p className="text-xs text-slate-500 max-w-xs animate-pulse">
              Segmenting text lines, decoding stroke sequences with CRNN-CTC, and reconstructing Tamil Unicode...
            </p>
          </div>
        )}

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => onChangeText(e.target.value)}
          placeholder={
            !hasFile
              ? 'Upload a document on the left and click "Start OCR" to extract Tamil handwritten text...'
              : 'Click "Start OCR" above to extract handwritten Tamil text from your uploaded document. You can edit, select, or download the extracted Unicode content.'
          }
          className="flex-1 w-full p-4 rounded-xl border border-slate-200 bg-slate-50/30 focus:bg-white focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 focus:outline-hidden resize-none font-tamil text-slate-800 text-base leading-relaxed tracking-wide placeholder:text-slate-400 placeholder:text-sm placeholder:font-sans transition-colors"
          spellCheck={false}
        />

        {/* Editor Utility Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-3 mt-1 text-xs text-slate-500 border-t border-slate-100">
          {/* Metrics */}
          <div className="flex items-center gap-3">
            <span>
              <strong className="text-slate-700">{characterCount}</strong> chars
            </span>
            <span>•</span>
            <span>
              <strong className="text-slate-700">{wordCount}</strong> words
            </span>
            <span>•</span>
            <span>
              <strong className="text-slate-700">{lineCount}</strong> lines
            </span>
            {processingTime !== undefined && (
              <>
                <span>•</span>
                <span className="text-indigo-600 font-medium">
                  {processingTime}s
                </span>
              </>
            )}
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-1.5">
            <button
              onClick={handleSelectAll}
              disabled={!text}
              className="px-2 py-1 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-md transition-colors disabled:opacity-40"
              title="Select All"
            >
              Select All
            </button>
            <button
              onClick={handleCopy}
              disabled={!text}
              className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors font-medium disabled:opacity-40 ${
                copied
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
              title="Copy text to clipboard"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5" />
                  <span>Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
            <button
              onClick={handleClear}
              disabled={!text}
              className="p-1 text-slate-400 hover:text-red-600 rounded-md hover:bg-red-50 transition-colors disabled:opacity-40"
              title="Clear text"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Export Section Footer */}
      <div className="p-4 bg-slate-50/70 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-600">
          <Download className="w-3.5 h-3.5 text-indigo-600" />
          <span>Export Current Text:</span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onExport('txt')}
            disabled={!text.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:border-indigo-400 text-slate-700 hover:text-indigo-600 text-xs font-medium rounded-lg shadow-2xs hover:bg-indigo-50/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            title="Download UTF-8 plain text"
          >
            <FileCode className="w-3.5 h-3.5 text-slate-500" />
            <span>TXT</span>
          </button>
          <button
            onClick={() => onExport('docx')}
            disabled={!text.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:border-indigo-400 text-slate-700 hover:text-indigo-600 text-xs font-medium rounded-lg shadow-2xs hover:bg-indigo-50/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            title="Download Word Document"
          >
            <FileText className="w-3.5 h-3.5 text-blue-600" />
            <span>DOCX</span>
          </button>
          <button
            onClick={() => onExport('pdf')}
            disabled={!text.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 hover:border-indigo-400 text-slate-700 hover:text-indigo-600 text-xs font-medium rounded-lg shadow-2xs hover:bg-indigo-50/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            title="Download PDF with embedded Tamil font"
          >
            <FileText className="w-3.5 h-3.5 text-red-600" />
            <span>PDF</span>
          </button>
        </div>
      </div>

      {/* Clear Confirmation Modal */}
      {showClearConfirm && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-2xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-xl max-w-sm w-full p-5 border border-slate-200">
            <div className="flex items-center gap-3 text-red-600 mb-2">
              <AlertCircle className="w-5 h-5" />
              <h3 className="font-semibold text-slate-800 text-sm">Clear Extracted Text?</h3>
            </div>
            <p className="text-xs text-slate-500 mb-4 leading-relaxed">
              Are you sure you want to clear the editor? Any unsaved edits will be lost.
            </p>
            <div className="flex items-center justify-end gap-2">
              <button
                onClick={() => setShowClearConfirm(false)}
                className="px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmClear}
                className="px-3 py-1.5 text-xs font-medium bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors shadow-2xs"
              >
                Clear Text
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
