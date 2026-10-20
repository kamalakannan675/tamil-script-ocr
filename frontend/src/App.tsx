import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { DocumentInputPanel } from './components/DocumentInputPanel';
import { OCRResultPanel } from './components/OCRResultPanel';
import { HowItWorksModal } from './components/HowItWorksModal';
import { AboutModal } from './components/AboutModal';
import { ModelStatus, OCRResponse } from './types';
import { fetchModelStatus, submitOCR, exportDocument } from './services/api';
import { AlertTriangle, X } from 'lucide-react';

export const App: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<boolean>(true);
  
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [extractedText, setExtractedText] = useState<string>('');
  const [unsavedEdits, setUnsavedEdits] = useState<boolean>(false);
  const [ocrResponse, setOcrResponse] = useState<OCRResponse | null>(null);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [pageCount, setPageCount] = useState<number>(1);
  const [processingTime, setProcessingTime] = useState<number | undefined>(undefined);

  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [howItWorksOpen, setHowItWorksOpen] = useState<boolean>(false);
  const [aboutOpen, setAboutOpen] = useState<boolean>(false);

  const loadStatus = async () => {
    setLoadingStatus(true);
    try {
      const status = await fetchModelStatus();
      setModelStatus(status);
    } catch (e: any) {
      console.error('Error fetching model status:', e);
      setModelStatus({
        available: false,
        model_name: null,
        loaded: false,
        device: 'unknown',
        missing_dependency: 'Cannot reach backend server. Check connection.',
        supported_formats: ['jpg', 'jpeg', 'png', 'webp', 'pdf'],
        max_pdf_pages: 20,
        max_file_size_mb: 25,
      });
    } finally {
      setLoadingStatus(false);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  const handleSelectFile = (file: File) => {
    setSelectedFile(file);
    setExtractedText('');
    setUnsavedEdits(false);
    setOcrResponse(null);
    setCurrentPage(1);
    setPageCount(1);
    setProcessingTime(undefined);
    setErrorMessage(null);
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setExtractedText('');
    setUnsavedEdits(false);
    setOcrResponse(null);
    setCurrentPage(1);
    setPageCount(1);
    setProcessingTime(undefined);
    setErrorMessage(null);
  };

  const handleChangePage = (newPage: number) => {
    if (!ocrResponse || !ocrResponse.pages) {
      setCurrentPage(newPage);
      return;
    }
    const targetPage = ocrResponse.pages.find((p) => p.page_number === newPage);
    if (targetPage) {
      setCurrentPage(newPage);
      setExtractedText(targetPage.text);
      setUnsavedEdits(false);
    }
  };

  const handleTextChange = (newText: string) => {
    setExtractedText(newText);
    setUnsavedEdits(true);
  };

  const handleStartOCR = async () => {
    if (!selectedFile) return;
    setIsProcessing(true);
    setErrorMessage(null);

    try {
      const response = await submitOCR(selectedFile);
      setOcrResponse(response);
      setExtractedText(response.full_text || '');
      setUnsavedEdits(false);
      setPageCount(response.page_count || 1);
      setCurrentPage(1);
      setProcessingTime(response.total_processing_time_sec);
    } catch (err: any) {
      setErrorMessage(err.message || 'An error occurred during OCR recognition.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExport = async (format: 'txt' | 'docx' | 'pdf') => {
    if (!extractedText.trim()) return;
    setErrorMessage(null);

    const baseName = selectedFile
      ? selectedFile.name.replace(/\.[^/.]+$/, '') + '_ocr'
      : 'tamil_script_ocr';

    try {
      await exportDocument(extractedText, format, baseName);
    } catch (err: any) {
      setErrorMessage(err.message || `Export failed for format ${format.toUpperCase()}`);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-100/70 text-slate-800">
      <Header
        modelStatus={modelStatus}
        loadingStatus={loadingStatus}
        onRefreshStatus={loadStatus}
        onOpenHowItWorks={() => setHowItWorksOpen(true)}
        onOpenAbout={() => setAboutOpen(true)}
      />

      {/* Global Error Notification */}
      {errorMessage && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-4 w-full">
          <div className="flex items-center justify-between p-3.5 bg-red-50 border border-red-200 text-red-800 text-xs rounded-xl shadow-xs">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="p-1 text-red-400 hover:text-red-700 rounded transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Main Two-Panel Workspace */}
      <main className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5 w-full flex flex-col">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 flex-1 min-h-[580px]">
          {/* Left Panel: Document Input */}
          <DocumentInputPanel
            selectedFile={selectedFile}
            onSelectFile={handleSelectFile}
            onClearFile={handleClearFile}
            pageCount={pageCount}
            currentPage={currentPage}
            onChangePage={handleChangePage}
            disabled={isProcessing}
          />

          {/* Right Panel: OCR Results & Editor */}
          <OCRResultPanel
            text={extractedText}
            onChangeText={handleTextChange}
            onStartOCR={handleStartOCR}
            onExport={handleExport}
            isProcessing={isProcessing}
            hasFile={!!selectedFile}
            modelReady={!!modelStatus?.available}
            unsavedEdits={unsavedEdits}
            processingTime={processingTime}
          />
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 border-t border-slate-200/80 bg-white text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div>
            TamilScript OCR Platform • Powered by PyTorch & OpenCV
          </div>
          <div className="text-[11px] text-slate-400">
            Compliant with Unicode Standard for Tamil Script (U+0B80–U+0BFF)
          </div>
        </div>
      </footer>

      {/* Modals */}
      <HowItWorksModal isOpen={howItWorksOpen} onClose={() => setHowItWorksOpen(false)} />
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
};

export default App;
