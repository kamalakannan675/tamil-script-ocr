import React, { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  FileText,
  RotateCw,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Sliders,
  FileCheck,
} from 'lucide-react';

interface Props {
  selectedFile: File | null;
  onSelectFile: (file: File) => void;
  onClearFile: () => void;
  pageCount: number;
  currentPage: number;
  onChangePage: (newPage: number) => void;
  disabled?: boolean;
}

export const DocumentInputPanel: React.FC<Props> = ({
  selectedFile,
  onSelectFile,
  onClearFile,
  pageCount,
  currentPage,
  onChangePage,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [rotation, setRotation] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [previewFilter, setPreviewFilter] = useState<'normal' | 'grayscale' | 'contrast'>('normal');
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle preview generation when a file is selected
  useEffect(() => {
    if (!selectedFile) {
      setImagePreviewUrl(null);
      setImageDimensions(null);
      setRotation(0);
      setZoom(1);
      return;
    }

    setFileError(null);
    if (selectedFile.type === 'application/pdf') {
      // PDF preview: we display document indicator or rendered first page
      setImagePreviewUrl(null);
      setImageDimensions(null);
    } else {
      const url = URL.createObjectURL(selectedFile);
      setImagePreviewUrl(url);

      const img = new Image();
      img.onload = () => {
        setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = url;

      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [selectedFile]);

  const validateAndProcessFile = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    const validExtensions = ['jpg', 'jpeg', 'png', 'webp', 'pdf'];
    const maxSize = 25 * 1024 * 1024; // 25 MB

    if (!ext || !validExtensions.includes(ext)) {
      setFileError(`Unsupported format '.${ext}'. Please upload JPG, PNG, WebP, or PDF.`);
      return;
    }

    if (file.size > maxSize) {
      setFileError(`File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds 25 MB limit.`);
      return;
    }

    setFileError(null);
    onSelectFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndProcessFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndProcessFile(e.target.files[0]);
    }
  };

  const handleRotate = (delta: number) => {
    setRotation((prev) => (prev + delta) % 360);
  };

  const handleZoom = (factor: number) => {
    setZoom((prev) => Math.min(Math.max(0.5, prev + factor), 3.0));
  };

  const resetTransform = () => {
    setRotation(0);
    setZoom(1);
    setPreviewFilter('normal');
  };

  const getFilterStyle = () => {
    switch (previewFilter) {
      case 'grayscale':
        return 'grayscale(100%)';
      case 'contrast':
        return 'grayscale(100%) contrast(200%)';
      default:
        return 'none';
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Panel Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100 bg-slate-50/50">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-indigo-600" />
          <h2 className="text-sm font-semibold text-slate-800">Document Input</h2>
        </div>
        {selectedFile && (
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              className="text-xs px-2.5 py-1 bg-white border border-slate-200 hover:border-indigo-400 text-slate-600 rounded-md font-medium transition-colors"
            >
              Replace
            </button>
            <button
              onClick={onClearFile}
              disabled={disabled}
              className="p-1 text-slate-400 hover:text-red-500 rounded transition-colors"
              title="Remove document"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div className="flex-1 p-5 flex flex-col justify-between overflow-y-auto">
        {!selectedFile ? (
          /* Upload State */
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 flex-1 min-h-[300px] ${
              isDragging
                ? 'border-indigo-500 bg-indigo-50/50 scale-[0.99]'
                : 'border-slate-300 hover:border-indigo-400 bg-slate-50/40 hover:bg-slate-50/80'
            }`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,.pdf"
              className="hidden"
              onChange={handleFileInputChange}
            />
            <div className="w-14 h-14 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center mb-4 ring-8 ring-indigo-50/40">
              <UploadCloud className="w-7 h-7" />
            </div>
            <h3 className="text-sm font-semibold text-slate-800 mb-1">
              Upload handwritten document
            </h3>
            <p className="text-xs text-slate-500 mb-4 max-w-xs">
              Drag and drop your scanned image or multipage PDF, or click to browse
            </p>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-white border border-slate-200 text-indigo-600 rounded-lg text-xs font-medium shadow-2xs hover:bg-indigo-50 transition-colors">
              Browse Files
            </div>
            <p className="text-[11px] text-slate-400 mt-4">
              Supported: JPG, PNG, WebP, PDF (up to 25 MB)
            </p>

            {fileError && (
              <div className="mt-3 p-2 bg-red-50 border border-red-200 text-red-600 text-xs rounded-lg max-w-sm">
                {fileError}
              </div>
            )}
          </div>
        ) : (
          /* Preview State */
          <div className="flex flex-col h-full gap-3">
            {/* Image / Document Canvas Preview */}
            <div className="relative flex-1 min-h-[340px] bg-slate-900/5 rounded-xl border border-slate-200 overflow-hidden flex items-center justify-center p-3 select-none">
              {selectedFile.type === 'application/pdf' ? (
                <div className="flex flex-col items-center text-center p-6 bg-white rounded-xl shadow-xs border border-slate-200 max-w-xs">
                  <div className="w-12 h-12 bg-red-50 text-red-600 rounded-xl flex items-center justify-center mb-3">
                    <FileText className="w-6 h-6" />
                  </div>
                  <h4 className="text-sm font-semibold text-slate-800 truncate w-full mb-1">
                    {selectedFile.name}
                  </h4>
                  <p className="text-xs text-slate-500 mb-2">
                    PDF Document (Image-based processing)
                  </p>
                  <span className="text-[11px] px-2 py-0.5 bg-slate-100 text-slate-600 rounded font-medium">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>
              ) : imagePreviewUrl ? (
                <div className="w-full h-full flex items-center justify-center overflow-hidden">
                  <img
                    src={imagePreviewUrl}
                    alt="Document preview"
                    className="max-h-full max-w-full object-contain transition-transform duration-200 shadow-sm rounded"
                    style={{
                      transform: `rotate(${rotation}deg) scale(${zoom})`,
                      filter: getFilterStyle(),
                    }}
                  />
                </div>
              ) : null}

              {/* Viewport Floating Controls */}
              {selectedFile.type !== 'application/pdf' && (
                <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1 bg-white/90 backdrop-blur-xs px-2.5 py-1.5 rounded-full shadow-md border border-slate-200/80">
                  <button
                    onClick={() => handleRotate(-90)}
                    className="p-1 hover:bg-slate-100 text-slate-600 rounded-full transition-colors"
                    title="Rotate Counter-Clockwise"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => handleRotate(90)}
                    className="p-1 hover:bg-slate-100 text-slate-600 rounded-full transition-colors"
                    title="Rotate Clockwise"
                  >
                    <RotateCw className="w-3.5 h-3.5" />
                  </button>
                  <span className="w-px h-3 bg-slate-200 mx-0.5" />
                  <button
                    onClick={() => handleZoom(-0.2)}
                    className="p-1 hover:bg-slate-100 text-slate-600 rounded-full transition-colors"
                    title="Zoom Out"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => handleZoom(0.2)}
                    className="p-1 hover:bg-slate-100 text-slate-600 rounded-full transition-colors"
                    title="Zoom In"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={resetTransform}
                    className="p-1 hover:bg-slate-100 text-slate-600 rounded-full transition-colors"
                    title="Reset View"
                  >
                    <Maximize2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>

            {/* Preview Filter Toggles */}
            {selectedFile.type !== 'application/pdf' && (
              <div className="flex items-center justify-between px-2 pt-1">
                <div className="flex items-center gap-1.5 text-xs text-slate-500">
                  <Sliders className="w-3.5 h-3.5 text-slate-400" />
                  <span>Preview Filter:</span>
                </div>
                <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-lg text-xs font-medium">
                  <button
                    onClick={() => setPreviewFilter('normal')}
                    className={`px-2 py-0.5 rounded-md transition-colors ${
                      previewFilter === 'normal'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Color
                  </button>
                  <button
                    onClick={() => setPreviewFilter('grayscale')}
                    className={`px-2 py-0.5 rounded-md transition-colors ${
                      previewFilter === 'grayscale'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Grayscale
                  </button>
                  <button
                    onClick={() => setPreviewFilter('contrast')}
                    className={`px-2 py-0.5 rounded-md transition-colors ${
                      previewFilter === 'contrast'
                        ? 'bg-white text-slate-900 shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    Threshold
                  </button>
                </div>
              </div>
            )}

            {/* Document Metadata Bar */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100 flex items-center justify-between text-xs text-slate-600">
              <div className="flex items-center gap-2 truncate mr-2">
                <FileCheck className="w-4 h-4 text-emerald-600 shrink-0" />
                <span className="font-medium text-slate-800 truncate" title={selectedFile.name}>
                  {selectedFile.name}
                </span>
                <span className="text-slate-400">
                  ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
                </span>
                {imageDimensions && (
                  <span className="text-slate-400 hidden sm:inline">
                    • {imageDimensions.width}×{imageDimensions.height}px
                  </span>
                )}
              </div>

              {/* Multipage PDF Page Controls */}
              {pageCount > 1 && (
                <div className="flex items-center gap-1.5 shrink-0 bg-white border border-slate-200 px-2 py-1 rounded-lg">
                  <button
                    disabled={currentPage <= 1 || disabled}
                    onClick={() => onChangePage(currentPage - 1)}
                    className="p-0.5 hover:bg-slate-100 rounded disabled:opacity-30"
                    title="Previous page"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                  </button>
                  <span className="font-medium text-slate-700 text-[11px]">
                    {currentPage} / {pageCount}
                  </span>
                  <button
                    disabled={currentPage >= pageCount || disabled}
                    onClick={() => onChangePage(currentPage + 1)}
                    className="p-0.5 hover:bg-slate-100 rounded disabled:opacity-30"
                    title="Next page"
                  >
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
