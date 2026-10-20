import { ModelStatus, OCRResponse, HealthStatus } from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/ocr/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch model status (HTTP ${res.status})`);
  }
  return res.json();
}

export async function submitOCR(file: File): Promise<OCRResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/ocr`, {
    method: 'POST',
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `OCR processing failed with HTTP ${res.status}`);
  }
  return data;
}

export async function exportDocument(
  text: string,
  format: 'txt' | 'docx' | 'pdf',
  filename: string = 'tamil_ocr_document'
): Promise<void> {
  const res = await fetch(`${API_BASE}/export`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      format,
      filename,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `Export failed with status ${res.status}`);
  }

  // Handle file download via blob
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${filename}.${format}`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}
