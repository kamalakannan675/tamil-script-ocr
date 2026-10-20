export interface ModelStatus {
  available: boolean;
  model_name: string | null;
  architecture?: string;
  framework?: string;
  loaded: boolean;
  device: string;
  checkpoint_path?: string | null;
  missing_dependency?: string | null;
  supported_formats: string[];
  max_pdf_pages: number;
  max_file_size_mb: number;
}

export interface BoundingBox {
  line_index: number;
  bbox: [number, number, number, number]; // [x, y, w, h]
  text: string;
  confidence: number;
}

export interface PageResult {
  page_number: number;
  width: number;
  height: number;
  skew_angle: number;
  line_count: number;
  text: string;
  lines: BoundingBox[];
  processing_time_sec: number;
}

export interface OCRResponse {
  success: boolean;
  filename: string;
  page_count: number;
  total_processing_time_sec: number;
  full_text: string;
  pages: PageResult[];
  error?: string;
  code?: string;
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  python_version: string;
  platform: string;
}
