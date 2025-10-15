import axios from 'axios';

// IMPORTANTE: Cambia esta IP por la IP local de tu Mac
// Para obtenerla: ejecuta "ifconfig | grep 'inet ' | grep -v 127.0.0.1" en tu Mac
const API_BASE_URL = 'http://172.16.132.152:5003'; // ← IP configurada automáticamente

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 segundos para videos
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error);
    
    if (error.code === 'ECONNABORTED') {
      error.message = 'Tiempo de espera agotado. Verifica tu conexión.';
    } else if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
      error.message = `No se puede conectar al servidor. Verifica que:\n1. El backend esté corriendo (python app.py)\n2. Tu celular y Mac estén en la misma WiFi\n3. La IP en api.ts sea correcta: ${API_BASE_URL}`;
    } else if (error.response?.status === 404) {
      error.message = 'Endpoint no encontrado en el servidor.';
    } else if (error.response?.status >= 500) {
      error.message = 'Error interno del servidor. Revisa los logs del backend.';
    }
    
    return Promise.reject(error);
  }
);

export interface ModelStatus {
  model_loaded: boolean;
}

export interface Detection {
  bbox: [number, number, number, number];
  class: string;
  confidence: number;
  model: string;
}

export interface DetectionResult {
  detections: Detection[];
  original_filename: string;
  original_image: string;
  processed_filename: string;
  processed_image: string;
  total_detections: number;
}

export interface VideoStatus {
  status: 'processing' | 'completed' | 'error' | 'idle' | 'failed';
  processed_frames: number;
  total_frames: number;
  progress_percent: number;
  progress: number;
  output_video_url?: string;
  error?: string;
  filename?: string;
}

export interface WebcamDetection {
  class_name: string;
  confidence: number;
  timestamp: number;
}

// Helper function to retry failed requests
const retryRequest = async <T>(
  requestFn: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await requestFn();
    } catch (error: any) {
      console.log(`🔄 Intento ${attempt}/${maxRetries} falló:`, error.message);
      
      if (attempt === maxRetries) {
        throw error;
      }
      
      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, delay * attempt));
    }
  }
  throw new Error('Max retries exceeded');
};

export const apiService = {
  // Model status
  async getModelStatus(): Promise<ModelStatus> {
    const response = await api.get('/api/model-status');
    return response.data;
  },

  // Image detection
  async detectImage(filename: string): Promise<DetectionResult> {
    const response = await api.post('/api/detect', { filename });
    return response.data;
  },

  // File upload with retry
  async uploadFile(file: { uri: string; name: string; type: string }): Promise<{ filename: string }> {
    return retryRequest(async () => {
      const formData = new FormData();
      formData.append('file', {
        uri: file.uri,
        name: file.name,
        type: file.type,
      } as any);
      
      const response = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutos para archivos grandes
      });
      return response.data;
    }, 3, 2000);
  },

  // Video processing with retry
  async processVideo(filename: string): Promise<{ message: string }> {
    return retryRequest(async () => {
      const response = await api.post('/api/process-video', { filename });
      return response.data;
    }, 3, 1000);
  },

  // Video status with retry
  async getVideoStatus(filename: string): Promise<VideoStatus> {
    return retryRequest(async () => {
      const response = await api.get(`/api/video-status/${filename}`);
      return response.data;
    }, 2, 500);
  },

  // Webcam detection (para imágenes de cámara)
  async detectWebcam(imageData: string): Promise<DetectionResult> {
    const response = await api.post('/api/webcam', { image: imageData });
    return response.data;
  },

  // Get webcam detections
  async getWebcamDetections(): Promise<{ detections: WebcamDetection[]; total: number }> {
    const response = await api.get('/api/webcam/detections');
    return response.data;
  },

  // Helper to get full URLs
  getDownloadUrl(filename: string): string {
    return `${API_BASE_URL}/download/${filename}`;
  },

  getVideoUrl(filename: string): string {
    return `${API_BASE_URL}/video/${filename}`;
  },

  // Get API base URL for debugging
  getBaseUrl(): string {
    return API_BASE_URL;
  },
};

export default apiService;

