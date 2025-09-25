import axios from 'axios';

const API_BASE_URL = 'http://localhost:5003';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  // Configuración adicional para mejorar la conectividad
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
    
    // Mejorar el mensaje de error para el usuario
    if (error.code === 'ERR_NETWORK' || error.message === 'Network Error') {
      error.message = 'No se puede conectar al servidor. Verifica que el backend esté ejecutándose en http://localhost:5003';
    } else if (error.code === 'ECONNREFUSED') {
      error.message = 'Conexión rechazada. El servidor backend no está disponible.';
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

export interface DetectionResult {
  detections: Array<{
    bbox: [number, number, number, number];
    class: string;
    confidence: number;
    model: string;
  }>;
  original_filename: string;
  original_image: string;
  processed_filename: string;
  processed_image: string;
  total_detections: number;
}

export interface VideoStatus {
  status: 'processing' | 'completed' | 'error' | 'idle';
  processed_frames: number; 
  total_frames: number;
  progress_percent: number;
  progress: number; 
  output_video_url?: string;
  error?: string;
  filename?: string;
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
  async uploadFile(file: File): Promise<{ filename: string }> {
    return retryRequest(async () => {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await api.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds for file upload
      });
      return response.data;
    }, 3, 2000); // 3 intentos, 2 segundos entre intentos
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
    }, 2, 500); // Menos reintentos para polling
  },

  // Webcam detection
  async detectWebcam(imageData: string): Promise<DetectionResult> {
    const response = await api.post('/api/webcam', { image: imageData });
    return response.data;
  },

  // Get webcam detections
  async getWebcamDetections(): Promise<{ detections: any[]; total: number }> {
    const response = await api.get('/api/webcam/detections');
    return response.data;
  },

  // Webcam stream URL
  getWebcamStreamUrl(): string {
    return `${API_BASE_URL}/api/webcam`;
  },

  // Download file
  getDownloadUrl(filename: string): string {
    return `${API_BASE_URL}/download/${filename}`;
  },

  // Video stream
  getVideoUrl(filename: string): string {
    return `${API_BASE_URL}/video/${filename}`;
  },
};

export default apiService;
