import { useEffect, useState, useCallback, Component } from 'react';
import type { ErrorInfo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  VideoCameraIcon, 
  CpuChipIcon,
  CheckCircleIcon,
  XCircleIcon,
  CloudArrowUpIcon,
  XMarkIcon,
  ArrowDownTrayIcon,
  PlayIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import WebcamDetection from './components/WebcamDetection';
import Header from './components/Header';
import Footer from './components/Footer';
import Navigation from './components/Navigation';

// Error Boundary to prevent app crashes
class ErrorBoundary extends Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('💥 App crashed:', error, errorInfo);
    console.error('💥 Error stack:', error.stack);
    console.error('💥 Component stack:', errorInfo.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center">
          <div className="max-w-md mx-auto text-center">
            <div className="bg-white rounded-lg shadow-lg p-8">
              <div className="text-red-500 text-6xl mb-4">⚠️</div>
              <h1 className="text-2xl font-bold text-gray-900 mb-4">
                Error en la aplicación
              </h1>
              <p className="text-gray-600 mb-6">
                Algo salió mal. Por favor, recarga la página para continuar.
              </p>
              <button
                onClick={() => window.location.reload()}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Recargar página
              </button>
              {this.state.error && (
                <details className="mt-4 text-left">
                  <summary className="text-sm text-gray-500 cursor-pointer">
                    Detalles del error
                  </summary>
                  <pre className="text-xs text-red-600 mt-2 overflow-auto">
                    {this.state.error.toString()}
                  </pre>
                </details>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Simple state management without external dependencies
const useAppState = () => {
  const [activeTab, setActiveTab] = useState<'image' | 'video' | 'webcam'>('webcam');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [currentDetection, setCurrentDetection] = useState<any>(null);
  const [videoStatus, setVideoStatus] = useState({
    status: 'idle' as 'idle' | 'processing' | 'completed' | 'error',
    progress: 0,
    processed_frames: 0, // Campo que envía el backend
    total_frames: 0,
    progress_percent: 0,
    error: '',
    output_video_url: null as string | null
  });
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [uploadedVideoFilename, setUploadedVideoFilename] = useState<string | null>(null);
  const [notifications, setNotifications] = useState<Array<{
    id: string;
    type: 'success' | 'error' | 'info' | 'warning';
    title: string;
    message: string;
    timestamp: Date;
  }>>([]);
  const [isConnected, setIsConnected] = useState(true);

  const addNotification = useCallback((notification: Omit<typeof notifications[0], 'id' | 'timestamp'>) => {
    const id = Math.random().toString(36).substr(2, 9);
    const newNotification = {
      ...notification,
      id,
      timestamp: new Date(),
    };
    setNotifications(prev => [...prev, newNotification]);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  }, []);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  const processVideo = useCallback(async () => {
    console.log('🎬 Iniciando processVideo:', { uploadedVideo, uploadedVideoFilename });
    
    if (!uploadedVideo || !uploadedVideoFilename) {
      console.error('❌ No hay video o filename:', { uploadedVideo: !!uploadedVideo, uploadedVideoFilename });
      addNotification({
        type: 'error',
        title: 'Error',
        message: 'No hay video seleccionado para procesar.',
      });
      return;
    }

    try {
      setIsLoading(true);
      console.log('🚀 Enviando solicitud de procesamiento para:', uploadedVideoFilename);
      
      // Start video processing
      const result = await apiService.processVideo(uploadedVideoFilename);
      console.log('✅ Respuesta del servidor:', result);
      
      addNotification({
        type: 'success',
        title: '¡Video en procesamiento!',
        message: 'Tu video se está analizando.',
      });
      
      // Start polling for status updates
      const pollStatus = async () => {
        try {
          console.log('🔍 Consultando estado para:', uploadedVideoFilename);
          const status = await apiService.getVideoStatus(uploadedVideoFilename);
          console.log('📊 Estado recibido:', status);
          
          // Mapear processed_frames del backend a progress para la UI
          setVideoStatus({
            ...status,
            progress: status.processed_frames || 0
          });
          
          if (status.status === 'processing') {
            // Continue polling
            setTimeout(pollStatus, 2000);
          } else if (status.status === 'completed') {
            console.log('🎉 Procesamiento completado!');
            addNotification({
              type: 'success',
              title: '¡Listo! Tu video está procesado',
              message: 'Ya podés ver todos los animales detectados en tu video',
            });
          } else if (status.status === 'failed') {
            console.error('💥 Procesamiento falló:', status.error);
            addNotification({
              type: 'error',
              title: 'Ups! Algo no salió bien',
              message: 'Hubo un problema al procesar tu video. ¿Podrías intentarlo de nuevo?',
            });
          }
        } catch (error) {
          console.error('❌ Error polling status:', error);
          addNotification({
            type: 'error',
            title: 'Error de conexión',
            message: 'No se pudo obtener el estado del procesamiento.',
          });
        }
      };
      
      // Start polling after a short delay
      setTimeout(pollStatus, 1000);
      
    } catch (error) {
      console.error('💥 Error processing video:', error);
      
      // Check if it's a connection error
      if (error instanceof Error && (error.message.includes('Network Error') || error.message.includes('ERR_NETWORK'))) {
        setIsConnected(false);
      }
      
      addNotification({
        type: 'error',
        title: 'Error al procesar video',
        message: `Error: ${error instanceof Error ? error.message : 'No se pudo iniciar el procesamiento del video.'}`,
      });
    } finally {
      setIsLoading(false);
    }
  }, [uploadedVideo, uploadedVideoFilename, addNotification, setIsLoading, setVideoStatus]);

  const handleVideoFileUpload = useCallback(async (file: File) => {
    console.log('📤 Iniciando upload de video:', file.name, file.type, `${(file.size / (1024 * 1024)).toFixed(2)}MB`);
    
    // Validate file type
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/wmv', 'video/flv'];
    if (!allowedTypes.includes(file.type)) {
      console.error('❌ Tipo de archivo no válido:', file.type);
      addNotification({
        type: 'error',
        title: 'Tipo de archivo no válido',
        message: 'Por favor, selecciona un video válido (MP4, AVI, MOV, MKV, WMV, FLV)',
      });
      return;
    }

    // Validate file size (150MB max)
    if (file.size > 150 * 1024 * 1024) {
      console.error('❌ Archivo demasiado grande:', `${(file.size / (1024 * 1024)).toFixed(2)}MB`);
      addNotification({
        type: 'error',
        title: 'Archivo demasiado grande',
        message: 'El video debe ser menor a 150MB',
      });
      return;
    }

    try {
      setIsLoading(true);
      setVideoStatus({ 
        status: 'idle', 
        progress: 0, 
        processed_frames: 0,
        total_frames: 0, 
        progress_percent: 0, 
        error: '', 
        output_video_url: null 
      });
      
      console.log('🚀 Enviando archivo al servidor...');
      
      // Upload file to backend
      const uploadResult = await apiService.uploadFile(file);
      console.log('✅ Upload completado:', uploadResult);
      
      setUploadedVideo(file);
      setUploadedVideoFilename(uploadResult.filename);
      
      console.log('📝 Estado actualizado:', { 
        originalName: file.name, 
        serverFilename: uploadResult.filename 
      });
      
      addNotification({
        type: 'success',
        title: 'Video subido',
        message: 'El video se ha subido correctamente. Haz click en "Procesar" para comenzar.',
      });
      
    } catch (error) {
      console.error('💥 Error uploading video:', error);
      
      // Check if it's a connection error
      if (error instanceof Error && (error.message.includes('Network Error') || error.message.includes('ERR_NETWORK') || error.message.includes('conexión'))) {
        setIsConnected(false);
      }
      
      addNotification({
        type: 'error',
        title: 'Error al subir video',
        message: `Error: ${error instanceof Error ? error.message : 'No se pudo subir el video. Inténtalo de nuevo.'}`,
      });
    } finally {
      setIsLoading(false);
    }
  }, [addNotification, setIsLoading, setVideoStatus, setUploadedVideo, setUploadedVideoFilename]);

  const resetVideoState = useCallback(() => {
    setVideoStatus({ 
      status: 'idle', 
      progress: 0, 
      processed_frames: 0,
      total_frames: 0, 
      progress_percent: 0, 
      error: '', 
      output_video_url: null 
    });
    setUploadedVideo(null);
    setUploadedVideoFilename(null);
  }, [setVideoStatus, setUploadedVideo, setUploadedVideoFilename]);

  return {
    activeTab,
    setActiveTab,
    modelLoaded,
    setModelLoaded,
    isLoading,
    setIsLoading,
    currentDetection,
    setCurrentDetection,
    videoStatus,
    setVideoStatus,
    uploadedVideo,
    setUploadedVideo,
    uploadedVideoFilename,
    setUploadedVideoFilename,
    handleVideoFileUpload,
    processVideo,
    resetVideoState,
    isConnected,
    setIsConnected,
    notifications,
    addNotification,
    removeNotification
  };
};

// Real API service that connects to the backend
const apiService = {
  async getModelStatus() {
    try {
      const response = await fetch('http://localhost:5003/api/model-status');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API Error:', error);
      throw error; // Re-throw to handle in component
    }
  },

  async uploadFile(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:5003/api/upload', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }
    
    return response.json();
  },

  async detectImage(filename: string) {
    const response = await fetch('http://localhost:5003/api/detect', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ filename }),
    });
    
    if (!response.ok) {
      throw new Error(`Detection failed: ${response.status}`);
    }
    
    return response.json();
  },

  async processVideo(filename: string) {
    const response = await fetch('http://localhost:5003/api/process-video', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ filename }),
    });
    
    if (!response.ok) {
      throw new Error(`Video processing failed: ${response.status}`);
    }
    
    return response.json();
  },

  async getVideoStatus(filename: string) {
    const response = await fetch(`http://localhost:5003/api/video-status/${filename}`);
    
    if (!response.ok) {
      throw new Error(`Status check failed: ${response.status}`);
    }
    
    return response.json();
  }
};

// Notification Center Component
const NotificationCenter = ({ 
  notifications, 
  removeNotification 
}: { 
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'info' | 'warning';
    title: string;
    message: string;
    timestamp: Date;
  }>;
  removeNotification: (id: string) => void;
}) => {
  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'error':
        return <XCircleIcon className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <XMarkIcon className="w-5 h-5 text-yellow-500" />;
      default:
        return <CpuChipIcon className="w-5 h-5 text-blue-500" />;
    }
  };

  const getNotificationColor = (type: string) => {
    switch (type) {
      case 'success':
        return 'bg-green-50 border-green-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200';
      default:
        return 'bg-blue-50 border-blue-200';
    }
  };

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2">
      <AnimatePresence>
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            initial={{ opacity: 0, x: 300, scale: 0.8 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 300, scale: 0.8 }}
            transition={{ duration: 0.3 }}
            className={`
              max-w-sm w-full bg-white rounded-lg shadow-lg border-l-4 p-4
              ${getNotificationColor(notification.type)}
            `}
          >
            <div className="flex items-start space-x-3">
              {getNotificationIcon(notification.type)}
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-medium text-gray-900">
                  {notification.title}
                </h4>
                <p className="text-sm text-gray-600 mt-1">
                  {notification.message}
                </p>
              </div>
                <button
                  onClick={() => removeNotification(notification.id)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                <XMarkIcon className="w-4 h-4" />
                </button>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

// Image Detection Component
const ImageDetection = ({ 
  isLoading, 
  setIsLoading, 
  currentDetection, 
  setCurrentDetection, 
  addNotification 
}: {
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  currentDetection: any;
  setCurrentDetection: (detection: any) => void;
  addNotification: (notification: any) => void;
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [, setUploadedFile] = useState<File | null>(null);

  const handleFileUpload = async (file: File) => {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      addNotification({
        type: 'error',
        title: 'Tipo de archivo no válido',
        message: 'Por favor, selecciona una imagen válida (JPG, PNG, GIF, WebP)',
      });
      return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
      addNotification({
        type: 'error',
        title: 'Archivo demasiado grande',
        message: 'La imagen debe ser menor a 10MB',
      });
      return;
    }

    try {
      setIsLoading(true);
      
      // Upload file to backend
      const uploadResult = await apiService.uploadFile(file);
      setUploadedFile(file);
      
      addNotification({
        type: 'success',
        title: 'Imagen subida',
        message: 'La imagen se ha subido correctamente. Procesando...',
      });
      
      // Detect objects in the image
      const detectionResult = await apiService.detectImage(uploadResult.filename);
      setCurrentDetection(detectionResult);
      
      addNotification({
        type: 'success',
        title: 'Detección completada',
        message: `Se detectaron ${detectionResult.total_detections} animales`,
      });
      
    } catch (error) {
      console.error('Error processing image:', error);
      addNotification({
        type: 'error',
        title: 'Error al procesar imagen',
        message: 'No se pudo procesar la imagen. Inténtalo de nuevo.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
    setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <div className="text-center">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Detección en Imágenes
          </h3>
          <p className="text-sm text-gray-600 mb-6">
            Sube una imagen para detectar automáticamente los animales que aparecen en ella
          </p>
          
          <div
            className={`
              relative border-2 border-dashed rounded-xl p-8 transition-colors duration-200
              ${dragActive 
                ? 'border-blue-400 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400'
              }
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
        >
          <input
            type="file"
            accept="image/*"
              onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              disabled={isLoading}
          />
          
          <div className="space-y-4">
              <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto" />
            <div>
                <p className="text-lg font-medium text-gray-900">
                  {isLoading ? 'Procesando...' : 'Arrastra una imagen aquí'}
                </p>
                <p className="text-sm text-gray-600">
                  o haz clic para seleccionar un archivo
              </p>
            </div>
          </div>
        </div>
          </div>
        </motion.div>

      {/* Results */}
      {currentDetection && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          {/* Detection Summary */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-gray-900">
                Resultados de la Detección
              </h4>
              <div className="flex items-center space-x-2 bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                <CheckCircleIcon className="w-4 h-4" />
                <span>{currentDetection.total_detections} animales detectados</span>
              </div>
            </div>
            
            {currentDetection.detections && currentDetection.detections.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {['cat', 'chicken', 'cow', 'dog', 'horse'].map((animal) => {
                  const count = currentDetection.detections.filter((d: any) => d.class === animal).length;
                  return (
                    <div key={animal} className="text-center">
                      <div className="w-12 h-12 rounded-full mx-auto mb-2 flex items-center justify-center text-white font-bold bg-blue-600">
                        {count}
                      </div>
                      <p className="text-xs text-gray-600 capitalize">{animal}</p>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Images */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Original Image */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h5 className="text-md font-semibold text-gray-900 mb-4">Imagen Original</h5>
              <img
                src={`http://localhost:5003${currentDetection.original_image}`}
                alt="Original"
                className="w-full h-auto rounded-lg shadow-sm"
              />
            </div>

            {/* Processed Image */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h5 className="text-md font-semibold text-gray-900 mb-4">Imagen con Detecciones</h5>
              <img
                src={`http://localhost:5003${currentDetection.processed_image}`}
                alt="Processed"
                className="w-full h-auto rounded-lg shadow-sm"
              />
            </div>
          </div>

          {/* Detection Details */}
          {currentDetection.detections && currentDetection.detections.length > 0 && (
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h5 className="text-md font-semibold text-gray-900 mb-4">
                Detalles de las Detecciones
              </h5>
            <div className="space-y-3">
              {currentDetection.detections.map((detection: any, index: number) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                      <div className="w-4 h-4 rounded-full bg-blue-600"></div>
                    <span className="font-medium text-gray-900 capitalize">
                      {detection.class}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {Math.round(detection.confidence * 100)}% confianza
                  </div>
                </div>
              ))}
            </div>
          </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

// Video Detection Component
const VideoDetection = ({ 
  isLoading, 
  videoStatus, 
  uploadedVideo, 
  handleVideoFileUpload, 
  processVideo,
  resetVideoState 
}: {
  isLoading: boolean;
  videoStatus: any;
  uploadedVideo: File | null;
  handleVideoFileUpload: (file: File) => Promise<void>;
  processVideo: () => Promise<void>;
  resetVideoState: () => void;
}) => {
  const [dragActive, setDragActive] = useState(false);

  const handleFileUpload = async (file: File) => {
    await handleVideoFileUpload(file);
  };

  const handleProcessVideo = async () => {
    await processVideo();
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
    setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <div className="text-center">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Procesamiento de Videos
          </h3>
          <p className="text-sm text-gray-600 mb-6">
            Sube un video para detectar y analizar animales a lo largo de toda la grabación
          </p>
          
          <div
            className={`
              relative border-2 border-dashed rounded-xl p-8 transition-colors duration-200
              ${dragActive 
                ? 'border-purple-400 bg-purple-50' 
                : 'border-gray-300 hover:border-gray-400'
              }
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
        >
          <input
            type="file"
            accept="video/*"
              onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              disabled={isLoading}
          />
          
          <div className="space-y-4">
              <VideoCameraIcon className="w-12 h-12 text-gray-400 mx-auto" />
            <div>
                <p className="text-lg font-medium text-gray-900">
                  {isLoading ? 'Subiendo...' : 'Arrastra un video aquí'}
                </p>
                <p className="text-sm text-gray-600">
                  o haz clic para seleccionar un archivo
              </p>
            </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Video Processing */}
      {uploadedVideo && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <div className="flex items-center justify-between mb-4">
              <div>
              <h4 className="text-lg font-semibold text-gray-900">
                Video Subido
              </h4>
              <p className="text-sm text-gray-600">
                {uploadedVideo.name} ({(uploadedVideo.size / (1024 * 1024)).toFixed(2)} MB)
                </p>
              </div>
              <button
              onClick={handleProcessVideo}
                disabled={isLoading || videoStatus.status === 'processing'}
              className="inline-flex items-center px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors duration-200"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Procesando...
                </>
              ) : (
                <>
                <PlayIcon className="w-4 h-4 mr-2" />
                  Procesar Video
                </>
              )}
              </button>
            </div>

          {/* Progress */}
            {videoStatus.status === 'processing' && (
          <div className="space-y-4">
              <div className="flex items-center justify-between text-sm text-gray-600">
                <span>Procesando frames...</span>
                <span>{videoStatus?.progress_percent ? videoStatus.progress_percent.toFixed(1) : '0.0'}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                  className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${videoStatus?.progress_percent || 0}%` }}
                  ></div>
                </div>
              <div className="text-xs text-gray-500">
                {videoStatus?.progress || 0} de {videoStatus?.total_frames || 0} frames procesados
                </div>
              </div>
            )}

          {/* Completed */}
          {videoStatus.status === 'completed' && videoStatus.output_video_url && (
              <div className="space-y-6">
                <div className="flex items-center space-x-2 text-green-600">
                <CheckCircleIcon className="w-5 h-5" />
                <span className="font-medium">Procesamiento completado</span>
                </div>
                
              {/* Video procesado integrado */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h5 className="text-md font-semibold text-gray-900 mb-4">
                  Video con Detecciones
                </h5>
                <div className="relative">
                    <video
                      controls
                    className="w-full h-auto rounded-lg shadow-sm"
                    style={{ maxHeight: '500px' }}
                  >
                    <source src={`http://localhost:5003${videoStatus.output_video_url}`} type="video/mp4" />
                    Tu navegador no soporta el elemento video.
                    </video>
                  </div>
                
                {/* Botones de acción */}
                <div className="flex space-x-4 mt-4">
                    <a
                    href={`http://localhost:5003${videoStatus.output_video_url}`}
                      download
                    className="inline-flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition-colors duration-200"
                  >
                    <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
                    Descargar Video
                  </a>
                  <button
                    onClick={() => {
                      const video = document.querySelector('video');
                      if (video) {
                        if (video.requestFullscreen) {
                          video.requestFullscreen();
                        }
                      }
                    }}
                    className="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors duration-200"
                  >
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                    </svg>
                    Pantalla Completa
                  </button>
                  </div>
                </div>
              
              {/* Estadísticas del procesamiento */}
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <h5 className="text-md font-semibold text-gray-900 mb-4 flex items-center">
                  <ChartBarIcon className="w-5 h-5 text-green-500 mr-2" />
                  Estadísticas del Procesamiento
                </h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {videoStatus?.total_frames || 0}
              </div>
                    <div className="text-sm text-gray-600">Frames Totales</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {videoStatus?.progress || 0}
                    </div>
                    <div className="text-sm text-gray-600">Frames Procesados</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {videoStatus?.progress_percent ? videoStatus.progress_percent.toFixed(1) : '0.0'}%
                    </div>
                    <div className="text-sm text-gray-600">Completado</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-600">
                      {uploadedVideo ? `${(uploadedVideo.size / (1024 * 1024)).toFixed(1)}MB` : 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Tamaño Original</div>
                  </div>
                </div>
              </div>
              
              {/* Botón para procesar otro video */}
              <div className="text-center">
                <button
                  onClick={resetVideoState}
                  className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-medium rounded-lg transition-all duration-200 transform hover:scale-105"
                >
                  <VideoCameraIcon className="w-5 h-5 mr-2" />
                  Procesar Otro Video
                </button>
              </div>
          </div>
          )}

          {/* Error */}
          {videoStatus.status === 'failed' && (
            <div className="flex items-center space-x-2 text-red-600">
              <XCircleIcon className="w-5 h-5" />
              <span className="font-medium">Error: {videoStatus.error}</span>
          </div>
            )}
        </motion.div>
      )}
    </div>
  );
};

const App = () => {
  const { 
    activeTab, 
    setActiveTab, 
    setModelLoaded, 
    isLoading,
    setIsLoading,
    currentDetection,
    setCurrentDetection,
    videoStatus,
    uploadedVideo,
    handleVideoFileUpload,
    processVideo,
    resetVideoState,
    isConnected,
    setIsConnected,
    notifications,
    addNotification,
    removeNotification
  } = useAppState();

  // Check model status and connection on mount
  useEffect(() => {
    let mounted = true;
    
    const checkModelStatus = async () => {
      try {
        const status = await apiService.getModelStatus();
        if (mounted) {
          setModelLoaded(status.model_loaded);
          setIsConnected(true); // Connection successful
          
          if (status.model_loaded) {
            addNotification({
              type: 'success',
              title: '¡Bienvenido al Sistema de Detección Veterinaria!',
              message: 'Todo está listo para detectar y analizar animales. ¡Comenzá subiendo una imagen o video!',
            });
          } else {
            addNotification({
              type: 'info',
              title: '¡Bienvenido!',
              message: 'Estamos preparando el sistema para vos. Solo unos momentos más...',
            });
          }
        }
      } catch (error) {
        console.error('Error checking model status:', error);
        if (mounted) {
          setIsConnected(false); // Connection failed
          addNotification({
            type: 'warning',
            title: 'Oops! No pudimos conectarnos',
            message: 'Parece que el servidor no está disponible. Por favor, verificá que esté funcionando correctamente.',
          });
        }
      }
    };

    checkModelStatus();

    // Check connection periodically
    const connectionCheck = setInterval(async () => {
      if (mounted) {
        try {
          await apiService.getModelStatus();
          setIsConnected(true);
        } catch (error) {
          console.warn('Connection check failed:', error);
          setIsConnected(false);
        }
      }
    }, 30000); // Check every 30 seconds

    return () => {
      mounted = false;
      clearInterval(connectionCheck);
    };
  }, []); // Empty dependency array - run only once

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'image':
        return (
          <ImageDetection 
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            currentDetection={currentDetection}
            setCurrentDetection={setCurrentDetection}
            addNotification={addNotification}
          />
        );
      case 'video':
        return (
          <VideoDetection 
            isLoading={isLoading}
            videoStatus={videoStatus}
            uploadedVideo={uploadedVideo}
            handleVideoFileUpload={handleVideoFileUpload}
            processVideo={processVideo}
            resetVideoState={resetVideoState}
          />
        );
      case 'webcam':
        return (
          <WebcamDetection />
        );
      default:
        return (
          <ImageDetection 
            isLoading={isLoading}
            setIsLoading={setIsLoading}
            currentDetection={currentDetection}
            setCurrentDetection={setCurrentDetection}
            addNotification={addNotification}
          />
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-yellow-50 to-white">
      {/* Header mejorado */}
      <Header isConnected={isConnected} />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Navegación mejorada */}
        <div className="mb-8">
          <Navigation 
            activeTab={activeTab} 
            onTabChange={(tab) => setActiveTab(tab as 'image' | 'video' | 'webcam')} 
          />
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {renderActiveTab()}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <Footer />

      {/* Notification Center */}
      <NotificationCenter 
        notifications={notifications} 
        removeNotification={removeNotification} 
      />
    </div>
  );
};

const AppWithErrorBoundary = () => (
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);

export default AppWithErrorBoundary;
