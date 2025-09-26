import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CameraIcon, 
  StopIcon,
  PlayIcon,
  ChartBarIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { useAppStore } from '../store/useAppStore';
import { apiService } from '../services/api';

const WebcamDetection = () => {
  const { 
    webcamActive, 
    setWebcamActive, 
    webcamDetections, 
    setWebcamDetections,
    isLoading,
    setIsLoading,
    addNotification 
  } = useAppStore();
  
  // Refs para elementos del DOM
  const streamImageRef = useRef<HTMLImageElement>(null);
  const detectionIntervalRef = useRef<number | null>(null);
  
  // Estados locales
  const [streamError, setStreamError] = useState<string | null>(null);
  const [recentDetections, setRecentDetections] = useState<any[]>([]);
  const [isStreamConnected, setIsStreamConnected] = useState(false);


  // Función para iniciar streaming continuo del backend
  const startWebcam = async () => {
    try {
      setIsLoading(true);
      setStreamError(null);
      setIsStreamConnected(false);
      
      console.log('🎥 Iniciando detección en vivo...');
      
      // Activar el estado inmediatamente para mostrar la UI
      setWebcamActive(true);
      
      // Configurar el stream con un pequeño delay para asegurar que el elemento img esté en el DOM
      setTimeout(() => {
      if (streamImageRef.current) {
          console.log('📡 Configurando stream URL:', apiService.getWebcamStreamUrl());
          
          // Configurar optimizaciones MÁXIMAS para fluidez
          const img = streamImageRef.current;
          img.style.imageRendering = 'pixelated';    // Render más rápido para video en tiempo real
          img.style.backfaceVisibility = 'hidden';
          img.style.transform = 'translateZ(0)';
          img.style.willChange = 'transform';        // Optimización GPU
          img.style.filter = 'contrast(1.1)';       // Mejorar contraste para compensar compresión
          
          // Set source con parámetros anti-cache para mejor streaming
          const streamUrl = apiService.getWebcamStreamUrl() + '?t=' + Date.now();
          img.src = streamUrl;
        
        img.onload = () => {
            console.log('✅ Stream conectado');
          setIsStreamConnected(true);
          addNotification({
            type: 'success',
              title: 'Cámara iniciada correctamente',
              message: 'La detección en tiempo real está funcionando perfectamente',
          });
        };

          img.onerror = (error) => {
            console.error('❌ Error en stream:', error);
          setStreamError('No se pudo conectar a la cámara');
          setIsStreamConnected(false);
          addNotification({
            type: 'error',
            title: 'Error de streaming',
            message: 'No se pudo conectar. Verifica que el backend esté corriendo.',
          });
        };
        } else {
          console.error('❌ streamImageRef.current es null');
          setStreamError('Error: elemento de video no disponible');
          setWebcamActive(false);
      }
      }, 100);
      
    } catch (error) {
      console.error('Error starting streaming webcam:', error);
      setStreamError('Error al iniciar el streaming');
      setWebcamActive(false);
      addNotification({
        type: 'error',
        title: 'Error de streaming',
        message: 'No se pudo iniciar el streaming de video.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Función para detener la cámara
  const stopWebcam = () => {
    console.log('🛑 Deteniendo cámara...');
    
    try {
      // Detener streaming
      if (streamImageRef.current) {
        streamImageRef.current.src = '';
        streamImageRef.current.onload = null;
        streamImageRef.current.onerror = null;
      }
      
      // Limpiar intervalos
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current);
        detectionIntervalRef.current = null;
      }
      
      // Resetear todos los estados
    setWebcamActive(false);
      setStreamError(null);
      setIsStreamConnected(false);
      setRecentDetections([]);
      setWebcamDetections([]);
      
      console.log('✅ Cámara detenida completamente');
      addNotification({
        type: 'info',
        title: 'Cámara detenida',
        message: 'La cámara se ha detenido correctamente',
      });
      
    } catch (error) {
      console.error('❌ Error al detener la cámara:', error);
      addNotification({
        type: 'warning',
        title: 'Cámara detenida',
        message: 'La cámara se detuvo pero hubo algunos problemas menores',
      });
    }
  };


  // Función para obtener detecciones del streaming
  const fetchStreamingDetections = async () => {
    try {
      const result = await apiService.getWebcamDetections();
      if (result.detections && result.detections.length > 0) {
        setRecentDetections(result.detections);
        setWebcamDetections(result.detections);
      }
    } catch (error) {
      console.error('Error fetching streaming detections:', error);
    }
  };


  // Effect para polling de detecciones en streaming
  useEffect(() => {
    if (webcamActive && isStreamConnected) {
      detectionIntervalRef.current = setInterval(fetchStreamingDetections, 3000);
      return () => {
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
        }
      };
    }
  }, [webcamActive, isStreamConnected]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  const getAnimalColor = (animalClass: string): string => {
    const colors: Record<string, string> = {
      cat: '#e879f9',      // Magenta brillante
      chicken: '#fb923c',  // Naranja
      cow: '#22c55e',      // Verde
      dog: '#3b82f6',      // Azul
      horse: '#facc15',    // Amarillo brillante (más visible)
    };
    return colors[animalClass] || '#6b7280';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between space-y-4 sm:space-y-0">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Detección en tiempo real
            </h3>
            <p className="text-sm text-gray-600 mt-1">
              Detecta animales usando tu cámara en tiempo real
            </p>
          </div>
        </div>

      </motion.div>

      {/* Webcam Controls */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
      >
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-md font-semibold text-gray-900">
              Control de cámara
            </h4>
            <p className="text-sm text-gray-600 mt-1">
              {webcamActive 
                   ? 'Streaming activo - Las detecciones aparecen automáticamente'
                : 'Inicia la cámara para comenzar la detección en vivo'
              }
            </p>
            {isStreamConnected && (
              <p className="text-xs text-green-500 mt-1">
                Stream conectado - Detección automática activa
              </p>
            )}
          </div>
          
          <div className="flex space-x-2">
            {!webcamActive ? (
              <button
                onClick={startWebcam}
                disabled={isLoading}
                className="inline-flex items-center px-4 py-2 bg-green-500 hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors duration-200"
              >
                {isLoading ? (
                  <ArrowPathIcon className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                <PlayIcon className="w-4 h-4 mr-2" />
                )}
                Iniciar detección en vivo
              </button>
            ) : (
              <button
                onClick={stopWebcam}
                  className="inline-flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors duration-200"
              >
                <StopIcon className="w-4 h-4 mr-2" />
                Detener cámara
              </button>
            )}
          </div>
        </div>

        {/* Error Display */}
        {streamError && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg"
          >
            <div className="flex items-center">
              <ExclamationTriangleIcon className="w-5 h-5 text-red-500 mr-2" />
              <span className="text-red-700 text-sm">{streamError}</span>
            </div>
          </motion.div>
        )}
      </motion.div>

      {/* Webcam Display */}
      <AnimatePresence>
        {webcamActive && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.5 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <div className="relative">
            {/* Streaming mode */}
              <div className="relative">
                <img
                  ref={streamImageRef}
                  alt="Streaming de cámara con detecciones"
                  className="w-full h-auto rounded-lg shadow-sm bg-gray-100"
                  style={{ 
                    minHeight: '300px',
                    imageRendering: 'pixelated',        // Render optimizado para video
                    backfaceVisibility: 'hidden',
                    transform: 'translateZ(0)',         // Aceleración GPU
                    willChange: 'transform',            // Optimización de animación
                    filter: 'contrast(1.1)'            // Mejorar contraste
                  }}
                />
                  {isStreamConnected && (
                    <div className="absolute top-4 right-4 bg-green-400 text-white px-3 py-1 rounded-full text-sm font-medium flex items-center">
                      <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></div>
                      En vivo
                    </div>
                  )}
              
              {/* Instructions overlay when not connected */}
              {!isStreamConnected && (
                    <div className="absolute inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center rounded-lg">
                      <div className="text-center text-white">
                        <CameraIcon className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg font-medium">Conectando a la cámara...</p>
                    <p className="text-sm opacity-75">Iniciando detección en tiempo real</p>
                      </div>
                    </div>
                  )}
                </div>
          </div>
        </motion.div>
      )}
      </AnimatePresence>

      {/* Detection Results */}
      {recentDetections.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          {/* Results Header */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div>
                <h4 className="text-lg font-semibold text-gray-900">
                Detecciones en vivo
                </h4>
                <p className="text-sm text-gray-600">
                {recentDetections.length} detecciones en los últimos 30 segundos
              </p>
            </div>
          </div>

          {/* Detection Details */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h5 className="text-md font-semibold text-gray-900 mb-4">
              Detalles de las detecciones
            </h5>
            <div className="space-y-3">
              {recentDetections?.map((detection: any, index: number) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: getAnimalColor((detection as any).class || (detection as any).class_name) }}
                    ></div>
                    <span className="font-medium text-gray-900 capitalize">
                      {(detection as any).class || (detection as any).class_name}
                    </span>
                    {detection.timestamp && (
                      <span className="text-xs text-gray-500">
                        {new Date(detection.timestamp * 1000).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                  <div className="text-sm text-gray-600">
                    {Math.round((detection.confidence || 0) * 100)}% confianza
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Recent Detections Summary */}
      {webcamDetections.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white rounded-xl shadow-sm border border-gray-200 p-6"
        >
          <div className="flex items-center space-x-2 mb-4">
            <ChartBarIcon className="w-5 h-5 text-green-500" />
            <h5 className="text-md font-semibold text-gray-900">
              Resumen de detecciones
            </h5>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {['cat', 'chicken', 'cow', 'dog', 'horse'].map((animal) => {
              const count = webcamDetections.filter(d => ((d as any).class || (d as any).class_name) === animal).length;
              return (
                <div key={animal} className="text-center">
                  <div
                    className="w-12 h-12 rounded-full mx-auto mb-2 flex items-center justify-center text-white font-bold"
                    style={{ backgroundColor: getAnimalColor(animal) }}
                  >
                    {count}
                  </div>
                  <p className="text-xs text-gray-600 capitalize">{animal}</p>
                </div>
              );
            })}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default WebcamDetection;