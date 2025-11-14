import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CameraIcon, 
  StopIcon,
  PlayIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  XMarkIcon,
  ChevronRightIcon,
  BellIcon
} from '@heroicons/react/24/outline';
import { useAppStore } from '../store/useAppStore';
import { apiService } from '../services/api';

const WebcamDetection = () => {
  const { 
    webcamActive, 
    setWebcamActive, 
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
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [toastNotifications, setToastNotifications] = useState<Array<{
    id: string;
    class: string;
    confidence: number;
    timestamp: number;
  }>>([]);
  const [previousDetectionsCount, setPreviousDetectionsCount] = useState(0);

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
          
          // Configuraciones optimizadas para velocidad
          const img = streamImageRef.current;
          img.style.imageRendering = 'auto';      // Renderizado estándar rápido
          img.style.backfaceVisibility = 'hidden';
          img.style.transform = 'translateZ(0)';
          img.style.willChange = 'transform';      // Optimización GPU básica
          img.style.filter = 'none';               // Sin filtros para máxima velocidad
          
          // Set source optimizado para velocidad
          const streamUrl = apiService.getWebcamStreamUrl() + '?speed=1&t=' + Date.now();
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
      setToastNotifications([]);
      setPreviousDetectionsCount(0);
      
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
        const newDetections = result.detections;
        setRecentDetections(newDetections);
        setWebcamDetections(newDetections);
        
        // Detectar nuevas detecciones para mostrar toasts
        if (newDetections.length > previousDetectionsCount) {
          const newOnes = newDetections.slice(previousDetectionsCount);
          newOnes.forEach((det: any) => {
            const toastId = `${det.timestamp || Date.now()}-${Math.random()}`;
            setToastNotifications(prev => [...prev, {
              id: toastId,
              class: det.class || det.class_name || 'unknown',
              confidence: det.confidence || 0,
              timestamp: det.timestamp || Date.now() / 1000
            }]);
            
            // Auto-remove toast after 4 seconds
            setTimeout(() => {
              setToastNotifications(prev => prev.filter(t => t.id !== toastId));
            }, 4000);
          });
        }
        
        setPreviousDetectionsCount(newDetections.length);
      }
    } catch (error) {
      console.error('Error fetching streaming detections:', error);
    }
  };


  // Effect para polling de detecciones - OPTIMIZADO PARA VELOCIDAD
  useEffect(() => {
    if (webcamActive && isStreamConnected) {
      detectionIntervalRef.current = setInterval(fetchStreamingDetections, 2000); // Cada 2 segundos para velocidad
      return () => {
        if (detectionIntervalRef.current) {
          clearInterval(detectionIntervalRef.current);
        }
      };
    }
  }, [webcamActive, isStreamConnected, previousDetectionsCount]);

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
      auto: '#a855f7',     // Púrpura para autos
      car: '#a855f7',
      caballo: '#facc15',
      perro: '#3b82f6',
      vaca: '#22c55e',
    };
    return colors[animalClass?.toLowerCase()] || '#6b7280';
  };

  const getAnimalName = (animalClass: string): string => {
    const names: Record<string, string> = {
      cat: 'Gato',
      chicken: 'Gallina',
      cow: 'Vaca',
      dog: 'Perro',
      horse: 'Caballo',
      auto: 'Auto',
      car: 'Auto',
      caballo: 'Caballo',
      perro: 'Perro',
      vaca: 'Vaca',
    };
    return names[animalClass?.toLowerCase()] || animalClass;
  };

  // Estadísticas rápidas
  const detectionStats = recentDetections.reduce((acc, det: any) => {
    const className = (det.class || det.class_name || 'unknown').toLowerCase();
    acc[className] = (acc[className] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const totalDetections = recentDetections.length;

  return (
    <div className="space-y-6 relative">
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

      {/* Webcam Display with Overlay Stats */}
      <AnimatePresence>
        {webcamActive && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.5 }}
            className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 relative"
        >
          <div className="relative">
            {/* Streaming mode */}
              <div className="relative">
                <img
                  ref={streamImageRef}
                  alt="Streaming de cámara con detecciones"
                  className="w-full h-auto rounded-lg shadow-sm bg-gray-100"
                  style={{ 
                    minHeight: '480px',
                    maxHeight: '600px',
                    imageRendering: 'auto',
                    backfaceVisibility: 'hidden',
                    transform: 'translateZ(0)',
                    willChange: 'transform'
                  }}
                />
                  
                  {/* Live indicator */}
                  {isStreamConnected && (
                    <div className="absolute top-4 right-4 bg-green-400 text-white px-3 py-1 rounded-full text-sm font-medium flex items-center z-10">
                      <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></div>
                      En vivo
                    </div>
                  )}

                  {/* Floating Stats Badge */}
                  {isStreamConnected && totalDetections > 0 && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg shadow-lg p-3 border border-gray-200 z-10"
                    >
                      <div className="flex items-center space-x-2 mb-2">
                        <BellIcon className="w-4 h-4 text-gray-600" />
                        <span className="text-sm font-semibold text-gray-900">
                          {totalDetections} detección{totalDetections !== 1 ? 'es' : ''}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                      {Object.entries(detectionStats).slice(0, 3).map(([animal, count]) => (
                        <div
                          key={animal}
                          className="flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium text-white"
                          style={{ backgroundColor: getAnimalColor(animal) }}
                        >
                          <span>{getAnimalName(animal)}</span>
                          <span className="bg-white/30 rounded-full px-1.5">{count as number}</span>
                        </div>
                      ))}
                        {Object.keys(detectionStats).length > 3 && (
                          <div className="px-2 py-1 rounded-full text-xs font-medium bg-gray-200 text-gray-700">
                            +{Object.keys(detectionStats).length - 3}
                          </div>
                        )}
                      </div>
                    </motion.div>
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

      {/* Toast Notifications - Top Right */}
      <div className="fixed top-20 right-4 z-50 space-y-2 pointer-events-none">
        <AnimatePresence>
          {toastNotifications.map((toast) => (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, x: 100, scale: 0.8 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 100, scale: 0.8 }}
              transition={{ duration: 0.3 }}
              className="pointer-events-auto bg-white rounded-lg shadow-xl border border-gray-200 p-4 min-w-[280px] max-w-[320px]"
            >
              <div className="flex items-start space-x-3">
                <div
                  className="w-3 h-3 rounded-full mt-1 flex-shrink-0"
                  style={{ backgroundColor: getAnimalColor(toast.class) }}
                ></div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold text-gray-900 capitalize">
                      {getAnimalName(toast.class)} detectado
                    </p>
                    <button
                      onClick={() => setToastNotifications(prev => prev.filter(t => t.id !== toast.id))}
                      className="text-gray-400 hover:text-gray-600 flex-shrink-0 ml-2"
                    >
                      <XMarkIcon className="w-4 h-4" />
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {Math.round(toast.confidence * 100)}% de confianza
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Sidebar de Detecciones - Deslizable desde la derecha */}
      <AnimatePresence>
        {webcamActive && (
          <>
            {/* Toggle Button */}
            <motion.button
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={`fixed right-0 top-1/2 -translate-y-1/2 z-40 bg-white rounded-l-lg shadow-lg border border-r-0 border-gray-200 p-3 transition-all duration-300 ${
                sidebarOpen ? 'translate-x-0' : 'translate-x-full'
              }`}
            >
              {sidebarOpen ? (
                <ChevronRightIcon className="w-5 h-5 text-gray-600" />
              ) : (
                <div className="relative">
                  <BellIcon className="w-5 h-5 text-gray-600" />
                  {totalDetections > 0 && (
                    <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-bold">
                      {totalDetections > 9 ? '9+' : totalDetections}
                    </span>
                  )}
                </div>
              )}
            </motion.button>

            {/* Sidebar Panel */}
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: sidebarOpen ? 0 : '100%' }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed right-0 top-0 h-full w-96 bg-white shadow-2xl z-50 border-l border-gray-200 overflow-hidden"
            >
              <div className="h-full flex flex-col">
                {/* Header */}
                <div className="bg-gradient-to-r from-primary-600 to-primary-700 text-white p-6 flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">Detecciones en Vivo</h3>
                    <p className="text-sm text-primary-100 mt-1">
                      {totalDetections} detección{totalDetections !== 1 ? 'es' : ''} recientes
                    </p>
                  </div>
                  <button
                    onClick={() => setSidebarOpen(false)}
                    className="text-white hover:text-primary-100 transition-colors"
                  >
                    <XMarkIcon className="w-6 h-6" />
                  </button>
                </div>

                {/* Stats Summary */}
                {Object.keys(detectionStats).length > 0 && (
                  <div className="p-4 bg-gray-50 border-b border-gray-200">
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(detectionStats).map(([animal, count]) => (
                        <div
                          key={animal}
                          className="flex items-center space-x-2 p-2 bg-white rounded-lg"
                        >
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getAnimalColor(animal) }}
                          ></div>
                          <span className="text-sm font-medium text-gray-700 capitalize flex-1">
                            {getAnimalName(animal)}
                          </span>
                          <span className="text-sm font-bold text-gray-900">{count as number}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Detections List */}
                <div className="flex-1 overflow-y-auto p-4">
                  {recentDetections.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                      <BellIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                      <p className="text-sm">Aún no hay detecciones</p>
                      <p className="text-xs mt-1">Las detecciones aparecerán aquí</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {recentDetections.slice().reverse().map((detection: any, index: number) => (
                        <motion.div
                          key={`${detection.timestamp || index}-${index}`}
                          initial={{ opacity: 0, x: 20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.05 }}
                          className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow"
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex items-start space-x-3 flex-1">
                              <div
                                className="w-4 h-4 rounded-full mt-1 flex-shrink-0"
                                style={{ backgroundColor: getAnimalColor(detection.class || detection.class_name) }}
                              ></div>
                              <div className="flex-1 min-w-0">
                                <p className="font-semibold text-gray-900 capitalize text-sm">
                                  {getAnimalName(detection.class || detection.class_name)}
                                </p>
                                <p className="text-xs text-gray-500 mt-1">
                                  {detection.timestamp 
                                    ? new Date(detection.timestamp * 1000).toLocaleTimeString()
                                    : 'Ahora'
                                  }
                                </p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="text-sm font-semibold text-gray-700">
                                {Math.round((detection.confidence || 0) * 100)}%
                              </p>
                              <p className="text-xs text-gray-400">confianza</p>
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  );
};

export default WebcamDetection;
