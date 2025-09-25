import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, PhotoIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useAppStore } from '../store/useAppStore';
import { apiService } from '../services/api';

const ImageDetection = () => {
  const { currentDetection, setCurrentDetection, isLoading, setIsLoading, addNotification } = useAppStore();
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    
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
        message: 'El archivo debe ser menor a 10MB',
      });
      return;
    }

    try {
      setIsLoading(true);
      
      // Upload file
      const uploadResult = await apiService.uploadFile(file);
      
      // Detect objects
      const detectionResult = await apiService.detectImage(uploadResult.filename);
      
      setCurrentDetection(detectionResult);
      
      addNotification({
        type: 'success',
        title: 'Detección completada',
        message: `Se detectaron ${detectionResult.total_detections} animales`,
      });
      
    } catch (error) {
      console.error('Error detecting image:', error);
      addNotification({
        type: 'error',
        title: 'Error en la detección',
        message: 'No se pudo procesar la imagen. Inténtalo de nuevo.',
      });
    } finally {
      setIsLoading(false);
    }
  }, [setCurrentDetection, setIsLoading, addNotification]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    },
    multiple: false,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  const clearDetection = () => {
    setCurrentDetection(null);
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="card"
      >
        <div
          {...getRootProps()}
          className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
            isDragActive || dragActive
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          
          <div className="space-y-4">
            <div className="mx-auto w-16 h-16 text-gray-400">
              {isDragActive || dragActive ? (
                <CloudArrowUpIcon className="w-full h-full animate-bounce-gentle" />
              ) : (
                <PhotoIcon className="w-full h-full" />
              )}
            </div>
            
            <div>
              <h3 className="text-lg font-medium text-gray-900">
                {isDragActive || dragActive
                  ? 'Suelta la imagen aquí'
                  : 'Arrastra una imagen o haz clic para seleccionar'
                }
              </h3>
              <p className="mt-2 text-sm text-gray-500">
                Soporta JPG, PNG, GIF, WebP (máx. 10MB)
              </p>
            </div>
            
            {!isDragActive && !dragActive && (
              <button className="btn-primary">
                Seleccionar imagen
              </button>
            )}
          </div>
        </div>
      </motion.div>

      {/* Loading State */}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card text-center"
        >
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
            <span className="text-gray-600">Procesando imagen<span className="loading-dots"></span></span>
          </div>
        </motion.div>
      )}

      {/* Detection Results */}
      {currentDetection && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          {/* Results Header */}
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">
                  Resultados de la detección
                </h3>
                <p className="text-sm text-gray-600">
                  {currentDetection.total_detections} animal{currentDetection.total_detections !== 1 ? 'es' : ''} detectado{currentDetection.total_detections !== 1 ? 's' : ''}
                </p>
              </div>
              <button
                onClick={clearDetection}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Image with detections */}
          <div className="card">
            <div className="relative">
              <img
                src={apiService.getDownloadUrl(currentDetection.processed_filename)}
                alt="Imagen procesada"
                className="w-full h-auto rounded-lg shadow-sm"
              />
              
              {/* Detection overlays */}
              {currentDetection.detections.map((detection, index) => (
                <div
                  key={index}
                  className="absolute border-2 rounded"
                  style={{
                    left: detection.bbox[0],
                    top: detection.bbox[1],
                    width: detection.bbox[2] - detection.bbox[0],
                    height: detection.bbox[3] - detection.bbox[1],
                    borderColor: getAnimalColor(detection.class),
                  }}
                >
                  <div
                    className="absolute -top-6 left-0 px-2 py-1 text-xs font-medium text-white rounded-t"
                    style={{ backgroundColor: getAnimalColor(detection.class) }}
                  >
                    {detection.class} ({Math.round(detection.confidence * 100)}%)
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Detection Details */}
          <div className="card">
            <h4 className="text-md font-semibold text-gray-900 mb-4">
              Detalles de las detecciones
            </h4>
            <div className="space-y-3">
              {currentDetection.detections.map((detection, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div
                      className="w-4 h-4 rounded-full"
                      style={{ backgroundColor: getAnimalColor(detection.class) }}
                    ></div>
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
        </motion.div>
      )}
    </div>
  );
};

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

export default ImageDetection;
