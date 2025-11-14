import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { 
  CloudArrowUpIcon, 
  VideoCameraIcon, 
  XMarkIcon, 
  PlayIcon,
  ArrowDownTrayIcon,
  ChartBarIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';
import { useAppStore } from '../store/useAppStore';
import { apiService } from '../services/api';
import type { VideoStatus } from '../services/api';

const VideoDetection = () => {
  const { videoStatus, setVideoStatus, isLoading, setIsLoading, addNotification } = useAppStore();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    
    // Validate file type
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/wmv', 'video/flv'];
    if (!allowedTypes.includes(file.type)) {
      addNotification({
        type: 'error',
        title: 'Tipo de archivo no válido',
        message: 'Por favor, selecciona un video válido (MP4, AVI, MOV, MKV, WMV, FLV)',
      });
      return;
    }

    // Validate file size (150MB max)
    if (file.size > 150 * 1024 * 1024) {
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
        output_video_url: undefined,
        processed_video_filename: null,
        report_pdf_filename: null,
        report_pdf_url: null
      } as VideoStatus);
      
      // Upload file
      await apiService.uploadFile(file);
      setUploadedFile(file);
      
      addNotification({
        type: 'success',
        title: 'Video subido',
        message: 'El video se ha subido correctamente. Haz clic en "Procesar" para comenzar.',
      });
      
    } catch (error) {
      console.error('Error uploading video:', error);
      addNotification({
        type: 'error',
        title: 'Error al subir video',
        message: 'No se pudo subir el video. Inténtalo de nuevo.',
      });
    } finally {
      setIsLoading(false);
    }
  }, [setVideoStatus, setIsLoading, addNotification]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    },
    multiple: false,
    onDragEnter: () => setDragActive(true),
    onDragLeave: () => setDragActive(false),
  });

  const processVideo = async () => {
    if (!uploadedFile) return;

    try {
      setIsLoading(true);
      setVideoStatus({ 
        status: 'processing', 
        progress: 0, 
        processed_frames: 0,
        total_frames: 0, 
        progress_percent: 0,
        output_video_url: undefined,
        processed_video_filename: null,
        report_pdf_filename: null,
        report_pdf_url: null
      } as VideoStatus);
      
      const uploadResult = await apiService.uploadFile(uploadedFile);
      await apiService.processVideo(uploadResult.filename);
      
      addNotification({
        type: 'info',
        title: 'Procesamiento iniciado',
        message: 'El video se está procesando. Puedes seguir el progreso abajo.',
      });
      
      // Start polling for status
      pollVideoStatus(uploadResult.filename);
      
    } catch (error) {
      console.error('Error processing video:', error);
      setVideoStatus({ 
        status: 'error', 
        progress: 0, 
        processed_frames: 0,
        total_frames: 0, 
        progress_percent: 0,
        output_video_url: undefined,
        processed_video_filename: null,
        report_pdf_filename: null,
        report_pdf_url: null,
        error: 'Error al procesar el video'
      } as VideoStatus);
      addNotification({
        type: 'error',
        title: 'Error al procesar video',
        message: 'No se pudo procesar el video. Inténtalo de nuevo.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const pollVideoStatus = async (filename: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await apiService.getVideoStatus(filename);
        
        // DEBUG: Ver qué datos llegan exactamente
        console.log('🔍 Raw backend data:', JSON.stringify(status, null, 2));
        
        // Usar los datos del backend directamente SIN normalización
        const statusWithProgress = {
          ...status,
          progress: status.processed_frames, // Mapear processed_frames a progress para la UI
        };
        
        console.log('🔍 Status for UI:', JSON.stringify(statusWithProgress, null, 2));
        setVideoStatus(statusWithProgress);
        
        if (status.status === 'completed' || status.status === 'error') {
          clearInterval(pollInterval);
          if (status.status === 'completed') {
            addNotification({
              type: 'success',
              title: 'Video procesado exitosamente',
              message: 'El video ha sido procesado correctamente.',
            });
          }
        }
      } catch (error) {
        console.error('Error polling video status:', error);
        clearInterval(pollInterval);
        setVideoStatus({
          status: 'error',
          progress: 0,
          processed_frames: 0,
          total_frames: 0,
          progress_percent: 0,
          output_video_url: undefined,
          processed_video_filename: null,
          report_pdf_filename: null,
          report_pdf_url: null,
          error: 'Error de conexión durante el procesamiento'
        } as VideoStatus);
      }
    }, 1000); // Reducir a 1 segundo para progreso más suave
  };

  const clearVideo = () => {
    setUploadedFile(null);
    setVideoStatus({ 
      status: 'idle', 
      progress: 0, 
      processed_frames: 0,
      total_frames: 0, 
      progress_percent: 0,
      output_video_url: undefined,
      processed_video_filename: null,
      report_pdf_filename: null,
      report_pdf_url: null
    } as VideoStatus);
  };

  const reportFilename = videoStatus.report_pdf_filename || undefined;
  const reportUrlFromStatus = videoStatus.report_pdf_url || undefined;
  const reportDownloadUrl = reportUrlFromStatus
    ? apiService.resolveUrl(reportUrlFromStatus)
    : reportFilename
    ? apiService.getDownloadUrl(reportFilename)
    : undefined;
  const hasReport = Boolean(reportDownloadUrl);
  const completedActionsClass = `grid grid-cols-1 ${hasReport ? 'sm:grid-cols-3' : 'sm:grid-cols-2'} gap-3`;

  const derivedFilenameFromUrl = videoStatus.output_video_url
    ? videoStatus.output_video_url.split('/').filter(Boolean).pop()
    : undefined;
  const processedVideoFilename =
    videoStatus.processed_video_filename ||
    derivedFilenameFromUrl ||
    (uploadedFile ? `processed_${uploadedFile.name}` : undefined);

  const viewVideoUrl = videoStatus.output_video_url
    ? apiService.resolveUrl(videoStatus.output_video_url)
    : processedVideoFilename
    ? apiService.getVideoUrl(processedVideoFilename)
    : undefined;

  const downloadVideoUrl = processedVideoFilename
    ? apiService.getDownloadUrl(processedVideoFilename)
    : undefined;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processing':
        return 'text-blue-600 bg-blue-100';
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'processing':
        return 'Procesando';
      case 'completed':
        return 'Completado';
      case 'error':
        return 'Error';
      default:
        return 'En espera';
    }
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
            isDragActive || isDragActive
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
                <VideoCameraIcon className="w-full h-full" />
              )}
            </div>
            
            <div>
              <h3 className="text-lg font-medium text-gray-900">
                {isDragActive || dragActive
                  ? 'Suelta el video aquí'
                  : 'Arrastra un video o haz clic para seleccionar'
                }
              </h3>
              <p className="mt-2 text-sm text-gray-500">
                Detección optimizada para mejores resultados<br/>
                Soporta MP4, AVI, MOV, MKV, WMV, FLV (máx. 150MB)
              </p>
            </div>
            
            {!isDragActive && !dragActive && (
              <button className="btn-primary">
                Seleccionar video
              </button>
            )}
          </div>
        </div>
      </motion.div>

      {/* Uploaded File Info */}
      {uploadedFile && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="card"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <VideoCameraIcon className="w-8 h-8 text-primary-600" />
              <div>
                <h4 className="font-medium text-gray-900">{uploadedFile.name}</h4>
                <p className="text-sm text-gray-500">
                  {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={processVideo}
                disabled={isLoading || videoStatus.status === 'processing'}
                className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <PlayIcon className="w-4 h-4 mr-2" />
                Procesar
              </button>
              <button
                onClick={clearVideo}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Processing Status */}
      {videoStatus.status !== 'idle' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="card"
        >
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="text-lg font-semibold text-gray-900">
                Estado del procesamiento
              </h4>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(videoStatus.status)}`}>
                {getStatusText(videoStatus.status)}
              </span>
            </div>

            {videoStatus.status === 'processing' && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>Progreso</span>
                  <span>{(videoStatus.progress_percent || 0).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${videoStatus.progress_percent || 0}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-gray-500">
                  <span>{videoStatus.progress || 0} frames procesados</span>
                  <span>{videoStatus.total_frames || 0} frames totales</span>
                </div>
                
                {/* Información del tracking */}
                <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded text-xs text-blue-700">
                  Procesando video - Analizando cada frame
                </div>
              </div>
            )}

            {videoStatus.status === 'completed' && (
              <div className="space-y-4">
                <div className="flex items-center space-x-2 text-green-600">
                  <ChartBarIcon className="w-5 h-5" />
                  <span className="font-medium">Video procesado exitosamente</span>
                </div>
                
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-3">
                  <p className="text-sm text-green-700">
                    <strong>Video procesado exitosamente:</strong> Detección optimizada aplicada en todos los frames
                  </p>
                </div>
                
                <div className={completedActionsClass}>
                  <a
                    href={viewVideoUrl || '#'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`btn-primary justify-center ${!viewVideoUrl ? 'pointer-events-none opacity-50' : ''}`}
                  >
                    <PlayIcon className="w-4 h-4 mr-2" />
                    Ver video procesado
                  </a>
                  <a
                    href={downloadVideoUrl || '#'}
                    download={processedVideoFilename || undefined}
                    className={`btn-secondary justify-center ${!downloadVideoUrl ? 'pointer-events-none opacity-50' : ''}`}
                  >
                    <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
                    Descargar
                  </a>
                  {hasReport && reportDownloadUrl && (
                    <a
                      href={reportDownloadUrl}
                      download={reportFilename || undefined}
                      className="btn-secondary justify-center"
                    >
                      <DocumentTextIcon className="w-4 h-4 mr-2" />
                      Reporte PDF
                    </a>
                  )}
                </div>
                
                {videoStatus.progress && videoStatus.total_frames && (
                  <div className="text-center">
                    <p className="text-xs text-gray-500">
                      Procesados {videoStatus.progress} de {videoStatus.total_frames} frames
                    </p>
                  </div>
                )}
              </div>
            )}

            {videoStatus.status === 'error' && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 text-red-600 mb-2">
                  <ChartBarIcon className="w-5 h-5" />
                  <span className="font-medium">Error en el procesamiento</span>
                </div>
                <p className="text-sm text-red-700 mb-3">
                  {videoStatus.error || 'Error desconocido durante el procesamiento del video'}
                </p>
                <div className="space-y-2 text-xs text-red-600">
                  <p>• Verifica que el backend esté ejecutándose</p>
                  <p>• Asegúrate de que el video no esté corrupto</p>
                  <p>• Intenta con un video más pequeño si el problema persiste</p>
                </div>
                <button
                  onClick={clearVideo}
                  className="mt-3 px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                >
                  Reintentar con otro video
                </button>
              </div>
            )}
          </div>
        </motion.div>
      )}

      {/* Loading State */}
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="card text-center"
        >
          <div className="flex items-center justify-center space-x-2">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
            <span className="text-gray-600">Inicializando sistema<span className="loading-dots"></span></span>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default VideoDetection;
