import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  Linking,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { useAppStore } from '../store/useAppStore';
import { apiService, VideoStatus } from '../services/api';

const VideoDetectionScreen: React.FC = () => {
  const {
    videoStatus,
    setVideoStatus,
    uploadedVideoFilename,
    setUploadedVideoFilename,
    isLoading,
    setIsLoading,
    addNotification,
  } = useAppStore();

  const [selectedVideoName, setSelectedVideoName] = useState<string | null>(null);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  const pickVideo = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'video/*',
        copyToCacheDirectory: true,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        setSelectedVideoName(asset.name);
        await uploadVideo(asset);
      }
    } catch (error) {
      console.error('Error picking video:', error);
      addNotification({
        type: 'error',
        title: 'Error',
        message: 'No se pudo seleccionar el video.',
      });
    }
  };

  const uploadVideo = async (asset: any) => {
    try {
      setIsLoading(true);
      setVideoStatus({
        status: 'idle',
        progress: 0,
        processed_frames: 0,
        total_frames: 0,
        progress_percent: 0,
      });

      // Upload video
      const uploadResult = await apiService.uploadFile({
        uri: asset.uri,
        name: asset.name,
        type: asset.mimeType || 'video/mp4',
      });

      setUploadedVideoFilename(uploadResult.filename);

      addNotification({
        type: 'success',
        title: 'Video subido',
        message: 'El video se subió correctamente. Haz clic en "Procesar" para comenzar.',
      });
    } catch (error: any) {
      console.error('Error uploading video:', error);
      addNotification({
        type: 'error',
        title: 'Error al subir',
        message: error.message || 'No se pudo subir el video.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const processVideo = async () => {
    if (!uploadedVideoFilename) return;

    try {
      setIsLoading(true);
      setVideoStatus({
        status: 'processing',
        progress: 0,
        processed_frames: 0,
        total_frames: 0,
        progress_percent: 0,
      });

      // Start processing
      await apiService.processVideo(uploadedVideoFilename);

      addNotification({
        type: 'info',
        title: 'Procesamiento iniciado',
        message: 'El video se está procesando. Puedes ver el progreso abajo.',
      });

      // Start polling for status
      startPolling(uploadedVideoFilename);
    } catch (error: any) {
      console.error('Error processing video:', error);
      setVideoStatus({
        status: 'error',
        progress: 0,
        processed_frames: 0,
        total_frames: 0,
        progress_percent: 0,
        error: error.message || 'Error al procesar el video',
      });
      addNotification({
        type: 'error',
        title: 'Error al procesar',
        message: error.message || 'No se pudo procesar el video.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const startPolling = (filename: string) => {
    const interval = setInterval(async () => {
      try {
        const status = await apiService.getVideoStatus(filename);
        const statusWithProgress = {
          ...status,
          progress: status.processed_frames,
        };

        setVideoStatus(statusWithProgress);

        if (status.status === 'completed') {
          clearInterval(interval);
          setPollingInterval(null);
          addNotification({
            type: 'success',
            title: 'Video procesado',
            message: 'El video ha sido procesado exitosamente.',
          });
        } else if (status.status === 'error' || status.status === 'failed') {
          clearInterval(interval);
          setPollingInterval(null);
          addNotification({
            type: 'error',
            title: 'Error',
            message: status.error || 'Error durante el procesamiento.',
          });
        }
      } catch (error) {
        console.error('Error polling status:', error);
      }
    }, 2000); // Poll every 2 seconds

    setPollingInterval(interval);
  };

  const clearVideo = () => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
    setVideoStatus({
      status: 'idle',
      progress: 0,
      processed_frames: 0,
      total_frames: 0,
      progress_percent: 0,
    });
    setUploadedVideoFilename(null);
    setSelectedVideoName(null);
  };

  const openProcessedVideo = () => {
    if (videoStatus.output_video_url) {
      const url = apiService.getVideoUrl(videoStatus.output_video_url);
      Linking.openURL(url).catch((err) =>
        addNotification({
          type: 'error',
          title: 'Error',
          message: 'No se pudo abrir el video.',
        })
      );
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processing':
        return '#3b82f6';
      case 'completed':
        return '#22c55e';
      case 'error':
      case 'failed':
        return '#ef4444';
      default:
        return '#9ca3af';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'processing':
        return 'Procesando';
      case 'completed':
        return 'Completado';
      case 'error':
      case 'failed':
        return 'Error';
      default:
        return 'En espera';
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Upload Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Procesamiento de Videos</Text>
        <Text style={styles.cardSubtitle}>
          Selecciona un video para detectar animales en cada frame
        </Text>

        <TouchableOpacity
          style={styles.uploadButton}
          onPress={pickVideo}
          disabled={isLoading || videoStatus.status === 'processing'}
        >
          <Text style={styles.uploadButtonIcon}>🎥</Text>
          <Text style={styles.uploadButtonText}>
            {isLoading ? 'Subiendo...' : 'Seleccionar Video'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Video Info */}
      {selectedVideoName && uploadedVideoFilename && (
        <View style={styles.card}>
          <View style={styles.videoHeader}>
            <View style={styles.videoInfo}>
              <Text style={styles.videoIcon}>🎬</Text>
              <View>
                <Text style={styles.videoName}>{selectedVideoName}</Text>
                <Text style={styles.videoSubtext}>Listo para procesar</Text>
              </View>
            </View>
            <View style={styles.actions}>
              <TouchableOpacity
                style={styles.processButton}
                onPress={processVideo}
                disabled={isLoading || videoStatus.status === 'processing'}
              >
                <Text style={styles.processButtonText}>▶ Procesar</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={clearVideo}>
                <Text style={styles.clearButton}>✕</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      )}

      {/* Processing Status */}
      {videoStatus.status !== 'idle' && (
        <View style={styles.card}>
          <View style={styles.statusHeader}>
            <Text style={styles.cardTitle}>Estado del Procesamiento</Text>
            <View
              style={[
                styles.statusBadge,
                { backgroundColor: getStatusColor(videoStatus.status) },
              ]}
            >
              <Text style={styles.statusBadgeText}>{getStatusText(videoStatus.status)}</Text>
            </View>
          </View>

          {videoStatus.status === 'processing' && (
            <View style={styles.progressSection}>
              <View style={styles.progressInfo}>
                <Text style={styles.progressLabel}>Progreso</Text>
                <Text style={styles.progressPercent}>
                  {(videoStatus.progress_percent || 0).toFixed(1)}%
                </Text>
              </View>
              <View style={styles.progressBarContainer}>
                <View
                  style={[
                    styles.progressBar,
                    { width: `${videoStatus.progress_percent || 0}%` },
                  ]}
                />
              </View>
              <Text style={styles.progressSubtext}>
                {videoStatus.progress || 0} de {videoStatus.total_frames || 0} frames procesados
              </Text>
            </View>
          )}

          {videoStatus.status === 'completed' && (
            <View style={styles.completedSection}>
              <Text style={styles.completedText}>✓ Video procesado exitosamente</Text>
              <TouchableOpacity style={styles.openButton} onPress={openProcessedVideo}>
                <Text style={styles.openButtonText}>🎬 Ver Video Procesado</Text>
              </TouchableOpacity>
              <View style={styles.statsGrid}>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{videoStatus.total_frames || 0}</Text>
                  <Text style={styles.statLabel}>Frames Totales</Text>
                </View>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{videoStatus.progress || 0}</Text>
                  <Text style={styles.statLabel}>Procesados</Text>
                </View>
              </View>
              <TouchableOpacity style={styles.newVideoButton} onPress={clearVideo}>
                <Text style={styles.newVideoButtonText}>Procesar Otro Video</Text>
              </TouchableOpacity>
            </View>
          )}

          {(videoStatus.status === 'error' || videoStatus.status === 'failed') && (
            <View style={styles.errorSection}>
              <Text style={styles.errorText}>✕ Error en el procesamiento</Text>
              <Text style={styles.errorMessage}>
                {videoStatus.error || 'Error desconocido'}
              </Text>
              <TouchableOpacity style={styles.retryButton} onPress={clearVideo}>
                <Text style={styles.retryButtonText}>Reintentar con otro video</Text>
              </TouchableOpacity>
            </View>
          )}
        </View>
      )}

      {/* Empty State */}
      {!selectedVideoName && videoStatus.status === 'idle' && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateIcon}>🎥</Text>
          <Text style={styles.emptyStateText}>
            Selecciona un video para comenzar el procesamiento
          </Text>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 30,
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 8,
  },
  cardSubtitle: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: 16,
  },
  uploadButton: {
    backgroundColor: '#8b5cf6',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  uploadButtonIcon: {
    fontSize: 24,
    marginRight: 8,
  },
  uploadButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  videoHeader: {
    flexDirection: 'column',
    gap: 12,
  },
  videoInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  videoIcon: {
    fontSize: 32,
    marginRight: 12,
  },
  videoName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
  },
  videoSubtext: {
    fontSize: 13,
    color: '#6b7280',
  },
  actions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  processButton: {
    flex: 1,
    backgroundColor: '#8b5cf6',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  processButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  clearButton: {
    fontSize: 24,
    color: '#9ca3af',
  },
  statusHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  statusBadgeText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },
  progressSection: {
    marginTop: 8,
  },
  progressInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  progressLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  progressPercent: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  progressBarContainer: {
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#8b5cf6',
  },
  progressSubtext: {
    fontSize: 12,
    color: '#9ca3af',
    marginTop: 8,
    textAlign: 'center',
  },
  completedSection: {
    alignItems: 'center',
  },
  completedText: {
    fontSize: 16,
    color: '#22c55e',
    fontWeight: '600',
    marginBottom: 16,
  },
  openButton: {
    backgroundColor: '#22c55e',
    borderRadius: 8,
    padding: 14,
    width: '100%',
    alignItems: 'center',
    marginBottom: 16,
  },
  openButtonText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '600',
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  statItem: {
    flex: 1,
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#8b5cf6',
  },
  statLabel: {
    fontSize: 11,
    color: '#6b7280',
    marginTop: 4,
  },
  newVideoButton: {
    backgroundColor: '#e5e7eb',
    borderRadius: 8,
    padding: 12,
    width: '100%',
    alignItems: 'center',
  },
  newVideoButtonText: {
    color: '#1f2937',
    fontSize: 14,
    fontWeight: '600',
  },
  errorSection: {
    alignItems: 'center',
  },
  errorText: {
    fontSize: 16,
    color: '#ef4444',
    fontWeight: '600',
    marginBottom: 8,
  },
  errorMessage: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 16,
  },
  retryButton: {
    backgroundColor: '#ef4444',
    borderRadius: 8,
    padding: 12,
    width: '100%',
    alignItems: 'center',
  },
  retryButtonText: {
    color: '#ffffff',
    fontSize: 14,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  emptyStateIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  emptyStateText: {
    fontSize: 16,
    color: '#9ca3af',
    textAlign: 'center',
  },
});

export default VideoDetectionScreen;

