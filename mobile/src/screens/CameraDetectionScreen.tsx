import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

type CameraType = 'back' | 'front';
import { useAppStore } from '../store/useAppStore';
import { apiService } from '../services/api';
import { getAnimalColor, getAnimalName } from '../utils/colors';

const CameraDetectionScreen: React.FC = () => {
  const {
    cameraActive,
    setCameraActive,
    cameraDetections,
    setCameraDetections,
    isLoading,
    setIsLoading,
    addNotification,
  } = useAppStore();

  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('back');
  const [isCapturing, setIsCapturing] = useState(false);
  const cameraRef = useRef<any>(null);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      if (!permission?.granted) {
        const result = await requestPermission();
        if (!result.granted) {
          addNotification({
            type: 'error',
            title: 'Permiso denegado',
            message: 'Necesitamos acceso a la cámara para detectar animales.',
          });
          return;
        }
      }

      setCameraActive(true);
      addNotification({
        type: 'success',
        title: 'Cámara iniciada',
        message: 'La detección en tiempo real está activa.',
      });

      // Start auto-capture every 3 seconds
      startAutoCapture();
    } catch (error) {
      console.error('Error starting camera:', error);
      addNotification({
        type: 'error',
        title: 'Error',
        message: 'No se pudo iniciar la cámara.',
      });
    }
  };

  const stopCamera = () => {
    setCameraActive(false);
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    setCameraDetections([]);
    addNotification({
      type: 'info',
      title: 'Cámara detenida',
      message: 'La detección se ha detenido.',
    });
  };

  const startAutoCapture = () => {
    // Capture frame every 3 seconds
    const interval = setInterval(() => {
      if (cameraRef.current && cameraActive) {
        captureAndDetect();
      }
    }, 3000);
    captureIntervalRef.current = interval;
  };

  const captureAndDetect = async () => {
    if (isCapturing || !cameraRef.current) return;

    try {
      setIsCapturing(true);

      // Take photo
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.6,
        base64: true,
        skipProcessing: true,
      });

      if (photo.base64) {
        // Send to backend for detection
        const imageData = `data:image/jpeg;base64,${photo.base64}`;
        const result = await apiService.detectWebcam(imageData);

        if (result.detections && result.detections.length > 0) {
          setCameraDetections(result.detections);
        }
      }
    } catch (error) {
      console.error('Error capturing and detecting:', error);
      // Don't show notification for every error to avoid spam
    } finally {
      setIsCapturing(false);
    }
  };

  const manualCapture = async () => {
    if (!cameraRef.current) return;

    try {
      setIsLoading(true);
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
        base64: true,
      });

      if (photo.base64) {
        addNotification({
          type: 'info',
          title: 'Analizando',
          message: 'Detectando animales en la imagen...',
        });

        const imageData = `data:image/jpeg;base64,${photo.base64}`;
        const result = await apiService.detectWebcam(imageData);

        setCameraDetections(result.detections);

        if (result.total_detections > 0) {
          addNotification({
            type: 'success',
            title: 'Detección completada',
            message: `Se detectaron ${result.total_detections} animales`,
          });
        } else {
          addNotification({
            type: 'info',
            title: 'Sin detecciones',
            message: 'No se detectaron animales en la imagen.',
          });
        }
      }
    } catch (error: any) {
      console.error('Error in manual capture:', error);
      addNotification({
        type: 'error',
        title: 'Error',
        message: error.message || 'No se pudo procesar la imagen.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const toggleCameraFacing = () => {
    setFacing((current) => (current === 'back' ? 'front' : 'back'));
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionContainer}>
          <ActivityIndicator size="large" color="#22c55e" />
          <Text style={styles.permissionText}>Cargando cámara...</Text>
        </View>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionContainer}>
          <Text style={styles.permissionIcon}>📷</Text>
          <Text style={styles.permissionTitle}>Permiso de Cámara Requerido</Text>
          <Text style={styles.permissionText}>
            Necesitamos acceso a tu cámara para detectar animales en tiempo real.
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Permitir Acceso</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {!cameraActive ? (
        <ScrollView contentContainerStyle={styles.contentContainer}>
          {/* Start Camera */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Detección en Tiempo Real</Text>
            <Text style={styles.cardSubtitle}>
              Usa tu cámara para detectar animales en vivo con inteligencia artificial
            </Text>

            <TouchableOpacity style={styles.startButton} onPress={startCamera}>
              <Text style={styles.startButtonIcon}>📸</Text>
              <Text style={styles.startButtonText}>Iniciar Cámara</Text>
            </TouchableOpacity>
          </View>

          {/* Instructions */}
          <View style={styles.card}>
            <Text style={styles.instructionsTitle}>Instrucciones</Text>
            <View style={styles.instructionItem}>
              <Text style={styles.instructionNumber}>1</Text>
              <Text style={styles.instructionText}>
                Apunta la cámara hacia el animal que quieres detectar
              </Text>
            </View>
            <View style={styles.instructionItem}>
              <Text style={styles.instructionNumber}>2</Text>
              <Text style={styles.instructionText}>
                La detección se realizará automáticamente cada 3 segundos
              </Text>
            </View>
            <View style={styles.instructionItem}>
              <Text style={styles.instructionNumber}>3</Text>
              <Text style={styles.instructionText}>
                También puedes capturar manualmente presionando el botón
              </Text>
            </View>
          </View>
        </ScrollView>
      ) : (
        <View style={styles.cameraContainer}>
          {/* Camera View */}
          <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
            {/* Camera Overlay */}
            <View style={styles.cameraOverlay}>
              {/* Top Bar */}
              <View style={styles.topBar}>
                <View style={styles.liveIndicator}>
                  <View style={styles.liveDot} />
                  <Text style={styles.liveText}>EN VIVO</Text>
                </View>
                {isCapturing && (
                  <View style={styles.analyzingBadge}>
                    <Text style={styles.analyzingText}>Analizando...</Text>
                  </View>
                )}
              </View>

              {/* Bottom Controls */}
              <View style={styles.bottomBar}>
                <TouchableOpacity style={styles.controlButton} onPress={toggleCameraFacing}>
                  <Text style={styles.controlButtonText}>🔄</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.captureButton}
                  onPress={manualCapture}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <ActivityIndicator color="#ffffff" />
                  ) : (
                    <View style={styles.captureButtonInner} />
                  )}
                </TouchableOpacity>

                <TouchableOpacity style={styles.controlButton} onPress={stopCamera}>
                  <Text style={styles.controlButtonText}>✕</Text>
                </TouchableOpacity>
              </View>
            </View>
          </CameraView>

          {/* Detection Results */}
          {cameraDetections.length > 0 && (
            <View style={styles.resultsContainer}>
              <Text style={styles.resultsTitle}>
                Detecciones Recientes: {cameraDetections.length}
              </Text>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.detectionsScroll}
              >
                {cameraDetections.map((detection, index) => (
                  <View
                    key={index}
                    style={[
                      styles.detectionCard,
                      { borderLeftColor: getAnimalColor(detection.class) },
                    ]}
                  >
                    <Text style={styles.detectionAnimal}>
                      {getAnimalName(detection.class)}
                    </Text>
                    <Text style={styles.detectionConfidence}>
                      {Math.round(detection.confidence * 100)}%
                    </Text>
                  </View>
                ))}
              </ScrollView>

              {/* Animal Summary */}
              <View style={styles.summaryContainer}>
                <Text style={styles.summaryTitle}>Resumen</Text>
                <View style={styles.summaryGrid}>
                  {['cat', 'chicken', 'cow', 'dog', 'horse'].map((animal) => {
                    const count = cameraDetections.filter((d) => d.class === animal).length;
                    if (count === 0) return null;
                    return (
                      <View key={animal} style={styles.summaryItem}>
                        <View
                          style={[
                            styles.summaryDot,
                            { backgroundColor: getAnimalColor(animal) },
                          ]}
                        />
                        <Text style={styles.summaryText}>
                          {getAnimalName(animal)}: {count}
                        </Text>
                      </View>
                    );
                  })}
                </View>
              </View>
            </View>
          )}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 30,
    backgroundColor: '#f9fafb',
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
  startButton: {
    backgroundColor: '#3b82f6',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  startButtonIcon: {
    fontSize: 24,
    marginRight: 8,
  },
  startButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  instructionsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  instructionItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  instructionNumber: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#3b82f6',
    color: '#ffffff',
    textAlign: 'center',
    lineHeight: 24,
    fontWeight: 'bold',
    marginRight: 12,
  },
  instructionText: {
    flex: 1,
    fontSize: 14,
    color: '#6b7280',
    lineHeight: 20,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
    backgroundColor: '#f9fafb',
  },
  permissionIcon: {
    fontSize: 64,
    marginBottom: 16,
  },
  permissionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 8,
    textAlign: 'center',
  },
  permissionText: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 24,
  },
  permissionButton: {
    backgroundColor: '#3b82f6',
    borderRadius: 12,
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  permissionButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  cameraContainer: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'space-between',
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    paddingTop: 50,
  },
  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(239, 68, 68, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#ffffff',
    marginRight: 6,
  },
  liveText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  analyzingBadge: {
    backgroundColor: 'rgba(59, 130, 246, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  analyzingText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '600',
  },
  bottomBar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 40,
  },
  controlButton: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  controlButtonText: {
    fontSize: 24,
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#ffffff',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#3b82f6',
  },
  resultsContainer: {
    backgroundColor: '#ffffff',
    padding: 16,
    maxHeight: 250,
  },
  resultsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 12,
  },
  detectionsScroll: {
    paddingRight: 16,
  },
  detectionCard: {
    backgroundColor: '#f9fafb',
    borderRadius: 8,
    padding: 12,
    marginRight: 8,
    borderLeftWidth: 4,
    minWidth: 100,
  },
  detectionAnimal: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 4,
  },
  detectionConfidence: {
    fontSize: 12,
    color: '#6b7280',
  },
  summaryContainer: {
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  summaryTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
    marginBottom: 8,
  },
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  summaryItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
  },
  summaryDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  summaryText: {
    fontSize: 12,
    color: '#4b5563',
  },
});

export default CameraDetectionScreen;

