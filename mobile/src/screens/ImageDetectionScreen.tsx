import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useAppStore } from '../store/useAppStore';
import { apiService } from '../services/api';
import { getAnimalColor, getAnimalName } from '../utils/colors';

const ImageDetectionScreen: React.FC = () => {
  const { currentDetection, setCurrentDetection, isLoading, setIsLoading, addNotification } = useAppStore();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const pickImage = async () => {
    try {
      // Request permissions
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        addNotification({
          type: 'error',
          title: 'Permiso denegado',
          message: 'Necesitamos permiso para acceder a tu galería.',
        });
        return;
      }

      // Pick image
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: false,
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        setSelectedImage(asset.uri);
        await processImage(asset.uri, asset.fileName || 'image.jpg');
      }
    } catch (error) {
      console.error('Error picking image:', error);
      addNotification({
        type: 'error',
        title: 'Error',
        message: 'No se pudo seleccionar la imagen.',
      });
    }
  };

  const processImage = async (uri: string, filename: string) => {
    try {
      setIsLoading(true);
      setCurrentDetection(null);

      // Upload image
      const uploadResult = await apiService.uploadFile({
        uri,
        name: filename,
        type: 'image/jpeg',
      });

      addNotification({
        type: 'info',
        title: 'Procesando',
        message: 'Analizando la imagen con IA...',
      });

      // Detect objects
      const detectionResult = await apiService.detectImage(uploadResult.filename);
      setCurrentDetection(detectionResult);

      addNotification({
        type: 'success',
        title: 'Detección completada',
        message: `Se detectaron ${detectionResult.total_detections} animales`,
      });
    } catch (error: any) {
      console.error('Error processing image:', error);
      addNotification({
        type: 'error',
        title: 'Error al procesar',
        message: error.message || 'No se pudo procesar la imagen.',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const clearDetection = () => {
    setCurrentDetection(null);
    setSelectedImage(null);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Upload Section */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Detección en Imágenes</Text>
        <Text style={styles.cardSubtitle}>
          Selecciona una imagen de tu galería para detectar animales
        </Text>

        <TouchableOpacity
          style={styles.uploadButton}
          onPress={pickImage}
          disabled={isLoading}
        >
          <Text style={styles.uploadButtonIcon}>📷</Text>
          <Text style={styles.uploadButtonText}>
            {isLoading ? 'Procesando...' : 'Seleccionar Imagen'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Loading */}
      {isLoading && (
        <View style={styles.card}>
          <ActivityIndicator size="large" color="#22c55e" />
          <Text style={styles.loadingText}>Analizando imagen...</Text>
        </View>
      )}

      {/* Results */}
      {currentDetection && !isLoading && (
        <>
          {/* Summary */}
          <View style={styles.card}>
            <View style={styles.resultHeader}>
              <Text style={styles.cardTitle}>Resultados</Text>
              <TouchableOpacity onPress={clearDetection}>
                <Text style={styles.clearButton}>✕</Text>
              </TouchableOpacity>
            </View>
            
            <View style={styles.detectionBadge}>
              <Text style={styles.detectionBadgeText}>
                ✓ {currentDetection.total_detections} animales detectados
              </Text>
            </View>

            {/* Animal counts */}
            <View style={styles.animalGrid}>
              {['cat', 'chicken', 'cow', 'dog', 'horse'].map((animal) => {
                const count = currentDetection.detections.filter((d) => d.class === animal).length;
                return (
                  <View key={animal} style={styles.animalCard}>
                    <View
                      style={[
                        styles.animalCircle,
                        { backgroundColor: getAnimalColor(animal) },
                      ]}
                    >
                      <Text style={styles.animalCount}>{count}</Text>
                    </View>
                    <Text style={styles.animalName}>{getAnimalName(animal)}</Text>
                  </View>
                );
              })}
            </View>
          </View>

          {/* Processed Image */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Imagen con Detecciones</Text>
            <Image
              source={{ uri: apiService.getDownloadUrl(currentDetection.processed_filename) }}
              style={styles.resultImage}
              resizeMode="contain"
            />
          </View>

          {/* Detection Details */}
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Detalles de las Detecciones</Text>
            {currentDetection.detections.map((detection, index) => (
              <View key={index} style={styles.detectionItem}>
                <View style={styles.detectionLeft}>
                  <View
                    style={[
                      styles.detectionDot,
                      { backgroundColor: getAnimalColor(detection.class) },
                    ]}
                  />
                  <Text style={styles.detectionClass}>
                    {getAnimalName(detection.class)}
                  </Text>
                </View>
                <Text style={styles.detectionConfidence}>
                  {Math.round(detection.confidence * 100)}% confianza
                </Text>
              </View>
            ))}
          </View>
        </>
      )}

      {/* Empty State */}
      {!currentDetection && !isLoading && !selectedImage && (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateIcon}>🖼️</Text>
          <Text style={styles.emptyStateText}>
            Selecciona una imagen para comenzar la detección
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
    backgroundColor: '#22c55e',
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
  loadingText: {
    textAlign: 'center',
    marginTop: 12,
    color: '#6b7280',
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  clearButton: {
    fontSize: 24,
    color: '#9ca3af',
  },
  detectionBadge: {
    backgroundColor: '#dcfce7',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    alignSelf: 'flex-start',
    marginBottom: 16,
  },
  detectionBadgeText: {
    color: '#16a34a',
    fontWeight: '600',
    fontSize: 14,
  },
  animalGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    gap: 12,
  },
  animalCard: {
    alignItems: 'center',
    width: 60,
  },
  animalCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 6,
  },
  animalCount: {
    color: '#ffffff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  animalName: {
    fontSize: 11,
    color: '#6b7280',
    textAlign: 'center',
  },
  resultImage: {
    width: '100%',
    height: 300,
    borderRadius: 8,
  },
  detectionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
  },
  detectionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  detectionDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 10,
  },
  detectionClass: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1f2937',
  },
  detectionConfidence: {
    fontSize: 13,
    color: '#6b7280',
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

export default ImageDetectionScreen;

