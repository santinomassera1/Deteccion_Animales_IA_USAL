import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import { useAppStore } from '../store/useAppStore';

const Header: React.FC = () => {
  const { modelLoaded, isConnected } = useAppStore();

  return (
    <View style={styles.container}>
      <View style={styles.content}>
        <View style={styles.titleSection}>
          <Text style={styles.title}>🐾 Detección de Animales</Text>
          <Text style={styles.subtitle}>Universidad del Salvador - IA Veterinaria</Text>
        </View>
        
        <View style={styles.statusSection}>
          {/* Connection Status */}
          <View style={styles.statusBadge}>
            <View style={[styles.statusDot, isConnected ? styles.connectedDot : styles.disconnectedDot]} />
            <Text style={styles.statusText}>
              {isConnected ? 'Conectado' : 'Desconectado'}
            </Text>
          </View>
          
          {/* Model Status */}
          {modelLoaded && (
            <View style={[styles.statusBadge, styles.modelBadge]}>
              <Text style={styles.modelText}>✓ Modelo AI activo</Text>
            </View>
          )}
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#ffffff',
    paddingTop: 50,
    paddingBottom: 15,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3,
  },
  content: {
    flexDirection: 'column',
  },
  titleSection: {
    marginBottom: 10,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1f2937',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 12,
    color: '#6b7280',
  },
  statusSection: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f3f4f6',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  connectedDot: {
    backgroundColor: '#22c55e',
  },
  disconnectedDot: {
    backgroundColor: '#ef4444',
  },
  statusText: {
    fontSize: 11,
    color: '#4b5563',
    fontWeight: '500',
  },
  modelBadge: {
    backgroundColor: '#dcfce7',
  },
  modelText: {
    fontSize: 11,
    color: '#16a34a',
    fontWeight: '600',
  },
});

export default Header;

