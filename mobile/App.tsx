import React, { useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, Text, StyleSheet } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useAppStore } from './src/store/useAppStore';
import { apiService } from './src/services/api';

// Screens
import ImageDetectionScreen from './src/screens/ImageDetectionScreen';
import VideoDetectionScreen from './src/screens/VideoDetectionScreen';
import CameraDetectionScreen from './src/screens/CameraDetectionScreen';

// Components
import Header from './src/components/Header';
import NotificationCenter from './src/components/NotificationCenter';

const Tab = createBottomTabNavigator();

// Tab Icons (using emojis for simplicity - you can replace with actual icons)
const TabIcon = ({ icon, focused }: { icon: string; focused: boolean }) => (
  <Text style={{ fontSize: 24, opacity: focused ? 1 : 0.5 }}>{icon}</Text>
);

const App: React.FC = () => {
  const { setModelLoaded, setIsConnected, addNotification } = useAppStore();

  // Check backend connection on mount
  useEffect(() => {
    let mounted = true;

    const checkConnection = async () => {
      try {
        console.log('🔍 Verificando conexión con backend...');
        const status = await apiService.getModelStatus();
        
        if (mounted) {
          setModelLoaded(status.model_loaded);
          setIsConnected(true);
          
          if (status.model_loaded) {
            addNotification({
              type: 'success',
              title: '¡Bienvenido!',
              message: 'Sistema de IA conectado y listo para detectar animales.',
            });
          } else {
            addNotification({
              type: 'warning',
              title: 'Modelo cargando',
              message: 'El modelo de IA está inicializando...',
            });
          }
        }
      } catch (error: any) {
        console.error('❌ Error conectando con backend:', error);
        if (mounted) {
          setIsConnected(false);
          addNotification({
            type: 'error',
            title: 'Error de conexión',
            message: `No se pudo conectar al servidor.\n\nVerifica:\n1. Backend corriendo (python app.py)\n2. Misma WiFi\n3. IP correcta en api.ts: ${apiService.getBaseUrl()}`,
          });
        }
      }
    };

    checkConnection();

    // Check connection every 30 seconds
    const interval = setInterval(() => {
      if (mounted) {
        apiService.getModelStatus()
          .then((status) => {
            setModelLoaded(status.model_loaded);
            setIsConnected(true);
          })
          .catch(() => {
            setIsConnected(false);
          });
      }
    }, 30000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <>
      <StatusBar style="dark" />
      <View style={styles.container}>
        <Header />
        <NavigationContainer>
          <Tab.Navigator
            screenOptions={{
              headerShown: false,
              tabBarActiveTintColor: '#22c55e',
              tabBarInactiveTintColor: '#9ca3af',
              tabBarStyle: {
                backgroundColor: '#ffffff',
                borderTopWidth: 1,
                borderTopColor: '#e5e7eb',
                paddingBottom: 8,
                paddingTop: 8,
                height: 70,
              },
              tabBarLabelStyle: {
                fontSize: 11,
                fontWeight: '600',
              },
            }}
          >
            <Tab.Screen
              name="Camera"
              component={CameraDetectionScreen}
              options={{
                tabBarLabel: 'Cámara',
                tabBarIcon: ({ focused }) => <TabIcon icon="📸" focused={focused} />,
              }}
            />
            <Tab.Screen
              name="Image"
              component={ImageDetectionScreen}
              options={{
                tabBarLabel: 'Imágenes',
                tabBarIcon: ({ focused }) => <TabIcon icon="🖼️" focused={focused} />,
              }}
            />
            <Tab.Screen
              name="Video"
              component={VideoDetectionScreen}
              options={{
                tabBarLabel: 'Videos',
                tabBarIcon: ({ focused }) => <TabIcon icon="🎥" focused={focused} />,
              }}
            />
          </Tab.Navigator>
        </NavigationContainer>
        <NotificationCenter />
      </View>
    </>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
});

export default App;

