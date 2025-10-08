# 📱 App Móvil - Detección de Animales USAL

Aplicación móvil React Native con Expo para detección de animales usando IA.

## 🚀 Inicio Rápido

### 1. Inicia el Backend (en otra terminal)

Desde el directorio raíz del proyecto:

```bash
cd ..
python3 app.py
```

El backend debe estar corriendo en el puerto 5003.

### 2. Inicia la App Móvil

Desde este directorio (`mobile/`):

```bash
./start.sh
```

Este script automáticamente:
- ✅ Detecta tu IP local
- ✅ Actualiza la configuración del backend
- ✅ Verifica las dependencias
- ✅ Inicia Expo con un código QR

### 3. Conecta tu Teléfono

1. **Descarga Expo Go**
   - [iOS (App Store)](https://apps.apple.com/app/expo-go/id982107779)
   - [Android (Play Store)](https://play.google.com/store/apps/details?id=host.exp.exponent)

2. **Asegúrate de estar en la misma WiFi** que tu Mac

3. **Escanea el código QR** que aparece en la terminal

## 📋 Requisitos

- Node.js v18.14.0 o superior
- npm o yarn
- Expo Go en tu dispositivo móvil
- Backend Python corriendo (ver directorio raíz)

## 🛠️ Instalación Manual

Si prefieres instalar manualmente:

```bash
# Instalar dependencias
npm install --legacy-peer-deps

# Iniciar Expo
npx expo start
```

## 🎨 Características

- **📸 Detección en Imágenes**: Sube o toma fotos para detectar animales
- **🎥 Detección en Videos**: Procesa videos grabados
- **📹 Cámara en Tiempo Real**: Detección continua usando la cámara
- **📊 Historial**: Revisa todas las detecciones previas
- **🔔 Notificaciones**: Alertas en tiempo real

## 🏗️ Estructura del Proyecto

```
mobile/
├── src/
│   ├── components/          # Componentes reutilizables
│   │   ├── Header.tsx
│   │   ├── NotificationCenter.tsx
│   │   └── LoadingSpinner.tsx
│   ├── screens/            # Pantallas principales
│   │   ├── ImageDetectionScreen.tsx
│   │   ├── VideoDetectionScreen.tsx
│   │   └── CameraDetectionScreen.tsx
│   ├── services/           # Servicios y API
│   │   └── api.ts         # Cliente API para backend
│   ├── store/             # Estado global (Zustand)
│   │   └── useAppStore.ts
│   └── utils/             # Utilidades
│       └── colors.ts
├── app.json               # Configuración de Expo
├── package.json           # Dependencias
├── tsconfig.json          # Configuración TypeScript
├── metro.config.js        # Configuración Metro bundler
└── start.sh              # Script de inicio

```

## 🔧 Configuración

### Cambiar la IP del Backend

Edita `src/services/api.ts`:

```typescript
const API_BASE_URL = 'http://TU_IP_AQUI:5003';
```

O usa el script `start.sh` que lo hace automáticamente.

### Configuración de la Cámara

Los permisos de cámara se configuran en `app.json`:

```json
"plugins": [
  [
    "expo-camera",
    {
      "cameraPermission": "Permitir acceso a cámara..."
    }
  ]
]
```

## 📱 Plataformas Soportadas

- ✅ iOS (iPhone/iPad)
- ✅ Android
- ⚠️  Web (limitado, algunas funciones no disponibles)

## 🐛 Solución de Problemas

### Error: "Cannot connect to backend"

1. Verifica que el backend esté corriendo:
   ```bash
   curl http://192.168.0.6:5003/api/model-status
   ```

2. Confirma que ambos dispositivos estén en la misma WiFi

3. Revisa la IP en `src/services/api.ts`

### Error: "ConfigError: package.json does not exist"

Asegúrate de ejecutar los comandos desde la carpeta `mobile/`:

```bash
cd mobile
./start.sh
```

### Error: "babel-preset-expo not found"

Reinstala las dependencias:

```bash
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps
```

### Error: "Camera permission denied"

1. Ve a la configuración de tu teléfono
2. Busca Expo Go
3. Habilita los permisos de cámara y galería

## 📚 Stack Tecnológico

- **Framework**: React Native + Expo SDK 54
- **Lenguaje**: TypeScript
- **Navegación**: React Navigation v6
- **Estado**: Zustand
- **HTTP**: Axios
- **Cámara**: expo-camera
- **Galería**: expo-image-picker

## 🔗 Enlaces Útiles

- [Documentación de Expo](https://docs.expo.dev/)
- [React Native Docs](https://reactnative.dev/docs/getting-started)
- [Expo Go App](https://expo.dev/client)

## 📝 Notas

- La app requiere que el backend Python esté corriendo
- Asegúrate de estar conectado a WiFi (no datos móviles)
- La primera compilación puede tardar algunos minutos
- Para builds de producción, consulta la documentación de Expo EAS Build

## 🆘 Soporte

Si encuentras problemas:

1. Revisa los logs: La terminal muestra errores detallados
2. Consulta `INSTRUCCIONES_RAPIDAS.md` para guías rápidas
3. Verifica que todas las dependencias estén instaladas
4. Asegúrate de tener la versión correcta de Node.js (`node --version`)

---

Desarrollado para Universidad de Salamanca (USAL)

