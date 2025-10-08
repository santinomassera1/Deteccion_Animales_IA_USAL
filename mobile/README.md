# 📱 Detección de Animales - Versión Mobile (React Native + Expo)

## 🎯 Descripción

Aplicación móvil para **detección de animales en tiempo real** utilizando inteligencia artificial. Compatible con Android e iOS, funciona con **Expo Go** para desarrollo rápido.

Esta app se conecta al mismo backend del sistema web, utilizando el **mismo modelo YOLOv8** para detección de:
- 🐱 Gatos
- 🐔 Gallinas  
- 🐄 Vacas
- 🐕 Perros
- 🐎 Caballos

---

## ✨ Características

### 📸 **Cámara en Tiempo Real**
- Detección automática cada 3 segundos
- Captura manual para mayor precisión
- Visualización en vivo de resultados
- Resumen de animales detectados

### 🖼️ **Detección en Imágenes**
- Selecciona fotos de tu galería
- Procesamiento con IA en tiempo real
- Resultados detallados con confianza
- Estadísticas por tipo de animal

### 🎥 **Procesamiento de Videos**
- Sube videos desde tu dispositivo
- Análisis frame por frame
- Barra de progreso en tiempo real
- Video procesado descargable

---

## 🚀 Instalación y Configuración

### **Requisitos Previos**

1. **Node.js** 18 o superior
2. **Expo Go** instalado en tu celular:
   - [iOS - App Store](https://apps.apple.com/app/expo-go/id982107779)
   - [Android - Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent)
3. **Backend Flask** corriendo en tu Mac (puerto 5003)
4. **Misma red WiFi** - Tu Mac y celular deben estar conectados a la misma red

### **Paso 1: Instalar Dependencias**

```bash
cd mobile/
npm install
```

### **Paso 2: Configurar la IP del Backend**

1. **Obtén la IP local de tu Mac:**

```bash
# En tu Mac, ejecuta:
ifconfig | grep "inet " | grep -v 127.0.0.1

# O más fácil:
ipconfig getifaddr en0  # WiFi
```

Ejemplo de salida: `192.168.1.45`

2. **Actualiza la IP en el archivo de API:**

Edita `mobile/src/services/api.ts`:

```typescript
// Línea 5 - Cambia esta IP por la tuya
const API_BASE_URL = 'http://192.168.1.45:5003'; // ← TU IP AQUÍ
```

### **Paso 3: Iniciar el Backend**

En otra terminal, desde la raíz del proyecto:

```bash
python app.py
```

Verifica que veas:
```
📱 Abre http://0.0.0.0:5003 en tu navegador
```

### **Paso 4: Iniciar la App Mobile**

```bash
cd mobile/
npm start
```

Verás un QR code en la terminal.

### **Paso 5: Abrir en tu Celular**

1. Abre **Expo Go** en tu celular
2. Escanea el **QR code** que apareció en la terminal
3. ¡Espera a que cargue la app!

---

## 🎮 Uso de la Aplicación

### **Pantalla 1: Cámara en Tiempo Real** 📸

1. Toca el botón **"Iniciar Cámara"**
2. Permite el acceso a la cámara cuando se solicite
3. Apunta la cámara hacia un animal
4. La detección se realiza automáticamente cada 3 segundos
5. También puedes capturar manualmente tocando el botón central
6. Los resultados aparecen en la parte inferior

**Controles:**
- **Botón central grande**: Captura manual
- **Botón 🔄**: Cambiar entre cámara frontal/trasera
- **Botón ✕**: Detener cámara

### **Pantalla 2: Detección en Imágenes** 🖼️

1. Toca **"Seleccionar Imagen"**
2. Elige una foto de tu galería
3. Espera el análisis (2-5 segundos)
4. Revisa los resultados:
   - Imagen con detecciones marcadas
   - Cantidad de cada animal
   - Porcentaje de confianza

### **Pantalla 3: Procesamiento de Videos** 🎥

1. Toca **"Seleccionar Video"**
2. Elige un video de tu dispositivo (máx. 150MB)
3. Toca **"▶ Procesar"**
4. Observa el progreso en tiempo real
5. Cuando termine:
   - Toca **"🎬 Ver Video Procesado"** para abrirlo
   - O descárgalo desde tu navegador

---

## 🔧 Solución de Problemas

### **Error: "No se puede conectar al servidor"**

**Solución:**

1. Verifica que el backend esté corriendo:
   ```bash
   python app.py
   ```

2. Confirma que tu Mac y celular estén en la **misma WiFi**

3. Verifica la IP en `mobile/src/services/api.ts`:
   ```bash
   # Obtén tu IP actual
   ipconfig getifaddr en0
   ```

4. Comprueba el firewall de tu Mac:
   - Ve a **Preferencias del Sistema** → **Seguridad y Privacidad** → **Firewall**
   - Asegúrate de que Python tenga permitido aceptar conexiones entrantes

### **Error: "Permiso de Cámara Denegado"**

**Solución:**

1. **iOS**: Ve a Ajustes → Expo Go → Permisos → Activa Cámara
2. **Android**: Ve a Ajustes → Apps → Expo Go → Permisos → Activa Cámara

### **Error: "La app no carga / QR no funciona"**

**Solución:**

1. Asegúrate de tener la última versión de Expo Go
2. Prueba presionar **"r"** en la terminal para recargar
3. Cierra y vuelve a abrir Expo Go
4. Intenta con el modo **"Tunnel"**:
   ```bash
   npx expo start --tunnel
   ```

### **Detección lenta o sin resultados**

**Solución:**

1. Verifica que el modelo esté cargado en el backend:
   - Abre http://TU-IP:5003/api/model-status
   - Debe decir `"model_loaded": true`

2. Asegúrate de tener buena iluminación en la cámara

3. Acércate más al animal si está muy lejos

4. Intenta con la captura manual en lugar de automática

---

## 📁 Estructura del Proyecto

```
mobile/
├── App.tsx                          # Punto de entrada principal
├── index.js                         # Registro de la app
├── package.json                     # Dependencias
├── app.json                         # Configuración de Expo
├── tsconfig.json                    # Configuración TypeScript
└── src/
    ├── screens/
    │   ├── CameraDetectionScreen.tsx    # Cámara en tiempo real
    │   ├── ImageDetectionScreen.tsx     # Detección de imágenes
    │   └── VideoDetectionScreen.tsx     # Procesamiento de videos
    ├── components/
    │   ├── Header.tsx                   # Encabezado con estado
    │   ├── NotificationCenter.tsx       # Sistema de notificaciones
    │   └── LoadingSpinner.tsx           # Indicador de carga
    ├── services/
    │   └── api.ts                       # Conexión con backend
    ├── store/
    │   └── useAppStore.ts               # Estado global (Zustand)
    └── utils/
        └── colors.ts                    # Colores de animales
```

---

## 🎨 Diseño y UI

La aplicación mantiene el **mismo estilo visual** que la versión web:

- **Colores de animales:**
  - Gato: Magenta (`#e879f9`)
  - Gallina: Naranja (`#fb923c`)
  - Vaca: Verde (`#22c55e`)
  - Perro: Azul (`#3b82f6`)
  - Caballo: Amarillo (`#facc15`)

- **Tema:** Claro y moderno
- **Tipografía:** System fonts nativos
- **Animaciones:** Suaves y naturales

---

## 🔬 Arquitectura Técnica

### **Stack Tecnológico:**

- **Framework**: React Native + Expo SDK 51
- **Lenguaje**: TypeScript
- **Navegación**: React Navigation (Bottom Tabs)
- **Estado**: Zustand (gestión de estado global)
- **HTTP**: Axios
- **Cámara**: expo-camera
- **Galería**: expo-image-picker
- **Videos**: expo-document-picker

### **Flujo de Datos:**

```
Mobile App (Expo)
      ↓
   Captura
   (Cámara/Galería)
      ↓
   HTTP Request
   (Axios)
      ↓
Backend Flask
 (localhost:5003)
      ↓
  Modelo YOLOv8
  (Detección IA)
      ↓
   JSON Response
      ↓
   Mobile App
   (Mostrar resultados)
```

### **El modelo NO corre en el celular:**
- ✅ Todo el procesamiento de IA es en el backend
- ✅ La app solo envía imágenes/videos
- ✅ Recibe resultados JSON
- ✅ Usa el **mismo modelo** que la versión web

---

## 🚢 Despliegue

### **Desarrollo (Expo Go)**

Ya configurado - solo escanea el QR.

### **Producción (Build Nativa)**

Si quieres crear una APK/IPA:

```bash
# Para Android
eas build --platform android

# Para iOS (necesitas cuenta Apple Developer)
eas build --platform ios
```

**Nota:** Para producción, necesitarás:
1. Cuenta de Expo (gratis)
2. Configurar `eas.json`
3. Backend en un servidor con IP/dominio público

---

## 📊 Rendimiento

### **Tiempos de Respuesta:**

- **Imagen (galería)**: 2-4 segundos
- **Cámara (captura manual)**: 2-3 segundos
- **Video corto (<30s)**: 1-2 minutos
- **Video largo (1-2min)**: 3-5 minutos

### **Optimizaciones:**

- ✅ Compresión de imágenes (80% quality)
- ✅ Captura automática cada 3 segundos (no sobrecargar)
- ✅ Polling inteligente para videos
- ✅ Reintentos automáticos en errores de red

---

## 🤝 Compatibilidad

### **Dispositivos Soportados:**

- ✅ **Android**: 5.0 (Lollipop) o superior
- ✅ **iOS**: 13.0 o superior
- ✅ **Tablets**: Android e iPad

### **Permisos Requeridos:**

- 📷 **Cámara**: Para detección en tiempo real
- 🖼️ **Galería**: Para seleccionar imágenes
- 📁 **Archivos**: Para seleccionar videos

---

## 📝 Notas Importantes

### ⚠️ **Limitaciones de Expo Go:**

1. **Red Local:** Solo funciona en tu WiFi local (no en internet)
2. **Tamaño:** Videos limitados a 150MB
3. **Velocidad:** Depende de tu red WiFi

### ✅ **Para Producción:**

Si necesitas que funcione fuera de tu red:
1. Despliega el backend en un servidor (AWS, Heroku, etc.)
2. Cambia `API_BASE_URL` a la URL del servidor
3. Crea un build nativo con `eas build`

---

## 🆘 Soporte

### **Logs útiles:**

```bash
# Ver logs del backend
python app.py  # Los logs aparecen aquí

# Ver logs de Expo
# Los logs aparecen automáticamente en la terminal
```

### **Archivos importantes:**

- `mobile/src/services/api.ts` - Configuración de conexión
- `app.py` - Backend Flask (puerto 5003)
- `mobile/App.tsx` - Punto de entrada

---

## 🎓 Desarrollado por

**Universidad del Salvador (USAL)**  
Facultad de Ciencias Veterinarias  
Proyecto de Inteligencia Artificial Aplicada  

**Autor:** Santino Massera  
**Año:** 2025

---

## 📄 Licencia

Uso Académico - USAL

---

## 🚀 ¡Listo para usar!

1. Inicia el backend: `python app.py`
2. Inicia Expo: `cd mobile/ && npm start`
3. Escanea el QR con Expo Go
4. ¡Comienza a detectar animales! 🐾

