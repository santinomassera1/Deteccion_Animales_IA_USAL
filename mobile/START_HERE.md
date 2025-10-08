# 🚀 EMPIEZA AQUÍ - Versión Mobile

## ⚡ Inicio Rápido (10 minutos)

### **Paso 1: Instalar Expo Go en tu celular**

Descarga **Expo Go** (gratis):
- [📱 iOS - App Store](https://apps.apple.com/app/expo-go/id982107779)
- [🤖 Android - Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent)

---

### **Paso 2: Instalar dependencias**

```bash
cd mobile/
npm install
```

Espera a que termine (tarda 2-3 minutos).

---

### **Paso 3: Configurar la IP**

#### 3.1 Obtén tu IP:

```bash
ipconfig getifaddr en0
```

Ejemplo de salida: `192.168.1.45`

#### 3.2 Edita el archivo de API:

Abre `mobile/src/services/api.ts`

Busca la línea 5 y reemplaza con TU IP:

```typescript
const API_BASE_URL = 'http://192.168.1.45:5003'; // ← CAMBIA ESTO
```

---

### **Paso 4: Iniciar el backend**

En la raíz del proyecto (NO en mobile/):

```bash
python app.py
```

Espera a ver:
```
📱 Abre http://0.0.0.0:5003 en tu navegador
```

✅ **Déjalo corriendo** - no cierres esta terminal.

---

### **Paso 5: Iniciar Expo**

Abre una **nueva terminal** y ejecuta:

```bash
cd mobile/
npm start
```

Verás un **QR code** en la terminal.

---

### **Paso 6: Escanear QR**

1. Abre **Expo Go** en tu celular
2. Escanea el **QR code** de la terminal
3. Espera a que cargue (30-60 segundos la primera vez)
4. ¡Listo! La app está corriendo

---

## 🎉 ¡Funciona!

Deberías ver la app con 3 pestañas en la parte inferior:

- **📸 Cámara** - Detección en tiempo real
- **🖼️ Imágenes** - Analiza fotos de tu galería
- **🎥 Videos** - Procesa videos

---

## ✅ Verificación Rápida

1. En el header verifica que diga:
   - **"Conectado"** con punto verde ✅
   - **"Modelo AI activo"** ✅

2. Si ves eso, **¡todo está funcionando!**

---

## ❌ Si algo falla...

### No se conecta al servidor:

1. **Verifica que ambos estén en la misma WiFi**
   - Tu Mac y tu celular deben estar en la misma red

2. **Confirma la IP:**
   ```bash
   ipconfig getifaddr en0
   ```
   
3. **Revisa que coincida con `api.ts` línea 5**

4. **Reinicia el backend:**
   - Presiona `Ctrl+C` para detenerlo
   - Vuelve a ejecutar `python app.py`

### Expo no inicia:

```bash
# Limpia caché y reinstala
cd mobile/
rm -rf node_modules
npm install
npm start
```

---

## 📖 Más Información

- **`README.md`** - Documentación completa
- **`INSTRUCCIONES_RAPIDAS.md`** - Guía de 5 pasos
- **`MOBILE_SETUP_GUIDE.md`** (en raíz) - Resumen del proyecto

---

## 🎓 Primera Vez con Expo?

**Expo Go** es como un navegador para apps React Native:
- No necesitas compilar nada
- Los cambios se ven al instante
- Perfecto para desarrollo

---

## 🔥 Consejos Pro

1. **Shake tu celular** - Abre el menú de desarrollador
2. **Presiona "r" en terminal** - Recarga la app
3. **Presiona "d" en terminal** - Abre Chrome DevTools

---

## 🆘 Necesitas Ayuda?

Lee el archivo `README.md` completo que tiene soluciones para problemas comunes.

---

**¡Disfruta detectando animales con IA en tu celular! 🐾**

