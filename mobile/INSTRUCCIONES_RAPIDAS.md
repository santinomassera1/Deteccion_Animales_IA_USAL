# ⚡ Guía Rápida - 5 Pasos

## 🎯 Configuración Inicial (Solo la primera vez)

### **1. Instala las dependencias**
```bash
cd mobile/
npm install
```

### **2. Obtén tu IP local**
```bash
ipconfig getifaddr en0
# Ejemplo: 192.168.1.45
```

### **3. Actualiza la IP en el código**

Edita `mobile/src/services/api.ts` línea 5:

```typescript
const API_BASE_URL = 'http://192.168.1.45:5003'; // ← TU IP AQUÍ
```

---

## 🚀 Uso Diario (Cada vez que uses la app)

### **4. Inicia el backend**

En la raíz del proyecto:
```bash
python app.py
```

Espera a ver: `📱 Abre http://0.0.0.0:5003 en tu navegador`

### **5. Inicia Expo**

En otra terminal:
```bash
cd mobile/
npm start
```

**Escanea el QR con Expo Go** en tu celular.

---

## 📱 Instalar Expo Go

### iOS
[Descargar desde App Store](https://apps.apple.com/app/expo-go/id982107779)

### Android
[Descargar desde Play Store](https://play.google.com/store/apps/details?id=host.exp.exponent)

---

## ✅ Checklist de Verificación

Antes de usar la app, asegúrate de:

- [ ] Backend corriendo (`python app.py`)
- [ ] Mac y celular en la misma WiFi
- [ ] IP correcta en `api.ts`
- [ ] Expo Go instalado en tu celular
- [ ] Permisos de cámara activados

---

## 🔧 Problema Común

**Error: "No se puede conectar al servidor"**

1. Verifica tu IP:
   ```bash
   ipconfig getifaddr en0
   ```

2. Comprueba que ambos dispositivos estén en la misma WiFi

3. Reinicia el backend:
   ```bash
   # Ctrl+C para detener
   python app.py  # Iniciar de nuevo
   ```

---

## 🎉 ¡Listo!

Una vez que veas la app en tu celular, puedes:

1. **📸 Cámara** - Detección en tiempo real
2. **🖼️ Imágenes** - Analiza fotos de tu galería
3. **🎥 Videos** - Procesa videos completos

---

**¿Necesitas más ayuda?** Lee el `README.md` completo.

