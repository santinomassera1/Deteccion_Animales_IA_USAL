# 📂 Assets

Esta carpeta contiene los recursos visuales de la aplicación.

## 🎨 Archivos Requeridos

Para que la app funcione correctamente con Expo, necesitas estos archivos:

### **Iconos y Splash Screen**

- `icon.png` - Icono de la app (1024x1024 px)
- `adaptive-icon.png` - Icono adaptable para Android (1024x1024 px)
- `splash.png` - Pantalla de carga (1284x2778 px)
- `favicon.png` - Favicon para web (48x48 px)

## 📝 Cómo Crear los Assets

### Opción 1: Usar el logo existente

Puedes usar el logo de USAL que ya existe en la raíz del proyecto:

```bash
# Desde la raíz del proyecto
cp usal-logo.jpg mobile/assets/icon.png
cp usal-logo.jpg mobile/assets/adaptive-icon.png
cp usal-logo.jpg mobile/assets/splash.png
cp usal-logo.jpg mobile/assets/favicon.png
```

### Opción 2: Generar automáticamente

Expo puede generar los assets por ti:

1. Coloca una imagen PNG de 1024x1024 px en `assets/icon.png`
2. Ejecuta:
   ```bash
   npx expo prebuild
   ```

### Opción 3: Por ahora usar placeholders

La app funcionará sin estos archivos, Expo usará placeholders predeterminados.

## ✅ Nota Importante

Los archivos de assets **NO son obligatorios** para que la app funcione en Expo Go durante el desarrollo. Solo son necesarios cuando hagas un build de producción.

Para desarrollo y testing, puedes omitir estos archivos completamente.

