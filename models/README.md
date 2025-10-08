# 📦 Carpeta de Modelos - Sistema USAL de Detección de Animales

## 🎯 **Propósito**
Esta carpeta contiene los modelos de **YOLO** necesarios para el sistema de detección de animales. El sistema está diseñado para ser **completamente portable** y funcionar en cualquier instalación.

## 📋 **Modelos Requeridos**

### 🥇 **Modelos Principales (Entrenados)**
- **`animals_best.pt`** - Modelo entrenado principal (mejor rendimiento)
- **`animals_last.pt`** - Modelo entrenado secundario (última época)

### 🏗️ **Modelos Base (YOLO)**
- **`yolov8m.pt`** - YOLO v8 Medium (equilibrio rendimiento/velocidad)
- **`yolov8s.pt`** - YOLO v8 Small (más rápido)  
- **`yolo11n.pt`** - YOLO 11 Nano (más liviano)

## 🚀 **Sistema de Descarga Automática**

### ✅ **¿Falta algún modelo?**
El sistema tiene **descarga automática** integrada:

1. **Al iniciar la aplicación**, el sistema detecta automáticamente qué modelos están disponibles
2. **Si falta un modelo**, intenta descargarlo automáticamente desde el repositorio oficial de YOLO
3. **Si falla la descarga**, usa modelos de fallback disponibles
4. **En caso extremo**, descarga un modelo de emergencia básico

### 🔧 **Comando de Diagnóstico**
```bash
# Ejecutar diagnóstico completo del sistema de modelos
python enhanced_model_handler.py
```

## 📊 **Verificar Estado del Sistema**
```bash
# Desde el navegador, acceder a:
http://localhost:5003/api/model-system-info

# Esto mostrará:
# - Qué modelos están disponibles
# - Rutas de archivos
# - Estado de carga
# - Recomendaciones
```

## ⚡ **Instalación Manual** (Opcional)

### 1. **Copiar modelos desde entrenamientos anteriores:**
```bash
# Si tienes acceso a modelos entrenados
cp "path/to/trained/best.pt" models/animals_best.pt
cp "path/to/trained/last.pt" models/animals_last.pt
```

### 2. **Descargar modelos base manualmente:**
```bash
# Descargar modelos YOLO base
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt -P models/
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -P models/
```

## 🎪 **Estructura Final Esperada**
```
models/
├── README.md                 (este archivo)
├── animals_best.pt          (modelo entrenado - principal)
├── animals_last.pt          (modelo entrenado - secundario) 
├── yolov8m.pt              (YOLO v8 Medium)
├── yolov8s.pt              (YOLO v8 Small)
└── yolo11n.pt              (YOLO 11 Nano)
```

## 🚨 **Troubleshooting**

### ❌ **Error: "No se encontraron modelos"**
1. Verificar que esta carpeta existe: `models/`
2. Ejecutar: `python enhanced_model_handler.py`
3. El sistema intentará descarga automática

### ⚠️ **Error: "Modelo corrupto"**  
1. Eliminar archivo: `rm models/modelo_corrupto.pt`
2. Reiniciar aplicación (descarga automática)

### 🌐 **Sin conexión a Internet**
- El sistema funciona con cualquier modelo `.pt` válido en esta carpeta
- Copiar manualmente archivos `.pt` de otras instalaciones de YOLO

## 📝 **Notas Técnicas**

- **Tamaño típico**: Los modelos pueden ocupar entre 50MB-500MB cada uno
- **Compatibilidad**: Todos los modelos YOLO v8, v9, v10, v11 son compatibles  
- **CPU/GPU**: El sistema funciona tanto en CPU como GPU automáticamente
- **Sistema operativo**: Compatible con Windows, Mac, Linux

---

## 🎓 **Proyecto USAL - Universidad del Salvador**
**Sistema de Detección de Animales con Inteligencia Artificial**

*Desarrollado como proyecto final - Ingeniería en Sistemas*
