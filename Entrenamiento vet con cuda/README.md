# Proyecto YOLOv8 - Detección de Animales en Video

Este proyecto utiliza YOLOv8 (Ultralytics) para entrenar un modelo de detección de animales y aplicarlo sobre videos. Incluye scripts para entrenamiento, análisis del dataset y testeo sobre video.

## Estructura del proyecto

```
├── DataSet_Veterinaria/           # Dataset en formato YOLO
├── generacionYolo.py              # Script de entrenamiento
├── analizar_dataset.py            # Análisis y estadísticas del dataset
├── estimar_tiempo.py              # Estimación de tiempo de entrenamiento
├── test_video.py                  # Inferencia sobre video
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Este archivo
```

## Requisitos

- Python 3.10+
- GPU NVIDIA con soporte CUDA (opcional, recomendado)

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Entrenamiento

Asegúrate de tener el dataset en la carpeta `DataSet_Veterinaria/entrenamiento Nacho/` y ejecuta:

```bash
python generacionYolo.py
```

El modelo entrenado se guardará en `runs/animals_training_m/weights/best.pt`.

## Análisis del dataset

Para obtener estadísticas y verificar la estructura del dataset:

```bash
python analizar_dataset.py
```

## Estimación de tiempo de entrenamiento

Puedes estimar el tiempo de entrenamiento con:

```bash
python estimar_tiempo.py
```

## Inferencia sobre video

Coloca tu video (por ejemplo, `videovet.mp4`) en la carpeta del proyecto y ejecuta:

```bash
python test_video.py
```

El video anotado se guardará como `output_video.mp4`.

## Créditos

- Basado en [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Dataset: Roboflow / propio

---

¡Listo para detectar animales en tus videos! 🐶🐱🐮🐔🐴
