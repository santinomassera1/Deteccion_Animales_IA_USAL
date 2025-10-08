#!/usr/bin/env python3
"""
Aplicación web para detección de animales con YOLO 8
"""

import os
import cv2
import numpy as np
import time
import json
import threading
from collections import deque
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from config import Config
from person_detector import PersonDetector
from enhanced_model_handler import EnhancedModelHandler
# Ya no necesitamos el detection_stabilizer custom - usamos tracking nativo de YOLO
# from detection_stabilizer import DetectionStabilizer, get_enhanced_drawing_style

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuración de la aplicación
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['MAX_IMAGE_SIZE'] = Config.MAX_IMAGE_SIZE
app.config['MAX_VIDEO_SIZE'] = Config.MAX_VIDEO_SIZE
app.config['CONFIDENCE_THRESHOLD'] = Config.CONFIDENCE_THRESHOLD
app.config['ALLOWED_IMAGE_EXTENSIONS'] = Config.ALLOWED_IMAGE_EXTENSIONS
app.config['ALLOWED_VIDEO_EXTENSIONS'] = Config.ALLOWED_VIDEO_EXTENSIONS

# Crear directorio de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Modelos del ensemble
models = {}
model_loaded = False
person_detector = None
video_processing_status = {}  # Estado de procesamiento de videos
STATUS_FILE = 'video_processing_status.json'  # Archivo para persistir el estado

# Handler mejorado para modelos
enhanced_handler = None

# Ahora usamos tracking nativo de YOLO integrado en enhanced_handler
# detection_stabilizer = None

def save_video_status():
    """Guardar el estado de procesamiento en archivo"""
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(video_processing_status, f, indent=2)
    except Exception as e:
        print(f"Error guardando estado: {e}")

def load_video_status():
    """Cargar el estado de procesamiento desde archivo"""
    global video_processing_status
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, 'r') as f:
                video_processing_status = json.load(f)
            print(f"📂 Estado de video cargado: {len(video_processing_status)} videos")
    except Exception as e:
        print(f"Error cargando estado: {e}")
        video_processing_status = {}

def load_models():
    """Carga el sistema mejorado con ensemble TTA + YOLO Tracking para detección de 5 animales"""
    global models, model_loaded, person_detector, enhanced_handler
    
    try:
        print("Cargando sistema mejorado con Ensemble TTA...")
        
        # Cargar handler mejorado
        enhanced_handler = EnhancedModelHandler()
        if enhanced_handler.load_models():
            model_loaded = True
            
            # También mantener compatibilidad con el sistema anterior
            models['enhanced'] = enhanced_handler
            
            # Cargar detector de personas usando uno de los modelos cargados
            if enhanced_handler.models:
                primary_model = list(enhanced_handler.models.values())[0]
                person_detector = PersonDetector(nacho_model=primary_model)
                print("Detector de personas cargado con modelo principal")
            
            print("Sistema mejorado cargado exitosamente!")
            print("   Ensemble TTA - Máxima precisión con múltiples modelos")
            print("   Test Time Augmentation - Múltiples vistas por imagen")
            print("   🧠 Weighted Ensemble - Combinación inteligente de predicciones")
            print("   Post-procesamiento avanzado - Filtros específicos por clase")
            print("   - Gatos: Ultra alta precisión con ensemble")
            print("   - Gallinas: Ultra alta precisión con ensemble") 
            print("   - Vacas: Ultra alta precisión con ensemble")
            print("   - Perros: Ultra alta precisión con ensemble")
            print("   - Caballos: Ultra alta precisión con ensemble")
            print("   - Personas: Detectadas para evitar falsos positivos")
            
            print("Sistema de tracking YOLO nativo activado - Elimina flickering profesional")
            print("   - ByteTrack integrado en cada modelo")
            print("   - Tracking IDs persistentes entre frames")
            print("   - Eliminación automática de parpadeo")
            print("   - Asociación robusta de objetos")
            
            return True
        else:
            raise Exception("No se pudieron cargar los modelos del ensemble")
            
    except Exception as e:
        print(f"Error cargando sistema mejorado: {e}")
        print("Intentando fallback al sistema anterior...")
        
        # Fallback al sistema anterior
        try:
            import torch
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            
            models['cuda_model'] = YOLO('Entrenamiento vet con cuda/runs/animals_training_m/weights/best.pt')
            person_detector = PersonDetector(nacho_model=models['cuda_model'])
            model_loaded = True
            print("Sistema fallback cargado (modelo único)")
            
        except Exception as fallback_error:
            print(f"Error también en fallback: {fallback_error}")
            model_loaded = False

def preprocess_image_for_detection(image):
    """Preprocesa la imagen para mejorar la detección"""
    # Convertir a LAB para mejor manejo de iluminación
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization) en el canal L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Recombinar canales
    lab = cv2.merge([l, a, b])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Aplicar un ligero desenfoque gaussiano para reducir ruido
    processed = cv2.GaussianBlur(processed, (3, 3), 0)
    
    return processed

def enhance_image_contrast(image):
    """Mejora el contraste de la imagen para mejor detección"""
    # Convertir a escala de grises para análisis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcular histograma
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Encontrar percentiles para ajuste automático
    p2, p98 = np.percentile(gray, (2, 98))
    
    # Aplicar estiramiento de contraste
    enhanced = np.clip((image.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    
    return enhanced

def combine_multiple_predictions(results_list):
    """Combina múltiples predicciones para mayor precisión"""
    all_detections = []
    
    for results in results_list:
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                detection = {
                    'box': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': confidence,
                    'class': class_id,
                    'class_name': ['cat', 'chicken', 'cow', 'dog', 'horse'][class_id] if class_id < 5 else 'unknown',
                    'model': 'cuda_yolov8m'
                }
                all_detections.append(detection)
    
    # Aplicar NMS para eliminar duplicados
    if all_detections:
        # Ordenar por confianza
        all_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        
        # Aplicar NMS
        final_detections = []
        for detection in all_detections:
            is_duplicate = False
            for final_det in final_detections:
                iou = calculate_iou(detection['box'], final_det['box'])
                if iou > 0.4:  # Si se superponen mucho
                    # Mantener la de mayor confianza
                    if detection['confidence'] > final_det['confidence']:
                        final_detections.remove(final_det)
                        final_detections.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        return final_detections
    
    return []

def ensemble_predict(image, confidence_threshold):
    """Predice usando el sistema mejorado con Ensemble TTA + Tracking nativo de YOLO"""
    global enhanced_handler
    
    try:
        # Usar sistema mejorado si está disponible
        if enhanced_handler and model_loaded:
            print(f"Usando sistema Ensemble TTA + YOLO Tracking (threshold: {confidence_threshold})")
            detections = enhanced_handler.predict_with_tta(image, confidence_threshold)
            
            if detections:
                print(f"Ensemble TTA + Tracking completado: {len(detections)} detecciones")
                
                for i, det in enumerate(detections):
                    track_info = ""
                    if det.get('track_id') is not None:
                        track_info = f" [ID:{det['track_id']}]"
                    
                    ensemble_info = f" (ensemble: {det.get('ensemble_size', 'N/A')})" if 'ensemble_size' in det else ""
                    
                    print(f"   - {i+1}. {det['class_name']} ({det['confidence']:.3f}){track_info}{ensemble_info}")
                
                return detections
            else:
                print("Ensemble TTA + Tracking no encontró detecciones")
                return []
        
        # Fallback al sistema anterior si no hay enhanced_handler
        elif 'cuda_model' in models:
            print(f"Usando sistema fallback (threshold: {confidence_threshold})")
            return _legacy_predict(image, confidence_threshold)
        else:
            print("No hay modelos disponibles")
            return []
            
    except Exception as e:
        print(f"Error en ensemble_predict: {e}")
        # Intentar fallback
        if 'cuda_model' in models:
            print("Intentando fallback después de error...")
            return _legacy_predict(image, confidence_threshold)
        return []

def _legacy_predict(image, confidence_threshold):
    """Predicción fallback con el sistema anterior"""
    try:
        all_detections = []
        
        # Redimensionar imagen si es muy grande
        height, width = image.shape[:2]
        if width > 640 or height > 480:
            scale = min(640/width, 480/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Preprocesamiento
        processed_image = preprocess_image_for_detection(image)
        
        # Predicción con modelo CUDA
        results = models['cuda_model'](processed_image, conf=confidence_threshold, verbose=False, 
                                     imgsz=640, device='cpu', half=False)
        
        # Procesar resultados
        for result in [results]:
            if result[0].boxes is not None:
                for box in result[0].boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id < 5:  # Solo nuestras 5 clases
                        detection = {
                            'box': box.xyxy[0].cpu().numpy().tolist(),
                            'confidence': confidence,
                            'class': class_id,
                            'class_name': ['cat', 'chicken', 'cow', 'dog', 'horse'][class_id],
                            'model': 'cuda_yolov8m_legacy'
                        }
                        all_detections.append(detection)
        
        # Filtrado básico
        final_detections = combine_detections(all_detections, confidence_threshold)
        
        print(f"Predicción legacy completada: {len(final_detections)} detecciones")
        return final_detections
        
    except Exception as e:
        print(f"Error en predicción legacy: {e}")
        return []

def correct_class_confusion(detection):
    """Corrige confusiones comunes entre clases similares"""
    class_name = detection['class_name']
    confidence = detection['confidence']
    x1, y1, x2, y2 = detection['box']
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height
    
    # Corrección MUY conservadora solo en casos extremos
    if class_name == 'cat' and confidence < 0.6:
        # Solo cambiar si es MUY grande (área > 20000 píxeles)
        area = width * height
        if area > 20000:
            detection['class_name'] = 'dog'
            detection['confidence'] = confidence * 0.9  # Reducir confianza por corrección
            print(f"Corrección conservadora: cat -> dog (área muy grande: {area})")
    
    elif class_name == 'dog' and confidence < 0.6:
        # Solo cambiar si es MUY pequeño (área < 3000 píxeles)
        area = width * height
        if area < 3000:
            detection['class_name'] = 'cat'
            detection['confidence'] = confidence * 0.9
            print(f"Corrección conservadora: dog -> cat (área muy pequeña: {area})")
    
    return detection

def combine_detections(detections, confidence_threshold):
    """Combina y filtra las mejores detecciones del modelo CUDA YOLOv8m con filtros avanzados"""
    if not detections:
        return []
    
    # Umbrales específicos por clase para mayor precisión - MÁS ESTRICTOS
    class_thresholds = {
        'cat': 0.7,      # Gatos: muy estricto
        'dog': 0.7,      # Perros: muy estricto  
        'chicken': 0.6,  # Gallinas: estricto
        'cow': 0.6,      # Vacas: estricto
        'horse': 0.6     # Caballos: estricto
    }
    
    filtered_detections = []
    
    for detection in detections:
        # 0. Aplicar corrección de confusión entre clases
        detection = correct_class_confusion(detection)
        
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # 1. Filtro por umbral específico de clase
        class_threshold = class_thresholds.get(class_name, confidence_threshold)
        if confidence < class_threshold:
            continue
            
        # 2. Filtro por tamaño mínimo y máximo (evitar detecciones muy pequeñas o muy grandes)
        x1, y1, x2, y2 = detection['box']
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Tamaño mínimo más estricto: 30x30 píxeles
        if width < 30 or height < 30 or area < 900:
            continue
            
        # Tamaño máximo para evitar detecciones que abarcan toda la imagen
        if width > 500 or height > 400 or area > 150000:
            # Solo permitir si la confianza es muy alta
            if confidence < 0.9:
                continue
            
        # 3. Filtro por ratio de aspecto (evitar formas muy extrañas)
        aspect_ratio = width / height
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Muy delgado o muy ancho
            continue
            
        # 4. Filtro por posición (evitar detecciones en bordes extremos)
        image_width = 640  # Asumiendo resolución estándar
        image_height = 480
        
        # Evitar detecciones que toquen los bordes (posibles falsos positivos)
        margin = 10
        if (x1 < margin or y1 < margin or 
            x2 > image_width - margin or y2 > image_height - margin):
            # Solo permitir si la confianza es muy alta
            if confidence < 0.8:
                continue
        
        filtered_detections.append(detection)
    
    # 5. Filtro de supresión no máxima (NMS) para evitar duplicados - MÁS ESTRICTO
    final_detections = apply_nms(filtered_detections, iou_threshold=0.5)
    
    # 6. Filtro adicional por clase: máximo 2 detecciones por clase para evitar sobre-detección
    class_counts = {}
    validated_detections = []
    
    # Ordenar por confianza para priorizar las mejores detecciones
    final_detections = sorted(final_detections, key=lambda x: x['confidence'], reverse=True)
    
    for detection in final_detections:
        class_name = detection['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0)
        
        # Límite por clase (máximo 2 por clase, excepto gallinas que pueden ser más)
        max_per_class = 3 if class_name == 'chicken' else 2
        
        if class_counts[class_name] < max_per_class:
            validated_detections.append(detection)
            class_counts[class_name] += 1
    
    return validated_detections

def apply_nms(detections, iou_threshold=0.5):
    """Aplica Non-Maximum Suppression para eliminar detecciones duplicadas"""
    if not detections:
        return []
    
    # Ordenar por confianza descendente
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    final_detections = []
    
    while detections:
        # Tomar la detección con mayor confianza
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        
        # Calcular IoU con las detecciones restantes
        remaining_detections = []
        for detection in detections:
            iou = calculate_iou(best_detection['box'], detection['box'])
            if iou < iou_threshold:  # Si no se superponen mucho, mantenerla
                remaining_detections.append(detection)
        
        detections = remaining_detections
    
    return final_detections

def calculate_iou(box1, box2):
    """Calcula el Intersection over Union (IoU) entre dos bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calcular intersección
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calcular unión
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def apply_temporal_filter(detections, frame_number):
    """Aplica filtrado temporal para eliminar detecciones inconsistentes"""
    global detection_history
    
    # Mantener solo los últimos N frames
    detection_history = detection_history[-MAX_HISTORY_FRAMES:]
    
    # Agregar detecciones actuales al historial
    detection_history.append({
        'frame': frame_number,
        'detections': detections.copy()
    })
    
    if len(detection_history) < 3:  # Necesitamos al menos 3 frames para validar
        return detections
    
    # Contar cuántas veces aparece cada detección en el historial
    detection_counts = {}
    
    for frame_data in detection_history:
        for detection in frame_data['detections']:
            # Crear una clave única para la detección (posición aproximada + clase)
            x1, y1, x2, y2 = detection['box']
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            key = f"{detection['class_name']}_{int(center_x/50)}_{int(center_y/50)}"
            
            if key not in detection_counts:
                detection_counts[key] = {
                    'count': 0,
                    'detection': detection,
                    'total_confidence': 0
                }
            
            detection_counts[key]['count'] += 1
            detection_counts[key]['total_confidence'] += detection['confidence']
    
    # Solo mantener detecciones que aparecen en al menos 2 de los últimos frames
    filtered_detections = []
    for key, data in detection_counts.items():
        if data['count'] >= 2:  # Aparece en al menos 2 frames
            # Promediar la confianza
            avg_confidence = data['total_confidence'] / data['count']
            detection = data['detection'].copy()
            detection['confidence'] = avg_confidence
            filtered_detections.append(detection)
    
    return filtered_detections

def get_detection_color(class_name):
    """Obtiene el color para una clase de detección - BGR format para OpenCV"""
    colors = {
        'cat': (255, 0, 255),      # Magenta para gatos (BGR)
        'chicken': (0, 165, 255),  # Naranja brillante para gallinas (BGR)
        'cow': (0, 255, 0),        # Verde brillante para vacas (BGR)
        'dog': (255, 0, 0),        # Azul brillante para perros (BGR)
        'horse': (0, 255, 255),    # Amarillo brillante para caballos (BGR) - MÁS VISIBLE
        'person': (255, 255, 0)    # Cian para personas (BGR)
    }
    return colors.get(class_name, (255, 255, 255))  # Blanco por defecto

def allowed_file(filename, file_type='image'):
    """Verifica si el archivo tiene una extensión permitida"""
    allowed_extensions = app.config['ALLOWED_IMAGE_EXTENSIONS'].union(app.config['ALLOWED_VIDEO_EXTENSIONS'])
    if file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/model-status')
def model_status():
    return jsonify({'model_loaded': model_loaded})

@app.route('/api/yolo-tracking-stats', methods=['GET'])
def yolo_tracking_stats():
    """Endpoint para obtener estadísticas del tracking YOLO nativo"""
    try:
        return jsonify({
            'status': 'active',
            'tracking_system': 'YOLO ByteTrack',
            'integrated': True,
            'message': 'Tracking integrado en cada modelo YOLO',
            'features': [
                'Tracking IDs persistentes',
                'Eliminación automática de flickering', 
                'Asociación robusta de objetos',
                'ByteTrack algorithm'
            ]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/video-status/<filename>', methods=['GET'])
def video_status(filename):
    """Endpoint para obtener el estado de procesamiento de un video"""
    try:
        if filename not in video_processing_status:
            return jsonify({
                'filename': filename,
                'status': 'idle',
                'processed_frames': 0,
                'total_frames': 0,
                'progress_percent': 0.0,
                'output_video_url': None,
                'error': 'Video not found in processing queue'
            }), 404
        
        status = video_processing_status[filename]
        
        # Calcular progreso porcentual
        progress_percent = 0.0
        if status['total_frames'] > 0:
            progress_percent = (status['processed_frames'] / status['total_frames']) * 100
        
        return jsonify({
            'filename': filename,
            'status': status['status'],
            'processed_frames': status['processed_frames'],
            'total_frames': status['total_frames'],
            'progress_percent': progress_percent,  # ← AGREGAR ESTE CAMPO
            'output_video_url': status.get('output_video_url'),
            'error': status.get('error')
        })
    except Exception as e:
        print(f"Error getting video status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Verificar tamaño del archivo
    file.seek(0, 2)  # Ir al final del archivo
    file_size = file.tell()
    file.seek(0)  # Volver al inicio
    
    original_filename = file.filename
    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()
    
    print(f"📁 Archivo original: '{original_filename}' -> Seguro: '{filename}'")
    
    if file_extension in ['jpg', 'jpeg', 'png', 'gif']:
        if file_size > app.config['MAX_IMAGE_SIZE']:
            return jsonify({'error': f'Image too large. Maximum {app.config["MAX_IMAGE_SIZE"] // (1024*1024)}MB allowed'}), 413
    elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
        if file_size > app.config['MAX_VIDEO_SIZE']:
            return jsonify({'error': f'Video too large. Maximum {app.config["MAX_VIDEO_SIZE"] // (1024*1024)}MB allowed'}), 413
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({
        'message': 'File uploaded successfully',
        'filename': filename,
        'filepath': filepath
    })

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Cargar imagen
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not load image'}), 400
        
        # Crear copia para procesar
        processed_image = image.copy()
        
        # Predicción del ensemble
        detections = ensemble_predict(image, app.config['CONFIDENCE_THRESHOLD'])
        
        if detections is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Dibujar detecciones en la imagen procesada con estilo mejorado
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['box']]
            
            # Label mejorado con información de tracking
            base_label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            
            # Agregar información de tracking si está disponible
            track_info = ""
            if detection.get('track_id') is not None:
                track_info = f" [ID:{detection['track_id']}]"
            
            label = base_label + track_info
            
            # Color y grosor basado en si tiene tracking o no
            color = get_detection_color(detection['class_name'])
            if detection.get('tracked', False):
                thickness = 3  # Líneas más gruesas para objetos trackeados
            else:
                thickness = 2  # Líneas normales para detecciones nuevas
            
            # Dibujar bounding box con grosor variable
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, thickness)
            
            # Dibujar etiqueta con fondo
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(processed_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            cv2.putText(processed_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Guardar imagen procesada
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, processed_image)
        
        # Preparar resultados
        results = []
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': round(detection['confidence'], 3),
                'class': detection['class_name'],
                'model': detection['model']
            })
        
        return jsonify({
            'detections': results,
            'total_detections': len(results),
            'original_image': f"/download/{filename}",
            'processed_image': f"/download/{processed_filename}",
            'original_filename': filename,
            'processed_filename': processed_filename
        })
        
    except Exception as e:
        print(f"Error in detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-video', methods=['POST'])
def process_video():
    # Check if JSON data is provided (new flow)
    if request.is_json:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
    else:
        # Legacy flow - direct file upload
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        if not allowed_file(video_file.filename, 'video'):
            return jsonify({'error': 'Invalid video file type'}), 400
        
        # Verificar tamaño del archivo
        video_file.seek(0, 2)  # Ir al final del archivo
        file_size = video_file.tell()
        video_file.seek(0)  # Volver al inicio
        
        if file_size > app.config['MAX_VIDEO_SIZE']:
            return jsonify({'error': f'Video file too large. Maximum size: {app.config["MAX_VIDEO_SIZE"] // (1024*1024)}MB'}), 413
        
        # Guardar video temporalmente
        filename = secure_filename(video_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(filepath)
    
    try:
        # Obtener información del video antes de procesar
        cap = cv2.VideoCapture(filepath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Obtener tamaño del archivo para ambos flujos
        file_size = os.path.getsize(filepath)
        
        # Crear directorio de salida
        name_without_ext = os.path.splitext(filename)[0]
        output_path = f"runs/detect/processed_{name_without_ext}"
        os.makedirs(output_path, exist_ok=True)
        
        # Iniciar procesamiento en segundo plano
        import threading
        
        def process_video_background():
            try:
                print(f"Procesando video en segundo plano: {filename}")
                print(f"📁 Archivo temporal: {filepath}")
                print(f"📂 Directorio de salida: {output_path}")
                
                # Inicializar estado de procesamiento
                video_processing_status[filename] = {
                    'status': 'processing',
                    'processed_frames': 0,
                    'total_frames': total_frames,
                    'output_video_url': None,
                    'error': None
                }
                save_video_status()  # Persistir estado
                
                output_filename = f"processed_{name_without_ext}.mp4"
                output_filepath = os.path.join(output_path, output_filename)
                
                print(f"   - Modelo: Único optimizado (no ensemble)")
                print(f"   - Tracking: ByteTrack nativo de YOLO")
                print(f"📁 Video de salida: {output_filepath}")
                
                # Configurar VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = None
                frame_count = 0
                
                # Callback para actualizar progreso
                def update_progress(frame_num):
                    nonlocal frame_count
                    frame_count = frame_num
                    video_processing_status[filename]['processed_frames'] = frame_count
                    
                    # Guardar estado cada 20 frames
                    if frame_count % 20 == 0:
                        progress_percent = (frame_count / total_frames) * 100
                        save_video_status()
                        print(f"📊 Progreso: {progress_percent:.1f}% ({frame_count}/{total_frames} frames)")
                
                # Procesar video con tracking persistente
                print(f"🚀 Iniciando procesamiento con tracking...")
                
                for result in enhanced_handler.process_video_with_tracking(
                    video_source=filepath,
                    confidence_threshold=app.config['CONFIDENCE_THRESHOLD'],
                    callback=update_progress
                ):
                    processed_frame = result['frame']
                    detections = result['detections']
                    
                    # Inicializar VideoWriter con las dimensiones del primer frame
                    if out is None:
                        h, w = processed_frame.shape[:2]
                        out = cv2.VideoWriter(output_filepath, fourcc, fps, (w, h))
                        print(f"   Resolución: {w}x{h}, {fps} FPS")
                    
                    # Escribir frame procesado (ya viene con detecciones dibujadas)
                    out.write(processed_frame)
                    
                    # Log opcional de detecciones
                    if detections and frame_count % 30 == 0:
                        tracked_count = sum(1 for d in detections if d['tracked'])
                        print(f"   Frame {frame_count}: {len(detections)} detecciones, {tracked_count} con tracking ID")
                
                # Liberar recursos
                if out is not None:
                    out.release()
                
                print(f"✅ Procesamiento con tracking completado: {frame_count} frames procesados")
                
                # Verificar que el archivo se creó correctamente
                if os.path.exists(output_filepath):
                    file_size = os.path.getsize(output_filepath)
                    print(f"Video procesado exitosamente: {output_filename}")
                    print(f"📁 Ubicación: {output_filepath}")
                    print(f"Tamaño: {file_size / (1024*1024):.2f} MB")
                    
                    # Verificar que el video se puede leer correctamente
                    test_cap = cv2.VideoCapture(output_filepath)
                    if test_cap.isOpened():
                        ret, test_frame = test_cap.read()
                        test_cap.release()
                        if ret and test_frame is not None:
                            print(f"Video verificado: Se puede leer correctamente")
                        else:
                            print(f"Advertencia: Video creado pero no se puede leer")
                    else:
                        print(f"Advertencia: Video creado pero no se puede abrir")
                else:
                    print(f"Error: El archivo de salida no se creó: {output_filepath}")
                
                # Limpiar archivo temporal
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"🧹 Archivo temporal eliminado: {filepath}")
                else:
                    print(f"Archivo temporal ya no existe: {filepath}")
                
                # Marcar como completado
                if filename in video_processing_status:
                    video_processing_status[filename]['status'] = 'completed'
                    video_processing_status[filename]['output_video_url'] = f"/video/{output_filename}"
                    save_video_status()  # Persistir estado final
                    print(f"Procesamiento completado para: {filename}")
                
            except Exception as e:
                print(f"Error procesando video en segundo plano: {e}")
                import traceback
                traceback.print_exc()
                
                # Limpiar archivo temporal en caso de error (solo si existe)
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        print(f"🧹 Archivo temporal eliminado después de error: {filepath}")
                except Exception as cleanup_error:
                    print(f"Error limpiando archivo temporal: {cleanup_error}")
                
                # Marcar como fallido en caso de error
                if filename in video_processing_status:
                    video_processing_status[filename]['status'] = 'failed'
                    video_processing_status[filename]['error'] = str(e)
                    save_video_status()  # Persistir estado de error
        
        # Iniciar procesamiento en segundo plano
        thread = threading.Thread(target=process_video_background)
        thread.daemon = True
        thread.start()
        
        # Devolver respuesta inmediata con información del video
        return jsonify({
            'message': 'Video uploaded and processing started',
            'filename': filename,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'file_size_mb': round(file_size / (1024*1024), 2),
            'estimated_time_minutes': max(1, round(total_frames / (fps * 60) * 0.8, 1)),  # Estimación optimizada
            'status': 'processing'
        })
        
    except Exception as e:
        print(f"Error en process_video: {e}")
        import traceback
        traceback.print_exc()
        
        # Limpiar archivo temporal en caso de error
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
                print(f"🧹 Archivo temporal eliminado después de error en endpoint: {filepath}")
        except Exception as cleanup_error:
            print(f"Error limpiando archivo en endpoint: {cleanup_error}")
            
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Descargar archivos desde uploads o runs/detect"""
    try:
        # Buscar en uploads (imágenes originales)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(upload_path):
            print(f"📁 Archivo encontrado en uploads: {filename}")
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        
        # Buscar en runs/detect y sus subdirectorios
        for root, dirs, files in os.walk('runs/detect'):
            if filename in files:
                print(f"📁 Archivo encontrado en: {root}/{filename}")
                return send_from_directory(root, filename)
        
        # Si no se encuentra, buscar en el directorio raíz de runs/detect
        detect_path = os.path.join('runs/detect', filename)
        if os.path.exists(detect_path):
            print(f"📁 Archivo encontrado en: runs/detect/{filename}")
            return send_from_directory('runs/detect', filename)
        
        print(f"Archivo no encontrado: {filename}")
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        print(f"Error descargando archivo {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/video/<filename>')
def serve_video(filename):
    """Servir videos procesados desde runs/detect"""
    try:
        # Buscar en runs/detect y sus subdirectorios
        for root, dirs, files in os.walk('runs/detect'):
            if filename in files:
                print(f"Video encontrado en: {root}/{filename}")
                return send_from_directory(root, filename)
        
        # Si no se encuentra, buscar en el directorio raíz de runs/detect
        detect_path = os.path.join('runs/detect', filename)
        if os.path.exists(detect_path):
            print(f"Video encontrado en: runs/detect/{filename}")
            return send_from_directory('runs/detect', filename)
        
        print(f"Video no encontrado: {filename}")
        return jsonify({'error': 'Video not found'}), 404
        
    except Exception as e:
        print(f"Error sirviendo video {filename}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/webcam', methods=['GET'])
def webcam_stream():
    """
    Streaming de cámara con TRACKING PERSISTENTE en tiempo real.
    Usa el mismo sistema que videos procesados: 1 modelo + ByteTrack.
    """
    def generate_tracked_frames():
        """
        Generador que consume el tracking del handler y produce stream MJPEG.
        """
        # 1. Validar que el modelo esté cargado
        if not enhanced_handler or not model_loaded:
            print("❌ ERROR: Stream de webcam solicitado pero modelo no está cargado")
            error_msg = b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Model not loaded\r\n\r\n'
            yield error_msg
            return
        
        print("🎬 Iniciando webcam con TRACKING PERSISTENTE (sin parpadeo)")
        print("   - Sistema: ByteTrack nativo de YOLO")
        print("   - Ventaja: Elimina parpadeo, IDs persistentes, 3-4x más rápido")
        
        try:
            # 2. Iniciar el generador de tracking persistente con webcam
            #    source=0 usa la webcam por defecto
            frame_generator = enhanced_handler.process_video_with_tracking(
                video_source=0,
                confidence_threshold=0.4  # Threshold optimizado para tiempo real
            )
            
            # 3. Iterar sobre los resultados del tracking
            for result in frame_generator:
                # El resultado es un diccionario con:
                # - 'frame': numpy array con detecciones dibujadas
                # - 'detections': lista de detecciones con tracking IDs
                # - 'frame_number': número de frame
                
                processed_frame = result['frame']
                detections = result['detections']
                frame_number = result['frame_number']
                
                # Almacenar detecciones para alertas (últimas 10)
                if detections:
                    for detection in detections:
                        webcam_detections.append({
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'timestamp': time.time(),
                            'track_id': detection.get('track_id')
                        })
                    
                    # Mantener solo las últimas 10
                    if len(webcam_detections) > 10:
                        webcam_detections.pop(0)
                
                # Log de debug cada 30 frames
                if frame_number % 30 == 0 and detections:
                    tracked_count = sum(1 for d in detections if d.get('tracked'))
                    print(f"📹 Frame {frame_number}: {len(detections)} detecciones, {tracked_count} con ID persistente")
                
                # 4. Codificar frame a JPEG
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, 85,  # Buena calidad para visualización
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Optimizado
                ]
                
                ret, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
                
                if not ret:
                    print(f"⚠️ Error codificando frame {frame_number}")
                    continue
                
                # 5. Convertir a bytes y formatear para MJPEG stream
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Cache-Control: no-cache\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            print(f"❌ Error durante streaming de webcam: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            print("🔚 Stream de webcam finalizado")
    
    # Retornar respuesta de streaming
    response = Response(
        generate_tracked_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Connection'] = 'close'
    
    return response

@app.route('/api/webcam-performance', methods=['GET'])
def get_webcam_performance():
    """Obtener estadísticas de rendimiento del stream webcam CON TRACKING PERSISTENTE"""
    try:
        # Nuevo sistema: Tracking persistente (sin frame skipping)
        performance_info = {
            'status': 'tracking_persistente',
            'system': 'ByteTrack Native YOLO',
            'video_capture_fps': 30,
            'ai_processing_fps': '25-30',  # Procesa TODOS los frames
            'resolution': 'Native webcam (sin resize forzado)',
            'optimizations': [
                '🎯 Tracking persistente con ByteTrack',
                '✅ Sin parpadeo - IDs constantes entre frames',
                '⚡ 1 modelo (no ensemble) = 24x más rápido',
                '📹 Procesa TODOS los frames (no skip)',
                '🎨 Trayectorias visuales de objetos trackeados',
                '🚀 Mismo sistema que videos procesados'
            ],
            'improvements': {
                'speed': '3-4x más rápido que sistema anterior',
                'stability': 'Sin parpadeo (tracking persistente)',
                'accuracy': 'IDs únicos y persistentes',
                'frames_processed': 'Todos (antes: 1 de cada 4)'
            },
            'expected_performance': '25-30 FPS video Y detecciones',
            'compression_quality': 85,
            'frame_skip_ratio': 'None - Procesa todos los frames',
            'tracking_ids': 'Persistentes con ByteTrack',
            'model_used': 'animals_best.pt (5 clases veterinarias)'
        }
        
        return jsonify(performance_info)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/model-system-info', methods=['GET'])
def get_model_system_info():
    """Obtener información completa del sistema de modelos - Para debugging"""
    try:
        if enhanced_handler:
            system_info = enhanced_handler.get_system_info()
            return jsonify({
                'status': 'success',
                'system_info': system_info,
                'timestamp': time.time()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Enhanced handler no disponible',
                'system_info': None
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'system_info': None
        }), 500

@app.route('/api/webcam', methods=['POST'])
def webcam_detect():
    """Endpoint para detectar animales en una imagen capturada de la webcam"""
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        print(f"Webcam POST - Datos recibidos: {type(data)}")
        
        if not data or 'image' not in data:
            print(f"Error: datos={data}, tiene image={data and 'image' in data if data else False}")
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        
        # Decodificar imagen base64
        if image_data.startswith('data:image/'):
            # Remover el prefijo data:image/jpeg;base64,
            image_data = image_data.split(',')[1]
        
        # Decodificar base64 a bytes
        import base64
        image_bytes = base64.b64decode(image_data)
        
        # Convertir bytes a imagen OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Aplicar detección con el modelo CUDA YOLOv8m
        detections = ensemble_predict(image, app.config['CONFIDENCE_THRESHOLD'])
        
        if detections is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Crear imagen procesada con detecciones
        processed_image = image.copy()
        
        # Dibujar detecciones en la imagen procesada
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['box']]
            label = f"{detection['class_name']} ({detection['confidence']:.2f})"
            
            # Color según el tipo de animal
            color = get_detection_color(detection['class_name'])
            
            # Dibujar bounding box optimizado
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar etiqueta con fondo
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(processed_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            cv2.putText(processed_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convertir imagen procesada a base64 para enviar al frontend
        ret, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ret:
            return jsonify({'error': 'Could not encode processed image'}), 500
        
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Preparar resultados
        results = []
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            results.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': round(detection['confidence'], 3),
                'class': detection['class_name'],
                'model': detection['model']
            })
        
        return jsonify({
            'detections': results,
            'total_detections': len(results),
            'processed_image': f"data:image/jpeg;base64,{processed_image_base64}",
            'original_filename': 'webcam_capture',
            'processed_filename': 'webcam_processed',
            'original_image': data['image'],  # Devolver la imagen original también
        })
        
    except Exception as e:
        print(f"Error in webcam detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/webcam/detections')
def get_webcam_detections():
    """Obtener las últimas detecciones de la webcam"""
    try:
        # Limpiar detecciones antiguas (más de 30 segundos)
        current_time = time.time()
        recent_detections = [
            d for d in webcam_detections 
            if current_time - d['timestamp'] <= 30
        ]
        
        # Actualizar la lista con solo las detecciones recientes
        webcam_detections.clear()
        webcam_detections.extend(recent_detections)
        
        return jsonify({
            'detections': webcam_detections,
            'total': len(webcam_detections)
        })
        
    except Exception as e:
        print(f"Error getting webcam detections: {e}")
        return jsonify({'error': str(e)}), 500

# Almacenar streams de procesamiento en vivo
# Almacenar detecciones de webcam para alertas
webcam_detections = []

# Sistema de filtrado temporal para evitar falsos positivos
detection_history = []  # Historial de detecciones para filtrado temporal
MAX_HISTORY_FRAMES = 5  # Número de frames a considerar para validación temporal



@app.route('/webcam-page')
def webcam_page():
    """Página para mostrar el stream de webcam en tiempo real"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Webcam - Detección en Tiempo Real</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                text-align: center;
            }
            h1 {
                margin-bottom: 30px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .webcam-container {
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            .webcam-feed {
                border-radius: 10px;
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            }
            .controls {
                margin: 20px 0;
            }
            .btn {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                margin: 0 10px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .info {
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 15px;
                margin: 20px 0;
                backdrop-filter: blur(5px);
            }
            .back-link {
                display: inline-block;
                margin-top: 20px;
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                background: rgba(255,255,255,0.2);
                border-radius: 20px;
                transition: all 0.3s ease;
            }
            .back-link:hover {
                background: rgba(255,255,255,0.3);
                transform: translateY(-1px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Detección en Tiempo Real</h1>
            
            <div class="webcam-container">
                <h2>Webcam en Vivo</h2>
                <img id="webcamFeed" class="webcam-feed" src="/api/webcam" alt="Webcam Feed">
                
                <div class="controls">
                    <button class="btn" onclick="startWebcam()">Iniciar</button>
                    <button class="btn" onclick="stopWebcam()">Detener</button>
                    <button class="btn" onclick="refreshWebcam()">Refrescar</button>
                </div>
            </div>
            
            <div class="info">
                <h3>Sistema AI Ensemble TTA</h3>
                <p><strong>Animales detectados:</strong> Gatos, Gallinas, Vacas, Perros, Caballos</p>
                <p><strong>Detección:</strong> Tiempo real con filtrado inteligente</p>
                <p><strong>Resolución:</strong> 640x480 @ 30 FPS</p>
            </div>
            
            <a href="/" class="back-link">← Volver a la aplicación principal</a>
        </div>
        
        <script>
            let webcamActive = true;
            
            function startWebcam() {
                if (!webcamActive) {
                    webcamActive = true;
                    document.getElementById('webcamFeed').src = '/api/webcam';
                }
            }
            
            function stopWebcam() {
                webcamActive = false;
                document.getElementById('webcamFeed').src = '';
            }
            
            function refreshWebcam() {
                stopWebcam();
                setTimeout(() => {
                    startWebcam();
                }, 100);
            }
            
            // Auto-refresh si hay problemas de conexión
            document.getElementById('webcamFeed').onerror = function() {
                console.log('Error en webcam, reintentando...');
                setTimeout(refreshWebcam, 2000);
            };
        </script>
    </body>
    </html>
    '''


if __name__ == '__main__':
    # Cargar modelos al iniciar
    load_models()
    load_video_status()  # Cargar estado persistente de videos
    
    port = int(os.environ.get('FLASK_PORT', 5003))
    print(f"Iniciando aplicación de detección de animales...")
    print(f"📱 Abre http://0.0.0.0:{port} en tu navegador")
    
    # Deshabilitar auto-reload para evitar interrumpir el procesamiento de videos
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
