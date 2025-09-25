#!/usr/bin/env python3
"""
Handler mejorado para modelos con múltiples técnicas de precisión
Incluye: Ensemble, TTA, mejor post-procesamiento
"""

import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
from pathlib import Path

class EnhancedModelHandler:
    def __init__(self):
        """Inicializar handler mejorado con múltiples modelos"""
        self.models = {}
        self.model_loaded = False
        
        # Fix para PyTorch 2.6 weights_only issue
        self._fix_torch_load()
        
        # FORZAR CPU - Deshabilitar CUDA completamente para Mac
        self._force_cpu_mode()
        
        # Paths de modelos disponibles
        self.model_paths = {
            'primary': 'Entrenamiento vet con cuda/runs/animals_training_m/weights/best.pt',
            'secondary': 'Entrenamiento vet con cuda/runs/animals_training_m/weights/last.pt',
            'yolo11n': 'Entrenamiento vet con cuda/yolo11n.pt',
            'yolov8m': 'Entrenamiento vet con cuda/yolov8m.pt',
            'yolov8s': 'Entrenamiento vet con cuda/yolov8s.pt'
        }
        
    def _fix_torch_load(self):
        """Fix para el problema de PyTorch 2.6 weights_only"""
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            # Forzar weights_only=False para compatibilidad
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        print("🔧 PyTorch load fix aplicado")
    
    def _force_cpu_mode(self):
        """Forzar modo CPU y deshabilitar CUDA completamente"""
        import os
        import torch
        
        # Deshabilitar CUDA en variables de entorno
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        print("🖥️  Modo CPU forzado - CUDA deshabilitado para compatibilidad Mac")
        print(f"   - torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"   - torch.cuda.device_count(): {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            print("⚠️  CUDA aún detectado, pero se usará CPU forzosamente")
        
    def load_models(self):
        """Cargar múltiples modelos para ensemble"""
        print("🔄 Cargando modelos para ensemble mejorado...")
        
        loaded_count = 0
        
        # Intentar cargar modelo principal (entrenado)
        for model_name, model_path in self.model_paths.items():
            if os.path.exists(model_path):
                try:
                    print(f"🔄 Intentando cargar {model_name}: {model_path}")
                    model = YOLO(model_path)
                    self.models[model_name] = model
                    
                    # Información del modelo
                    if hasattr(model.model, 'names'):
                        classes = list(model.model.names.values())
                        print(f"✅ {model_name} cargado: {len(classes)} clases")
                        
                        # Verificar si tiene nuestras 5 clases de animales
                        target_classes = ['cat', 'chicken', 'cow', 'dog', 'horse']
                        if all(cls in classes for cls in target_classes):
                            print(f"   🎯 Modelo compatible con detección de animales")
                            loaded_count += 1
                        else:
                            print(f"   ⚠️ Clases disponibles: {classes[:10]}")
                            
                except Exception as e:
                    print(f"❌ Error cargando {model_name}: {e}")
            else:
                print(f"⚠️ Modelo no encontrado: {model_path}")
        
        if loaded_count > 0:
            self.model_loaded = True
            print(f"✅ {loaded_count} modelos cargados para ensemble")
            return True
        else:
            print("❌ No se pudo cargar ningún modelo")
            return False
    
    def predict_with_tta(self, image, confidence_threshold=0.5):
        """Predicción con Test Time Augmentation para mayor precisión"""
        if not self.model_loaded or not self.models:
            return []
            
        all_predictions = []
        
        # Generar variaciones de la imagen para TTA
        augmented_images = self._generate_tta_images(image)
        
        # Predecir con cada modelo en cada variación
        for model_name, model in self.models.items():
            for aug_name, aug_image in augmented_images.items():
                try:
                    # Usar tracking nativo de YOLO para eliminar flickering
                    results = model.track(
                        aug_image,
                        conf=confidence_threshold,
                        iou=0.4,  # IoU threshold para NMS
                        imgsz=640,  # Tamaño optimizado para CPU
                        device='cpu',  # Forzar CPU (Mac compatible)
                        half=False,  # No half precision en CPU
                        verbose=False,
                        tracker="bytetrack.yaml",  # Usar ByteTrack para estabilidad
                        persist=True,  # Mantener tracking entre frames
                        max_det=50  # Máximo 50 detecciones
                    )
                    
                    # Procesar resultados
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                
                                # Solo procesar nuestras 5 clases
                                if class_id < 5:
                                    # Ajustar coordenadas según la augmentación
                                    adjusted_box = self._adjust_box_coordinates(
                                        box.xyxy[0].cpu().numpy().tolist(),
                                        aug_name,
                                        image.shape
                                    )
                                    
                                    # Extraer información de tracking si está disponible
                                    track_id = None
                                    if hasattr(box, 'id') and box.id is not None:
                                        track_id = int(box.id[0])
                                    
                                    detection = {
                                        'box': adjusted_box,
                                        'confidence': confidence,
                                        'class': class_id,
                                        'class_name': ['cat', 'chicken', 'cow', 'dog', 'horse'][class_id],
                                        'model': f'{model_name}_yolo_track_{aug_name}',
                                        'tta_weight': self._get_tta_weight(aug_name),
                                        'track_id': track_id,  # ID de tracking de YOLO
                                        'tracked': track_id is not None  # Bandera de tracking
                                    }
                                    all_predictions.append(detection)
                                    
                except Exception as e:
                    print(f"⚠️ Error en predicción {model_name}-{aug_name}: {e}")
                    continue
        
        # Post-procesamiento avanzado
        final_predictions = self._advanced_post_processing(all_predictions)
        
        print(f"🔍 TTA Ensemble: {len(all_predictions)} predicciones brutas -> {len(final_predictions)} finales")
        
        return final_predictions
    
    def _generate_tta_images(self, image):
        """Generar variaciones de la imagen para TTA"""
        height, width = image.shape[:2]
        
        augmentations = {
            'original': image,
            'flip_horizontal': cv2.flip(image, 1),
            'brightness_up': cv2.convertScaleAbs(image, alpha=1.2, beta=20),
            'brightness_down': cv2.convertScaleAbs(image, alpha=0.8, beta=-20),
            'contrast_up': cv2.convertScaleAbs(image, alpha=1.3, beta=0),
        }
        
        # Escala múltiple si la imagen es grande
        if width > 800 or height > 600:
            scale_90 = cv2.resize(image, (int(width*0.9), int(height*0.9)))
            augmentations['scale_90'] = cv2.resize(scale_90, (width, height))
        
        return augmentations
    
    def _adjust_box_coordinates(self, box, aug_name, original_shape):
        """Ajustar coordenadas según el tipo de augmentación"""
        x1, y1, x2, y2 = box
        height, width = original_shape[:2]
        
        if aug_name == 'flip_horizontal':
            # Invertir coordenadas X
            new_x1 = width - x2
            new_x2 = width - x1
            return [new_x1, y1, new_x2, y2]
        
        # Para otras augmentaciones, mantener coordenadas originales
        return box
    
    def _get_tta_weight(self, aug_name):
        """Peso para cada tipo de augmentación"""
        weights = {
            'original': 1.0,
            'flip_horizontal': 0.8,
            'brightness_up': 0.6,
            'brightness_down': 0.6,
            'contrast_up': 0.7,
            'scale_90': 0.8
        }
        return weights.get(aug_name, 0.5)
    
    def _advanced_post_processing(self, predictions):
        """Post-procesamiento avanzado con weighted ensemble y NMS"""
        if not predictions:
            return []
        
        # Agrupar por clase
        class_predictions = {}
        for pred in predictions:
            class_name = pred['class_name']
            if class_name not in class_predictions:
                class_predictions[class_name] = []
            class_predictions[class_name].append(pred)
        
        final_predictions = []
        
        # Procesar cada clase por separado
        for class_name, class_preds in class_predictions.items():
            # Aplicar weighted ensemble
            weighted_preds = self._apply_weighted_ensemble(class_preds)
            
            # Aplicar NMS avanzado
            nms_preds = self._advanced_nms(weighted_preds, iou_threshold=0.4)
            
            # Filtros específicos por clase
            filtered_preds = self._apply_class_specific_filters(nms_preds, class_name)
            
            final_predictions.extend(filtered_preds)
        
        # Limitar número total de detecciones
        final_predictions = sorted(final_predictions, key=lambda x: x['confidence'], reverse=True)[:20]
        
        return final_predictions
    
    def _apply_weighted_ensemble(self, predictions):
        """Aplicar ensemble ponderado"""
        if len(predictions) <= 1:
            return predictions
            
        # Agrupar predicciones similares por IoU
        groups = []
        used = set()
        
        for i, pred in enumerate(predictions):
            if i in used:
                continue
                
            group = [pred]
            for j, other_pred in enumerate(predictions[i+1:], i+1):
                if j in used:
                    continue
                    
                iou = self._calculate_iou(pred['box'], other_pred['box'])
                if iou > 0.3:  # Threshold para agrupar
                    group.append(other_pred)
                    used.add(j)
            
            groups.append(group)
            used.add(i)
        
        # Crear predicción promediada para cada grupo
        ensemble_predictions = []
        for group in groups:
            if len(group) == 1:
                ensemble_predictions.append(group[0])
            else:
                # Promedio ponderado
                total_weight = sum(pred['tta_weight'] for pred in group)
                avg_confidence = sum(pred['confidence'] * pred['tta_weight'] for pred in group) / total_weight
                
                # Promedio de cajas bounding
                avg_box = [0, 0, 0, 0]
                for i in range(4):
                    avg_box[i] = sum(pred['box'][i] * pred['tta_weight'] for pred in group) / total_weight
                
                ensemble_pred = {
                    'box': avg_box,
                    'confidence': avg_confidence,
                    'class': group[0]['class'],
                    'class_name': group[0]['class_name'],
                    'model': 'ensemble_tta',
                    'ensemble_size': len(group)
                }
                ensemble_predictions.append(ensemble_pred)
        
        return ensemble_predictions
    
    def _advanced_nms(self, predictions, iou_threshold=0.4):
        """NMS avanzado con soft-NMS"""
        if len(predictions) <= 1:
            return predictions
            
        # Ordenar por confianza
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        final_preds = []
        
        while predictions:
            best_pred = predictions.pop(0)
            final_preds.append(best_pred)
            
            # Filtrar predicciones que se superponen mucho
            filtered_preds = []
            for pred in predictions:
                iou = self._calculate_iou(best_pred['box'], pred['box'])
                if iou < iou_threshold:
                    filtered_preds.append(pred)
                elif iou < 0.7:  # Soft suppression
                    pred['confidence'] *= (1.0 - iou)
                    if pred['confidence'] > 0.1:
                        filtered_preds.append(pred)
            
            predictions = filtered_preds
        
        return final_preds
    
    def _apply_class_specific_filters(self, predictions, class_name):
        """Filtros específicos por clase"""
        filtered = []
        
        # Umbrales específicos mejorados
        thresholds = {
            'cat': 0.4,     # Gatos: menos estricto
            'dog': 0.4,     # Perros: menos estricto
            'chicken': 0.3, # Gallinas: menos estricto (más difíciles)
            'cow': 0.3,     # Vacas: menos estricto
            'horse': 0.3    # Caballos: menos estricto
        }
        
        min_confidence = thresholds.get(class_name, 0.3)
        
        for pred in predictions:
            if pred['confidence'] >= min_confidence:
                # Filtros adicionales por tamaño
                x1, y1, x2, y2 = pred['box']
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Tamaños mínimos más permisivos
                if width >= 20 and height >= 20 and area >= 400:
                    # Ratio de aspecto más permisivo
                    aspect_ratio = width / height
                    if 0.1 <= aspect_ratio <= 10.0:
                        filtered.append(pred)
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """Calcular IoU entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersección
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Unión
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

# Función de prueba
def test_enhanced_model():
    """Probar el modelo mejorado"""
    print("🧪 PROBANDO MODELO MEJORADO...")
    
    handler = EnhancedModelHandler()
    
    if handler.load_models():
        print("✅ Modelos cargados exitosamente")
        
        # Imagen de prueba
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Predicción mejorada
        predictions = handler.predict_with_tta(test_image, confidence_threshold=0.5)
        
        print(f"🎯 Predicciones finales: {len(predictions)}")
        for i, pred in enumerate(predictions):
            print(f"   {i+1}. {pred['class_name']}: {pred['confidence']:.3f} ({pred.get('model', 'unknown')})")
        
        return True
    else:
        print("❌ No se pudieron cargar los modelos")
        return False

if __name__ == "__main__":
    test_enhanced_model()