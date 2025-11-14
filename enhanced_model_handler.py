#!/usr/bin/env python3
"""
Handler para modelo especializado yolov8n.pt con filtros COCO
Usa yolov8n.pt con filtrado por clases COCO: car(2), cow(19), horse(17), dog(16)
Incluye: TTA, mejor post-procesamiento
"""

import cv2
import numpy as np
import torch
import os
import time
from collections import deque
from ultralytics import YOLO
from pathlib import Path
from config import Config

# Intentar cargar configuración de modelos especializados sin depender de sys.path
import importlib.util

# Importar configuración de modelos especializados
COCO_CLASS_MAPPING = {
    'car': 2,      # COCO class 2 = car
    'cow': 19,     # COCO class 19 = cow  
    'horse': 17,   # COCO class 17 = horse
    'dog': 16      # COCO class 16 = dog
}


class EnhancedModelHandler:
    def __init__(self):
        """Inicializar handler con modelo entrenado bestnov.pt"""
        self.model = None
        self.model_loaded = False
        self.model_class_names = {}  # Diccionario ID -> nombre de clase del modelo
        self.model_class_list = []   # Lista ordenada de nombres de clases
        
        # Flag para indicar si es modelo entrenado (bestnov.pt) o COCO (yolov8n.pt)
        self.is_trained_model = True  # bestnov.pt es un modelo entrenado
        
        # Clases COCO (solo para fallback si se usa yolov8n.pt)
        self.coco_class_ids = list(COCO_CLASS_MAPPING.values())  # [2, 16, 17, 19]
        self.coco_to_name = {v: k for k, v in COCO_CLASS_MAPPING.items()}  # {2: 'car', 16: 'dog', 17: 'horse', 19: 'cow'}
        
        # Filtro temporal: historial de detecciones por frame
        self.temporal_history = deque(maxlen=Config.TEMPORAL_HISTORY_SIZE)
        
        # Estadísticas de depuración
        self.debug_stats = {
            'inference_time': [],
            'nms_time': [],
            'temporal_time': [],
            'total_time': []
        }
        
        # Fix para PyTorch 2.6 weights_only issue
        self._fix_torch_load()
        
        # FORZAR CPU - Deshabilitar CUDA completamente para Mac
        self._force_cpu_mode()
        
        # Sistema robusto de detección de modelos - PORTABLE
        self.base_dir = Path(__file__).parent.absolute()
        self.models_dir = self.base_dir / "models"
        
        # Asegurar que la carpeta models existe
        self.models_dir.mkdir(exist_ok=True)
        
        # Configuración del modelo entrenado bestnov.pt (prioridad)
        self.model_path = self.base_dir / 'models' / 'bestnov.pt'
        self.fallback_path = self.base_dir / 'models' / 'yolov8n.pt'  # Fallback a COCO si no existe bestnov.pt
        
        # Detectar modelo disponible
        self.model_path_final = self._discover_model()
        
    def _fix_torch_load(self):
        """Fix para el problema de PyTorch 2.6 weights_only"""
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            # Forzar weights_only=False para compatibilidad
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        print("PyTorch load fix aplicado")
    
    def _force_cpu_mode(self):
        """Forzar modo CPU y deshabilitar CUDA completamente"""
        import os
        import torch
        
        # Deshabilitar CUDA en variables de entorno
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        print("Modo CPU forzado - CUDA deshabilitado para compatibilidad Mac")
        print(f"   - torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"   - torch.cuda.device_count(): {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            print("CUDA aún detectado, pero se usará CPU forzosamente")

    def _discover_model(self):
        """Detectar el modelo bestnov.pt (entrenado) con fallback a yolov8n.pt (COCO)"""
        print("\n🔍 Buscando modelo bestnov.pt (entrenado)...")
        print(f"   - Ruta principal: {self.model_path}")
        
        # 1. Verificar ruta principal (bestnov.pt - modelo entrenado)
        if self.model_path.exists():
            print(f"   ✅ Modelo entrenado encontrado: {self.model_path}")
            self.is_trained_model = True
            return str(self.model_path)
        
        # 2. Verificar fallback (yolov8n.pt - COCO con filtros)
        if self.fallback_path.exists():
            print(f"   ⚠️  Usando fallback COCO: {self.fallback_path}")
            print(f"   - Clases COCO: {self.coco_class_ids} ({list(COCO_CLASS_MAPPING.keys())})")
            self.is_trained_model = False
            return str(self.fallback_path)
        
        # 3. Error si no se encuentra ningún modelo
        print(f"   ❌ ERROR: No se encontró bestnov.pt ni yolov8n.pt")
        print(f"   Verificar que el archivo existe en: {self.model_path}")
        return None

    def load_models(self):
        """Cargar el modelo entrenado bestnov.pt o fallback a yolov8n.pt (COCO)"""
        if self.is_trained_model:
            print("\n📦 Cargando modelo entrenado bestnov.pt...")
        else:
            print("\n📦 Cargando modelo yolov8n.pt (especializado con filtros COCO)...")
        
        if not self.model_path_final:
            print("❌ FATAL: No se encontró el modelo")
            print(f"   Verificar que el archivo existe en: {self.model_path}")
            return False
        
        try:
            print(f"   📂 Ruta: {self.model_path_final}")
            
            # Verificar que el archivo existe
            if not os.path.exists(self.model_path_final):
                print(f"   ❌ Archivo no existe en el momento de carga")
                return False
            
            # Cargar modelo
            self.model = YOLO(self.model_path_final)
            
            # Verificar que el modelo se cargó correctamente
            if not hasattr(self.model, 'model') or self.model.model is None:
                print(f"   ❌ Modelo cargado pero estructura inválida")
                return False
            
            # Información del modelo y extraer clases
            if hasattr(self.model.model, 'names'):
                model_names = self.model.model.names
                if model_names:
                    # Para modelo entrenado: usar todas las clases del modelo
                    # Para modelo COCO: filtrar solo las clases deseadas
                    if self.is_trained_model:
                        # Modelo entrenado: usar todas las clases que tiene
                        self.model_class_names = {int(k): str(v) for k, v in model_names.items()}
                        self.model_class_list = [self.model_class_names[i] for i in sorted(self.model_class_names.keys())]
                        
                        print(f"   ✅ Modelo entrenado cargado exitosamente")
                        print(f"   📊 Total de clases: {len(self.model_class_names)}")
                        print(f"   🎯 Clases del modelo entrenado:")
                        for class_id, class_name in sorted(self.model_class_names.items()):
                            print(f"      - ID {class_id}: {class_name}")
                    else:
                        # Modelo COCO: filtrar solo las clases deseadas
                        all_coco_classes = {int(k): str(v) for k, v in model_names.items()}
                        self.model_class_names = {}
                        for coco_id in self.coco_class_ids:
                            if coco_id in all_coco_classes:
                                normalized_name = self.coco_to_name.get(coco_id, all_coco_classes[coco_id])
                                self.model_class_names[coco_id] = normalized_name
                        
                        self.model_class_list = [self.model_class_names[i] for i in sorted(self.model_class_names.keys())]
                        
                        print(f"   ✅ Modelo COCO cargado exitosamente")
                        print(f"   📊 Clases COCO totales: {len(all_coco_classes)}")
                        print(f"   🎯 Clases filtradas (especializadas): {len(self.model_class_names)}")
                        print(f"   📝 Mapeo COCO -> Normalizado:")
                        for coco_id, normalized_name in sorted(self.model_class_names.items()):
                            coco_name = all_coco_classes.get(coco_id, 'unknown')
                            print(f"      - COCO ID {coco_id} ({coco_name}) -> {normalized_name}")
                else:
                    self.model_class_names = {}
                    self.model_class_list = []
                    print(f"   ⚠️  Modelo sin información de clases")
            else:
                self.model_class_names = {}
                self.model_class_list = []
                print(f"   ⚠️  Modelo sin atributo 'names'")
            
            self.model_loaded = True
            if self.is_trained_model:
                print(f"\n✅ ¡Modelo bestnov.pt entrenado cargado exitosamente!")
                print(f"   🚀 Sistema: Modelo entrenado personalizado")
                print(f"   🎯 Clases detectadas: {', '.join(sorted(self.model_class_list))}")
            else:
                print(f"\n✅ ¡Modelo yolov8n.pt especializado cargado exitosamente!")
                print(f"   🚀 Sistema: YOLOv8n + Filtros COCO")
                print(f"   🎯 Clases detectadas: {', '.join(sorted(self.model_class_list))}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error cargando modelo: {e}")
            print(f"   Tipo de error: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_single_image(self, image, use_tta=True):
        """
        Predicción para imágenes estáticas (offline).
        Usa TTA opcionalmente según configuración.
        """
        if not self.model_loaded or self.model is None:
            return []
        
        start_time = time.time()
        
        # Usar TTA solo si está habilitado y se solicita
        if use_tta and Config.IMAGE_USE_TTA:
            return self.predict_with_tta(image, Config.IMAGE_CONFIDENCE_THRESHOLD)
        else:
            # Predicción simple sin TTA
            # Usar tamaño de imagen mayor para mejor detección (más lento pero más preciso)
            predict_kwargs = {
                'conf': Config.IMAGE_CONFIDENCE_THRESHOLD,
                'iou': Config.IMAGE_IOU_THRESHOLD,
                'imgsz': Config.IMAGE_SIZE,  # Usar tamaño configurado (1280 para mejor detección)
                'device': 'cpu',
                'half': False,
                'verbose': False,
                'max_det': 300,  # Aumentado de 100 a 300 para capturar más detecciones
            }
            
            # Solo filtrar por clases COCO si es modelo COCO (no modelo entrenado)
            if not self.is_trained_model:
                predict_kwargs['classes'] = self.coco_class_ids  # FILTRO: Solo clases COCO deseadas [2, 16, 17, 19]
            
            results = self.model.predict(image, **predict_kwargs)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Procesar todas las clases del modelo (entrenado) o solo COCO filtradas
                        if class_id in self.model_class_names:
                            class_name = self.model_class_names.get(class_id, f'class_{class_id}')
                            model_name = 'bestnov.pt (entrenado)' if self.is_trained_model else 'yolov8n.pt (COCO specialized)'
                            detections.append({
                                'box': box.xyxy[0].cpu().numpy().tolist(),
                                'confidence': confidence,
                                'class': class_id,
                                'class_name': class_name,
                                'model': model_name
                            })
            
            # Aplicar filtros geométricos
            detections = self._apply_geometric_filters(detections, image.shape)
            
            # Aplicar NMS mejorado
            detections = self._soft_nms(detections, Config.IMAGE_IOU_THRESHOLD)
            
            # Resolver conflictos entre especies (DOG vs COW/HORSE)
            detections = self._resolve_species_conflicts(detections)
            
            # Aplicar NMS adicional después de resolver conflictos para eliminar superposiciones restantes
            detections = self._soft_nms(detections, Config.IMAGE_IOU_THRESHOLD)
            
            # Aplicar filtro de exclusión mutua vaca/perro (legacy, complementa _resolve_species_conflicts)
            detections = self._apply_cow_dog_exclusion_filter(detections)
            
            inference_time = time.time() - start_time
            if Config.DEBUG_LOGS_ENABLED:
                print(f"📸 Imagen: {len(detections)} detecciones en {inference_time*1000:.1f}ms")
            
            return detections
    
    def predict_with_tta(self, image, confidence_threshold=0.5):
        """Predicción con Test Time Augmentation - SOLO para imágenes estáticas"""
        if not self.model_loaded or self.model is None:
            return []
            
        all_predictions = []
        
        # Generar variaciones de la imagen para TTA
        augmented_images = self._generate_tta_images(image)
        
        # Predecir con el modelo único en cada variación TTA
        for aug_name, aug_image in augmented_images.items():
            try:
                # NO usar tracking en TTA (solo para imágenes estáticas)
                # Usar tamaño de imagen mayor y más detecciones para mejor precisión
                predict_kwargs = {
                    'conf': max(0.15, confidence_threshold - 0.15),  # Reducir threshold más agresivamente para TTA
                    'iou': Config.IMAGE_IOU_THRESHOLD,
                    'imgsz': Config.IMAGE_SIZE,  # Usar tamaño configurado (1280 para mejor detección)
                    'device': 'cpu',
                    'half': False,
                    'verbose': False,
                    'max_det': 300,  # Aumentado de 100 a 300 para capturar más detecciones
                }
                
                # Solo filtrar por clases COCO si es modelo COCO (no modelo entrenado)
                if not self.is_trained_model:
                    predict_kwargs['classes'] = self.coco_class_ids  # FILTRO: Solo clases COCO deseadas [2, 16, 17, 19]
                
                results = self.model.predict(aug_image, **predict_kwargs)
                
                # Procesar resultados
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            
                            # Procesar todas las clases del modelo
                            if class_id in self.model_class_names:
                                # Ajustar coordenadas según la augmentación
                                adjusted_box = self._adjust_box_coordinates(
                                    box.xyxy[0].cpu().numpy().tolist(),
                                    aug_name,
                                    image.shape
                                )
                                
                                # Obtener nombre de clase del modelo
                                class_name = self.model_class_names.get(class_id, f'class_{class_id}')
                                model_name = 'bestnov.pt (entrenado)' if self.is_trained_model else 'yolov8n.pt (COCO specialized)'
                                
                                detection = {
                                    'box': adjusted_box,
                                    'confidence': confidence,
                                    'class': class_id,
                                    'class_name': class_name,
                                    'model': model_name,
                                    'tta_weight': self._get_tta_weight(aug_name)
                                }
                                all_predictions.append(detection)
                                
            except Exception as e:
                print(f"Error en predicción TTA {aug_name}: {e}")
                continue
        
        # Post-procesamiento avanzado
        final_predictions = self._advanced_post_processing(all_predictions)
        
        print(f"TTA: {len(all_predictions)} predicciones brutas -> {len(final_predictions)} finales")
        
        return final_predictions
    
    def _generate_tta_images(self, image):
        """Generar variaciones de la imagen para TTA - MEJORADO para mejor detección"""
        height, width = image.shape[:2]
        
        augmentations = {
            'original': image,
            'flip_horizontal': cv2.flip(image, 1),
            'brightness_up': cv2.convertScaleAbs(image, alpha=1.3, beta=30),  # Más agresivo
            'brightness_down': cv2.convertScaleAbs(image, alpha=0.7, beta=-30),  # Más agresivo
            'contrast_up': cv2.convertScaleAbs(image, alpha=1.4, beta=0),  # Más contraste
            'contrast_down': cv2.convertScaleAbs(image, alpha=0.8, beta=0),  # Menos contraste
        }
        
        # Escalas múltiples para mejor detección de objetos de diferentes tamaños
        # Escala 110% (zoom in)
        scale_110 = cv2.resize(image, (int(width*1.1), int(height*1.1)))
        augmentations['scale_110'] = cv2.resize(scale_110, (width, height))
        
        # Escala 90% (zoom out)
        scale_90 = cv2.resize(image, (int(width*0.9), int(height*0.9)))
        augmentations['scale_90'] = cv2.resize(scale_90, (width, height))
        
        # Escala 80% (más zoom out para objetos grandes)
        scale_80 = cv2.resize(image, (int(width*0.8), int(height*0.8)))
        augmentations['scale_80'] = cv2.resize(scale_80, (width, height))
        
        # Rotación ligera (±5 grados) para mejor detección
        center = (width // 2, height // 2)
        rotation_matrix_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
        augmentations['rotate_5'] = cv2.warpAffine(image, rotation_matrix_5, (width, height))
        
        rotation_matrix_minus5 = cv2.getRotationMatrix2D(center, -5, 1.0)
        augmentations['rotate_minus5'] = cv2.warpAffine(image, rotation_matrix_minus5, (width, height))
        
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
        
        # Para escalas, las coordenadas ya están ajustadas porque la imagen se redimensionó
        # Para rotaciones, las coordenadas también están ajustadas por la transformación
        # Para otras augmentaciones (brightness, contrast), mantener coordenadas originales
        return box
    
    def _get_tta_weight(self, aug_name):
        """Peso para cada tipo de augmentación - AJUSTADO para mejor detección"""
        weights = {
            'original': 1.0,  # Peso máximo para imagen original
            'flip_horizontal': 0.85,  # Aumentado - flip es muy útil
            'brightness_up': 0.7,  # Aumentado
            'brightness_down': 0.7,  # Aumentado
            'contrast_up': 0.75,  # Aumentado
            'contrast_down': 0.7,  # Nuevo
            'scale_110': 0.8,  # Escala zoom in
            'scale_90': 0.75,  # Escala zoom out
            'scale_80': 0.7,  # Escala más zoom out
            'rotate_5': 0.65,  # Rotación ligera
            'rotate_minus5': 0.65  # Rotación ligera inversa
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
        
        # Aplicar filtro de exclusión mutua vaca/perro para reducir falsos positivos
        final_predictions = self._apply_cow_dog_exclusion_filter(final_predictions)
        
        # Limitar número total de detecciones - AUMENTADO para mejor recall
        final_predictions = sorted(final_predictions, key=lambda x: x['confidence'], reverse=True)[:50]  # Aumentado de 20 a 50
        
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
                
                model_name = 'bestnov.pt (entrenado)' if self.is_trained_model else 'yolov8n.pt (COCO specialized)'
                ensemble_pred = {
                    'box': avg_box,
                    'confidence': avg_confidence,
                    'class': group[0]['class'],
                    'class_name': group[0]['class_name'],
                    'model': model_name,
                    'tta_size': len(group)
                }
                ensemble_predictions.append(ensemble_pred)
        
        return ensemble_predictions
    
    def _advanced_nms(self, predictions, iou_threshold=0.4):
        """NMS avanzado con soft-NMS - MÁS ESTRICTO para eliminar superposiciones"""
        if len(predictions) <= 1:
            return predictions
            
        # Ordenar por confianza
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        final_preds = []
        
        while predictions:
            best_pred = predictions.pop(0)
            final_preds.append(best_pred)
            
            # Filtrar predicciones que se superponen mucho - MÁS ESTRICTO
            filtered_preds = []
            for pred in predictions:
                iou = self._calculate_iou(best_pred['box'], pred['box'])
                same_class = (best_pred.get('class') == pred.get('class'))
                
                # Si es la misma clase y IoU >= 0.35, eliminar duplicado
                if same_class and iou >= 0.35:  # Más estricto para misma clase
                    if Config.DEBUG_LOGS_ENABLED:
                        print(f"  🗑️  Advanced NMS: eliminado duplicado {pred.get('class_name', 'unknown')} (IOU={iou:.2f})")
                    continue
                
                # Si IoU < umbral, mantener
                if iou < iou_threshold:
                    filtered_preds.append(pred)
                # Si IoU >= umbral pero < 0.6: soft suppression más agresiva
                elif iou < 0.6:  # Reducido de 0.7 a 0.6
                    pred['confidence'] *= (1.0 - iou * 0.8)  # Reducción más agresiva
                    if pred['confidence'] > 0.15:  # Aumentado de 0.1 a 0.15
                        filtered_preds.append(pred)
                # Si IoU >= 0.6: eliminar directamente
                else:
                    if Config.DEBUG_LOGS_ENABLED:
                        print(f"  🗑️  Advanced NMS: eliminado por superposición alta (IOU={iou:.2f}): {pred.get('class_name', 'unknown')}")
            
            predictions = filtered_preds
        
        return final_preds
    
    def _apply_class_specific_filters(self, predictions, class_name):
        """Filtros específicos por clase - para imágenes estáticas con TTA"""
        filtered = []
        
        # Umbrales específicos mejorados - AJUSTADOS para mejor detección de perros
        # Dataset: Auto, Caballo, Perro, Vaca
        thresholds = {
            'Auto': 0.20,      # Autos: más sensible
            'auto': 0.20,      # Variante en minúsculas
            'car': 0.20,       # Variante en inglés
            'Caballo': 0.25,   # Caballos: umbral medio
            'caballo': 0.25,   # Variante en minúsculas
            'horse': 0.25,     # Variante en inglés
            'Perro': 0.30,     # Perros: umbral medio-alto para mejor precisión
            'perro': 0.30,     # Variante en minúsculas
            'dog': 0.30,       # Variante en inglés
            'Vaca': 0.70,      # Vacas: UMBRAL ALTO para reducir falsos positivos
            'vaca': 0.70,      # Variante en minúsculas
            'cow': 0.70        # Variante en inglés - SOLO marcar vacas con alta confianza
        }
        
        min_confidence = thresholds.get(class_name, 0.3)
        
        for pred in predictions:
            if pred['confidence'] >= min_confidence:
                # Filtros adicionales por tamaño
                x1, y1, x2, y2 = pred['box']
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Tamaños mínimos más permisivos - REDUCIDOS para capturar objetos más pequeños
                if width >= 15 and height >= 15 and area >= 225:  # Reducido de 20x20/400 a 15x15/225
                    # Ratio de aspecto más permisivo
                    aspect_ratio = width / height
                    if 0.1 <= aspect_ratio <= 10.0:
                        filtered.append(pred)
        
        return filtered
    
    def _resolve_species_conflicts(self, detections):
        """
        Resolver conflictos entre especies: DOG vs {COW, HORSE}
        Reglas:
        - Si IoU > 0.45 entre DOG y {COW, HORSE}:
          * Si área_dog <= 0.55*área_other y conf_dog >= conf_other - 0.10 → favorecer DOG (degradar otra)
          * Si conf_other >= conf_dog + 0.20 → degradar DOG (evita dog fantasma)
        """
        if not detections:
            return detections
        
        # Separar detecciones por clase
        dog_detections = []
        cow_detections = []
        horse_detections = []
        other_detections = []
        
        for det in detections:
            class_name = det.get('class_name', '').lower()
            if class_name in ['dog', 'perro']:
                dog_detections.append(det)
            elif class_name in ['cow', 'vaca']:
                cow_detections.append(det)
            elif class_name in ['horse', 'caballo']:
                horse_detections.append(det)
            else:
                other_detections.append(det)
        
        # Si no hay perros o no hay vacas/caballos, no hay conflicto
        if not dog_detections or (not cow_detections and not horse_detections):
            return detections
        
        # Procesar conflictos DOG vs COW
        filtered_dogs = []
        filtered_cows = []
        
        for dog_det in dog_detections:
            dog_box = dog_det['box']
            dog_conf = dog_det['confidence']
            dog_area = (dog_box[2] - dog_box[0]) * (dog_box[3] - dog_box[1])
            dog_degraded = False
            
            # Verificar conflictos con vacas
            for cow_det in cow_detections[:]:  # Copia para poder modificar
                cow_box = cow_det['box']
                cow_conf = cow_det['confidence']
                cow_area = (cow_box[2] - cow_box[0]) * (cow_box[3] - cow_box[1])
                
                iou = self._calculate_iou(dog_box, cow_box)
                
                if iou > 0.45:  # Superposición significativa
                    # Regla 1: Favorecer DOG si es más pequeño y tiene confianza similar
                    if dog_area <= 0.55 * cow_area and dog_conf >= cow_conf - 0.10:
                        # Degradar vaca (reducir confianza significativamente)
                        cow_det['confidence'] *= 0.3
                        if Config.DEBUG_LOGS_ENABLED:
                            print(f"  🐕 Favorecido DOG sobre COW: dog_conf={dog_conf:.3f}, cow_conf={cow_conf:.3f}→{cow_det['confidence']:.3f}, IoU={iou:.2f}, área_dog={dog_area:.0f} <= 0.55*área_cow={0.55*cow_area:.0f}")
                    # Regla 2: Si vaca tiene confianza mucho mayor, degradar perro
                    elif cow_conf >= dog_conf + 0.20:
                        dog_degraded = True
                        dog_det['confidence'] *= 0.3
                        if Config.DEBUG_LOGS_ENABLED:
                            print(f"  🐄 Favorecido COW sobre DOG: cow_conf={cow_conf:.3f} >= dog_conf={dog_conf:.3f}+0.20, IoU={iou:.2f}")
            
            # Verificar conflictos con caballos
            for horse_det in horse_detections[:]:  # Copia para poder modificar
                horse_box = horse_det['box']
                horse_conf = horse_det['confidence']
                horse_area = (horse_box[2] - horse_box[0]) * (horse_box[3] - horse_box[1])
                
                iou = self._calculate_iou(dog_box, horse_box)
                
                if iou > 0.45:  # Superposición significativa
                    # Regla 1: Favorecer DOG si es más pequeño y tiene confianza similar
                    if dog_area <= 0.55 * horse_area and dog_conf >= horse_conf - 0.10:
                        # Degradar caballo (reducir confianza significativamente)
                        horse_det['confidence'] *= 0.3
                        if Config.DEBUG_LOGS_ENABLED:
                            print(f"  🐕 Favorecido DOG sobre HORSE: dog_conf={dog_conf:.3f}, horse_conf={horse_conf:.3f}→{horse_det['confidence']:.3f}, IoU={iou:.2f}, área_dog={dog_area:.0f} <= 0.55*área_horse={0.55*horse_area:.0f}")
                    # Regla 2: Si caballo tiene confianza mucho mayor, degradar perro
                    elif horse_conf >= dog_conf + 0.20:
                        dog_degraded = True
                        dog_det['confidence'] *= 0.3
                        if Config.DEBUG_LOGS_ENABLED:
                            print(f"  🐴 Favorecido HORSE sobre DOG: horse_conf={horse_conf:.3f} >= dog_conf={dog_conf:.3f}+0.20, IoU={iou:.2f}")
            
            if not dog_degraded:
                filtered_dogs.append(dog_det)
            elif dog_det['confidence'] > 0.1:  # Mantener si aún tiene confianza mínima
                filtered_dogs.append(dog_det)
        
        # Filtrar vacas y caballos degradados (confianza muy baja)
        filtered_cows = [cow for cow in cow_detections if cow['confidence'] > 0.1]
        filtered_horses = [horse for horse in horse_detections if horse['confidence'] > 0.1]
        
        # Combinar todas las detecciones
        return filtered_dogs + filtered_cows + filtered_horses + other_detections
    
    def _apply_cow_dog_exclusion_filter(self, predictions):
        """
        Filtro de exclusión mutua: Si hay un perro con alta confianza en la misma área,
        eliminar o reducir la confianza de detecciones de vaca cercanas.
        Esto reduce falsos positivos de vacas detectadas en perros.
        """
        if not predictions:
            return predictions
        
        # Separar detecciones de perros y vacas
        dog_detections = []
        cow_detections = []
        other_detections = []
        
        for pred in predictions:
            class_name = pred['class_name'].lower()
            if class_name in ['dog', 'perro']:
                dog_detections.append(pred)
            elif class_name in ['cow', 'vaca']:
                cow_detections.append(pred)
            else:
                other_detections.append(pred)
        
        # Si no hay vacas o no hay perros, no hay conflicto
        if not cow_detections or not dog_detections:
            return predictions
        
        # Filtrar vacas que están cerca de perros con alta confianza
        filtered_cows = []
        for cow_pred in cow_detections:
            cow_box = cow_pred['box']
            cow_conf = cow_pred['confidence']
            
            # Buscar perros cercanos con alta confianza
            has_nearby_high_conf_dog = False
            for dog_pred in dog_detections:
                dog_box = dog_pred['box']
                dog_conf = dog_pred['confidence']
                
                # Calcular IoU entre vaca y perro
                iou = self._calculate_iou(cow_box, dog_box)
                
                # Si hay superposición significativa (IoU > 0.3) y el perro tiene alta confianza (>0.5)
                # entonces es probable que la vaca sea un falso positivo
                if iou > 0.3 and dog_conf > 0.5:
                    # Si la confianza de la vaca es menor que la del perro, es muy probable que sea falso positivo
                    if cow_conf < dog_conf + 0.1:  # Vaca con confianza similar o menor al perro
                        has_nearby_high_conf_dog = True
                        if Config.DEBUG_LOGS_ENABLED:
                            print(f"  🚫 Filtro vaca/perro: Eliminada vaca (conf={cow_conf:.3f}) cerca de perro (conf={dog_conf:.3f}, IoU={iou:.2f})")
                        break
            
            # Solo mantener vacas que no están cerca de perros con alta confianza
            # O vacas con confianza muy alta (>0.75) que probablemente son reales
            if not has_nearby_high_conf_dog or cow_conf > 0.75:
                filtered_cows.append(cow_pred)
        
        # Combinar todas las detecciones filtradas
        return dog_detections + filtered_cows + other_detections
    
    def _apply_geometric_filters(self, detections, image_shape):
        """
        Aplicar filtros geométricos con mínimos por clase para mejor detección de perros:
        - dog: 12x12/144, horse: 22x22/484, cow: 24x24/576, car: 20x20/400
        - Aspect ratio: 0.25 - 6.0
        """
        if not detections:
            return []
        
        filtered = []
        h, w = image_shape[:2]
        
        # Mínimos por clase (width, height, area)
        class_minimums = {
            'dog': (12, 12, 144),
            'perro': (12, 12, 144),
            'horse': (22, 22, 484),
            'caballo': (22, 22, 484),
            'cow': (24, 24, 576),
            'vaca': (24, 24, 576),
            'car': (20, 20, 400),
            'auto': (20, 20, 400)
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Obtener mínimos por clase o usar valores por defecto
            class_name_lower = det.get('class_name', '').lower()
            min_width, min_height, min_area = class_minimums.get(
                class_name_lower,
                (Config.MIN_BOX_WIDTH, Config.MIN_BOX_HEIGHT, Config.MIN_BOX_AREA)
            )
            
            # Filtro de tamaño mínimo por clase
            if width < min_width or height < min_height:
                if Config.DEBUG_LOGS_ENABLED:
                    print(f"  ❌ Rechazado por tamaño: {width:.0f}x{height:.0f} < {min_width}x{min_height} (clase: {class_name_lower})")
                continue
            
            # Filtro de área mínima por clase
            if area < min_area:
                if Config.DEBUG_LOGS_ENABLED:
                    print(f"  ❌ Rechazado por área: {area:.0f} < {min_area} (clase: {class_name_lower})")
                continue
            
            # Filtro de aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < Config.MIN_ASPECT_RATIO or aspect_ratio > Config.MAX_ASPECT_RATIO:
                if Config.DEBUG_LOGS_ENABLED:
                    print(f"  ❌ Rechazado por aspect ratio: {aspect_ratio:.2f} fuera de [{Config.MIN_ASPECT_RATIO}, {Config.MAX_ASPECT_RATIO}]")
                continue
            
            # Filtro de posición (evitar bordes extremos)
            margin = 5
            if (x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin):
                # Solo rechazar si la confianza es baja
                if det['confidence'] < 0.5:
                    if Config.DEBUG_LOGS_ENABLED:
                        print(f"  ❌ Rechazado por posición en borde con confianza baja: {det['confidence']:.3f}")
                    continue
            
            filtered.append(det)
        
        return filtered
    
    def _soft_nms(self, detections, iou_threshold=0.5):
        """
        Soft-NMS mejorado para reducir FPs y duplicados manteniendo recall.
        - Elimina duplicados del mismo objeto (misma clase + IOU alto)
        - Usa supresión suave para detecciones parcialmente superpuestas
        - MÁS ESTRICTO para eliminar superposiciones visibles
        """
        if not detections or len(detections) <= 1:
            return detections
        
        # Ordenar por confianza descendente
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        remaining = detections.copy()
        
        while remaining:
            # Tomar la detección con mayor confianza
            best = remaining.pop(0)
            final_detections.append(best)
            
            # Aplicar soft-NMS a las restantes - MÁS ESTRICTO
            new_remaining = []
            for det in remaining:
                iou = self._calculate_iou(best['box'], det['box'])
                same_class = (best['class'] == det['class'])
                
                # REGLA 1: Si son de la misma clase y se superponen significativamente (IoU >= 0.4), eliminar duplicado
                if same_class and iou >= 0.4:  # Reducido de 0.5 a 0.4 para ser más estricto
                    # Duplicado claro: eliminar la de menor confianza
                    if Config.DEBUG_LOGS_ENABLED:
                        print(f"  🗑️  NMS: eliminado duplicado {det['class_name']} (IOU={iou:.2f}, conf={det['confidence']:.3f} < {best['confidence']:.3f})")
                    # No agregar a new_remaining (eliminado)
                    continue
                
                # REGLA 2: Superposición moderada de la misma clase (IoU >= 0.25): soft suppression más agresiva
                elif same_class and iou >= 0.25:  # Reducido de 0.3 a 0.25
                    det['confidence'] *= (1.0 - iou * 0.7)  # Reducción más agresiva (0.7 en lugar de 0.5)
                    if det['confidence'] > 0.2:  # Aumentado de 0.15 a 0.2 para ser más estricto
                        new_remaining.append(det)
                    elif Config.DEBUG_LOGS_ENABLED:
                        print(f"  ❌ Soft-NMS: eliminado {det['class_name']} por confianza baja tras supresión: {det['confidence']:.3f}")
                    continue
                
                # REGLA 3: Superposición muy alta (IoU >= 0.6) incluso de diferentes clases: eliminar
                elif iou >= 0.6:  # Reducido de 0.7 a 0.6 para eliminar más superposiciones
                    if Config.DEBUG_LOGS_ENABLED:
                        print(f"  🗑️  NMS: eliminado por superposición alta (IOU={iou:.2f}): {det['class_name']} conf={det['confidence']:.3f} vs {best['class_name']} conf={best['confidence']:.3f}")
                    continue
                
                # REGLA 4: Superposición moderada-alta (IoU >= iou_threshold y < 0.6): soft suppression
                elif iou >= iou_threshold and iou < 0.6:
                    det['confidence'] *= (1.0 - iou * 0.8)  # Reducción más agresiva
                    if det['confidence'] > 0.15:  # Mantener si aún tiene confianza mínima
                        new_remaining.append(det)
                    elif Config.DEBUG_LOGS_ENABLED:
                        print(f"  ❌ Soft-NMS: eliminado por confianza baja tras supresión: {det['confidence']:.3f}")
                else:
                    # No se superponen mucho, mantener
                    new_remaining.append(det)
            
            remaining = new_remaining
        
        return final_detections
    
    def _apply_roi_filter(self, detections, image_shape):
        """
        Aplicar filtro de ROI (Región de Interés) adaptativo.
        Mantiene detecciones en la parte inferior, pero permite excepciones:
        - Alta confianza (>ROI_HIGH_CONFIDENCE_THRESHOLD)
        - Objetos grandes (>ROI_LARGE_OBJECT_AREA)
        - Animales (dog, horse, cow) con confianza >= ROI_ANIMAL_BYPASS_CONF
        """
        if not Config.ROI_ENABLED or not detections:
            return detections
        
        h, w = image_shape[:2]
        roi_y_start = int(h * Config.ROI_Y_START)
        
        # Clases de animales que tienen bypass específico
        animal_classes = {'dog', 'perro', 'horse', 'caballo', 'cow', 'vaca'}
        
        filtered = []
        rejected_count = 0
        for det in detections:
            x1, y1, x2, y2 = det['box']
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            area = width * height
            conf = det['confidence']
            class_name_lower = det.get('class_name', '').lower()
            
            # Verificar si está en ROI
            in_roi = center_y >= roi_y_start
            
            # ROI adaptativo: permitir excepciones
            if not in_roi and Config.ROI_ADAPTIVE:
                # Excepción 1: Alta confianza
                if conf >= Config.ROI_HIGH_CONFIDENCE_THRESHOLD:
                    filtered.append(det)
                    continue
                
                # Excepción 2: Objeto grande (probablemente válido)
                if area >= Config.ROI_LARGE_OBJECT_AREA:
                    filtered.append(det)
                    continue
                
                # Excepción 3: Animales con confianza suficiente (bypass específico)
                if class_name_lower in animal_classes and conf >= Config.ROI_ANIMAL_BYPASS_CONF:
                    filtered.append(det)
                    if Config.DEBUG_LOGS_ENABLED:
                        print(f"  ✅ ROI bypass animal: {class_name_lower} (conf={conf:.3f} >= {Config.ROI_ANIMAL_BYPASS_CONF})")
                    continue
            
            # Mantener si está en ROI
            if in_roi:
                filtered.append(det)
            else:
                rejected_count += 1
                # Solo loggear si hay muchos rechazos o en frames de debug
                if Config.DEBUG_LOGS_ENABLED and rejected_count <= 2:
                    print(f"  ❌ ROI: rechazado (centro_y={center_y:.0f} < {roi_y_start}, conf={conf:.3f}, área={area:.0f})")
        
        return filtered
    
    def _apply_temporal_filter(self, detections, frame_number):
        """
        Filtro temporal: voto de persistencia.
        Requiere que una detección aparezca en ≥3 de los últimos 5 frames
        con IoU ≥ 0.3 para ser aceptada.
        """
        if not Config.TEMPORAL_FILTER_ENABLED:
            return detections
        
        if not detections:
            # Agregar frame vacío al historial
            self.temporal_history.append({
                'frame': frame_number,
                'detections': []
            })
            return []
        
        # Agregar detecciones actuales al historial
        self.temporal_history.append({
            'frame': frame_number,
            'detections': detections.copy()
        })
        
        # Necesitamos al menos TEMPORAL_HISTORY_SIZE frames para validar
        if len(self.temporal_history) < Config.TEMPORAL_HISTORY_SIZE:
            return detections
        
        # Contar votos de persistencia para cada detección actual
        validated_detections = []
        
        for current_det in detections:
            votes = 0
            total_confidence = current_det['confidence']
            
            # Buscar en el historial
            for hist_frame in self.temporal_history:
                for hist_det in hist_frame['detections']:
                    # Misma clase
                    if hist_det['class_name'] != current_det['class_name']:
                        continue
                    
                    # Calcular IoU
                    iou = self._calculate_iou(current_det['box'], hist_det['box'])
                    if iou >= Config.TEMPORAL_IOU_THRESHOLD:
                        votes += 1
                        total_confidence += hist_det['confidence']
                        break  # Solo contar una vez por frame
            
            # Promediar confianza
            avg_confidence = total_confidence / (votes + 1)
            
            # Validar si tiene suficientes votos
            if votes >= Config.TEMPORAL_VOTE_THRESHOLD:
                # Actualizar confianza promediada
                current_det['confidence'] = avg_confidence
                validated_detections.append(current_det)
                
                # Solo loggear en frames de debug para reducir ruido
                if Config.DEBUG_LOGS_ENABLED and len(validated_detections) <= 3:
                    print(f"  ✅ Temporal: aceptado {current_det['class_name']} con {votes}/{Config.TEMPORAL_HISTORY_SIZE} votos, conf={avg_confidence:.3f}")
            else:
                # Solo loggear rechazos en frames de debug
                if Config.DEBUG_LOGS_ENABLED and len([d for d in detections if d['class_name'] == current_det['class_name']]) <= 2:
                    print(f"  ❌ Temporal: rechazado {current_det['class_name']} con {votes}/{Config.TEMPORAL_HISTORY_SIZE} votos (necesita ≥{Config.TEMPORAL_VOTE_THRESHOLD})")
        
        return validated_detections
    
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
    
    def process_video_with_tracking(self, video_source, confidence_threshold=None, callback=None):
        """
        Procesa un video o stream con tracking persistente SIN TTA.
        Aplica todos los filtros: geométricos, NMS mejorado, ROI, y temporal.
        
        Args:
            video_source: Path al video o índice de cámara (0 para webcam)
            confidence_threshold: Umbral de confianza (None = usar Config.STREAMING_CONFIDENCE_THRESHOLD)
            callback: Función opcional para reportar progreso (frame_num, total_frames)
        
        Yields:
            dict: {
                'frame': numpy array del frame procesado con detecciones dibujadas,
                'detections': lista de detecciones con tracking IDs,
                'frame_number': número de frame actual
            }
        """
        if not self.model_loaded or self.model is None:
            print("❌ No hay modelo cargado para tracking")
            return
        
        # Usar threshold de configuración si no se especifica
        if confidence_threshold is None:
            confidence_threshold = Config.STREAMING_CONFIDENCE_THRESHOLD
        
        if self.is_trained_model:
            print(f"🎬 Iniciando streaming SIN TTA con bestnov.pt (modelo entrenado)")
        else:
            print(f"🎬 Iniciando streaming SIN TTA con yolov8n.pt (COCO specialized)")
        print(f"   - Confidence threshold: {confidence_threshold} (global)")
        print(f"   - IOU threshold: {Config.STREAMING_IOU_THRESHOLD}")
        print(f"   - Video source: {video_source}")
        print(f"   - Filtros: geométricos, ROI, temporal, soft-NMS")
        
        try:
            # Resetear tracker e historial temporal para iniciar un stream limpio
            self.temporal_history.clear()
            try:
                # Opción 1: Intentar resetear el tracker existente
                if hasattr(self.model, 'tracker') and self.model.tracker is not None:
                    # Limpiar todas las listas de tracks
                    if hasattr(self.model.tracker, 'tracked_tracks'):
                        self.model.tracker.tracked_tracks = []
                    if hasattr(self.model.tracker, 'lost_tracks'):
                        self.model.tracker.lost_tracks = []
                    if hasattr(self.model.tracker, 'removed_tracks'):
                        self.model.tracker.removed_tracks = []
                    # Limpiar contador de IDs si existe
                    if hasattr(self.model.tracker, '_id_counter'):
                        self.model.tracker._id_counter = 0
                    if hasattr(self.model.tracker, 'frame_id'):
                        self.model.tracker.frame_id = 0
                
                # Opción 2: Forzar recreación del tracker estableciéndolo en None
                # Esto hará que YOLO cree un nuevo tracker en la próxima llamada a track()
                self.model.tracker = None
                print("✅ Tracker reiniciado - IDs comenzarán desde 0 en el nuevo stream")
                
            except Exception as tracker_reset_error:
                print(f"⚠️  Error al reiniciar tracker: {tracker_reset_error}")
                # Último recurso: intentar recrear el tracker manualmente
                try:
                    self.model.tracker = None
                    print("✅ Tracker forzado a None - se creará uno nuevo")
                except Exception as final_error:
                    print(f"⚠️  No se pudo reiniciar el tracker: {final_error}")
            
            # NO usar TTA en streaming - usar predict directo con ByteTrack
            track_kwargs = {
                'source': video_source,
                'stream': True,          # Procesar como stream para eficiencia
                'persist': True,         # CRÍTICO: Mantiene IDs entre frames
                'tracker': "bytetrack.yaml",  # ByteTrack algorithm
                'conf': confidence_threshold,  # Usar threshold exacto (sin ajustes)
                'iou': Config.STREAMING_IOU_THRESHOLD,  # NMS más estricto
                'imgsz': 640,           # Tamaño de imagen optimizado
                'device': 'cpu',        # Forzar CPU (Mac compatible)
                'half': False,          # No half precision en CPU
                'verbose': False,       # Sin logs excesivos
                'max_det': 100,         # Máximo de detecciones
            }
            
            # Solo filtrar por clases COCO si es modelo COCO (no modelo entrenado)
            if not self.is_trained_model:
                track_kwargs['classes'] = self.coco_class_ids  # FILTRO: Solo clases COCO deseadas [2, 16, 17, 19]
            
            results_generator = self.model.track(**track_kwargs)
            
            frame_number = 0
            
            # Iterar sobre los resultados del tracking
            for results in results_generator:
                frame_start = time.time()
                frame_number += 1
                
                # Obtener frame original para procesamiento
                if hasattr(results, 'orig_img'):
                    original_frame = results.orig_img
                else:
                    # Si no hay orig_img, usar el plot (menos ideal)
                    original_frame = results.plot()
                    print("⚠️  Advertencia: usando frame plot en lugar de original")
                
                # Extraer información de las detecciones brutas
                raw_detections = []
                
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes
                    
                    for idx in range(len(boxes)):
                        # Extraer datos de cada detección
                        box = boxes.xyxy[idx].cpu().numpy().tolist()
                        conf = float(boxes.conf[idx])
                        class_id = int(boxes.cls[idx])
                        
                        # Tracking ID (si está disponible)
                        track_id = None
                        if boxes.id is not None:
                            track_id = int(boxes.id[idx])
                        
                        # Procesar todas las clases del modelo
                        if class_id in self.model_class_names:
                            class_name = self.model_class_names.get(class_id, f'class_{class_id}')
                            
                            # Obtener box primero
                            box = boxes.xyxy[idx].cpu().numpy().tolist()
                            
                            # Aplicar umbral por clase para streaming
                            class_threshold = Config.STREAMING_CLASS_THRESHOLDS.get(
                                class_name, 
                                Config.STREAMING_CONFIDENCE_THRESHOLD
                            )
                            
                            # Filtro adaptativo: REDUCIR umbral para objetos grandes (RESTAR, no sumar)
                            if Config.ADAPTIVE_SIZE_THRESHOLD:
                                width = box[2] - box[0]
                                height = box[3] - box[1]
                                area = width * height
                                
                                if area >= Config.LARGE_OBJECT_AREA:
                                    # RESTAR el bono (es negativo, así que reduce el umbral)
                                    class_threshold += Config.LARGE_OBJECT_CONFIDENCE_BONUS
                                    # Clamp al mínimo absoluto
                                    class_threshold = max(class_threshold, Config.STREAMING_MIN_CONFIDENCE)
                            
                            if conf >= class_threshold:
                                raw_detections.append({
                                    'box': box,
                                    'confidence': conf,
                                    'class': class_id,
                                    'class_name': class_name,
                                    'track_id': track_id,
                                    'tracked': track_id is not None
                                })
                
                inference_time = time.time() - frame_start
                
                # === PIPELINE DE FILTROS (en orden) ===
                # 1. Filtros geométricos
                nms_start = time.time()
                detections_after_geom = self._apply_geometric_filters(raw_detections, original_frame.shape)
                
                # 2. Soft-NMS mejorado
                detections_after_nms = self._soft_nms(detections_after_geom, Config.STREAMING_IOU_THRESHOLD)
                nms_time = time.time() - nms_start
                
                # 2.5. Resolver conflictos entre especies (DOG vs COW/HORSE)
                detections_after_conflicts = self._resolve_species_conflicts(detections_after_nms)
                
                # 2.6. Aplicar NMS adicional después de resolver conflictos para eliminar superposiciones restantes
                detections_after_conflicts_nms = self._soft_nms(detections_after_conflicts, Config.STREAMING_IOU_THRESHOLD)
                
                # 3. Filtro ROI (mitad inferior)
                detections_after_roi = self._apply_roi_filter(detections_after_conflicts_nms, original_frame.shape)
                
                # 3.5. Filtro de exclusión mutua vaca/perro (antes del filtro temporal)
                detections_after_exclusion = self._apply_cow_dog_exclusion_filter(detections_after_roi)
                
                # 4. Filtro temporal (voto de persistencia)
                temporal_start = time.time()
                detections = self._apply_temporal_filter(detections_after_exclusion, frame_number)
                temporal_time = time.time() - temporal_start
                
                # Dibujar detecciones validadas en el frame
                processed_frame = original_frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = [int(coord) for coord in det['box']]
                    class_name = det['class_name']
                    conf = det['confidence']
                    
                    # Color según clase
                    color = self._get_detection_color(class_name)
                    
                    # Dibujar bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Etiqueta con confianza
                    label = f"{class_name} {conf:.2f}"
                    if det.get('track_id') is not None:
                        label += f" [ID:{det['track_id']}]"
                    
                    # Fondo para texto
                    (label_width, label_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        processed_frame, 
                        (x1, y1 - label_height - 10), 
                        (x1 + label_width, y1), 
                        color, -1
                    )
                    cv2.putText(
                        processed_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                
                total_time = time.time() - frame_start
                
                # Logs de depuración
                if Config.DEBUG_LOGS_ENABLED and frame_number % Config.DEBUG_LOG_INTERVAL == 0:
                    print(f"\n📊 Frame {frame_number}:")
                    print(f"   Pipeline: Raw={len(raw_detections)} → Geométrico={len(detections_after_geom)} → NMS={len(detections_after_nms)} → ROI={len(detections_after_roi)} → Temporal={len(detections)}")
                    print(f"   Tiempos: infer={inference_time*1000:.1f}ms, NMS={nms_time*1000:.1f}ms, temporal={temporal_time*1000:.1f}ms, total={total_time*1000:.1f}ms")
                    if detections:
                        for det in detections:
                            x1, y1, x2, y2 = det['box']
                            area = (x2 - x1) * (y2 - y1)
                            aspect = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                            print(f"   ✅ {det['class_name']}: conf={det['confidence']:.3f}, área={area:.0f}, aspect={aspect:.2f}")
                    elif raw_detections:
                        print(f"   ⚠️  {len(raw_detections)} detecciones raw fueron filtradas")
                
                # Guardar estadísticas
                self.debug_stats['inference_time'].append(inference_time)
                self.debug_stats['nms_time'].append(nms_time)
                self.debug_stats['temporal_time'].append(temporal_time)
                self.debug_stats['total_time'].append(total_time)
                
                # Llamar al callback si existe (para reportar progreso)
                if callback:
                    callback(frame_number)
                
                # Yield el resultado
                yield {
                    'frame': processed_frame,
                    'detections': detections,
                    'frame_number': frame_number
                }
        
        except Exception as e:
            print(f"❌ Error en tracking persistente: {e}")
            import traceback
            traceback.print_exc()
            return
    
    def _get_detection_color(self, class_name):
        """Obtener color para una clase (BGR format)"""
        class_lower = class_name.lower() if class_name else ''
        return Config.CLASS_COLORS.get(class_lower, (255, 255, 255))
    
    def get_system_info(self):
        """Obtener información completa del sistema de modelos - Para debugging"""
        print("\nINFORMACIÓN COMPLETA DEL SISTEMA DE MODELOS")
        print("=" * 60)
        
        info = {
            'system_status': 'operational' if self.model_loaded else 'error',
            'base_directory': str(self.base_dir),
            'models_directory': str(self.models_dir),
            'models_dir_exists': self.models_dir.exists(),
            'model_path': str(self.model_path),
            'model_path_final': str(self.model_path_final) if self.model_path_final else None,
            'model_loaded': self.model_loaded
        }
        
        # Información del modelo
        print(f"\n📦 MODELO CONFIGURADO:")
        status = "NO DISPONIBLE"
        if self.model_path.exists():
            status = "DISPONIBLE"
        elif self.fallback_path.exists():
            status = "FALLBACK"
        
        if self.is_trained_model:
            print(f"   {status} bestnov.pt (modelo entrenado)")
        else:
            print(f"   {status} yolov8n.pt (COCO specialized)")
        print(f"      🔗 Ruta principal: {self.model_path}")
        if self.fallback_path.exists():
            print(f"      🔗 Ruta fallback: {self.fallback_path}")
        if self.model_path_final:
            print(f"      ✅ Encontrado: {self.model_path_final}")
        if self.model_loaded:
            print(f"      ✅ Estado: CARGADO")
            if self.is_trained_model:
                print(f"      🎯 Clases del modelo entrenado: {', '.join(sorted(self.model_class_list))}")
            else:
                print(f"      🎯 Clases COCO filtradas: {', '.join(sorted(self.model_class_list))}")
                print(f"      📊 IDs COCO: {sorted(self.coco_class_ids)}")
                    
        # Estado de la carpeta models
        print(f"\n📁 CARPETA MODELS:")
        print(f"   🔗 Ruta: {self.models_dir}")
        print(f"   Existe: {self.models_dir.exists()}")
        
        if self.models_dir.exists():
            pt_files = list(self.models_dir.glob("*.pt"))
            print(f"   📦 Archivos .pt encontrados: {len(pt_files)}")
            for pt_file in pt_files[:10]:  # Mostrar máximo 10
                size_mb = pt_file.stat().st_size / (1024*1024)
                print(f"      - {pt_file.name} ({size_mb:.1f} MB)")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        if info['system_status'] == 'operational':
            print("   Sistema funcionando correctamente")
        else:
            print("   Sistema con problemas")
            if not self.model_loaded:
                print("   No hay modelo cargado - verificar rutas")
                print("   💾 Ejecutar: python enhanced_model_handler.py para diagnóstico")
            
        print("=" * 60)
        return info

# Función de prueba y diagnóstico
def test_enhanced_model():
    """Probar el modelo mejorado con diagnóstico completo"""
    print("🧪 PROBANDO SISTEMA ROBUSTO DE MODELOS...")
    
    handler = EnhancedModelHandler()
    
    # Mostrar información completa del sistema
    system_info = handler.get_system_info()
    
    print(f"\nINTENTANDO CARGAR MODELOS...")
    if handler.load_models():
        print("\n¡MODELOS CARGADOS EXITOSAMENTE!")
        
        # Mostrar información post-carga
        if handler.model_loaded:
            if handler.is_trained_model:
                print(f"✅ Modelo activo: bestnov.pt (entrenado)")
            else:
                print(f"✅ Modelo activo: yolov8n.pt (COCO specialized)")
            print(f"   - Ruta: {handler.model_path_final}")
            print(f"   - Clases: {', '.join(sorted(handler.model_class_list))}")
        
        # Imagen de prueba
        print(f"\nProbando predicción con imagen de prueba...")
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Predicción mejorada
        predictions = handler.predict_with_tta(test_image, confidence_threshold=0.3)
        
        print(f"Predicciones finales: {len(predictions)}")
        for i, pred in enumerate(predictions):
            print(f"   {i+1}. {pred['class_name']}: {pred['confidence']:.3f} ({pred.get('model', 'unknown')})")
        
        print(f"\n¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        return True
    else:
        print(f"\nFALLA EN CARGA DE MODELOS")
        print(f"Información del sistema:")
        print(f"   - Directorio base: {system_info['base_directory']}")
        print(f"   - Directorio models: {system_info['models_directory']}")
        print(f"   - Models existe: {system_info['models_dir_exists']}")
        print(f"   - Modelos detectados: {system_info['models_detected']}")
        print(f"   - Modelos cargados: {system_info['models_loaded']}")
        
        print(f"\n💡 SOLUCIONES:")
        print(f"   1. Verificar que la carpeta 'models' existe")
        print(f"   2. Verificar que hay archivos .pt en la carpeta")
        print(f"   3. Ejecutar 'git pull' para obtener los modelos")
        print(f"   4. La descarga automática se intentará si falta algo")
        
        return False

if __name__ == "__main__":
    test_enhanced_model()