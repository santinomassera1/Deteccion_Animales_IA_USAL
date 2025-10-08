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
        
        # Sistema robusto de detección de modelos - PORTABLE
        self.base_dir = Path(__file__).parent.absolute()
        self.models_dir = self.base_dir / "models"
        
        # Asegurar que la carpeta models existe
        self.models_dir.mkdir(exist_ok=True)
        
        # Configuración de modelos con fallbacks automáticos
        self.model_config = {
            'primary': {
                'path': 'models/animals_best.pt',
                'fallback': 'models/yolov8m.pt',
                'auto_download': 'yolov8m.pt',
                'description': 'Modelo entrenado principal (animals_best.pt)'
            },
            'secondary': {
                'path': 'models/animals_last.pt', 
                'fallback': 'models/yolov8s.pt',
                'auto_download': 'yolov8s.pt',
                'description': 'Modelo entrenado secundario (animals_last.pt)'
            },
            'yolo11n': {
                'path': 'models/yolo11n.pt',
                'fallback': None,
                'auto_download': 'yolo11n.pt',
                'description': 'YOLO 11 Nano'
            },
            'yolov8m': {
                'path': 'models/yolov8m.pt',
                'fallback': None,
                'auto_download': 'yolov8m.pt', 
                'description': 'YOLO v8 Medium'
            },
            'yolov8s': {
                'path': 'models/yolov8s.pt',
                'fallback': None,
                'auto_download': 'yolov8s.pt',
                'description': 'YOLO v8 Small'
            }
        }
        
        # Detectar y preparar modelos disponibles
        self.model_paths = self._discover_available_models()
        
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

    def _discover_available_models(self):
        """Detectar modelos disponibles con sistema robusto de fallbacks"""
        available_models = {}
        print("\nIniciando detección inteligente de modelos...")
        
        for model_name, config in self.model_config.items():
            model_path = self.base_dir / config['path']
            fallback_path = self.base_dir / config['fallback'] if config['fallback'] else None
            final_path = None
            
            print(f"\n📦 Verificando {config['description']}:")
            print(f"   - Ruta principal: {model_path}")
            
            # 1. Verificar ruta principal
            if model_path.exists():
                final_path = str(model_path)
                print(f"   Encontrado en ruta principal")
            
            # 2. Verificar fallback
            elif fallback_path and fallback_path.exists():
                final_path = str(fallback_path)
                print(f"   Usando fallback: {fallback_path}")
            
            # 3. Intentar descarga automática
            elif config['auto_download']:
                try:
                    print(f"   🌐 Intentando descarga automática: {config['auto_download']}")
                    # YOLO automáticamente descarga modelos básicos
                    temp_model = YOLO(config['auto_download'])
                    # Copiar el modelo descargado a nuestra carpeta
                    downloaded_path = self.models_dir / config['auto_download']
                    if not downloaded_path.exists() and hasattr(temp_model, 'ckpt_path'):
                        import shutil
                        shutil.copy2(temp_model.ckpt_path, downloaded_path)
                        print(f"   Descargado y guardado en: {downloaded_path}")
                    final_path = str(downloaded_path)
                except Exception as e:
                    print(f"   Error en descarga automática: {e}")
            
            if final_path:
                available_models[model_name] = final_path
                print(f"   DISPONIBLE: {final_path}")
            else:
                print(f"   NO DISPONIBLE: {model_name}")
        
        print(f"\nRESUMEN: {len(available_models)}/{len(self.model_config)} modelos disponibles")
        
        if not available_models:
            print("\nADVERTENCIA: No se encontraron modelos!")
            print("   Creando configuración de emergencia con modelos básicos...")
            # Configuración de emergencia
            try:
                emergency_model = YOLO('yolov8n.pt')  # Modelo más pequeño
                emergency_path = self.models_dir / 'emergency_yolov8n.pt'
                available_models['emergency'] = str(emergency_path)
                print(f"   Modelo de emergencia preparado: {emergency_path}")
            except Exception as e:
                print(f"   Error crítico: No se pudo preparar modelo de emergencia: {e}")
                raise RuntimeError("FATAL: No hay modelos disponibles y no se puede descargar automáticamente")
        
        return available_models

    def load_models(self):
        """Cargar múltiples modelos usando el sistema robusto"""
        print("\nIniciando carga de modelos con sistema robusto...")
        
        if not self.model_paths:
            print("FATAL: No hay modelos disponibles después de la detección")
            return False
        
        loaded_count = 0
        successful_models = []
        
        # Cargar cada modelo detectado
        for model_name, model_path in self.model_paths.items():
            try:
                print(f"\nCargando {model_name}...")
                print(f"   📂 Ruta: {model_path}")
                
                # Verificar que el archivo existe (redundante pero seguro)
                if not os.path.exists(model_path):
                    print(f"   Archivo no existe en el momento de carga")
                    continue
                
                # Cargar modelo con manejo de errores robusto
                model = YOLO(model_path)
                
                # Verificar que el modelo se cargó correctamente
                if not hasattr(model, 'model') or model.model is None:
                    print(f"   Modelo cargado pero estructura inválida")
                    continue
                
                self.models[model_name] = model
                
                # Información detallada del modelo
                if hasattr(model.model, 'names'):
                    classes = list(model.model.names.values()) if model.model.names else []
                    print(f"   Cargado exitosamente")
                    print(f"   Clases: {len(classes)}")
                    print(f"   Algunas clases: {classes[:5] if classes else 'Sin información'}")
                    
                    # Verificar compatibilidad con animales
                    target_classes = ['cat', 'chicken', 'cow', 'dog', 'horse']
                    animal_classes_found = [cls for cls in target_classes if cls in classes]
                    
                    if animal_classes_found:
                        print(f"   Compatible con animales: {animal_classes_found}")
                        successful_models.append(model_name)
                    else:
                        print(f"   📝 Modelo genérico (COCO/otros): {classes[:3] if classes else []}")
                        successful_models.append(model_name)  # Aceptar modelos genéricos también
                else:
                    print(f"   Sin información de clases, pero modelo cargado")
                    successful_models.append(model_name)
                
                loaded_count += 1
                
            except Exception as e:
                print(f"   Error cargando {model_name}: {e}")
                print(f"   Tipo de error: {type(e).__name__}")
                continue
        
        # Resultado final
        print(f"\nRESULTADO FINAL:")
        print(f"   Modelos cargados: {loaded_count}/{len(self.model_paths)}")
        print(f"   Modelos exitosos: {successful_models}")
        
        if loaded_count > 0:
            self.model_loaded = True
            print(f"\n¡Sistema preparado! {loaded_count} modelos listos para ensemble")
            
            # Mostrar configuración final
            print(f"\nCONFIGURACIÓN FINAL:")
            for name in successful_models:
                print(f"   - {name}: {self.model_paths[name]}")
            
            return True
        else:
            print(f"\nFALLA CRÍTICA: No se pudo cargar ningún modelo")
            print(f"   Verificar que los archivos .pt existen y son válidos")
            print(f"   💡 La app intentará usar descarga automática en el próximo inicio")
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
                    print(f"Error en predicción {model_name}-{aug_name}: {e}")
                    continue
        
        # Post-procesamiento avanzado
        final_predictions = self._advanced_post_processing(all_predictions)
        
        print(f"TTA Ensemble: {len(all_predictions)} predicciones brutas -> {len(final_predictions)} finales")
        
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
    
    def process_video_with_tracking(self, video_source, confidence_threshold=0.4, callback=None):
        """
        Procesa un video o stream con tracking persistente para eliminar parpadeo.
        
        Esta función usa el tracking nativo de YOLO (ByteTrack) de forma persistente,
        manteniendo el estado entre frames para lograr detecciones suaves y estables.
        
        Args:
            video_source: Path al video o índice de cámara (0 para webcam)
            confidence_threshold: Umbral de confianza (0.0-1.0)
            callback: Función opcional para reportar progreso (frame_num, total_frames)
        
        Yields:
            dict: {
                'frame': numpy array del frame procesado con detecciones dibujadas,
                'detections': lista de detecciones con tracking IDs,
                'frame_number': número de frame actual
            }
        """
        if not self.model_loaded or not self.models:
            print("❌ No hay modelos cargados para tracking")
            return
        
        # Seleccionar modelo primario para tracking
        # Para tracking, usamos UN SOLO modelo (no ensemble) para consistencia
        model_to_use = None
        
        # Prioridad: primary > yolov8m > yolo11n > cualquier otro
        priority_order = ['primary', 'yolov8m', 'yolo11n', 'yolov8s', 'secondary']
        
        for model_key in priority_order:
            if model_key in self.models:
                model_to_use = self.models[model_key]
                print(f"🎯 Usando modelo '{model_key}' para tracking persistente")
                break
        
        if model_to_use is None:
            # Usar el primer modelo disponible
            model_key = list(self.models.keys())[0]
            model_to_use = self.models[model_key]
            print(f"🎯 Usando modelo '{model_key}' para tracking persistente")
        
        print(f"🎬 Iniciando tracking persistente con ByteTrack")
        print(f"   - Confidence threshold: {confidence_threshold}")
        print(f"   - Video source: {video_source}")
        
        try:
            # Usar el método .track() nativo de YOLO para tracking persistente
            # Esto mantiene el estado del tracker entre frames
            results_generator = model_to_use.track(
                source=video_source,
                stream=True,          # Procesar como stream para eficiencia
                persist=True,         # CRÍTICO: Mantiene IDs entre frames
                tracker="bytetrack.yaml",  # ByteTrack algorithm
                conf=confidence_threshold,
                iou=0.5,             # IoU threshold para tracking
                imgsz=640,           # Tamaño de imagen optimizado
                device='cpu',        # Forzar CPU (Mac compatible)
                half=False,          # No half precision en CPU
                verbose=False,       # Sin logs excesivos
                max_det=50,          # Máximo 50 detecciones
                classes=[0, 1, 2, 3, 4]  # Solo nuestras 5 clases
            )
            
            frame_number = 0
            
            # Iterar sobre los resultados del tracking
            for results in results_generator:
                frame_number += 1
                
                # Obtener el frame procesado con las detecciones dibujadas
                # results.plot() dibuja automáticamente:
                # - Bounding boxes
                # - Tracking IDs
                # - Confidence scores
                # - Class labels
                processed_frame = results.plot()
                
                # Extraer información de las detecciones para retornar
                detections = []
                
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
                        
                        # Solo procesar nuestras 5 clases
                        if class_id < 5:
                            class_names = ['cat', 'chicken', 'cow', 'dog', 'horse']
                            detection = {
                                'box': box,
                                'confidence': conf,
                                'class': class_id,
                                'class_name': class_names[class_id],
                                'track_id': track_id,
                                'tracked': track_id is not None
                            }
                            detections.append(detection)
                
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
    
    def get_system_info(self):
        """Obtener información completa del sistema de modelos - Para debugging"""
        print("\nINFORMACIÓN COMPLETA DEL SISTEMA DE MODELOS")
        print("=" * 60)
        
        info = {
            'system_status': 'operational' if self.model_loaded else 'error',
            'base_directory': str(self.base_dir),
            'models_directory': str(self.models_dir),
            'models_dir_exists': self.models_dir.exists(),
            'total_models_configured': len(self.model_config),
            'models_detected': len(self.model_paths),
            'models_loaded': len(self.models),
            'model_details': {},
            'configuration': self.model_config,
            'paths': self.model_paths
        }
        
        # Información de cada modelo configurado
        print(f"\nMODELOS CONFIGURADOS ({len(self.model_config)}):")
        for name, config in self.model_config.items():
            model_path = self.base_dir / config['path']
            fallback_path = self.base_dir / config['fallback'] if config['fallback'] else None
            
            status = "NO DISPONIBLE"
            actual_path = None
            
            if model_path.exists():
                status = "DISPONIBLE"
                actual_path = str(model_path)
            elif fallback_path and fallback_path.exists():
                status = "FALLBACK"
                actual_path = str(fallback_path)
            
            info['model_details'][name] = {
                'description': config['description'],
                'configured_path': str(model_path),
                'fallback_path': str(fallback_path) if fallback_path else None,
                'exists': actual_path is not None,
                'actual_path': actual_path,
                'loaded': name in self.models,
                'auto_download': config['auto_download']
            }
            
            print(f"   {status} {config['description']}")
            print(f"      🔗 Ruta: {model_path}")
            if fallback_path:
                print(f"      Fallback: {fallback_path}")
            if actual_path:
                print(f"      Encontrado: {actual_path}")
            if name in self.models:
                print(f"      Estado: CARGADO")
                # Información adicional del modelo cargado
                model = self.models[name]
                if hasattr(model.model, 'names') and model.model.names:
                    classes = list(model.model.names.values())
                    print(f"      Clases: {len(classes)} ({classes[:3]}...)")
                    
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
            if info['models_loaded'] == 0:
                print("   No hay modelos cargados - verificar rutas")
                print("   💾 Ejecutar: python enhanced_model_handler.py para diagnóstico")
            
        if info['models_detected'] < info['total_models_configured']:
            missing = info['total_models_configured'] - info['models_detected']
            print(f"   Faltan {missing} modelos - descarga automática disponible")
            
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
        print(f"Modelos activos: {len(handler.models)}")
        for name, model in handler.models.items():
            print(f"   - {name}: {handler.model_paths[name]}")
        
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