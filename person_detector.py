#!/usr/bin/env python3
"""
Detector de personas para distinguir personas de animales
Usa YOLO pre-entrenado y filtra solo detecciones de personas
"""

import cv2
import numpy as np
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, nacho_model=None):
        """Inicializar detector de personas usando el modelo principal"""
        print("Cargando detector de personas...")
        if nacho_model is not None:
            # Usar el modelo CUDA YOLOv8m si está disponible
            self.model = nacho_model
            print("Detector de personas usando modelo CUDA YOLOv8m")
        else:
            # Fallback al modelo pre-entrenado si no hay modelo principal
            self.model = YOLO('yolov8n.pt')  # Modelo pre-entrenado con COCO
            print("Detector de personas usando modelo pre-entrenado (fallback)")
        self.person_class_id = 0  # Clase 'person' en COCO dataset
        print("Detector de personas cargado exitosamente!")
    
    def detect_persons(self, image, confidence_threshold=0.5):
        """Detectar solo personas en la imagen"""
        try:
            # Si estamos usando el modelo CUDA YOLOv8m, no detecta personas
            # El modelo CUDA YOLOv8m solo detecta: cat, chicken, cow, dog, horse
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                model_classes = list(self.model.model.names.values())
                if 'person' not in model_classes:
                    # El modelo CUDA YOLOv8m no detecta personas, retornar lista vacía
                    return []
            
            results = self.model(image, conf=confidence_threshold, verbose=False)
            
            person_detections = []
            
            for result in results:
                if result.boxes is not None:
                    # Filtrar solo detecciones de personas
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id == self.person_class_id and confidence > confidence_threshold:
                            detection = {
                                'box': box.xyxy[0].cpu().numpy().tolist(),
                                'confidence': confidence,
                                'class': 'person',
                                'class_id': class_id,
                                'model': 'person_detector'
                            }
                            person_detections.append(detection)
            
            return person_detections
            
        except Exception as e:
            print(f"Error detectando personas: {e}")
            return []
    
    def draw_person_detections(self, image, detections):
        """Dibujar detecciones de personas en la imagen"""
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['box']]
            confidence = detection['confidence']
            
            # Color verde para personas
            color = (0, 255, 0)
            
            # Dibujar bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar etiqueta
            label = f"Person ({confidence:.2f})"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image
    
    def filter_animal_detections(self, animal_detections, person_detections, iou_threshold=0.3):
        """Filtrar detecciones de animales que se superponen con personas"""
        filtered_animals = []
        
        for animal_detection in animal_detections:
            is_person = False
            
            # Verificar si la detección de animal se superpone con una persona
            for person_detection in person_detections:
                iou = self.calculate_iou(animal_detection['box'], person_detection['box'])
                
                if iou > iou_threshold:
                    is_person = True
                    print(f"🚫 Filtrando detección de animal como persona (IoU: {iou:.2f})")
                    break
            
            if not is_person:
                filtered_animals.append(animal_detection)
        
        return filtered_animals
    
    def calculate_iou(self, box1, box2):
        """Calcular Intersection over Union entre dos bounding boxes"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        # Calcular intersección
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calcular unión
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

# Función de prueba
def test_person_detection():
    """Probar detección de personas"""
    detector = PersonDetector()
    
    # Crear imagen de prueba
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Detectar personas
    detections = detector.detect_persons(test_image)
    print(f"Personas detectadas: {len(detections)}")
    
    return detections

if __name__ == "__main__":
    test_person_detection()
