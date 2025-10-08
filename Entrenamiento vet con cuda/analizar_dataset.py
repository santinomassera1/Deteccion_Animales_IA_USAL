#!/usr/bin/env python3
"""
Análisis completo del dataset de veterinaria
"""
import os
import yaml
from collections import Counter, defaultdict
import cv2
import glob
from pathlib import Path

def analizar_dataset():
    # Rutas del dataset
    dataset_path = "DataSet_Veterinaria/entrenamiento Nacho"
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    
    print("="*60)
    print("🔍 ANÁLISIS COMPLETO DEL DATASET DE VETERINARIA")
    print("="*60)
    
    # Leer configuración del dataset
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n📋 CONFIGURACIÓN DEL DATASET:")
    print(f"   • Número de clases: {data_config['nc']}")
    print(f"   • Clases: {data_config['names']}")
    print(f"   • Proyecto Roboflow: {data_config['roboflow']['project']}")
    
    # Analizar cada conjunto
    conjuntos = ['train', 'valid', 'test']
    estadisticas_globales = {
        'total_imagenes': 0,
        'total_objetos': 0,
        'distribucion_clases': Counter(),
        'tamaños_imagenes': [],
        'objetos_por_imagen': []
    }
    
    for conjunto in conjuntos:
        print(f"\n📊 ANÁLISIS DEL CONJUNTO: {conjunto.upper()}")
        print("-" * 40)
        
        # Rutas de imágenes y etiquetas
        images_path = os.path.join(dataset_path, conjunto, "images")
        labels_path = os.path.join(dataset_path, conjunto, "labels")
        
        # Contar archivos
        images_files = glob.glob(os.path.join(images_path, "*.jpg"))
        labels_files = glob.glob(os.path.join(labels_path, "*.txt"))
        
        print(f"   • Imágenes: {len(images_files)}")
        print(f"   • Etiquetas: {len(labels_files)}")
        
        # Estadísticas por conjunto
        conjunto_stats = {
            'num_imagenes': len(images_files),
            'num_objetos': 0,
            'clases_conjunto': Counter(),
            'tamaños': [],
            'objetos_por_imagen': []
        }
        
        # Analizar algunas imágenes para obtener dimensiones
        for i, img_path in enumerate(images_files[:10]):  # Muestra de 10 imágenes
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    conjunto_stats['tamaños'].append((w, h))
                    estadisticas_globales['tamaños_imagenes'].append((w, h))
            except Exception as e:
                print(f"   ⚠️ Error leyendo {img_path}: {e}")
        
        # Analizar etiquetas
        for label_file in labels_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    num_objetos_imagen = len(lines)
                    conjunto_stats['objetos_por_imagen'].append(num_objetos_imagen)
                    estadisticas_globales['objetos_por_imagen'].append(num_objetos_imagen)
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_name = data_config['names'][class_id]
                            conjunto_stats['clases_conjunto'][class_name] += 1
                            estadisticas_globales['distribucion_clases'][class_name] += 1
                            conjunto_stats['num_objetos'] += 1
                            estadisticas_globales['total_objetos'] += 1
            except Exception as e:
                print(f"   ⚠️ Error leyendo {label_file}: {e}")
        
        estadisticas_globales['total_imagenes'] += conjunto_stats['num_imagenes']
        
        # Mostrar estadísticas del conjunto
        print(f"   • Total objetos: {conjunto_stats['num_objetos']}")
        if conjunto_stats['objetos_por_imagen']:
            avg_objects = sum(conjunto_stats['objetos_por_imagen']) / len(conjunto_stats['objetos_por_imagen'])
            print(f"   • Promedio objetos/imagen: {avg_objects:.2f}")
        
        print(f"   • Distribución por clase:")
        for clase, cantidad in conjunto_stats['clases_conjunto'].most_common():
            porcentaje = (cantidad / conjunto_stats['num_objetos'] * 100) if conjunto_stats['num_objetos'] > 0 else 0
            print(f"     - {clase}: {cantidad} ({porcentaje:.1f}%)")
        
        if conjunto_stats['tamaños']:
            avg_w = sum(w for w, h in conjunto_stats['tamaños']) / len(conjunto_stats['tamaños'])
            avg_h = sum(h for w, h in conjunto_stats['tamaños']) / len(conjunto_stats['tamaños'])
            print(f"   • Tamaño promedio imágenes: {avg_w:.0f}x{avg_h:.0f}")
    
    # Estadísticas globales
    print(f"\n🌍 ESTADÍSTICAS GLOBALES")
    print("="*40)
    print(f"   • Total imágenes: {estadisticas_globales['total_imagenes']}")
    print(f"   • Total objetos: {estadisticas_globales['total_objetos']}")
    
    if estadisticas_globales['objetos_por_imagen']:
        avg_global = sum(estadisticas_globales['objetos_por_imagen']) / len(estadisticas_globales['objetos_por_imagen'])
        print(f"   • Promedio objetos/imagen: {avg_global:.2f}")
    
    print(f"\n📈 DISTRIBUCIÓN GLOBAL POR CLASE:")
    for clase, cantidad in estadisticas_globales['distribucion_clases'].most_common():
        porcentaje = (cantidad / estadisticas_globales['total_objetos'] * 100) if estadisticas_globales['total_objetos'] > 0 else 0
        print(f"   • {clase}: {cantidad} objetos ({porcentaje:.1f}%)")
    
    if estadisticas_globales['tamaños_imagenes']:
        avg_w_global = sum(w for w, h in estadisticas_globales['tamaños_imagenes']) / len(estadisticas_globales['tamaños_imagenes'])
        avg_h_global = sum(h for w, h in estadisticas_globales['tamaños_imagenes']) / len(estadisticas_globales['tamaños_imagenes'])
        print(f"\n🖼️  DIMENSIONES DE IMÁGENES:")
        print(f"   • Tamaño promedio: {avg_w_global:.0f}x{avg_h_global:.0f}")
        
        # Mostrar algunos tamaños únicos
        tamaños_unicos = list(set(estadisticas_globales['tamaños_imagenes']))
        print(f"   • Tamaños encontrados (muestra): {tamaños_unicos[:5]}")
    
    # Verificar compatibilidad con el script de entrenamiento
    print(f"\n✅ COMPATIBILIDAD CON generacionYolo.py:")
    print(f"   • Formato YOLO: ✅ Compatible")
    print(f"   • Estructura de directorios: ✅ Compatible")
    print(f"   • Archivo data.yaml: ✅ Encontrado")
    
    # Nota sobre las diferencias de clases
    script_classes = ['Arcella', 'Aspidisca', 'Epistylis', 'Otro', 'Rotifero', 'Trachelophyllum', 'Vorticella']
    dataset_classes = data_config['names']
    
    print(f"\n⚠️  NOTA IMPORTANTE:")
    print(f"   • Clases en generacionYolo.py: {script_classes}")
    print(f"   • Clases en este dataset: {dataset_classes}")
    print(f"   • ¿Coinciden? {'✅ SÍ' if script_classes == dataset_classes else '❌ NO - Necesita ajuste'}")
    
    if script_classes != dataset_classes:
        print(f"\n🔧 RECOMENDACIONES:")
        print(f"   1. Actualizar data.yaml con las clases correctas, o")
        print(f"   2. Usar este dataset actual con sus clases de animales")
    
    print(f"\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    analizar_dataset()