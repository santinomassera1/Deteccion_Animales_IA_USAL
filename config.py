"""
Configuración de la aplicación de detección de animales
"""

import os

class Config:
    """Configuración base"""
    
    # Configuración Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'veterinaria_detection_app_dev_key'
    
    # Configuración de archivos
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 150 * 1024 * 1024  # 150MB
    
    # Límites específicos por tipo de archivo
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB para imágenes
    MAX_VIDEO_SIZE = 150 * 1024 * 1024  # 150MB para videos
    
    # Configuración del modelo - ACTUALIZADO CON NUEVO MODELO MEJORADO (50MB)
    MODEL_PATH = 'models/animals_best.pt'  # Ahora es el modelo mejorado (50MB, 25.9M params)
    CONFIDENCE_THRESHOLD = 0.3  # Umbral base para modelo optimizado
    
    # Configuración de clases (nuevo modelo mejorado - 4 clases)
    # Modelo: best.pt (50MB) - YOLOv8m con 25.9M parámetros
    # Velocidad: 5.31 FPS (40% más rápido que el anterior)
    CLASS_NAMES = {
        0: 'car',      # Autos/Gatos
        1: 'cow',      # Vacas
        2: 'dog',      # Perros
        3: 'horse'     # Caballos
    }
    
    CLASS_COLORS = {
        'car': (255, 0, 255),      # Magenta para autos
        'cow': (0, 255, 0),        # Verde para vacas
        'dog': (0, 0, 255),        # Azul para perros
        'horse': (255, 255, 0)     # Amarillo para caballos
    }
    
    # Tipos de archivo permitidos
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
    
    @classmethod
    def get_allowed_extensions(cls):
        """Obtener todas las extensiones permitidas"""
        return cls.ALLOWED_IMAGE_EXTENSIONS.union(cls.ALLOWED_VIDEO_EXTENSIONS)
    
    # Configuración de la aplicación
    APP_NAME = "Detección de Animales y Vehículos - Veterinaria (Modelo YOLOv8m Mejorado)"
    APP_VERSION = "3.0.0"
    APP_DESCRIPTION = "Sistema de detección inteligente optimizado para 4 clases: autos, vacas, perros y caballos - Modelo 40% más rápido"
    
    # Configuración de desarrollo
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    PORT = int(os.environ.get('FLASK_PORT', 5003))
    
    # Configuración de seguridad
    SESSION_COOKIE_SECURE = False  # Cambiar a True en producción con HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Configuración de logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configuración de caché
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutos
    
    # Configuración de límites
    MAX_DETECTIONS_PER_IMAGE = 50
    MAX_HISTORY_ITEMS = 100
    
    # Configuración de la cámara
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

class DevelopmentConfig(Config):
    """Configuración para desarrollo"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Configuración para producción"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    LOG_LEVEL = 'WARNING'

class TestingConfig(Config):
    """Configuración para testing"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False

# Diccionario de configuraciones
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
