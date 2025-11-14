"""
Configuración de la aplicación de detección de animales
"""

import os

class Config:
    """Configuración base"""
    # Configuración de archivos
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 150 * 1024 * 1024  
    
    # Límites específicos por tipo de archivo
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 
    MAX_VIDEO_SIZE = 150 * 1024 * 1024  
    
    MODEL_PATH = 'models/bestnov.pt'  # Modelo entrenado (fallback a yolov8n.pt si no existe)  
    
    # ===== THRESHOLDS UNIFICADOS - FUENTE ÚNICA DE VERDAD =====
    # Configuración para imágenes estáticas (offline) - OPTIMIZADO PARA MEJOR DETECCIÓN
    IMAGE_CONFIDENCE_THRESHOLD = 0.25  # Reducido para capturar más detecciones (más lento pero mejor)
    IMAGE_IOU_THRESHOLD = 0.40  # Reducido de 0.45 a 0.40 para eliminar más superposiciones
    IMAGE_USE_TTA = True  # TTA solo para imágenes estáticas
    IMAGE_SIZE = 1280  # Aumentado de 640 a 1280 para mejor detección (más lento pero más preciso)
    
    # Configuración para streaming (webcam/video) - SIN TTA
    STREAMING_CONFIDENCE_THRESHOLD = 0.40  # Global para streaming
    STREAMING_IOU_THRESHOLD = 0.40  # NMS más estricto para reducir FPs y duplicados (reducido de 0.45 a 0.40)
    STREAMING_USE_TTA = False  # NO usar TTA en streaming
    
    # Umbrales por clase para streaming (ajustados según rendimiento real)
    STREAMING_CLASS_THRESHOLDS = {
        # Español
        'Perro': 0.42, 'perro': 0.42, 'dog': 0.42, 'dog detect': 0.42,  # Perros: ligeramente más permisivo
        'Auto': 0.33, 'auto': 0.33, 'car': 0.33, 'cars detect': 0.33, 'cars': 0.33,  # Autos: más permisivo
        'Caballo': 0.33, 'caballo': 0.33, 'horse': 0.33, 'horse detect': 0.33,  # Caballos: más permisivo
        'Vaca': 0.70, 'vaca': 0.70, 'cow': 0.70, 'cow detection': 0.70,  # Vacas: UMBRAL ALTO para reducir falsos positivos
    }
    
    # Filtro adaptativo: umbrales más bajos para objetos grandes
    ADAPTIVE_SIZE_THRESHOLD = True
    LARGE_OBJECT_AREA = 30000  # px² - objetos más grandes que esto pueden tener umbral reducido
    LARGE_OBJECT_CONFIDENCE_BONUS = -0.05  # Reducción de umbral para objetos grandes (se RESTA, no se suma)
    
    # Umbral mínimo para streaming (clamp para umbral adaptativo)
    STREAMING_MIN_CONFIDENCE = 0.20  # Umbral mínimo absoluto después de aplicar bono adaptativo
    
    # Filtros geométricos (ajustados para mejor recall)
    MIN_BOX_WIDTH = 35  # Reducido de 40 para capturar objetos más pequeños
    MIN_BOX_HEIGHT = 35  # Reducido de 40 para capturar objetos más pequeños
    MIN_BOX_AREA = 1500  # Reducido de 2000 para capturar objetos más pequeños
    MIN_ASPECT_RATIO = 0.2  # Más permisivo
    MAX_ASPECT_RATIO = 7.0  # Más permisivo
    
    # Filtro temporal (voto de persistencia)
    TEMPORAL_FILTER_ENABLED = True
    TEMPORAL_HISTORY_SIZE = 5  # Últimos N frames
    TEMPORAL_VOTE_THRESHOLD = 3  # Mínimo de frames con detección
    TEMPORAL_IOU_THRESHOLD = 0.3  # IoU mínimo para considerar misma detección
    
    # ROI (Región de Interés) - parte inferior del frame (menos restrictivo)
    ROI_ENABLED = True
    ROI_Y_START = 0.3  # Empezar desde 30% del frame (parte inferior, menos restrictivo)
    ROI_ADAPTIVE = True  # Permitir detecciones fuera de ROI si tienen alta confianza o son grandes
    ROI_HIGH_CONFIDENCE_THRESHOLD = 0.65  # Confianza mínima para ignorar ROI
    ROI_LARGE_OBJECT_AREA = 50000  # Área mínima (px²) para considerar objeto "grande" y permitir fuera de ROI
    ROI_ANIMAL_BYPASS_CONF = 0.35  # Confianza mínima para animales (dog, horse, cow) para bypass de ROI
    
    # Logs de depuración
    DEBUG_LOGS_ENABLED = True
    DEBUG_LOG_INTERVAL = 30  # Log cada N frames
    DEBUG_VERBOSE_FILTERS = False  # Reducir logs verbosos de filtros individuales
    
    # Thresholds legacy (para compatibilidad)
    CONFIDENCE_THRESHOLD = IMAGE_CONFIDENCE_THRESHOLD  # Por defecto para imágenes 
    

    CLASS_NAME_MAPPING = {
        # Español a inglés
        'Auto': 'car', 'auto': 'car',
        'Caballo': 'horse', 'caballo': 'horse',
        'Perro': 'dog', 'perro': 'dog',
        'Vaca': 'cow', 'vaca': 'cow',
        # Inglés estándar
        'car': 'car', 'horse': 'horse', 'dog': 'dog', 'cow': 'cow',
        # Nombres reales del modelo a inglés normalizado
        'cars detect': 'car', 'cars': 'car',
        'cow detection': 'cow',
        'dog detect': 'dog',
        'horse detect': 'horse'
    }
    
    CLASS_COLORS = {
        # Español
        'Auto': (255, 0, 255), 'auto': (255, 0, 255),
        'Caballo': (0, 255, 255), 'caballo': (0, 255, 255),
        'Perro': (255, 0, 0), 'perro': (255, 0, 0),
        'Vaca': (0, 255, 0), 'vaca': (0, 255, 0),
        # Inglés estándar
        'car': (255, 0, 255), 'cars': (255, 0, 255),      # Magenta para autos
        'horse': (0, 255, 0),        # Verde para caballos
        'dog': (255, 0, 0),        # Rojo para perros
        'cow': (0, 255, 255),     # Amarillo/Cyan para vacas
        # Nombres reales del modelo
        'cars detect': (255, 0, 255),
        'cow detection': (0, 255, 255),
        'dog detect': (255, 0, 0),
        'horse detect': (0, 255, 0)
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
    APP_DESCRIPTION = "Sistema de detección inteligente optimizado para 4 clases: autos, caballos, perros y vacas - Modelo 40% más rápido"
    
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
