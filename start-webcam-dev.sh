#!/bin/bash

# Script para iniciar el desarrollo de la aplicación con funcionalidad de webcam

echo "🚀 Iniciando aplicación de detección de animales con webcam..."

# Función para limpiar procesos al salir
cleanup() {
    echo "🧹 Limpiando procesos..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Configurar trap para cleanup
trap cleanup SIGINT SIGTERM

# Verificar que Python y Node están instalados
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no está instalado"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js no está instalado"
    exit 1
fi

# Verificar que las dependencias de Python están instaladas
if [ ! -f "requirements.txt" ]; then
    echo "❌ archivo requirements.txt no encontrado"
    exit 1
fi

# Instalar dependencias de Python si no están instaladas
echo "📦 Verificando dependencias de Python..."
pip3 install -r requirements.txt > /dev/null 2>&1

# Verificar que el directorio frontend existe
if [ ! -d "frontend" ]; then
    echo "❌ Directorio frontend no encontrado"
    exit 1
fi

# Instalar dependencias de Node.js si no están instaladas
echo "📦 Verificando dependencias de Node.js..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "📦 Instalando dependencias de Node.js..."
    npm install
fi
cd ..

echo ""
echo "🎥 Funcionalidades de webcam disponibles:"
echo "   • Streaming continuo con detección automática"
echo "   • Captura manual con análisis bajo demanda"
echo "   • Detección de 5 animales: gato, gallina, vaca, perro, caballo"
echo ""

# Iniciar el backend de Flask
echo "🐍 Iniciando servidor backend (Flask)..."
python3 app.py &
BACKEND_PID=$!

# Esperar a que el backend esté listo
echo "⏳ Esperando que el backend esté listo..."
sleep 5

# Iniciar el frontend de React
echo "⚛️  Iniciando servidor frontend (React)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "✅ Aplicación iniciada exitosamente!"
echo ""
echo "🔗 URLs disponibles:"
echo "   • Frontend React: http://localhost:5173"
echo "   • Backend Flask:  http://localhost:5003"
echo "   • Streaming webcam: http://localhost:5003/api/webcam"
echo ""
echo "📋 Para usar la webcam:"
echo "   1. Ve a la pestaña 'Cámara' en la interfaz web"
echo "   2. Selecciona modo 'Streaming' para detección automática"
echo "   3. O selecciona modo 'Manual' para usar tu cámara local"
echo "   4. ¡Disfruta detectando animales en tiempo real!"
echo ""
echo "⚠️  Presiona Ctrl+C para detener ambos servidores"
echo ""

# Esperar que ambos procesos estén corriendo
wait
