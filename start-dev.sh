#!/bin/bash

# Script para iniciar el desarrollo completo (Backend + Frontend)

echo "🚀 Iniciando aplicación de detección de animales..."
echo "=================================================="

# Función para limpiar procesos al salir
cleanup() {
    echo ""
    echo "🛑 Deteniendo servidores..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Capturar Ctrl+C
trap cleanup SIGINT

# Iniciar backend
echo "📡 Iniciando servidor backend (Puerto 5003)..."
cd "$(dirname "$0")"
python3 app.py &
BACKEND_PID=$!

# Esperar un poco para que el backend se inicie
sleep 3

# Iniciar frontend
echo "🎨 Iniciando servidor frontend (Puerto 3000)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Esperar un poco para que el frontend se inicie
sleep 3

echo ""
echo "✅ Aplicación iniciada correctamente!"
echo "=================================================="
echo "🌐 Frontend: http://localhost:3000"
echo "📡 Backend:  http://localhost:5003"
echo ""
echo "Presiona Ctrl+C para detener ambos servidores"
echo ""

# Mantener el script corriendo
wait
