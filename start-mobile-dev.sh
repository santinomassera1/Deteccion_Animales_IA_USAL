#!/bin/bash

# Script para iniciar el backend y la app móvil simultáneamente
# Autor: Sistema de Detección de Animales USAL

echo "🚀 Iniciando Sistema de Detección de Animales - Modo Móvil"
echo "=================================================="
echo ""

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mostrar mensajes con color
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "app.py" ]; then
    log_error "Error: app.py no encontrado. Ejecuta este script desde el directorio raíz del proyecto."
    exit 1
fi

# Obtener la IP local de la Mac
log_info "Detectando IP local..."
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -n1 | awk '{print $2}')

if [ -z "$LOCAL_IP" ]; then
    log_error "No se pudo detectar la IP local. Verifica tu conexión WiFi."
    exit 1
fi

log_success "IP local detectada: $LOCAL_IP"

# Actualizar la IP en el archivo api.ts del móvil
log_info "Actualizando configuración de la app móvil..."
API_FILE="mobile/src/services/api.ts"

if [ -f "$API_FILE" ]; then
    # Usar sed compatible con macOS
    sed -i '' "s|const API_BASE_URL = 'http://.*:5003'|const API_BASE_URL = 'http://$LOCAL_IP:5003'|g" "$API_FILE"
    log_success "Configuración actualizada en $API_FILE"
else
    log_warning "No se encontró $API_FILE. Asegúrate de configurar la IP manualmente."
fi

echo ""
log_info "Configuración completada:"
echo "  - Backend URL: http://$LOCAL_IP:5003"
echo "  - Frontend móvil: Expo Go"
echo ""

# Función para limpiar procesos al salir
cleanup() {
    echo ""
    log_info "Deteniendo servicios..."
    kill $BACKEND_PID 2>/dev/null
    kill $MOBILE_PID 2>/dev/null
    log_success "Servicios detenidos. ¡Hasta pronto!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Iniciar el backend Python
log_info "Iniciando backend Python en el puerto 5003..."
python3 app.py > backend.log 2>&1 &
BACKEND_PID=$!

# Esperar a que el backend se inicie
sleep 3

# Verificar que el backend está corriendo
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    log_error "El backend falló al iniciar. Revisa backend.log para más detalles."
    cat backend.log
    exit 1
fi

log_success "Backend Python corriendo (PID: $BACKEND_PID)"
echo ""

# Iniciar la app móvil con Expo
log_info "Iniciando app móvil con Expo..."
cd mobile

# Verificar que node_modules existe
if [ ! -d "node_modules" ]; then
    log_warning "node_modules no encontrado. Instalando dependencias..."
    npm install --legacy-peer-deps
fi

npx expo start --clear > ../mobile.log 2>&1 &
MOBILE_PID=$!
cd ..

log_success "App móvil iniciada (PID: $MOBILE_PID)"
echo ""

log_success "=================================================="
log_success "🎉 Sistema iniciado correctamente!"
log_success "=================================================="
echo ""
echo "📱 Para conectar tu dispositivo móvil:"
echo "   1. Descarga 'Expo Go' desde App Store o Play Store"
echo "   2. Asegúrate de estar conectado a la misma WiFi"
echo "   3. Escanea el código QR que aparece en mobile.log"
echo ""
echo "🔍 Para ver los logs:"
echo "   Backend:  tail -f backend.log"
echo "   Mobile:   tail -f mobile.log"
echo ""
echo "🛑 Para detener: presiona Ctrl+C"
echo ""

# Mostrar los primeros logs del móvil
log_info "Esperando a que Expo genere el código QR..."
sleep 5

# Mostrar el QR si está disponible
if [ -f "mobile.log" ]; then
    echo ""
    log_info "Últimas líneas de mobile.log:"
    tail -30 mobile.log
fi

# Mantener el script corriendo
log_info "Servicios en ejecución. Presiona Ctrl+C para detener."
wait

