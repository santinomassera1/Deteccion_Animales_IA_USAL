#!/bin/bash

# Script para iniciar la app móvil de detección de animales
# Ejecutar desde la carpeta mobile/

echo "🚀 Iniciando App Móvil de Detección de Animales USAL"
echo "===================================================="
echo ""

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

# Verificar que estamos en mobile/
if [ ! -f "package.json" ] || [ ! -f "app.json" ]; then
    log_error "Error: Ejecuta este script desde la carpeta mobile/"
    exit 1
fi

# Detectar IP local
log_info "Detectando IP local de tu Mac..."
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -n1 | awk '{print $2}')

if [ -z "$LOCAL_IP" ]; then
    log_error "No se pudo detectar la IP local. Verifica tu conexión WiFi."
    exit 1
fi

log_success "IP detectada: $LOCAL_IP"

# Actualizar IP en api.ts
log_info "Actualizando configuración del backend..."
API_FILE="src/services/api.ts"

if [ -f "$API_FILE" ]; then
    sed -i '' "s|const API_BASE_URL = 'http://.*:5003'|const API_BASE_URL = 'http://$LOCAL_IP:5003'|g" "$API_FILE"
    log_success "Backend configurado en: http://$LOCAL_IP:5003"
else
    log_warning "No se encontró $API_FILE"
fi

echo ""
log_warning "IMPORTANTE: Asegúrate de que el backend Python esté corriendo:"
echo "  cd .. && python3 app.py"
echo ""

# Verificar node_modules
if [ ! -d "node_modules" ]; then
    log_info "Instalando dependencias..."
    npm install --legacy-peer-deps
fi

log_info "Iniciando Expo..."
echo ""
log_success "===================================================="
log_success "Para conectar tu dispositivo móvil:"
log_success "===================================================="
echo ""
echo "📱 1. Descarga 'Expo Go' en tu teléfono"
echo "   iOS: App Store"
echo "   Android: Play Store"
echo ""
echo "📱 2. Conéctate a la misma WiFi que tu Mac"
echo ""
echo "📱 3. Escanea el código QR que aparecerá abajo"
echo ""
echo "🛑 Para detener: presiona Ctrl+C"
echo ""

# Iniciar Expo
npx expo start
