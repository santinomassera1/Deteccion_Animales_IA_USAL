import { motion } from 'framer-motion';
import { CpuChipIcon, AcademicCapIcon } from '@heroicons/react/24/outline';

interface HeaderProps {
  isConnected?: boolean;
}

const Header = ({ isConnected = true }: HeaderProps) => {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="bg-gradient-to-r from-green-500 via-green-400 to-green-500 shadow-lg border-b-4 border-yellow-300"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo y título principal */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              {/* Logo USAL */}
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-lg">
                <img 
                  src="/usal-logo.jpg" 
                  alt="Logo USAL" 
                  className="w-12 h-12 object-contain rounded-full"
                />
              </div>
              
              {/* Título principal */}
              <div className="text-white">
                <h1 className="text-2xl font-bold tracking-tight">
                  Detección de Animales
                </h1>
                <p className="text-green-100 text-sm font-medium">
                  Universidad del Salvador
                </p>
              </div>
            </div>
          </div>

          {/* Información académica */}
          <div className="hidden md:flex items-center space-x-6 text-white">
            <div className="text-right">
              <div className="flex items-center space-x-2 text-yellow-200">
                <AcademicCapIcon className="w-5 h-5" />
                <span className="font-semibold">Ingeniería en Informática</span>
              </div>
              <p className="text-green-100 text-sm">Tecnologías Emergentes 2025</p>
            </div>
            
            {/* Indicador de modelo activo */}
            <div className="flex items-center space-x-2 bg-white bg-opacity-20 rounded-lg px-4 py-2">
              <CpuChipIcon className="w-5 h-5 text-yellow-200" />
              <span className="text-sm font-medium">Sistema de Detección Listo</span>
            </div>
            
            {/* Indicador de conexión */}
            <div className={`flex items-center space-x-2 rounded-lg px-3 py-2 ${
              isConnected 
                ? 'bg-green-500 bg-opacity-20' 
                : 'bg-red-500 bg-opacity-20'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-300' : 'bg-red-300'
              } ${isConnected ? 'animate-pulse' : ''}`}></div>
              <span className="text-xs font-medium text-white">
                {isConnected ? 'Conectado' : 'Desconectado'}
              </span>
            </div>
          </div>

          {/* Versión móvil */}
          <div className="md:hidden">
            <div className="text-white text-right">
              <p className="text-sm font-semibold">USAL</p>
              <p className="text-xs text-green-100">Ing. Informática</p>
            </div>
          </div>
        </div>
      </div>

      {/* Barra decorativa inferior */}
      <div className="h-1 bg-gradient-to-r from-yellow-300 via-yellow-200 to-yellow-300"></div>
    </motion.header>
  );
};

export default Header;
