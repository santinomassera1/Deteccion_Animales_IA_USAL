import { motion } from 'framer-motion';
import { 
  PhotoIcon, 
  VideoCameraIcon, 
  CameraIcon,
  SparklesIcon
} from '@heroicons/react/24/outline';

interface NavigationProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const Navigation = ({ activeTab, onTabChange }: NavigationProps) => {
  const tabs = [
    {
      id: 'image',
      name: 'Imagen',
      icon: PhotoIcon,
      description: 'Detección en imágenes estáticas',
      color: 'from-green-400 to-green-500',
      hoverColor: 'hover:from-green-500 hover:to-green-600'
    },
    {
      id: 'video',
      name: 'Video',
      icon: VideoCameraIcon,
      description: 'Procesamiento de videos',
      color: 'from-yellow-400 to-yellow-500',
      hoverColor: 'hover:from-yellow-500 hover:to-yellow-600'
    },
    {
      id: 'webcam',
      name: 'Cámara',
      icon: CameraIcon,
      description: 'Detección en tiempo real',
      color: 'from-red-400 to-red-500',
      hoverColor: 'hover:from-red-500 hover:to-red-600'
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.1 }}
      className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden"
    >
      {/* Header de navegación */}
      <div className="bg-gradient-to-r from-green-500 to-green-600 px-6 py-4">
        <div className="flex items-center space-x-3">
          <SparklesIcon className="w-6 h-6 text-yellow-200" />
          <h2 className="text-xl font-bold text-white">Sistema de Detección AI</h2>
        </div>
        <p className="text-green-100 text-sm mt-1">
          Elige el modo de detección que mejor se adapte a tus necesidades
        </p>
      </div>

      {/* Pestañas */}
      <div className="p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            
            return (
              <motion.button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={`
                  relative p-6 rounded-xl border-2 transition-all duration-300
                  ${isActive 
                    ? `bg-gradient-to-r ${tab.color} text-white border-transparent shadow-lg` 
                    : 'bg-gray-50 text-gray-700 border-gray-200 hover:border-gray-300 hover:shadow-md'
                  }
                `}
              >
                {/* Indicador activo */}
                {isActive && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-gradient-to-r from-yellow-300 to-yellow-400 rounded-xl opacity-20"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                
                <div className="relative z-10">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`
                      p-3 rounded-lg
                      ${isActive 
                        ? 'bg-white bg-opacity-20' 
                        : 'bg-white shadow-sm'
                      }
                    `}>
                      <Icon className={`
                        w-6 h-6
                        ${isActive ? 'text-white' : 'text-gray-600'}
                      `} />
                    </div>
                    <h3 className={`
                      font-bold text-lg
                      ${isActive ? 'text-white' : 'text-gray-900'}
                    `}>
                      {tab.name}
                    </h3>
                  </div>
                  
                  <p className={`
                    text-sm leading-relaxed
                    ${isActive ? 'text-white text-opacity-90' : 'text-gray-600'}
                  `}>
                    {tab.description}
                  </p>
                  
                  {/* Indicador de estado */}
                  {isActive && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="flex items-center space-x-1 mt-3"
                    >
                      <div className="w-2 h-2 bg-yellow-200 rounded-full animate-pulse"></div>
                      <span className="text-xs text-yellow-100 font-medium">Activo</span>
                    </motion.div>
                  )}
                </div>
              </motion.button>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
};

export default Navigation;
