import { motion } from 'framer-motion';
import { 
  HeartIcon, 
  AcademicCapIcon, 
  CpuChipIcon,
  GlobeAltIcon,
  BookOpenIcon
} from '@heroicons/react/24/outline';

const Footer = () => {
  return (
    <motion.footer
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.2 }}
      className="bg-gradient-to-r from-gray-700 via-gray-800 to-gray-700 text-white mt-16"
    >
      {/* Contenido principal del footer */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* Información de la universidad */}
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center">
                <img 
                  src="/usal-logo.jpg" 
                  alt="Logo USAL" 
                  className="w-8 h-8 object-contain rounded-full"
                />
              </div>
              <div>
                <h3 className="text-lg font-bold text-yellow-300">Universidad del Salvador</h3>
                <p className="text-gray-300 text-sm">Educación de Excelencia</p>
              </div>
            </div>
            <p className="text-gray-400 text-sm leading-relaxed">
              Institución educativa comprometida con la formación integral de profesionales 
              en el campo de la tecnología y la innovación.
            </p>
          </div>

          {/* Información del proyecto */}
          <div className="space-y-4">
            <h3 className="text-lg font-bold text-yellow-300 flex items-center">
              <CpuChipIcon className="w-5 h-5 mr-2" />
              Proyecto Académico
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center text-gray-300">
                <AcademicCapIcon className="w-4 h-4 mr-2 text-yellow-300" />
                <span>Ingeniería en Informática</span>
              </div>
              <div className="flex items-center text-gray-300">
                <BookOpenIcon className="w-4 h-4 mr-2 text-yellow-300" />
                <span>Tecnologías Emergentes</span>
              </div>
              <div className="flex items-center text-gray-300">
                <GlobeAltIcon className="w-4 h-4 mr-2 text-yellow-300" />
                <span>Año 2025</span>
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg p-3">
              <p className="text-xs text-gray-400">
                <strong className="text-yellow-300">Modelo AI:</strong> YOLO v8 entrenado 
                para detección de 5 especies animales con alta precisión.
              </p>
            </div>
          </div>

          {/* Tecnologías utilizadas */}
          <div className="space-y-4">
            <h3 className="text-lg font-bold text-yellow-300">Tecnologías</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="bg-gray-800 rounded-lg p-2 text-center">
                <div className="text-yellow-300 font-semibold">Frontend</div>
                <div className="text-gray-400 text-xs">React + TypeScript</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-2 text-center">
                <div className="text-yellow-300 font-semibold">Backend</div>
                <div className="text-gray-400 text-xs">Python + Flask</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-2 text-center">
                <div className="text-yellow-300 font-semibold">AI Model</div>
                <div className="text-gray-400 text-xs">YOLO v8</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-2 text-center">
                <div className="text-yellow-300 font-semibold">Computer Vision</div>
                <div className="text-gray-400 text-xs">OpenCV</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Barra inferior */}
      <div className="border-t border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center space-x-2 text-gray-400 text-sm">
              <span>Desarrollado con</span>
              <HeartIcon className="w-4 h-4 text-red-500" />
              <span>por estudiantes de USAL</span>
            </div>
            <div className="text-gray-500 text-xs mt-2 md:mt-0">
              © 2025 Universidad del Salvador - Todos los derechos reservados
            </div>
          </div>
        </div>
      </div>
    </motion.footer>
  );
};

export default Footer;
