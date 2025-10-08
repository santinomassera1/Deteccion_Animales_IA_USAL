// Colores para cada clase de animal (consistentes con la versión web)
export const ANIMAL_COLORS: Record<string, string> = {
  cat: '#e879f9',      // Magenta brillante
  chicken: '#fb923c',  // Naranja
  cow: '#22c55e',      // Verde
  dog: '#3b82f6',      // Azul
  horse: '#facc15',    // Amarillo brillante
};

export const getAnimalColor = (animalClass: string): string => {
  return ANIMAL_COLORS[animalClass] || '#6b7280';
};

// Traducciones de animales
export const ANIMAL_NAMES: Record<string, string> = {
  cat: 'Gato',
  chicken: 'Gallina',
  cow: 'Vaca',
  dog: 'Perro',
  horse: 'Caballo',
};

export const getAnimalName = (animalClass: string): string => {
  return ANIMAL_NAMES[animalClass] || animalClass;
};

