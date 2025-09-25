import { create } from 'zustand';

export interface Detection {
  bbox: [number, number, number, number];
  class: string;
  confidence: number;
  model: string;
}

export interface DetectionResult {
  detections: Detection[];
  original_filename: string;
  original_image: string;
  processed_filename: string;
  processed_image: string;
  total_detections: number;
}

export interface VideoStatus {
  status: 'processing' | 'completed' | 'error' | 'idle';
  progress: number;
  total_frames: number;
  progress_percent: number;
  error?: string;
}

interface AppState {
  // Model status
  modelLoaded: boolean;
  setModelLoaded: (loaded: boolean) => void;
  
  // Current detection
  currentDetection: DetectionResult | null;
  setCurrentDetection: (detection: DetectionResult | null) => void;
  
  // Video processing
  videoStatus: VideoStatus;
  setVideoStatus: (status: VideoStatus) => void;
  
  // Webcam
  webcamActive: boolean;
  setWebcamActive: (active: boolean) => void;
  webcamDetections: Detection[];
  setWebcamDetections: (detections: Detection[]) => void;
  
  // UI state
  activeTab: 'image' | 'video' | 'webcam';
  setActiveTab: (tab: 'image' | 'video' | 'webcam') => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  
  // Notifications
  notifications: Array<{
    id: string;
    type: 'success' | 'error' | 'info' | 'warning';
    title: string;
    message: string;
    timestamp: Date;
  }>;
  addNotification: (notification: Omit<AppState['notifications'][0], 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  // Model status
  modelLoaded: false,
  setModelLoaded: (loaded) => set({ modelLoaded: loaded }),
  
  // Current detection
  currentDetection: null,
  setCurrentDetection: (detection) => set({ currentDetection: detection }),
  
  // Video processing
  videoStatus: {
    status: 'idle',
    progress: 0,
    total_frames: 0,
    progress_percent: 0,
  },
  setVideoStatus: (status) => set({ videoStatus: status }),
  
  // Webcam
  webcamActive: false,
  setWebcamActive: (active) => set({ webcamActive: active }),
  webcamDetections: [],
  setWebcamDetections: (detections) => set({ webcamDetections: detections }),
  
  // UI state
  activeTab: 'image',
  setActiveTab: (tab) => set({ activeTab: tab }),
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
  
  // Notifications
  notifications: [],
  addNotification: (notification) => {
    const id = Math.random().toString(36).substr(2, 9);
    const newNotification = {
      ...notification,
      id,
      timestamp: new Date(),
    };
    set((state) => ({
      notifications: [...state.notifications, newNotification],
    }));
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
      get().removeNotification(id);
    }, 5000);
  },
  removeNotification: (id) => {
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    }));
  },
}));
