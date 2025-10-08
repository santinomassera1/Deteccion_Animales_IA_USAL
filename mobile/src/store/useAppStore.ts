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
  status: 'processing' | 'completed' | 'error' | 'idle' | 'failed';
  progress: number;
  processed_frames: number;
  total_frames: number;
  progress_percent: number;
  output_video_url?: string;
  error?: string;
}

export type NotificationType = 'success' | 'error' | 'info' | 'warning';

export interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  timestamp: Date;
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
  uploadedVideoFilename: string | null;
  setUploadedVideoFilename: (filename: string | null) => void;
  
  // Camera
  cameraActive: boolean;
  setCameraActive: (active: boolean) => void;
  cameraDetections: Detection[];
  setCameraDetections: (detections: Detection[]) => void;
  
  // UI state
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  isConnected: boolean;
  setIsConnected: (connected: boolean) => void;
  
  // Notifications
  notifications: Notification[];
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
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
    processed_frames: 0,
    total_frames: 0,
    progress_percent: 0,
  },
  setVideoStatus: (status) => set({ videoStatus: status }),
  uploadedVideoFilename: null,
  setUploadedVideoFilename: (filename) => set({ uploadedVideoFilename: filename }),
  
  // Camera
  cameraActive: false,
  setCameraActive: (active) => set({ cameraActive: active }),
  cameraDetections: [],
  setCameraDetections: (detections) => set({ cameraDetections: detections }),
  
  // UI state
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
  isConnected: false,
  setIsConnected: (connected) => set({ isConnected: connected }),
  
  // Notifications
  notifications: [],
  addNotification: (notification) => {
    const id = Math.random().toString(36).substr(2, 9);
    const newNotification: Notification = {
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
  clearNotifications: () => set({ notifications: [] }),
}));

