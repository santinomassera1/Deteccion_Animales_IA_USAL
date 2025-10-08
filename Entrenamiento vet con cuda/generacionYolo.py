from ultralytics import YOLO
import yaml
import os

def train_model(data_yaml_path, epochs=90, img_size=416):
    """
    Train YOLO model with hardware-optimized parameters for animals dataset
    Dataset: 5 classes (cat, chicken, cow, dog, horse)
    Images: 416x416 pixels, 2765 total images
    """
    # Set CUDA memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    
    # Initialize model with pretrained weights (YOLOv8m)
    model = YOLO('yolov8m.pt')
    
    print("\nStarting hardware-optimized training with configuration:")
    print(f"- Image size: {img_size}px")
    print(f"- Batch size: 8")
    print(f"- Epochs: {epochs}")
    print(f"- Dataset: {data_yaml_path}")

 
    # Train with hardware-optimized parameters for animals dataset (YOLOv8m)
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,        # 416px optimized for animals dataset
        device=0,              # GPU acceleration
        batch=8,               # Batch size 8 para YOLOv8m
        workers=6,             # Optimized for i7-7700K (8 threads)
        project='runs',
        name='animals_training_m',   # Training results for animals dataset (medium)
        exist_ok=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,       # Reduced for faster start
        cos_lr=True,
        half=True,            # Mixed precision for memory efficiency
        augment=True,
        mixup=0.1,
        degrees=10,
        shear=2.0,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        copy_paste=0.3,
        patience=20,           # Reduced for faster convergence
        save_period=10,
        verbose=True,
        cache=True,           # Cache enabled for NVMe
        amp=True,             # Automatic mixed precision
        overlap_mask=False,   # Reduce memory usage
        deterministic=False   # Better performance
    )
    return results



if __name__ == "__main__":
    # Path to your data.yaml file
    data_yaml_path = 'DataSet_Veterinaria/entrenamiento Nacho/data.yaml'
    
    print("\n=== Starting YOLOv8 Training ===")
    print("Dataset: Animals Dataset (Veterinary)")
    print("Model: YOLOv8m (MEDIUM)")
    print("GPU: RTX 2080 SUPER")
    print("Classes: cat, chicken, cow, dog, horse")
    print("="*30 + "\n")
    
    try:
        # Train the model
        results = train_model(data_yaml_path)
        print("\n=== Training Completed Successfully ===")
        print(f"Results saved in: {results.save_dir}")
        print(f"Best model: {results.save_dir}/weights/best.pt")
        print("="*30)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {str(e)}")
