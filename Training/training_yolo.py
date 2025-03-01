from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,    # Reduzido
    batch=8,      # Reduzido
    workers=4,    # Reduzido
    device=0      # Garante que está usando a GPU
)