from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=512,    # Reduzido
    batch=8,      # Reduzido
    workers=4,    # Reduzido
    device=0      # Garante que está usando a GPU
)