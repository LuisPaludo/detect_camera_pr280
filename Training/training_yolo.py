from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='/caminho/para/seu/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)