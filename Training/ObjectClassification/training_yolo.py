from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(
    data='data/data.yaml',
    epochs=200,
    imgsz=640,
    batch=8,
    workers=4,
    device=0
)
