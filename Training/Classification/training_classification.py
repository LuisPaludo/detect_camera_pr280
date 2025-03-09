from ultralytics import YOLO

model = YOLO()
# model = YOLO('/home/paludo/projects/detect_camera_pr280/runs/classify/train8/weights/best.pt')

results = model.train(
    data='data',
    epochs=300,
    translate=0.0,
    scale=0.0,
    mosaic=0.0,
    fliplr=0.0,
    patience=0
)