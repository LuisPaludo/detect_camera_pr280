from ultralytics import YOLO

# Inicializar o modelo de classificação
model = YOLO('yolov8s-cls.pt')  # modelo base de classificação

# Treinar o modelo com seu dataset do Roboflow
results = model.train(data='data', epochs=100)