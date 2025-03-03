from ultralytics import YOLO

class TrainedObjectDetectionModel:
    BIG_CONE = 0
    GO_SIGN = 1
    SMALL_CONE = 2
    STOP_SIGN = 3

    def __init__(self):
        self.model = YOLO('/home/paludo/projects/detect_camera_pr280/runs/detect/train9/weights/best.pt')
        self.model.export(format='onnx')
        self.pato_branco_detections = []
        self.clevelandia_detections = []
        self.confidence = None
        self.colors = {
            TrainedObjectDetectionModel.STOP_SIGN: (0, 0, 255),  # Vermelho para PARE
            TrainedObjectDetectionModel.BIG_CONE: (0, 165, 255),  # Laranja para cones grandes
            TrainedObjectDetectionModel.SMALL_CONE: (0, 255, 255),  # Amarelo para cones pequenos
            TrainedObjectDetectionModel.GO_SIGN: (0, 255, 0)  # Verde para siga
        }

    def detect_objects(self, region_of_interest):
        detected_objects = []
        self.confidence = 0

        results = self.model.predict(region_of_interest)

        for r in results:
            boxes = r.boxes
            for b in boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])

                class_name = self.model.names[cls]
                box = b.xyxy[0].cpu().numpy()

                detected_objects.append({
                    'class': cls,
                    'class_name': class_name,
                    'confidence': conf,
                    'box': box
                })
        return detected_objects