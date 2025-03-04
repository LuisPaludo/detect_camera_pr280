from ultralytics import YOLO


class TrainedClassifyModel:
    PATO_BRANCO = 'pato_branco'
    CLEVELANDIA = 'clevelandia'
    CLOSED = 'closed'

    def __init__(self):
        self.model = YOLO('/home/paludo/projects/detect_camera_pr280/runs/classify/train5/weights/best.pt')
        self.confidence = None
        self.selected_class = None

    def classify_frame(self, region_of_interest):
        return self.model.predict(region_of_interest)
