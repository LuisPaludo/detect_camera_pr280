from ultralytics import YOLO
from collections import defaultdict

class TrainedClassifyModel:
    PATO_BRANCO = 'pato_branco'
    CLEVELANDIA = 'clevelandia'
    CLOSED = 'closed'

    def __init__(self):
        self.model = YOLO('/home/paludo/projects/detect_camera_pr280/runs/classify/train9/weights/best.pt')
        self.confidence = None
        self.selected_class = None

        self.class_data = defaultdict(list)

    def classify_frame(self, region_of_interest):
        return self.model.predict(region_of_interest)

    def add_classification(self, selected_class, confidence):
        if selected_class and confidence:
            self.class_data[selected_class].append(confidence)

    def get_most_frequent_class(self):
        if not self.class_data:
            return None, 0.0

        most_frequent_class = max(self.class_data, key=lambda k: len(self.class_data[k]))
        avg_confidence = sum(self.class_data[most_frequent_class]) / len(self.class_data[most_frequent_class])
        return most_frequent_class, avg_confidence

    def clear_data(self):
        self.class_data.clear()
