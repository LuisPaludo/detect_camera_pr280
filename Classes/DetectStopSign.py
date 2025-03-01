import pytesseract
import time
from datetime import datetime
from ultralytics import YOLO
import cv2
import logging

from Classes.DetectClasses import DetectClasses
from Classes.Position import Position


class DetectStopSign:
    def __init__(self):
        # 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'

        self.model = YOLO('/home/paludo/projects/detect_camera_pr280/runs/detect/train6/weights/best.pt')
        self.model.export(format='onnx')
        self.url = "https://camera1.pr280.com.br/index.m3u8"
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 50000)  # 50 segundos
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 50000)  # 50 segundos
        self.frame_count = 0
        self.last_datetime_read = 0
        self.stop_sign_detected = False
        self.confidence = 0
        self.box = None
        self.detect_classes = DetectClasses()
        self.detected_objects = []
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

    def execute(self):
        while True:
            reading_success, frame = self.cap.read()
            if not reading_success:
                print("Tentando reconectar ao stream...")
                self.cap.release()
                time.sleep(2)  # Esperar antes de tentar novamente
                self.cap = cv2.VideoCapture(self.url)
                continue

            real_frame = cv2.resize(frame, (1080, 720))
            self.frame_count += 1

            self.zoom_image(real_frame)

            current_time = time.time()
            if current_time - self.last_datetime_read >= 5.0:
                self.get_datetime(real_frame)
                self.last_datetime_read = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def reconnect_if_needed(self):
        """Verifica se o stream está funcionando e tenta reconectar se necessário"""
        if not self.cap.isOpened():
            return False

        reading_success, frame = self.cap.read()
        return reading_success

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def get_datetime(self, real_frame):
        datetime_position = Position(x=800, y=10, width=275, height=30)
        datetime_roi = real_frame[
                       datetime_position.y : datetime_position.y + datetime_position.height,
                       datetime_position.x : datetime_position.x + datetime_position.width
                       ]
        gray = cv2.cvtColor(datetime_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        datetime_text = pytesseract.image_to_string(thresh, config='--psm 7')
        datetime_text = datetime_text.strip()
        if datetime_text:
            print(f"Data/Hora extraída: {datetime_text}")
            self.detect_time_diff(datetime_text)

    @staticmethod
    def detect_time_diff(datetime_text):
        try:
            extracted_datetime = datetime.strptime(datetime_text, "%d-%m-%Y %H:%M:%S")
            current_datetime = datetime.now()
            time_diff = (current_datetime - extracted_datetime).total_seconds()
            if time_diff > 0:
                minutes = int(time_diff // 60)
                seconds = int(time_diff % 60)
                if minutes > 0:
                    print(f"Atraso: {minutes} minutos e {seconds} segundos")
                else:
                    print(f"Atraso: {seconds} segundos")

        except ValueError as e:
            print(f"Erro ao processar a data/hora: {e}")

    def detect_objects(self, region_of_interest):
        self.detected_objects = []
        self.stop_sign_detected = False
        self.confidence = 0
        self.box = None

        results = self.model(region_of_interest)

        for r in results:
            boxes = r.boxes
            for b in boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])

                class_name = self.model.names[cls]
                box = b.xyxy[0].cpu().numpy()

                self.detected_objects.append({
                    'class': cls,
                    'class_name': class_name,
                    'confidence': conf,
                    'box': box
                })

    def zoom_image(self, real_frame):
        position = Position(x=250, y=150, width=450, height=300, zoom_factor=3)

        roi = real_frame[
              position.y : position.y+position.height,
              position.x : position.x+position.width
              ]

        should_process = self.frame_count % 5 == 0

        if should_process:
            self.detect_objects(roi)

        zoomed_roi = cv2.resize(
            roi,
            None,
            fx=position.zoom_factor,
            fy=position.zoom_factor,
            interpolation=cv2.INTER_LINEAR
        )

        if hasattr(self, 'detected_objects') and self.detected_objects:
            for obj in self.detected_objects:
                class_name = obj['class_name']
                confidence = obj['confidence']
                x1, y1, x2, y2 = obj['box']

                x1 = int(x1 * position.zoom_factor)
                y1 = int(y1 * position.zoom_factor)
                x2 = int(x2 * position.zoom_factor)
                y2 = int(y2 * position.zoom_factor)

                colors = {
                    self.detect_classes.stop_sign: (0, 0, 255),  # Vermelho para PARE
                    self.detect_classes.big_cone: (0, 165, 255),  # Laranja para cones grandes
                    self.detect_classes.small_cone: (0, 255, 255),  # Amarelo para cones pequenos
                    self.detect_classes.go_sign: (0, 255, 0)  # Verde para siga
                }

                color = colors.get(obj['class'], (255, 0, 0))

                cv2.rectangle(zoomed_roi, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(zoomed_roi, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Zoom', zoomed_roi)

