import pytesseract
import time
from datetime import datetime
from ultralytics import YOLO
import cv2
import logging
import numpy as np

from Classes.DetectClasses import DetectClasses
from Classes.Position import Position
from Classes.SafeBox import SafeBox


class DetectStopSign:
    PATO_BRANCO = 'pato_branco'
    CLEVELANDIA = 'clevelandia'
    CLOSED = 'closed'

    def __init__(self):
        # 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
        # self.model = YOLO('/home/paludo/projects/detect_camera_pr280/runs/detect/train9/weights/best.pt')
        self.model = YOLO('/home/paludo/projects/detect_camera_pr280/runs/classify/train3/weights/best.pt')
        # self.model.export(format='onnx')
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
        self.pato_branco_detections = []
        self.clevelandia_detections = []
        self.clevelandia_safe_box = SafeBox(x1 = 90, y1 = 180, x2 = 275, y2 = 35)
        self.pato_branco_safe_triangle = SafeBox(x1 = 100, y1 = 140, x2 = 270, y2 = 140, x3 = 270, y3 = 60)
        self.zoom_factor = 3
        self.confidence = None
        self.selected_class = None

        logging.getLogger("ultralytics").setLevel(logging.WARNING)

    def execute(self):
        while True:
            reading_success, frame = self.cap.read()
            if not reading_success:
                print("Trying to reconnect stream...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url)
                continue

            real_frame = cv2.resize(frame, (1080, 720))
            self.frame_count += 1

            # self.roi_pato_branco_direction(real_frame)
            # self.roi_clevelandia_direction(real_frame)
            selected_class, confidence = self.region_of_intereset(real_frame)

            if selected_class and confidence:
                self.selected_class = selected_class
                self.confidence = confidence
                match self.selected_class:
                    case DetectStopSign.PATO_BRANCO:
                        print(f'O caminho para Pato Branco está liberado. Confiança do resultado: {self.confidence}')
                    case DetectStopSign.CLEVELANDIA:
                        print(f'O caminho para Clevelândia está liberado. Confiança do resultado: {self.confidence}')
                    case DetectStopSign.CLOSED:
                        print(f'Ambos os caminhos estão bloqueados, existe a possibilidade da barreira de clevelândia estar aberta. Confiança do resultado: {confidence}')


            current_time = time.time()
            if current_time - self.last_datetime_read >= 10.0:
                self.get_datetime(real_frame)
                self.last_datetime_read = current_time
                if self.selected_class and self.confidence:
                    match self.selected_class:
                        case DetectStopSign.PATO_BRANCO:
                            print(f'O caminho para Pato Branco está liberado. Confiança do resultado: {self.confidence}')
                        case DetectStopSign.CLEVELANDIA:
                            print(f'O caminho para Clevelândia está liberado. Confiança do resultado: {self.confidence}')
                        case DetectStopSign.CLOSED:
                            print(f'Ambos os caminhos estão bloqueados, existe a possibilidade da barreira de clevelândia estar aberta. Confiança do resultado: {confidence}')

                # self.verify_if_safe_box_contains_object(self.clevelandia_detections, self.clevelandia_safe_box, 'Clevelândia')
                # self.verify_if_safe_box_contains_object(self.pato_branco_detections, self.pato_branco_safe_triangle, 'Pato Branco')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

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
            self.detect_time_diff(datetime_text)

    @staticmethod
    def detect_time_diff(datetime_text):
        try:
            extracted_datetime = datetime.strptime(datetime_text, "%d-%m-%Y %H:%M:%S")
            current_datetime = datetime.now()
            time_diff = (current_datetime - extracted_datetime).total_seconds()

            camera_misconfigured = extracted_datetime.year == 2000

            if time_diff > 0:
                minutes = int(time_diff // 60)
                seconds = int(time_diff % 60)
                if camera_misconfigured:
                        print(f"Atraso: {seconds} segundos (câmera com data desconfigurada)")
                else:
                    if minutes > 0:
                        print(f"Data/Hora: {datetime_text} -> Atraso: {minutes} minutos e {seconds} segundos")
                    else:
                        print(f"Data/Hora: {datetime_text} -> Atraso: {seconds} segundos")

        except ValueError as e:
            print(f"Error on processing date/hour: {e}")

    def detect_objects(self, region_of_interest):
        detected_objects = []
        self.stop_sign_detected = False
        self.confidence = 0
        self.box = None

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

    def classify_frame(self, region_of_interest):
        return self.model.predict(region_of_interest)

    def roi_pato_branco_direction(self, real_frame):
        position = Position(x=275, y=220, width=125, height=50, zoom_factor=self.zoom_factor)
        zoomed_roi, detected_objects = self.zoom_roi_and_detect_objects(real_frame, position)
        if detected_objects:
            self.pato_branco_detections = detected_objects
        color = (0, 255, 0)
        self.print_objects(zoomed_roi, self.pato_branco_detections)

        p1 = (self.pato_branco_safe_triangle.x1, self.pato_branco_safe_triangle.y1)
        p2 = (self.pato_branco_safe_triangle.x2, self.pato_branco_safe_triangle.y2)
        p3 = (self.pato_branco_safe_triangle.x3, self.pato_branco_safe_triangle.y3)
        points = np.array([p1, p2, p3], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(zoomed_roi, [points], isClosed=True, color=color, thickness=2)

        cv2.imshow('PatoBranco', zoomed_roi)

    def roi_clevelandia_direction(self, real_frame):
        position = Position(x=360, y=220, width=125, height=70, zoom_factor=self.zoom_factor)
        zoomed_roi, detected_objects = self.zoom_roi_and_detect_objects(real_frame, position)
        if detected_objects:
            self.clevelandia_detections = detected_objects
        color = (0, 255, 0)
        self.print_objects(zoomed_roi, self.clevelandia_detections)
        cv2.rectangle(
            zoomed_roi,
            (self.clevelandia_safe_box.x1, self.clevelandia_safe_box.y1),
            (self.clevelandia_safe_box.x2, self.clevelandia_safe_box.y2),
            color,
            3
        )
        cv2.imshow('Clevelandia', zoomed_roi)

    def region_of_intereset(self, real_frame):
        position = Position(x=200, y=150, width=460, height=200, zoom_factor=1)
        zoomed_roi, results = self.zoom_roi_and_classify_frame(real_frame, position)

        selected_class = None
        confidence = None
        if results:
            for result in results :
                predicted = result.probs.top1
                selected_class = result.names[predicted]
                confidence = result.probs.top1conf
        cv2.imshow('ROI', zoomed_roi)
        return selected_class, confidence

    def zoom_roi_and_detect_objects(self, real_frame, position):
        roi = real_frame[
              position.y : position.y+position.height,
              position.x : position.x+position.width
              ]
        detected_objects = []
        should_process = self.frame_count % 5 == 0
        if should_process:
            detected_objects = self.detect_objects(roi)
        zoomed_roi = cv2.resize(
            roi,
            None,
            fx=position.zoom_factor,
            fy=position.zoom_factor,
            interpolation=cv2.INTER_LINEAR
        )
        return zoomed_roi, detected_objects

    def zoom_roi_and_classify_frame(self, real_frame, position):
        results = None
        roi = real_frame[
              position.y : position.y+position.height,
              position.x : position.x+position.width
              ]
        should_process = self.frame_count % 5 == 0
        if should_process:
            results = self.classify_frame(roi)
        zoomed_roi = cv2.resize(
            roi,
            None,
            fx=position.zoom_factor,
            fy=position.zoom_factor,
            interpolation=cv2.INTER_LINEAR
        )
        return zoomed_roi, results

    def print_objects(self, roi, detected_objects):
        if detected_objects:
            for obj in detected_objects:
                class_name = obj['class_name']
                confidence = obj['confidence']
                x1, y1, x2, y2 = obj['box']

                x1 = int(x1 * self.zoom_factor)
                y1 = int(y1 * self.zoom_factor)
                x2 = int(x2 * self.zoom_factor)
                y2 = int(y2 * self.zoom_factor)

                colors = {
                    self.detect_classes.stop_sign: (0, 0, 255),  # Vermelho para PARE
                    self.detect_classes.big_cone: (0, 165, 255),  # Laranja para cones grandes
                    self.detect_classes.small_cone: (0, 255, 255),  # Amarelo para cones pequenos
                    self.detect_classes.go_sign: (0, 255, 0)  # Verde para siga
                }

                color = colors.get(obj['class'], (255, 0, 0))

                cv2.rectangle(roi, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(roi, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def verify_if_safe_box_contains_object(self, detected_objects, safe_box: SafeBox, via):
        obj_contained_list = []
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['box']
            x1 = int(x1 * self.zoom_factor)
            y1 = int(y1 * self.zoom_factor)
            x2 = int(x2 * self.zoom_factor)
            y2 = int(y2 * self.zoom_factor)
            contains = safe_box.contains(x2, y2, x1, y1)

            match contains:
                case SafeBox.PARTIAL:
                    obj_contained_list.append(obj)

                case SafeBox.CONTAINS:
                    obj_contained_list.append(obj)

        if obj_contained_list:
            print(f'Sentido {via} bloqueado')
        else:
            print(f'Sentido {via} liberada')
