import logging

import os
import uuid
import time
import cv2
from datetime import datetime

from Classes.DatetimeManagement import DatetimeManagement
from Classes.Position import Position
from Classes.TrainedClassifyModel import TrainedClassifyModel
from decouple import config

from Database.RoadStateModel import RoadStateModel

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "road_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class CameraDetection:

    def __init__(self):
        self.model = TrainedClassifyModel()
        self.date_time_management = DatetimeManagement()

        self.url = config('URL')
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 50000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 50000)
        self.frame_count = 0
        self.measurement_interval = 15

        self.db_road_state = RoadStateModel()

        logging.getLogger("ultralytics").setLevel(logging.WARNING)

    def execute(self):
        while True:
            reading_success, frame = self.cap.read()
            if not reading_success:
                self.reconnect()
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            real_frame = cv2.resize(gray_frame, (1080, 720))
            self.frame_count += 1

            selected_class, confidence, zoomed_roi = self.region_of_interest(real_frame)

            if selected_class and confidence:
                self.model.add_classification(selected_class, confidence)

            current_time = time.time()
            time_interval = current_time - self.date_time_management.last_datetime_read
            if time_interval >= self.measurement_interval:
                self.date_time_management.last_datetime_read = current_time

                most_frequent_class, avg_confidence = self.get_measurement()
                img_url_path = self.generate_img(zoomed_roi, most_frequent_class)

                if most_frequent_class and avg_confidence:
                    self.db_road_state.add(most_frequent_class, avg_confidence, img_url_path)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def reconnect(self):
        print("Trying to reconnect stream...")
        self.cap.release()
        time.sleep(2)
        self.cap = cv2.VideoCapture(self.url)

    @staticmethod
    def generate_img(frame, most_frequent_class):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4().hex[:8])
        filename = f'{timestamp}_road_{most_frequent_class}_{unique_id}.jpg'
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        success = cv2.imwrite(file_path, frame)
        return f'/road_images/{filename}' if success else ''

    @staticmethod
    def get_datetime_frame(real_frame):
        datetime_position = Position(x=800, y=10, width=275, height=30)
        datetime_roi = real_frame[
                       datetime_position.y: datetime_position.y + datetime_position.height,
                       datetime_position.x: datetime_position.x + datetime_position.width
                       ]
        _, date_time_frame = cv2.threshold(datetime_roi, 150, 255, cv2.THRESH_BINARY_INV)
        return date_time_frame

    def get_date_time(self, real_frame, current_time):
        date_time_frame = self.get_datetime_frame(real_frame)
        date_time_text = self.date_time_management.get_date_time_text(date_time_frame)
        self.date_time_management.get_time_diff(date_time_text)
        self.date_time_management.last_datetime_read = current_time
        return self.date_time_management.datetime_text_to_datetime(date_time_text)

    def get_measurement(self):
        most_frequent_class, avg_confidence = self.model.get_most_frequent_class()
        self.model.clear_data()
        return most_frequent_class, avg_confidence

    def get_zoomed_frame(self, real_frame):
        position = Position(x=200, y=150, width=460, height=200, zoom_factor=1)
        return self.zoom_roi(real_frame, position)

    @staticmethod
    def zoom_roi(real_frame, position):
        roi = real_frame[
              position.y: position.y + position.height,
              position.x: position.x + position.width
              ]
        return cv2.resize(
            roi,
            None,
            fx=position.zoom_factor,
            fy=position.zoom_factor,
            interpolation=cv2.INTER_LINEAR
        )

    def region_of_interest(self, real_frame):
        zoomed_roi = self.get_zoomed_frame(real_frame)
        results = self.classify_frame(zoomed_roi)

        selected_class = None
        confidence = None
        if results:
            for result in results:
                predicted = result.probs.top1
                selected_class = result.names[predicted]
                confidence = result.probs.top1conf
        cv2.imshow('ROI', zoomed_roi)
        return selected_class, confidence, zoomed_roi

    def classify_frame(self, zoomed_roi):
        results = None
        should_process = self.frame_count % 2 == 0
        if should_process:
            results = self.model.classify_frame(zoomed_roi)
        return results
