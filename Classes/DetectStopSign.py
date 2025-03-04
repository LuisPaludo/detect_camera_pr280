import time
import cv2
import logging

from Classes.DatetimeManagement import DatetimeManagement
from Classes.Position import Position
from Classes.TrainedClassifyModel import TrainedClassifyModel
from decouple import config

class DetectStopSign:

    def __init__(self):
        self.model = TrainedClassifyModel()
        self.date_time_management = DatetimeManagement()

        self.url =config('URL')
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 50000)  # 50 segundos
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 50000)  # 50 segundos
        self.frame_count = 0

        logging.getLogger("ultralytics").setLevel(logging.WARNING)

    def execute(self):
        while True:
            reading_success, frame = self.cap.read()
            if not reading_success:
                self.reconnect()
                continue

            real_frame = cv2.resize(frame, (1080, 720))
            self.frame_count += 1

            selected_class, confidence = self.region_of_intereset(real_frame)

            if selected_class and confidence:
                self.model.selected_class = selected_class
                self.model.confidence = confidence
                match self.model.selected_class:
                    case TrainedClassifyModel.PATO_BRANCO:
                        print(f'O caminho para Pato Branco está liberado. Confiança do resultado: {self.model.confidence}')
                    case TrainedClassifyModel.CLEVELANDIA:
                        print(f'O caminho para Clevelândia está liberado. Confiança do resultado: {self.model.confidence}')
                    case TrainedClassifyModel.CLOSED:
                        print(f'Ambos os caminhos estão bloqueados, existe a possibilidade da barreira de clevelândia estar aberta. Confiança do resultado: {confidence}')

            current_time = time.time()
            if current_time - self.date_time_management.last_datetime_read >= 1:
                date_time_frame = self.get_datetime_frame(real_frame)
                date_time_text = self.date_time_management.get_date_time_text(date_time_frame)
                self.date_time_management.get_time_diff(date_time_text)

                self.date_time_management.last_datetime_read = current_time
                # if self.model.selected_class and self.model.confidence:
                #     match self.model.selected_class:
                #         case TrainedClassifyModel.PATO_BRANCO:
                #             print(f'O caminho para Pato Branco está liberado. Confiança do resultado: {self.model.confidence}')
                #         case TrainedClassifyModel.CLEVELANDIA:
                #             print(f'O caminho para Clevelândia está liberado. Confiança do resultado: {self.model.confidence}')
                #         case TrainedClassifyModel.CLOSED:
                #             print(f'Ambos os caminhos estão bloqueados, existe a possibilidade da barreira de clevelândia estar aberta. Confiança do resultado: {confidence}')

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
    def get_datetime_frame(real_frame):
        datetime_position = Position(x=800, y=10, width=275, height=30)
        datetime_roi = real_frame[
                       datetime_position.y : datetime_position.y + datetime_position.height,
                       datetime_position.x : datetime_position.x + datetime_position.width
                       ]
        gray = cv2.cvtColor(datetime_roi, cv2.COLOR_BGR2GRAY)
        _, date_time_frame = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        return date_time_frame

    def get_zoomed_frame(self, real_frame):
        position = Position(x=200, y=150, width=460, height=200, zoom_factor=1)
        return self.zoom_roi(real_frame, position)

    @staticmethod
    def zoom_roi(real_frame, position):
        roi = real_frame[
              position.y : position.y+position.height,
              position.x : position.x+position.width
              ]
        return cv2.resize(
            roi,
            None,
            fx=position.zoom_factor,
            fy=position.zoom_factor,
            interpolation=cv2.INTER_LINEAR
        )

    def region_of_intereset(self, real_frame):
        zoomed_roi = self.get_zoomed_frame(real_frame)
        results = self.classify_frame(zoomed_roi)

        selected_class = None
        confidence = None
        if results:
            for result in results :
                predicted = result.probs.top1
                selected_class = result.names[predicted]
                confidence = result.probs.top1conf
        cv2.imshow('ROI', zoomed_roi)
        return selected_class, confidence

    def classify_frame(self, zoomed_roi):
        results = None
        should_process = self.frame_count % 5 == 0
        if should_process:
            results = self.model.classify_frame(zoomed_roi)
        return results
