import pytesseract
from datetime import datetime


class DatetimeManagement:
    def __init__(self):
        self.last_datetime_read = 0

    @staticmethod
    def get_date_time_text(frame):
        datetime_text = pytesseract.image_to_string(frame, config='--psm 7')
        return datetime_text.strip()

    @staticmethod
    def get_time_diff(datetime_text):
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
