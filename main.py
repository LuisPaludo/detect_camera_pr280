from Classes.DetectStopSign import DetectStopSign
from Database.DatabaseManager import DatabaseManager

if __name__ == "__main__":
    detect_stop_sign = DetectStopSign()
    detect_stop_sign.execute()
    data_base_manager = DatabaseManager()
    data_base_manager.connect()