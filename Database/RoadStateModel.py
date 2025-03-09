from Database.DatabaseManager import DatabaseManager
from sqlalchemy import Column, Integer, String, DateTime, Float, Interval
from sqlalchemy.sql import func
from collections import Counter
import datetime

db_manager = DatabaseManager()
Base = db_manager.base


class RoadStateModel(Base):
    __tablename__ = 'road_state'

    id = Column(Integer, primary_key=True, autoincrement=True)
    road_state = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    img_url = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    moving_avg_state = Column(String(50), nullable=True)
    moving_avg_percentage = Column(Float, nullable=True)
    state_duration = Column(Interval, nullable=True)

    def __repr__(self):
        return (
            f"<RoadState(id={self.id},"
            f" road_state='{self.road_state}',"
            f" confidence={self.confidence},"
            f" created_at='{self.created_at}',"
            f" state_duration='{self.state_duration}',"
            f" moving_avg_state='{self.moving_avg_state}',"
            f" moving_avg_percentage='{self.moving_avg_percentage}',)>"
        )

    def to_dict(self):
        return {
            'id': self.id,
            'road_state': self.road_state,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'img_url': self.img_url,
            'updated_at': self.updated_at,
            'moving_avg_state': self.moving_avg_state,
            'moving_avg_percentage': self.moving_avg_percentage,
            'state_duration': self.state_duration,
            'formatted_duration': self.format_duration() if self.state_duration else "00:00:00"
        }

    @staticmethod
    def add(road_state, confidence, img_url):
        try:
            new_state = RoadStateModel(
                road_state=road_state,
                img_url=img_url,
                confidence=confidence.detach().cpu().item()
            )

            db_manager.session.add(new_state)
            db_manager.session.flush()

            moving_avg = RoadStateModel.calculate_moving_average()
            new_state.moving_avg_state = moving_avg["state"]
            new_state.moving_avg_percentage = moving_avg["percentage"]

            new_state.calculate_state_duration()

            db_manager.session.commit()

            print(f'RoadState Added: {new_state}')
            return new_state

        except Exception as e:
            db_manager.session.rollback()
            print(f"Error adding to database: {e}")
            return None

    @staticmethod
    def calculate_moving_average(n=5):
        try:
            session = db_manager.session
            records = session.query(RoadStateModel) \
                .order_by(RoadStateModel.created_at.desc()) \
                .limit(n).all()

            if not records:
                return {"state": None, "percentage": 0}

            states = [record.road_state for record in records]
            counter = Counter(states)

            most_common_state, count = counter.most_common(1)[0]
            percentage = (count / len(records)) * 100

            return {
                "state": most_common_state,
                "percentage": round(percentage, 2)
            }
        except Exception as e:
            print(f"Error calculating moving average: {e}")
            return {"state": None, "percentage": 0}

    @staticmethod
    def update_all_moving_averages(n=5):
        session = db_manager.session
        try:
            records = session.query(RoadStateModel).all()

            for i, record in enumerate(records):
                if i < n - 1:
                    subset = records[:i + 1]
                else:
                    subset = records[i - (n - 1):i + 1]

                states = [r.road_state for r in subset]
                counter = Counter(states)
                most_common_state, count = counter.most_common(1)[0]
                percentage = (count / len(subset)) * 100

                record.moving_avg_state = most_common_state
                record.moving_avg_percentage = round(percentage, 2)

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error updating all moving averages: {e}")
            return False

    def update_moving_average(self, n=5):
        try:
            moving_avg = RoadStateModel.calculate_moving_average(n)
            self.moving_avg_state = moving_avg["state"]
            self.moving_avg_percentage = moving_avg["percentage"]
            db_manager.session.commit()
            return True
        except Exception as e:
            db_manager.session.rollback()
            print(f"Error updating moving average: {e}")
            return False

    def calculate_state_duration(self):
        try:
            session = db_manager.session
            current_avg_state = self.moving_avg_state

            if not current_avg_state:
                self.state_duration = datetime.timedelta(0)
                return

            previous_different_state = session.query(RoadStateModel) \
                .filter(RoadStateModel.moving_avg_state != current_avg_state) \
                .filter(RoadStateModel.id < self.id) \
                .order_by(RoadStateModel.id.desc()) \
                .first()

            if not previous_different_state:
                first_record = session.query(RoadStateModel) \
                    .filter(RoadStateModel.moving_avg_state == current_avg_state) \
                    .order_by(RoadStateModel.id) \
                    .first()

                if first_record and first_record.id != self.id:
                    self.state_duration = self.created_at - first_record.created_at
                else:
                    self.state_duration = datetime.timedelta(0)
            else:
                first_record_with_current_state = session.query(RoadStateModel) \
                    .filter(RoadStateModel.moving_avg_state == current_avg_state) \
                    .filter(RoadStateModel.id > previous_different_state.id) \
                    .filter(RoadStateModel.id <= self.id) \
                    .order_by(RoadStateModel.id) \
                    .first()

                if first_record_with_current_state:
                    self.state_duration = self.created_at - first_record_with_current_state.created_at
                else:
                    self.state_duration = datetime.timedelta(0)

        except Exception as e:
            print(f"Error calculating state duration: {e}")
            self.state_duration = datetime.timedelta(0)

    def format_duration(self):
        if self.state_duration is None:
            return "00:00:00"

        total_seconds = self.state_duration.total_seconds()

        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"