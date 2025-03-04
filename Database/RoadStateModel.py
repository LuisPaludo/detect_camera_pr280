from Database.DatabaseManager import DatabaseManager
from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func

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

    def __repr__(self):
        return f"<RoadState(id={self.id}, road_state='{self.road_state}', created_at='{self.created_at}')>"

    def to_dict(self):
        return {
            'id': self.id,
            'road_state': self.road_state,
            'confidence': self.confidence,
            'created_at': self.created_at,
            'img_url': self.img_url,
            'updated_at': self.updated_at
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
            db_manager.session.commit()
            print(f'RoadState Added: {new_state}')
            return new_state

        except Exception as e:
            db_manager.session.rollback()
            print(f"Error adding to database: {e}")
            return None