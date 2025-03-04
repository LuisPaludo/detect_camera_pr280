from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from decouple import config

class DatabaseManager:
    def __init__(self):
        username = config("DB_USERNAME")
        password = config("DB_PASSWORD")
        host = config("DB_HOST")
        port = config("DB_PORT")
        database = config("DB_NAME")
        database_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(database_url)
        self.base = declarative_base()
        session = sessionmaker(bind=self.engine)
        self.session = session()

    def connect(self):
        try:
            connection = self.engine.connect()
            connection.close()
        except Exception as e:
            print(f"Error on connecting db: {e}")

