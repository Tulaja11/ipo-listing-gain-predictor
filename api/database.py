"""
Database setup using SQLAlchemy + SQLite.
Logs every prediction made through the API — demonstrates persistence /
database integration, not just a stateless model wrapper.
"""
from sqlalchemy import create_engine, Column, Integer, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    issue_size_crores = Column(Float)
    qib_subscription = Column(Float)
    hni_subscription = Column(Float)
    rii_subscription = Column(Float)
    total_subscription = Column(Float)
    offer_price = Column(Float)
    predicted_listing_gain_pct = Column(Float)
    listed_at_premium = Column(Boolean)
    premium_probability = Column(Float)
    created_at = Column(DateTime)


def init_db():
    Base.metadata.create_all(bind=engine)
