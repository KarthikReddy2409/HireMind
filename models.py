from sqlalchemy import create_engine, Column, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import uuid
from datetime import datetime

Base = declarative_base()

class Resume(Base):
    __tablename__ = 'resumes'
    resume_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    profile = relationship("CandidateProfile", uselist=False, back_populates="resume")

class CandidateProfile(Base):
    __tablename__ = 'candidate_profiles'
    resume_id = Column(String, ForeignKey('resumes.resume_id'), primary_key=True)
    extracted_data = Column(String)  # JSON string
    fit_score = Column(Float, default=0.0)
    secondary_index = Column(String)  # Anonymized identifier
    reason = Column(String, default='No reason provided.')
    resume = relationship("Resume", back_populates="profile")

engine = create_engine('sqlite:///hiring_tool.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)