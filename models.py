from sqlalchemy import create_engine, Column, String, DateTime, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session
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
    extracted_data = Column(Text)  # JSON string (can be large)
    fit_score = Column(Float, default=0.0)
    secondary_index = Column(String)  # Anonymized identifier
    reason = Column(Text, default='No reason provided.')
    # Explainability vectors (as JSON strings for SQLite)
    subscores = Column(Text, default='{}')
    penalties = Column(Text, default='{}')
    evidence_snippets = Column(Text, default='[]')
    evidence_spans = Column(Text, default='[]')  # Offsets for UI highlighting
    top_factors = Column(Text, default='[]')
    uncertainty = Column(Float, default=0.0)
    scoring_version = Column(String, default='1.0.0')
    # Store the exact redacted text sent to the LLM and a hash for integrity/debug
    redacted_text = Column(Text)
    redacted_text_hash = Column(String(64))
    resume = relationship("Resume", back_populates="profile")

engine = create_engine('sqlite:///hiring_tool.db', future=True)
Base.metadata.create_all(engine)
# Thread-safe session factory
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))