

import os
import re
import uuid
import json
import logging
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session
from sqlalchemy.orm import Session
from dotenv import load_dotenv, find_dotenv
from models import Resume, CandidateProfile, engine, Session
from resume_processor import extract_text, parse_resume_with_gemini, rank_candidates_with_gemini

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

# Create Flask application
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['UPLOAD_FOLDER'] = 'resumes/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def secure_filename(filename):
    filename = os.path.basename(filename)
    filename = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    return filename

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if not os.getenv('GEMINI_API_KEY'):
            logger.error('GEMINI_API_KEY is missing. Please set it in your .env file.')
            flash('GEMINI_API_KEY is missing. Please set it in your .env file.', 'danger')
            return render_template('index.html')
        job_description = request.form.get('job_description')
        custom_prompt = request.form.get('custom_prompt', '')
        resumes = request.files.getlist('resumes')

        if not job_description:
            logger.warning('Job description is missing.')
            flash('Job description is required.', 'danger')
            return redirect(url_for('upload'))
        if not resumes or all(not r.filename for r in resumes):
            logger.warning('No resumes uploaded.')
            flash('At least one resume is required.', 'danger')
            return redirect(url_for('upload'))

        session_db = Session(bind=engine)
        try:
            candidate_data = []
            for resume in resumes:
                if not resume.filename.lower().endswith(('.pdf', '.txt')):
                    logger.warning(f'Invalid file type for {resume.filename}. Only PDF and TXT are supported.')
                    flash(f'Invalid file type for {resume.filename}. Only PDF and TXT are supported.', 'danger')
                    continue

                resume_id = str(uuid.uuid4())
                safe_name = secure_filename(resume.filename)
                resume_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{resume_id}_{safe_name}")
                resume.save(resume_path)
                logger.info(f'Saved resume: {resume_path}')

                resume_text = extract_text(resume_path)
                if not resume_text:
                    flash(f'Could not extract text from {resume.filename}.', 'danger')
                    continue

                resume_record = Resume(
                    resume_id=resume_id,
                    file_path=resume_path
                )
                session_db.add(resume_record)

                extracted_data = parse_resume_with_gemini(resume_text)
                if not extracted_data:
                    flash(f'Failed to process {resume.filename} with Gemini API.', 'danger')
                    continue

                extracted_json = json.dumps(extracted_data)
                profile = CandidateProfile(
                    resume_id=resume_id,
                    extracted_data=extracted_json,
                    secondary_index=extracted_data.get('anonymized_id', f'Candidate_{resume_id[:8]}')
                )
                session_db.add(profile)
                candidate_data.append({
                    'resume_id': resume_id,
                    'extracted_data': extracted_data
                })

            session_db.commit()
            logger.info('All resumes processed and saved.')

            rankings = rank_candidates_with_gemini(job_description, custom_prompt, candidate_data)
            if rankings:
                logger.info('Ranking candidates.')
                for ranking in rankings:
                    profile = session_db.query(CandidateProfile).filter_by(resume_id=ranking.get('resume_id')).first()
                    if profile:
                        profile.fit_score = ranking.get('fit_score', 0.0)
                session_db.commit()
                logger.info('Candidate rankings committed to database.')

            logger.info('Redirecting to dashboard.')
            return redirect(url_for('dashboard'))
        except Exception as e:
            session_db.rollback()
            logger.error(f"Error processing resumes: {str(e)}", exc_info=True)
            flash('An error occurred while processing your request.', 'danger')
        finally:
            session_db.close()
            logger.info('Database session closed.')
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        expected_user = os.getenv('ADMIN_USER', 'admin')
        expected_pass = os.getenv('ADMIN_PASS', 'password')
        if username == expected_user and password == expected_pass:
            session['logged_in'] = True
            logger.info(f'User {username} logged in successfully.')
            return redirect(url_for('upload'))
        else:
            logger.warning(f'Failed login attempt for user {username}.')
            flash('Invalid credentials.', 'danger')
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    db_session = Session(bind=engine)
    try:
        candidates = db_session.query(CandidateProfile).order_by(CandidateProfile.fit_score.desc()).all()
        logger.info('Loaded candidates for dashboard')
        candidates_data = []
        for idx, candidate in enumerate(candidates, 1):
            resume_obj = db_session.query(Resume).filter_by(resume_id=candidate.resume_id).first()
            if not resume_obj:
                logger.warning(f'No resume found for candidate {candidate.resume_id}')
                continue
            try:
                data = json.loads(candidate.extracted_data)
                candidate_info = format_candidate_data(idx, candidate, resume_obj, data)
                candidates_data.append(candidate_info)
            except json.JSONDecodeError:
                logger.error(f'Invalid JSON data for candidate {candidate.resume_id}')
                continue
            except Exception as e:
                logger.error(f'Error processing candidate {candidate.resume_id}: {str(e)}')
                continue
        return render_template('dashboard.html', candidates=candidates_data)
    except Exception as e:
        logger.error(f'Error loading dashboard: {str(e)}', exc_info=True)
        flash('An error occurred while loading the dashboard.', 'danger')
        return render_template('dashboard.html', candidates=[])
    finally:
        db_session.close()
        logger.info('Database session closed (dashboard).')

def format_candidate_data(idx, candidate, resume_obj, data):
    fit_score = 0
    if candidate.fit_score:
        try:
            fit_score = round(float(candidate.fit_score), 2)
        except (TypeError, ValueError):
            logger.warning(f'Invalid fit score for candidate {candidate.resume_id}')
    projects = []
    for project in data.get('projects', []):
        if isinstance(project, dict) and 'name' in project:
            projects.append(project['name'])
    return {
        'rank': idx,
        'candidate_id': candidate.secondary_index,
        'resume_filename': os.path.basename(resume_obj.file_path),
        'resume_path': resume_obj.file_path,
        'fit_score': fit_score,
        'reason': candidate.reason,  # This will show None/null if not set
        'years_of_experience': int(data.get('years_of_experience', 0)),
        'skills': ', '.join(data.get('skills', [])),
        'projects': '; '.join(projects)
    }

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)