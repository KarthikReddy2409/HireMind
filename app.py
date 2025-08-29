import os
import re
import uuid
import json
import logging
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
# SQLAlchemy session is imported from models as DBSession
from dotenv import load_dotenv, find_dotenv
from models import Resume, CandidateProfile, engine, SessionLocal
try:
    from werkzeug.utils import secure_filename as _wk_secure_filename
except Exception:
    _wk_secure_filename = None

# Load environment variables ASAP to ensure OPENAI key is available before importing LLM code
dotenv_path = find_dotenv()
load_dotenv(dotenv_path, override=True)

# Import LLM processors after .env is loaded to avoid missing keys at import time
from resume_processor import extract_text, parse_resume_with_openai, rank_candidates_with_openai
import html
from markupsafe import Markup
import hashlib
import importlib

# Create Flask application
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')
app.config['UPLOAD_FOLDER'] = 'resumes/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

# Configure logging
logging.basicConfig(level=logging.INFO, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional CSRF protection if Flask-WTF is installed
try:
    csrf_mod = importlib.import_module("flask_wtf")
    CSRFProtect = getattr(csrf_mod, "CSRFProtect", None)
    if CSRFProtect:
        app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", app.secret_key)
        app.config["WTF_CSRF_TIME_LIMIT"] = 60 * 60
        CSRFProtect(app)
except Exception:
    csrf_mod = None

def secure_filename(filename):
    """Prefer Werkzeug's secure_filename if available; else fallback to regex sanitizer."""
    if _wk_secure_filename:
        return _wk_secure_filename(filename)
    filename = os.path.basename(filename)
    return re.sub(r'[^A-Za-z0-9_.-]', '_', filename)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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
        # Ensure correct API key for OpenAI is present
        if not os.getenv('OPENAI_API_KEY'):
            logger.error('OPENAI_API_KEY is missing. Please set it in your .env file.')
            flash('OPENAI_API_KEY is missing. Please set it in your .env file.', 'danger')
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
        # Lightweight abuse guard: cap batch size
        if len(resumes) > 100:
            logger.warning('Too many files in one batch upload.')
            flash('Please upload at most 100 files per batch.', 'warning')
            return redirect(url_for('upload'))
        session_db = SessionLocal()
        try:
            candidate_data = []
            for resume in resumes:
                if not resume or not resume.filename:
                    continue
                if not allowed_file(resume.filename):
                    logger.warning(f'Invalid file type for {resume.filename}. Only PDF, DOCX and TXT are supported.')
                    flash(f'Invalid file type for {resume.filename}. Only PDF, DOCX and TXT are supported.', 'danger')
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

                extracted_data = parse_resume_with_openai(resume_text)
                if not extracted_data:
                    flash(f'Failed to process {resume.filename} with OpenAI API.', 'danger')
                    continue

                # Pull the exact redacted body the model saw to keep offsets stable
                red_txt = extracted_data.pop('_redacted_text', None)
                red_hash = hashlib.sha256((red_txt or '').encode('utf-8')).hexdigest() if red_txt else None

                extracted_json = json.dumps(extracted_data)
                profile = CandidateProfile(
                    resume_id=resume_id,
                    extracted_data=extracted_json,
                    secondary_index=f'Candidate_{resume_id[:8]}',
                    redacted_text=red_txt,
                    redacted_text_hash=red_hash,
                )
                session_db.add(profile)
                candidate_data.append({
                    'resume_id': resume_id,
                    'extracted_data': extracted_data
                })

            session_db.commit()
            logger.info('All resumes processed and saved.')

            rankings = rank_candidates_with_openai(job_description, custom_prompt, candidate_data)
            if rankings:
                logger.info('Ranking candidates.')
                for ranking in rankings:
                    profile = session_db.query(CandidateProfile).filter_by(resume_id=ranking.get('resume_id')).first()
                    if profile:
                        profile.fit_score = ranking.get('fit_score', 0.0)
                        profile.reason = ranking.get('reason', profile.reason or 'No reason provided.')
                        # Persist explainability vectors if present
                        subs = ranking.get('subscores')
                        pens = ranking.get('penalties')
                        evid = ranking.get('evidence_snippets')
                        tf = ranking.get('top_factors')
                        spans = ranking.get('evidence_spans')
                        if subs is not None:
                            try:
                                profile.subscores = json.dumps(subs)
                            except Exception:
                                profile.subscores = '{}'
                        if pens is not None:
                            try:
                                profile.penalties = json.dumps(pens)
                            except Exception:
                                profile.penalties = '{}'
                        if evid is not None:
                            try:
                                profile.evidence_snippets = json.dumps(evid)
                            except Exception:
                                profile.evidence_snippets = '[]'
                        if spans is not None:
                            try:
                                profile.evidence_spans = json.dumps(spans)
                            except Exception:
                                profile.evidence_spans = '[]'
                        if tf is not None:
                            try:
                                profile.top_factors = json.dumps(tf)
                            except Exception:
                                profile.top_factors = '[]'
                        if 'uncertainty' in ranking:
                            try:
                                profile.uncertainty = float(ranking.get('uncertainty') or 0.0)
                            except Exception:
                                profile.uncertainty = 0.0
                        if ranking.get('scoring_version'):
                            profile.scoring_version = str(ranking.get('scoring_version'))
                        # Audit log for this scoring event
                        try:
                            logger.info(
                                'scored',
                                extra={
                                    'resume_id': profile.resume_id,
                                    'scoring_version': getattr(profile, 'scoring_version', ''),
                                    'subscores': subs,
                                    'penalties': pens,
                                    'uncertainty': getattr(profile, 'uncertainty', 0.0),
                                    'fit_score': getattr(profile, 'fit_score', 0.0),
                                }
                            )
                        except Exception:
                            # Avoid breaking flow if logger fails on extras
                            logger.info(f"scored resume_id={profile.resume_id} fit={profile.fit_score}")
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

# Expose a csrf_token() helper to templates when CSRF is enabled
@app.context_processor
def inject_csrf_token():
    def _csrf_token():
        try:
            if csrf_mod:
                csrf_csrf = importlib.import_module('flask_wtf.csrf')
                return getattr(csrf_csrf, 'generate_csrf')()
        except Exception:
            return ''
        return ''
    return dict(csrf_token=_csrf_token)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        expected_user = os.getenv('ADMIN_USER', 'admin')
        expected_pass = os.getenv('ADMIN_PASS', 'password')
        is_prod = os.getenv('FLASK_ENV', os.getenv('ENV', '')).lower() == 'production'
        using_defaults = (expected_user == 'admin' and expected_pass == 'password')
        if is_prod and using_defaults:
            logger.error('Default demo credentials are disabled in production.')
            flash('Login disabled: configure ADMIN_USER and ADMIN_PASS for production.', 'danger')
            return render_template('login.html')
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
    db_session = SessionLocal()
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
    # Derive a flat skills list if not present
    skills_list = data.get('skills')
    if not skills_list:
        tech = data.get('tech', {}) if isinstance(data.get('tech', {}), dict) else {}
        skills_list = []
        for k in ('languages', 'ml', 'or', 'cloud'):
            vals = tech.get(k, [])
            if isinstance(vals, list):
                skills_list.extend([str(v) for v in vals])

    # Ensure .0/.5 display of YOE
    yoe = data.get('years_of_experience', 0)
    yoe_display = (int(yoe * 2) / 2.0) if isinstance(yoe, (int, float)) else 0
    # Derive a non-PII display name from the uploaded filename (without extension)
    base_name = os.path.splitext(os.path.basename(resume_obj.file_path))[0]
    # Strip the UUID prefix if present (pattern: <uuid>_originalName)
    parts = base_name.split('_', 1)
    if len(parts) == 2 and re.match(r"^[0-9a-fA-F-]{8}", parts[0] or ""):
        base_name = parts[1]
    display_name = base_name or candidate.secondary_index

    return {
        'rank': idx,
        'candidate_id': candidate.secondary_index,
        'display_name': display_name,
    'resume_id': candidate.resume_id,
        'resume_filename': os.path.basename(resume_obj.file_path),
        'resume_path': resume_obj.file_path,
        'fit_score': fit_score,
        'reason': candidate.reason,  # This will show None/null if not set
        'years_of_experience': yoe_display,
        'skills': ', '.join(skills_list or []),
        'projects': '; '.join(projects)
    }


def highlight_with_spans(text: str, spans: list[dict]) -> str:
    """Render redacted text with <mark> highlights for non-overlapping spans.

    Offsets must match the stored redacted_text (exactly what was sent to the LLM) to avoid drift.
    """
    if not text or not spans:
        return html.escape(text or "")
    # sanitize, sort & merge overlaps
    clean = []
    for s in spans:
        try:
            a = int(s.get('start'))
            b = int(s.get('end'))
            if b > a:
                clean.append((max(0, a), max(0, b)))
        except Exception:
            continue
    clean.sort(key=lambda x: x[0])
    merged = []
    for s,e in clean:
        if not merged or s > merged[-1][1]:
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    # build HTML
    out = []
    prev = 0
    for s,e in merged:
        out.append(html.escape(text[prev:s]))
        out.append(f"<mark>{html.escape(text[s:e])}</mark>")
        prev = e
    out.append(html.escape(text[prev:]))
    return ''.join(out)


@app.route('/candidate/<resume_id>')
@login_required
def candidate_detail(resume_id: str):
    db_session = SessionLocal()
    try:
        profile = db_session.query(CandidateProfile).filter_by(resume_id=resume_id).first()
        if not profile:
            flash('Candidate not found.', 'warning')
            return redirect(url_for('dashboard'))
        # Parse explainability artifacts
        try:
            spans = json.loads(profile.evidence_spans or '[]')
        except Exception:
            spans = []
        try:
            subscores = json.loads(profile.subscores or '{}')
        except Exception:
            subscores = {}
        try:
            penalties = json.loads(profile.penalties or '{}')
        except Exception:
            penalties = {}
        try:
            top_factors = json.loads(getattr(profile, 'top_factors', '[]') or '[]')
        except Exception:
            top_factors = []

        highlighted = ''
        if profile.redacted_text:
            try:
                highlighted = highlight_with_spans(profile.redacted_text, spans)
            except Exception:
                highlighted = html.escape(profile.redacted_text)

        # Pull YoE from extracted_data, if present
        try:
            data = json.loads(profile.extracted_data or '{}')
            yoe = data.get('years_of_experience', 0)
            yoe_display = (int(yoe * 2) / 2.0) if isinstance(yoe, (int, float)) else 0
        except Exception:
            yoe_display = 0

        # Compute display name similar to dashboard
        try:
            base_name = os.path.splitext(os.path.basename(profile.resume.file_path))[0]
            parts = base_name.split('_', 1)
            if len(parts) == 2 and re.match(r"^[0-9a-fA-F-]{8}", parts[0] or ""):
                base_name = parts[1]
            display_name = base_name or profile.secondary_index
        except Exception:
            display_name = profile.secondary_index

        return render_template(
            'candidate.html',
            candidate=profile,
            highlighted_text=highlighted,
            subscores=subscores,
            penalties=penalties,
            top_factors=top_factors,
            yoe_display=yoe_display,
            display_name=display_name,
        )
    except Exception as e:
        logger.error(f"Error loading candidate detail: {e}", exc_info=True)
        flash('Could not load candidate details.', 'danger')
        return redirect(url_for('dashboard'))
    finally:
        db_session.close()
        logger.info('Database session closed (candidate detail).')

@app.teardown_appcontext
def remove_session(exc=None):
    """Ensure the scoped session is removed at the end of the request context."""
    try:
        SessionLocal.remove()
    except Exception:
        pass


@app.after_request
def set_security_headers(resp):
    """Add minimal security headers for basic hardening, with CSP tuned for our templates.

    Notes:
    - Allows Bootstrap from jsdelivr CDN
    - Allows inline styles for the simple highlight view
    - Adjust if you later add other CDNs
    """
    try:
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        csp = (
            "default-src 'self' data:; "
            "img-src 'self' data:; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "script-src 'self' https://cdn.jsdelivr.net"
        )
        resp.headers["Content-Security-Policy"] = csp
    except Exception:
        pass
    return resp


def _highlight_ranges(text: str, spans: list) -> str:
    """Render text with <mark> around spans and escape safely; returns Markup."""
    if not text or not isinstance(spans, list):
        return text or ""
    # sanitize spans and sort by start
    clean = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        try:
            a = int(s.get("start"))
            b = int(s.get("end"))
        except Exception:
            continue
        if b > a:
            clean.append({"start": max(0, a), "end": max(0, b), "type": s.get("type", "evidence")})
    clean.sort(key=lambda d: d["start"]) 

    out, cur = [], 0
    for s in clean:
        st, en = s["start"], s["end"]
        if st > cur:
            out.append(html.escape(text[cur:st]))
        frag = html.escape(text[st:en])
        t = html.escape(str(s.get("type", "evidence")))
        out.append(f"<mark data-type='{t}'>{frag}</mark>")
        cur = en
    if cur < len(text):
        out.append(html.escape(text[cur:]))
    return Markup("".join(out))


@app.route("/candidate/<resume_id>/highlight")
@login_required
def candidate_highlight(resume_id):
    db = SessionLocal()
    try:
        prof = db.query(CandidateProfile).filter_by(resume_id=resume_id).first()
        if not prof or not getattr(prof, 'redacted_text', None):
            return "No redacted text available", 404
        try:
            spans = json.loads(getattr(prof, 'evidence_spans', '[]') or '[]')
        except Exception:
            spans = []
        body = _highlight_ranges(prof.redacted_text, spans)
        # Collect unique types for legend and total count
        try:
            types = sorted({str(s.get('type', 'evidence')) for s in spans if isinstance(s, dict)})
        except Exception:
            types = []
        # Compute display name for title
        try:
            base_name = os.path.splitext(os.path.basename(prof.resume.file_path))[0]
            parts = base_name.split('_', 1)
            if len(parts) == 2 and re.match(r"^[0-9a-fA-F-]{8}", parts[0] or ""):
                base_name = parts[1]
            display_name = base_name or prof.secondary_index
        except Exception:
            display_name = prof.secondary_index

        return render_template(
            'highlight.html',
            candidate=prof,
            highlighted_text=body,
            types=types,
            span_count=len(spans),
            display_name=display_name,
        )
    finally:
        db.close()


@app.route('/resumes/<path:filename>')
@login_required
def download_resume(filename: str):
    # Serve files only from the configured UPLOAD_FOLDER
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    debug_flag = os.getenv('FLASK_DEBUG', 'true').lower() in ('1', 'true', 'yes')
    app.run(debug=debug_flag)