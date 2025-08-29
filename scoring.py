"""
Deterministic scoring for HireMind v1.2.
Computes subscores, penalties, aggregate score, and generates reasons.

Notes:
- We normalize weights/penalties to keep totals sane and enforce non-negativity.
- We expose SCORING_VERSION for audit logs and persistence.
"""
from typing import Dict, Any, List, Set
import os

SCORING_VERSION = "1.2.0"


def clamp01(x) -> float:
    try:
        f = float(x)
    except Exception:
        f = 0.0
    return max(0.0, min(1.0, f))


def jaccard(a: Set[str], b: Set[str]) -> float:
    return 0.0 if not a or not b else len(a & b) / len(a | b)


def _lower_set(vals) -> Set[str]:
    if not isinstance(vals, list):
        return set()
    # Normalize common synonyms so JD and resume tech align
    SYN = {
        "sklearn": "scikit-learn",
        "azure ml": "azure",
        "ms azure": "azure",
        "aws sagemaker": "aws",
        "sagemaker": "aws",
    }
    out: Set[str] = set()
    for v in vals:
        s = str(v).strip().lower()
        if not s:
            continue
        s = SYN.get(s, s)
        out.add(s)
    return out


def compute_subscores(parsed: Dict[str, Any], job: Dict[str, Any], team_gap_vec: Dict[str, float]) -> Dict[str, float]:
    tech = parsed.get("tech", {}) if isinstance(parsed.get("tech", {}), dict) else {}
    skills = set()
    for bucket in ("languages", "ml", "or", "cloud"):
        bucket_vals = tech.get(bucket, [])
        if isinstance(bucket_vals, list):
            skills.update(map(lambda s: str(s).lower(), bucket_vals))

    req = _lower_set(job.get("required_skills", []))
    nice = _lower_set(job.get("nice_to_have", []))
    job_match = 0.7 * jaccard(skills, req) + 0.3 * jaccard(skills, nice)

    impacts = parsed.get("impacts", []) if isinstance(parsed.get("impacts", []), list) else []
    has_numbers = sum(1 for i in impacts if isinstance(i, dict) and isinstance(i.get("delta"), (int, float)))
    impact = clamp01(0.2 + 0.2 * min(len(impacts), 5) / 5 + 0.6 * min(has_numbers, 5) / 5)

    ds = parsed.get("data_scale", {}) if isinstance(parsed.get("data_scale", {}), dict) else {}
    rows = str(ds.get("rows", "0")).lower()
    storage = str(ds.get("storage", "0")).lower()

    def to_score(token: str) -> float:
        if "pb" in token or "petabyte" in token:
            return 1.0
        if "tb" in token or "trillion" in token:
            return 0.8
        if "gb" in token or "billion" in token:
            return 0.5
        if "mb" in token or "million" in token:
            return 0.3
        return 0.0

    scale = clamp01(max(to_score(rows), to_score(storage)))

    td = 0.0
    ml_set = _lower_set(tech.get("ml", []))
    ml_signals = {"causal", "bayesian", "transformer", "rl", "llm", "bert", "gpt", "diffusion"}
    mlops_signals = {"feature store", "mlops", "ci/cd", "docker", "kubernetes", "mlflow"}
    if ml_signals.intersection(ml_set):
        td += 0.5
    if mlops_signals.intersection(skills):
        td += 0.3
    if {"xgboost", "lightgbm", "prophet", "tensorflow", "pytorch"}.intersection(ml_set):
        td += 0.2
    technical_depth = clamp01(td)

    or_set = _lower_set(tech.get("or", []))
    opt = 0.0
    or_signals = {"milp", "minlp", "constraint programming", "column generation", "benders", "lagrangian"}
    if or_signals.intersection(or_set):
        opt += 0.7
    if {"gurobi", "cplex", "or-tools", "mosek"}.intersection(skills):
        opt += 0.3
    optimization_rigor = clamp01(opt)

    sys = parsed.get("systems", {}) if isinstance(parsed.get("systems", {}), dict) else {}
    arch = sys.get("arch")
    micro = isinstance(arch, str) and ("microservices" in arch.lower())
    regions = sys.get("regions")
    try:
        regions_n = int(regions or 1)
    except Exception:
        regions_n = 1
    system_complexity = clamp01(0.3 * (1.0 if micro else 0.0) + 0.7 * min(regions_n, 3) / 3)

    lead = parsed.get("leadership", {}) if isinstance(parsed.get("leadership", {}), dict) else {}
    mentored = 0
    try:
        mentored = int(lead.get("mentored", 0) or 0)
    except Exception:
        mentored = 0
    leadership = clamp01(0.5 * (1.0 if lead.get("tech_lead", False) else 0.0) + 0.5 * min(mentored, 5) / 5)

    learning_velocity = clamp01(0.2 + 0.8 * min(len(skills), 40) / 40)

    # Reliability considers oncall + SLO + postmortems led (capped at 5)
    post_led = 0
    try:
        post_led = int(parsed.get("postmortems_led", 0) or 0)
    except Exception:
        post_led = 0
    reliability = clamp01(
        0.2
        + 0.4 * (1.0 if parsed.get("oncall") else 0.0)
        + 0.2 * (1.0 if parsed.get("slo_experience") else 0.0)
        + 0.2 * min(post_led, 5) / 5.0
    )

    # Communication is provided by extractor (0..1), fallback to 0.5
    communication = clamp01(parsed.get("communication_score", 0.5))

    career_progression = 0.5

    domain_fit = jaccard(_lower_set(parsed.get("domains", [])), _lower_set(job.get("domains", [])))

    cover = sum(min(team_gap_vec.get(k, 0.0), 1.0) for k in skills)
    denom = sum(team_gap_vec.values()) or 1.0
    complementarity = clamp01(cover / denom)

    oss = parsed.get("oss_patents", {}) if isinstance(parsed.get("oss_patents", {}), dict) else {}
    oss_count = len(oss.get("oss", [])) if isinstance(oss.get("oss", []), list) else 0
    patent = oss.get("patents", 0)
    try:
        patent_count = int(patent or 0)
    except Exception:
        patent_count = 0
    proof_of_work = clamp01((oss_count + patent_count) / 3.0)

    recency = clamp01(parsed.get("recency_score", 0.5))

    return {
        "job_match": job_match,
        "impact": impact,
        "scale": scale,
        "technical_depth": technical_depth,
        "optimization_rigor": optimization_rigor,
        "system_complexity": system_complexity,
        "leadership": leadership,
        "learning_velocity": learning_velocity,
        "reliability": reliability,
        "communication": communication,
        "domain_fit": domain_fit,
        "career_progression": career_progression,
        "proof_of_work": proof_of_work,
        "recency": recency,
        "complementarity": complementarity,
    }


def compute_penalties(parsed: Dict[str, Any]) -> Dict[str, float]:
    return {
        "vagueness": clamp01(parsed.get("vagueness_density", 0.0)),
        "keyword_stuffing": clamp01(parsed.get("stuffing_score", 0.0)),
        "timeline_inconsistency": clamp01(parsed.get("timeline_issues", 0.0)),
        "unverifiable_claims": clamp01(parsed.get("unverifiable", 0.0)),
        "job_hopping": clamp01(parsed.get("job_hop_score", 0.0)),
    }


def aggregate_score(sub: Dict[str, float], pen: Dict[str, float], cfg: Dict[str, Any], uncertainty: float) -> float:
    W = cfg["weights"]
    P = cfg["penalties"]
    I = cfg.get("interactions", {})
    s = sum(W.get(k, 0.0) * sub.get(k, 0.0) for k in W)
    for key, w in I.items():
        try:
            a, b = key.split("*")
        except ValueError:
            continue
        s += w * sub.get(a, 0.0) * sub.get(b, 0.0)
    s += cfg.get("beta_complementarity", 0.0) * sub.get("complementarity", 0.0)
    s -= sum(P.get(k, 0.0) * pen.get(k, 0.0) for k in P)
    s *= (1.0 - cfg.get("uncertainty_discount", 0.25) * clamp01(uncertainty))
    return s


def scale_to_0_100(scores: List[float]) -> List[float]:
    if not scores:
        return []
    xs = sorted(scores)
    lo_idx = max(0, int(0.01 * len(xs)) - 1)
    hi_idx = min(len(xs) - 1, int(0.99 * len(xs)))
    lo = xs[lo_idx]
    hi = xs[hi_idx]
    if hi - lo <= 1e-12:
        # Flat distribution: assign neutral 50 to avoid all-zeros UI
        return [50.0 for _ in scores]
    rng = hi - lo
    return [round(100.0 * (x - lo) / rng, 2) for x in scores]


def generate_reason(candidate: Dict[str, Any], subscores: Dict[str, float], penalties: Dict[str, float], fit_score: float) -> str:
    top_strengths = sorted(
        [(k, v) for k, v in subscores.items() if k != "complementarity"], key=lambda kv: kv[1], reverse=True
    )[:3]
    concerns = sorted([(k, v) for k, v in penalties.items() if v > 0.2], key=lambda kv: kv[1], reverse=True)[:2]

    impacts = candidate.get("impacts", []) if isinstance(candidate.get("impacts", []), list) else []
    evidences = []
    for i in impacts:
        if isinstance(i, dict) and i.get("evidence"):
            evidences.append(str(i.get("evidence")))
        if len(evidences) >= 2:
            break

    strength_labels = {
        "job_match": "skills alignment",
        "impact": "demonstrated impact",
        "scale": "scale of experience",
        "technical_depth": "technical depth",
        "leadership": "leadership experience",
        "optimization_rigor": "optimization expertise",
    }

    strengths_txt = ", ".join(f"{strength_labels.get(k, k)} ({v:.2f})" for k, v in top_strengths)
    concern_labels = {
        "vagueness": "vague accomplishments",
        "keyword_stuffing": "keyword stuffing",
        "timeline_inconsistency": "timeline gaps",
        "unverifiable_claims": "unverifiable claims",
        "job_hopping": "frequent job changes",
    }
    concerns_txt = ", ".join(concern_labels.get(k, k) for k, v in concerns)

    reason = f"Fit score: {fit_score:.1f}/100. "
    if strengths_txt:
        reason += f"Strengths: {strengths_txt}. "
    if evidences:
        reason += f"Evidence: {'; '.join(evidences)}. "
    if concerns_txt:
        reason += f"Concerns: {concerns_txt}. "
    return reason.strip()


def default_config() -> Dict[str, Any]:
    base = {
        "weights": {
            "job_match": 0.18,
            "impact": 0.16,
            "scale": 0.10,
            "technical_depth": 0.12,
            "system_complexity": 0.06,
            "leadership": 0.09,
            "learning_velocity": 0.05,
            "reliability": 0.06,
            "communication": 0.04,
            "domain_fit": 0.04,
            "career_progression": 0.05,
            "recency": 0.03,
            "proof_of_work": 0.02,
        },
        "interactions": {"technical_depth*scale": 0.04, "impact*leadership": 0.03},
        "penalties": {
            "vagueness": 0.12,
            "keyword_stuffing": 0.08,
            "timeline_inconsistency": 0.10,
            "unverifiable_claims": 0.10,
            "job_hopping": 0.06,
        },
        "beta_complementarity": 0.08,
    "uncertainty_discount": float(os.getenv("UNCERTAINTY_DISCOUNT", "0.25")),
    }
    return _validate_and_normalize_config(base)


def _validate_and_normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize weights to sum to 1, clamp negatives to 0, and validate interactions.

    This prevents accidental over/under-weighting and guards against typos in interactions.
    """
    W = {k: max(0.0, float(v)) for k, v in cfg.get("weights", {}).items()}
    s = sum(W.values()) or 1.0
    cfg["weights"] = {k: (v / s) for k, v in W.items()}
    cfg["penalties"] = {k: max(0.0, float(v)) for k, v in cfg.get("penalties", {}).items()}
    inter = cfg.get("interactions", {}) or {}
    valid_keys = set(cfg["weights"].keys())
    for key in list(inter.keys()):
        try:
            a, b = key.split("*")
        except ValueError:
            raise ValueError(f"Bad interaction term {key}")
        if a not in valid_keys or b not in valid_keys:
            raise ValueError(f"Bad interaction term {key}")
    return cfg
