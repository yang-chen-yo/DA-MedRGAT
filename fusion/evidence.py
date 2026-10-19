import math
import re
from dataclasses import dataclass
from typing import Dict, Any, Set, Optional

from config import (
    SUPPORT_RETRIEVAL_WEIGHT,
    SUPPORT_NLI_WEIGHT,
    SUPPORT_LITERATURE_WEIGHT,
    SUPPORT_KB_WEIGHT,
    SUPPORT_SCORE_THRESHOLD,
    SUPPORT_ACCEPT_THRESHOLD,
    SUPPORT_REJECT_THRESHOLD,
    SUPPORT_REVIEW_ENABLED,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class SupportResult:
    score: float
    status: str
    components: Dict[str, float]


def _tokens(text: str) -> Set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}


def _sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)


def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _entity_surface(triple: Any) -> str:
    head = getattr(triple, "head", "") or ""
    tail = getattr(triple, "tail", "") or ""
    relation = getattr(triple, "relation", "") or ""
    return f"{head} {relation} {tail}".strip()


def retrieval_support_score(retrieval_score: Optional[float]) -> float:
    if retrieval_score is None:
        return 0.0
    try:
        value = float(retrieval_score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    value = max(0.0, min(1.0, value))
    return _clip(_sigmoid(value))


def nli_proxy_score(
    triple: Any,
    evidence_text: Optional[str],
    evidence_title: Optional[str] = None,
) -> float:
    claim_text = _entity_surface(triple).lower()
    text_blob = f"{evidence_title or ''} {evidence_text or ''}".strip().lower()
    if not text_blob:
        return 0.0
    if claim_text and claim_text in text_blob:
        return 1.0

    claim_tokens = _tokens(claim_text)
    text_tokens = _tokens(text_blob)
    if not claim_tokens or not text_tokens:
        return 0.0

    overlap = len(claim_tokens & text_tokens)
    union = len(claim_tokens | text_tokens)
    jaccard = overlap / union if union else 0.0
    coverage = overlap / max(len(claim_tokens), 1)
    if coverage >= 0.50 or jaccard >= 0.30:
        return 1.0
    score = 0.15 + 0.55 * jaccard + 0.30 * coverage
    return _clip(score)


def kb_consistency_score(
    triple: Any,
    patient_context: Optional[str],
    evidence_source: Optional[str],
) -> float:
    if (evidence_source or "").upper() == "UMLS":
        return 1.0

    context_tokens = _tokens(patient_context or "")
    head_tokens = _tokens(getattr(triple, "head", "") or "")
    tail_tokens = _tokens(getattr(triple, "tail", "") or "")
    relation_tokens = _tokens(getattr(triple, "relation", "") or "")

    if not context_tokens:
        return 0.50

    head_hit = len(head_tokens & context_tokens) > 0
    tail_hit = len(tail_tokens & context_tokens) > 0
    relation_hit = len(relation_tokens & context_tokens) > 0

    score = 0.25
    score += 0.30 if head_hit else 0.0
    score += 0.25 if tail_hit else 0.0
    score += 0.20 if relation_hit else 0.0
    if head_hit or tail_hit or relation_hit:
        score = max(score, 0.75)
    return _clip(score)


def literature_support_score(
    triple: Any,
    patient_context: Optional[str],
    evidence_text: Optional[str],
    evidence_title: Optional[str] = None,
    evidence_source: Optional[str] = None,
) -> float:
    """Quantify how strongly a published or literature-derived evidence supports the triple."""
    if not evidence_text and not evidence_title:
        return 0.0

    claim_text = _entity_surface(triple).lower()
    text_blob = f"{evidence_title or ''} {evidence_text or ''}".strip().lower()
    if not text_blob:
        return 0.0

    exact_match = 1.0 if claim_text and claim_text in text_blob else 0.0
    claim_tokens = _tokens(claim_text)
    text_tokens = _tokens(text_blob)
    if not claim_tokens or not text_tokens:
        return 0.0

    overlap = len(claim_tokens & text_tokens)
    coverage = overlap / max(len(claim_tokens), 1)
    source_boost = 0.18 if (evidence_source or "").upper() in {"PUBMED", "LITERATURE", "MEDICAL_LITERATURE"} else 0.0

    patient_tokens = _tokens(patient_context or "")
    context_overlap = len(patient_tokens & text_tokens) / max(len(patient_tokens | text_tokens), 1)

    # Strong literature support should be close to 1.0 when the same disease and context are present.
    if coverage >= 0.50 or context_overlap >= 0.25:
        return _clip(0.90 + source_boost)
    score = 0.20 + 0.50 * coverage + 0.35 * exact_match + 0.25 * context_overlap + source_boost
    return _clip(score)


def assess_triple_support(
    triple: Any,
    retrieval_score: Optional[float],
    patient_context: Optional[str],
    evidence_text: Optional[str],
    evidence_title: Optional[str] = None,
    evidence_source: Optional[str] = None,
) -> SupportResult:
    ret_score = retrieval_support_score(retrieval_score)
    nli_score = nli_proxy_score(triple, evidence_text=evidence_text, evidence_title=evidence_title)
    literature_score = literature_support_score(
        triple,
        patient_context=patient_context,
        evidence_text=evidence_text,
        evidence_title=evidence_title,
        evidence_source=evidence_source,
    )
    kb_score = kb_consistency_score(triple, patient_context=patient_context, evidence_source=evidence_source)

    final_score = (
        SUPPORT_RETRIEVAL_WEIGHT * ret_score
        + SUPPORT_NLI_WEIGHT * nli_score
        + SUPPORT_LITERATURE_WEIGHT * literature_score
        + SUPPORT_KB_WEIGHT * kb_score
    )
    final_score = _clip(final_score)
    evidence_source_key = (evidence_source or "").upper()
    has_valid_evidence = bool((evidence_text or evidence_title) and evidence_source_key in {"UMLS", "PUBMED", "LITERATURE", "MEDICAL_LITERATURE"})
    has_correlated_support = (nli_score > 0.0) or (literature_score > 0.0) or (kb_score >= 0.75)

    if SUPPORT_REVIEW_ENABLED:
        if final_score >= SUPPORT_ACCEPT_THRESHOLD:
            status = "ACCEPTED"
        elif final_score < SUPPORT_REJECT_THRESHOLD:
            if has_valid_evidence and has_correlated_support:
                status = "ACCEPTED"
            else:
                status = "REJECTED"
        else:
            if has_valid_evidence and has_correlated_support:
                status = "ACCEPTED"
            else:
                status = "REVIEW"
    else:
        status = "ACCEPTED" if final_score >= SUPPORT_SCORE_THRESHOLD else "REJECTED"

    return SupportResult(
        score=final_score,
        status=status,
        components={
            "ret": ret_score,
            "nli": nli_score,
            "literature": literature_score,
            "kb": kb_score,
            "accept_threshold": SUPPORT_ACCEPT_THRESHOLD,
            "reject_threshold": SUPPORT_REJECT_THRESHOLD,
        },
    )
