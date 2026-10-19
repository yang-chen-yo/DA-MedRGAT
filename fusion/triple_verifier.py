# === fusion/triple_verifier.py ===
"""
Triple ↔ sentence entailment verifier（Phase 2b）。

確認每個 generated triple 都有文獻中的句子支持（sentence-level entailment）。

流程
  1. 把 (h, r, t) 用 relation template 轉成一句自然語言 claim
  2. 把證據（PubMed title + abstract）切成句子
  3. 對每個 (sentence → claim) 跑 biomedical NLI cross-encoder，取 entailment 機率
  4. 取機率最高的句子當 supporting sentence
  5. 依門檻決定 ACCEPTED / REVIEW / REJECTED

可追溯欄位：pmid、supporting_sentence、entail 機率、verifier 模型名、decision。

沒有 transformers / 模型下載不到時，get_verifier() 回傳 None，
呼叫端請自行退回舊的 lexical scorer。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from config import (
    VERIFIER_ENABLED,
    VERIFIER_MODEL,
    VERIFIER_FALLBACK_MODEL,
    VERIFIER_ACCEPT_THRESHOLD,
    VERIFIER_REVIEW_THRESHOLD,
    VERIFIER_CONTRA_THRESHOLD,
    VERIFIER_MAX_SENTENCES,
    VERIFIER_BATCH_SIZE,
    VERIFIER_DEVICE,
)

# ----------------------------------------------------------------------
# 1) claim 語言化
# ----------------------------------------------------------------------
_ABBREV = {"e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "fig.", "no.", "al."}
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WS_RE = re.compile(r"\s+")


def split_sentences(text: str, max_sentences: int = 0) -> List[str]:
    """輕量斷句（不需 nltk）。針對 PubMed abstract 夠用。"""
    text = _WS_RE.sub(" ", (text or "").strip())
    if not text:
        return []
    raw = _SENT_SPLIT_RE.split(text)
    out: List[str] = []
    for seg in raw:
        seg = seg.strip()
        if not seg:
            continue
        # 把因縮寫誤切的句子接回上一句
        if out and out[-1].split()[-1].lower() in _ABBREV:
            out[-1] = f"{out[-1]} {seg}"
        else:
            out.append(seg)
    if max_sentences and len(out) > max_sentences:
        out = out[:max_sentences]
    return out


def _g(triple: Any, key: str) -> str:
    if isinstance(triple, dict):
        return str(triple.get(key, "") or "")
    return str(getattr(triple, key, "") or "")


def _clean_entity(value: str) -> str:
    value = _WS_RE.sub(" ", (value or "").strip())
    # 去掉 "Type:xxx" 前綴
    if ":" in value and len(value.split(":", 1)[0].split()) == 1:
        value = value.split(":", 1)[1].strip()
    return value


# relation（正規化後）→ claim 樣板；{h} {t} 會被代入
_REL_TEMPLATES = {
    # === Phase 2 封閉式詞彙 ===
    "has_disease": "The patient has {t}.",
    "has_symptom": "{h} can cause the symptom {t}.",
    "has_complication": "{t} is a complication of {h}.",
    "has_risk_factor": "{t} is a risk factor for {h}.",
    "has_lab_abnormality": "{h} is associated with the laboratory abnormality {t}.",
    "treated_with": "{h} can be treated with {t}.",
    "treats": "{h} is used to treat {t}.",
    "causes_adverse_effect": "{h} can cause {t} as an adverse effect.",
    "contraindicated_in": "{h} is contraindicated in patients with {t}.",
    "interacts_with": "{h} interacts with {t}.",
    "associated_with": "{h} is associated with {t}.",
    # === 舊式 / 變體（相容）===
    "patient_disease": "The patient has {t}.",
    "disease_symptom": "{h} can cause the symptom {t}.",
    "symptom_of": "{t} can cause the symptom {h}.",
    "used_drug": "The patient is treated with {t}.",
    "patient_drug": "The patient is treated with {t}.",
    "drug_disease": "{h} is used to treat {t}.",
    "disease_drug": "{t} is used to treat {h}.",
    "causes_side_effect": "{h} can cause {t} as an adverse effect.",
    "drug_adverseeffect": "{h} can cause {t} as an adverse effect.",
    "has_adverse_effect": "{h} can cause {t} as an adverse effect.",
    "received_treatment": "The patient received {t}.",
    "received_procedure": "The patient underwent {t}.",
    "patient_treatment": "The patient received {t}.",
    "patient_procedure": "The patient underwent {t}.",
    "has_lab_result": "The patient had a laboratory result of {t}.",
    "patient_laboratoryresult": "The patient had a laboratory result of {t}.",
    "patient_labresult": "The patient had a laboratory result of {t}.",
}


def _norm_rel(rel: str) -> str:
    rel = (rel or "").strip().lower()
    rel = rel.replace("–", "_").replace("-", "_").replace(" ", "_")
    rel = re.sub(r"_+", "_", rel).strip("_")
    return rel


def verbalize_triple(triple: Any) -> str:
    """(h, r, t) → 一句英文 claim。"""
    h = _clean_entity(_g(triple, "head"))
    t = _clean_entity(_g(triple, "tail"))
    rel = _g(triple, "relation")
    key = _norm_rel(rel)

    tmpl = _REL_TEMPLATES.get(key)
    if tmpl:
        return tmpl.format(h=h or "the patient", t=t or "the finding")

    # 泛用退回：relation 拆成單字
    rel_words = _norm_rel(rel).replace("_", " ").strip() or "is related to"
    h = h or "the patient"
    t = t or "the finding"
    return f"{h} {rel_words} {t}."


# ----------------------------------------------------------------------
# 2) verifier
# ----------------------------------------------------------------------
@dataclass
class VerifierResult:
    score: float                       # entailment 機率（0..1，取最佳句子）
    decision: str                      # ACCEPTED / REVIEW / REJECTED
    claim: str
    supporting_sentence: Optional[str]
    sentence_index: Optional[int]
    model_name: str
    per_sentence: List[float] = field(default_factory=list)


class TripleEvidenceVerifier:
    def __init__(
        self,
        model_name: str = VERIFIER_MODEL,
        fallback_model: str = VERIFIER_FALLBACK_MODEL,
        accept_threshold: float = VERIFIER_ACCEPT_THRESHOLD,
        review_threshold: float = VERIFIER_REVIEW_THRESHOLD,
        contra_threshold: float = VERIFIER_CONTRA_THRESHOLD,
        max_sentences: int = VERIFIER_MAX_SENTENCES,
        batch_size: int = VERIFIER_BATCH_SIZE,
        device: str = VERIFIER_DEVICE,
    ):
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold
        self.contra_threshold = contra_threshold
        self.max_sentences = max_sentences
        self.batch_size = batch_size
        self._device = device or None
        self._tok = None
        self._model = None
        self._entail_idx = None
        self._contra_idx = None
        self._loaded_name = None

    # ---- 載入（lazy）------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        last_err = None
        for name in (self.model_name, self.fallback_model):
            if not name:
                continue
            try:
                tok = AutoTokenizer.from_pretrained(name)
                model = AutoModelForSequenceClassification.from_pretrained(name)
                model.eval()
                if self._device is None:
                    self._device = "cuda:0" if _cuda_ok() else "cpu"
                model.to(self._device)
                self._tok, self._model, self._loaded_name = tok, model, name
                self._entail_idx = _find_label_index(model.config, "entail", default=model.config.num_labels - 1)
                self._contra_idx = _find_label_index(model.config, "contra", default=0)
                print(f"[TripleEvidenceVerifier] loaded {name} on {self._device} "
                      f"(entail_idx={self._entail_idx}, contra_idx={self._contra_idx})")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[TripleEvidenceVerifier] 載入 {name} 失敗：{e}")
                last_err = e
        raise RuntimeError(f"NLI verifier 模型載入失敗：{last_err}")

    # ---- NLI 機率 ------------------------------------------------
    def _softmax_cols(self, premises: List[str], hypothesis: str):
        """回傳每個 premise 的完整 softmax（list of list）。"""
        import torch

        self._ensure_loaded()
        rows = []
        for i in range(0, len(premises), self.batch_size):
            chunk = premises[i:i + self.batch_size]
            enc = self._tok(
                chunk, [hypothesis] * len(chunk),
                return_tensors="pt", truncation=True, max_length=256, padding=True,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            sm = torch.softmax(logits, dim=-1).cpu().tolist()
            rows.extend(sm)
        return rows

    def _entail_probs(self, premises: List[str], hypothesis: str) -> List[float]:
        return [float(r[self._entail_idx]) for r in self._softmax_cols(premises, hypothesis)]

    def _contra_probs(self, premises: List[str], hypothesis: str) -> List[float]:
        return [float(r[self._contra_idx]) for r in self._softmax_cols(premises, hypothesis)]

    # ---- 對外 API -----------------------------------------------
    def verify(
        self,
        triple: Any,
        evidence_text: Optional[str],
        evidence_title: Optional[str] = None,
        evidence_id: Optional[str] = None,
    ) -> VerifierResult:
        return self._score(verbalize_triple(triple), evidence_text, evidence_title, tail_of=triple)

    def verify_claim(
        self,
        claim: str,
        evidence_text: Optional[str],
        evidence_title: Optional[str] = None,
    ) -> VerifierResult:
        """自由文字 claim（給 benchmark / LLM-judge 用；不做實體接地 gate）。"""
        return self._score(str(claim or "").strip(), evidence_text, evidence_title, tail_of=None)

    def _score(self, claim, evidence_text, evidence_title, tail_of) -> VerifierResult:
        sentences: List[str] = []
        if evidence_title:
            sentences.append(_WS_RE.sub(" ", evidence_title.strip()))
        sentences.extend(split_sentences(evidence_text, self.max_sentences))
        seen, uniq = set(), []
        for s in sentences:
            if s and s not in seen:
                seen.add(s)
                uniq.append(s)
        sentences = uniq[: self.max_sentences] if self.max_sentences else uniq

        if not sentences or not claim:
            return VerifierResult(0.0, "REJECTED", claim, None, None,
                                  self._loaded_name or self.model_name, [])

        cols = self._softmax_cols(sentences, claim)
        scores = [float(r[self._entail_idx]) for r in cols]
        contras = [float(r[self._contra_idx]) for r in cols]
        best_i = max(range(len(scores)), key=lambda k: scores[k])
        best = scores[best_i]
        best_sent = sentences[best_i]

        contra_i = max(range(len(contras)), key=lambda k: contras[k])
        max_contra = contras[contra_i]

        # 實體接地只在有 triple（tail_of）時做
        def _grounded(sent):
            return True if tail_of is None else _tail_grounded(tail_of, f"{evidence_title or ''} {sent}")
        grounded = _grounded(best_sent)

        if (max_contra >= self.contra_threshold and max_contra > best and _grounded(sentences[contra_i])):
            decision = "CONTRADICTED"
            best_i, best, best_sent = contra_i, scores[contra_i], sentences[contra_i]
        elif best >= self.accept_threshold and grounded:
            decision = "ACCEPTED"
        elif best >= self.review_threshold and grounded:
            decision = "REVIEW"
        elif best >= self.accept_threshold and not grounded:
            decision = "REVIEW"
        else:
            decision = "REJECTED"

        return VerifierResult(
            score=round(best, 4),
            decision=decision,
            claim=claim,
            supporting_sentence=best_sent,
            sentence_index=best_i,
            model_name=self._loaded_name or self.model_name,
            per_sentence=[round(s, 4) for s in scores],
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "to", "with", "for", "by",
         "is", "are", "was", "were", "be", "as", "at", "from", "that", "this",
         "patients", "patient", "disease", "syndrome", "disorder", "chronic", "acute"}


def _content_tokens(text: str):
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in _STOP}


def _tail_grounded(triple: Any, evidence: str) -> bool:
    """tail 實體的關鍵詞是否真的出現在支持文字裡（容忍 US/UK 拼法差異）。"""
    tail = _clean_entity(_g(triple, "tail")).lower()
    if not tail:
        return False
    ev = (evidence or "").lower()
    if tail in ev:
        return True
    tt = _content_tokens(tail)
    if not tt:                       # tail 全是停用詞/短詞 → 無法判斷，放行
        return True
    et = _content_tokens(ev)
    et_pref = {w[:6] for w in et}

    def _hit(w):
        return w in et or w[:6] in et_pref or any(w[:6] == x[:6] for x in et)

    hit = sum(1 for w in tt if _hit(w))
    return hit >= 1 and hit / len(tt) >= 0.5


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _find_label_index(cfg, prefix: str, default: int = 0) -> int:
    """從 model.config.id2label 找某類別（entail / contra …）的 index。"""
    id2label = getattr(cfg, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).lower().startswith(prefix):
            return int(idx)
    return default


def _find_entail_index(cfg) -> int:  # 向後相容
    return _find_label_index(cfg, "entail", default=getattr(cfg, "num_labels", 3) - 1)


# ----------------------------------------------------------------------
# module-level singleton
# ----------------------------------------------------------------------
_VERIFIER: Optional[TripleEvidenceVerifier] = None
_VERIFIER_FAILED = False


def get_verifier() -> Optional[TripleEvidenceVerifier]:
    """回傳已載入的 verifier；未啟用或載入失敗回 None。"""
    global _VERIFIER, _VERIFIER_FAILED
    if not VERIFIER_ENABLED or _VERIFIER_FAILED:
        return None
    if _VERIFIER is None:
        v = TripleEvidenceVerifier()
        try:
            v._ensure_loaded()
        except Exception as e:  # noqa: BLE001
            print(f"[TripleEvidenceVerifier] 停用（載入失敗）：{e}")
            _VERIFIER_FAILED = True
            return None
        _VERIFIER = v
    return _VERIFIER
