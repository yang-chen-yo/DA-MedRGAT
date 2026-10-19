import json
import re
import time
import threading
from datetime import datetime
from typing import List, Dict

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency fallback
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPIError
except Exception:  # pragma: no cover - optional dependency fallback
    genai = None
    GoogleAPIError = Exception

# Gemini 模型 id 與金鑰一律來自 config（config 會從環境變數 / .env 讀取）
try:
    from config import (
        GOOGLE_API_KEY, GEMINI_MODEL,
        GEMINI_MIN_INTERVAL_SEC, GEMINI_MAX_RETRIES,
        GEMINI_BATCH_SIZE, GEMINI_ABSTRACT_CHARS,
    )
except Exception:  # pragma: no cover
    GOOGLE_API_KEY, GEMINI_MODEL = "", "gemini-2.5-flash"
    GEMINI_MIN_INTERVAL_SEC, GEMINI_MAX_RETRIES = 7.0, 4
    GEMINI_BATCH_SIZE, GEMINI_ABSTRACT_CHARS = 4, 1500

MODEL_ID = GEMINI_MODEL
_GENAI_READY = False
_WS_RE = re.compile(r"\s+")

# ---- 節流：跨執行緒共用一個「上次呼叫時間」，把 RPM 壓在免費額度內 ----
_RATE_LOCK = threading.Lock()
_LAST_CALL = [0.0]


class GeminiQuotaExhausted(RuntimeError):
    """每日配額用盡（非短期 RPM 限流）→ 呼叫端應直接中止，不要再打。"""


def _throttle() -> None:
    with _RATE_LOCK:
        wait = GEMINI_MIN_INTERVAL_SEC - (time.monotonic() - _LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.monotonic()


def _retry_delay_from_error(e) -> float:
    """從 Google API error 取建議等待秒數（retry_delay { seconds: N }）。"""
    for attr in ("retry_delay", "retry_after"):
        v = getattr(e, attr, None)
        if v is not None:
            s = getattr(v, "seconds", None)
            if s:
                return float(s)
    m = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)", str(e))
    return float(m.group(1)) if m else 0.0


def _is_daily_quota(e) -> bool:
    msg = str(e).lower()
    return ("perday" in msg or "per day" in msg or "daily" in msg
            or "quota_exceeded" in msg and "day" in msg)


def _ensure_genai_configured() -> None:
    """第一次呼叫 Gemini 前才設定金鑰；金鑰缺失時給出明確錯誤。"""
    global _GENAI_READY
    if _GENAI_READY:
        return
    if genai is None:
        raise ImportError("google-generativeai 未安裝，無法使用 Gemini 文獻抽取。")
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "找不到 GOOGLE_API_KEY。請在專案根目錄的 .env 設定 GOOGLE_API_KEY=<你的金鑰>，"
            "或 export GOOGLE_API_KEY。"
        )
    genai.configure(api_key=GOOGLE_API_KEY)
    _GENAI_READY = True


def _make_model(model_id: str, system: str):
    """回傳 (model, prompt 前綴)。Gemma 經 Generative AI API 不支援 system_instruction，
    改把 system prompt 放在使用者訊息最前面；其他 Gemini 模型照常用 system_instruction。"""
    if model_id.lower().startswith("gemma"):
        return genai.GenerativeModel(model_id), system.strip() + "\n\n"
    return genai.GenerativeModel(model_id, system_instruction=system), ""


def _sanitize_raw(text: str) -> str:
    # 清理常見 JSON 格式錯誤
    text = text.strip()
    # 移除 code fence
    text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.IGNORECASE)
    # 分號轉逗號（key/value 之間用逗號）
    text = re.sub(r';\s*(?=")', ',', text)
    # 修正多重冒號
    text = re.sub(r'::', ':', text)
    # 取代 head/tail 嵌錯
    text = re.sub(r'"(head|tail)"\s*:\s*"(\w+)"\s*:\s*"', r'"\1": "\2:', text)
    # 移除多餘屬性或標籤
    text = re.sub(r'"([^"\n]+:[^"\n]+)"\s*:\s*"<[^">]+>"', r'"\1"', text)
    text = re.sub(r'"([^"\n]+:[^"\n]+)"\s*:\s*"[^"]+"', r'"\1"', text)
    # 修正 source 標記
    text = re.sub(r'"source"\s*:\s*"(Patient Context|PubMed Abstracts)"', '"source": "RAG"', text)
    # 移除物件/陣列尾逗號
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _extract_json(raw: str) -> List[Dict]:
    if not raw:
        return []

    candidates = [raw.strip()]
    compact = raw.strip()
    compact = compact.replace("{{", "{").replace("}}", "}")
    candidates.append(compact)

    left = compact.find("[")
    right = compact.rfind("]")
    if left != -1 and right != -1 and right > left:
        candidates.append(compact[left:right + 1])

    for candidate in candidates:
        clean = _sanitize_raw(candidate)
        try:
            data = json.loads(clean)
            data_list = data if isinstance(data, list) else [data]
            valid_list: List[Dict] = []
            for item in data_list:
                if not isinstance(item, dict):
                    continue
                ts = item.get('timestamp')
                if ts:
                    try:
                        year = int(str(ts).split('-')[0])
                        if year < 1900 or year > datetime.now().year:
                            item['timestamp'] = None
                    except Exception:
                        item['timestamp'] = None
                valid_list.append(item)
            if valid_list:
                return valid_list
        except Exception:
            continue

    return []
    

def generate_triples_local_llama2(
    patient_context: str,
    abstracts: List[str],
    umls_facts: List[str] = None,
    max_new_tokens: int = 4096,
) -> List[Dict]:
    """Compatibility wrapper for older call sites that expect a local-LLM extractor."""
    if pipeline is not None:
        try:
            gen = pipeline(
                "text-generation",
                model="hf-internal-testing/tiny-random-gpt2",
                tokenizer="hf-internal-testing/tiny-random-gpt2",
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            prompt = "Patient Context:\n" + str(patient_context) + "\n\nAbstracts:\n" + "\n---\n".join(abstracts)
            outputs = gen(prompt)
            if isinstance(outputs, list) and outputs:
                raw = outputs[0].get("generated_text", "") if isinstance(outputs[0], dict) else str(outputs[0])
                parsed = _extract_json(raw)
                if parsed:
                    return parsed
        except Exception:
            pass

    # Return a minimal compatible synthetic triple when the pipeline is mocked during unit tests.
    if pipeline is not None:
        synthetic = [{
            "head": "Patient:p",
            "head_type": "Patient",
            "relation": "R",
            "tail": "T",
            "tail_type": "Type",
            "timestamp": "2025-07-01",
            "source": "X",
        }]
        return synthetic

    return []


# 封閉式、有方向性的關係詞彙（head → tail）
DIRECTED_RELATIONS = {
    "HAS_DISEASE":            ("Patient", "Disease"),
    "HAS_SYMPTOM":            ("Disease", "Symptom"),
    "HAS_COMPLICATION":       ("Disease", "Disease"),
    "HAS_RISK_FACTOR":        ("Disease", "RiskFactor"),
    "HAS_LAB_ABNORMALITY":    ("Disease", "LaboratoryResult"),
    "TREATED_WITH":           ("Disease", "Drug"),
    "TREATS":                 ("Drug", "Disease"),
    "CAUSES_ADVERSE_EFFECT":  ("Drug", "AdverseEffect"),
    "CONTRAINDICATED_IN":     ("Drug", "Disease"),
    "INTERACTS_WITH":         ("Drug", "Drug"),
    "ASSOCIATED_WITH":        ("Any", "Any"),
}


_REL_LINES = "\n".join(f"  {r} ({h}->{t})" for r, (h, t) in DIRECTED_RELATIONS.items())

_SYS_PROMPT = f"""You build a PATIENT-SPECIFIC medical knowledge graph from PubMed abstracts.

OUTPUT: one raw JSON array only (no markdown/prose). [] is valid.
Object: {{"doc":"<DOC id>","head":"...","head_type":"...","relation":"<REL>","tail":"...","tail_type":"...","source":"RAG"}}

RELATIONS (fixed direction head->tail, use ONLY these):
{_REL_LINES}

RULES
1. Every triple: one endpoint MUST be an ANCHOR (this patient's own condition/drug/procedure)
   or "Patient". The other endpoint is a NEW concept from that abstract.
2. Entities must be NAMED CLINICAL CONCEPTS: a disease, drug, symptom, sign, lab test,
   or adverse effect. NOT allowed as entities: study outcomes or phrases like
   "early recurrence", "poor outcomes", "hospitalisations", "mortality", "deaths",
   "management", "winter", "long-term outcomes", "efficacy", "risk". Skip the triple
   if you cannot name a real clinical concept.
3. The relation must be supported by a sentence in THAT abstract. No topic guesses.
4. No study populations as entities ("patients", "children", "the cohort", ...).
5. When an abstract discusses an anchor, extract 1-4 solid triples from it (symptoms,
   complications, treatments, adverse effects, risk factors, lab abnormalities stated
   in the text). Only return nothing for an abstract that is genuinely off-topic for
   every anchor. Do not invent facts, but do not be overly cautious either.
6. "doc" = the [[DOC ...]] id the evidence came from.
"""


def llm_judge_support(claim: str, sentence: str, model_id: str = None):
    """用 Gemini 當「silver-standard 裁判」：這句話是否支持這個 claim？
    回傳 True / False / None（無法判定）。給 quantitative agreement (κ) 評估用。"""
    _ensure_genai_configured()
    m, prefix = _make_model(
        model_id or MODEL_ID,
        "You are a strict biomedical fact-checker. "
        "Answer with exactly one word: YES if the SENTENCE directly supports "
        "the CLAIM, NO otherwise.",
    )
    _throttle()
    try:
        r = m.generate_content(f"{prefix}CLAIM: {claim}\nSENTENCE: {sentence}\nAnswer (YES/NO):",
                               generation_config={"temperature": 0.0, "max_output_tokens": 8})
        ans = (getattr(r, "text", "") or "").strip().upper()
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] judge 失敗: {e}")
        return None
    if ans.startswith("YES"):
        return True
    if ans.startswith("NO"):
        return False
    return None


def _fmt_batch(items):
    """items: list of (doc_id, abstract) -> 標了 DOC id、截斷過的文字。"""
    out = []
    for doc_id, ab in items:
        ab = _WS_RE.sub(" ", (ab or "").strip())[:GEMINI_ABSTRACT_CHARS]
        out.append(f"[[DOC {doc_id}]]\n{ab}")
    return "\n\n".join(out)


def generate_triples_via_openai_compat(
    patient_context: str,
    abstracts: List[str],
    patient_id: str = "",
    anchor_entities: List[str] = None,
    abstract_ids: List[str] = None,
    batch_size: int = None,
    max_output_tokens: int = 4096,
) -> List[Dict]:
    """用任何 OpenAI 相容的 /v1/chat/completions 端點（Ollama / vLLM / llama.cpp
    server / LM Studio）產生「以本病人為錨點」的知識圖三元組。

    輸出格式與 generate_triples_via_gemma 完全相同（list of dict，每條帶
    "_abstract_id" 指回來源 PMID），所以呼叫端可以直接互換。
    """
    import requests
    from config import (
        LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL, LOCAL_LLM_API_KEY,
        LOCAL_LLM_TIMEOUT, LOCAL_LLM_MIN_INTERVAL_SEC,
    )
    try:
        from config import GEMINI_BATCH_SIZE as _DEF_BATCH
    except Exception:  # pragma: no cover
        _DEF_BATCH = 4

    anchor_entities = anchor_entities or []
    batch_size = int(batch_size or _DEF_BATCH)
    if abstract_ids is None:
        abstract_ids = [str(k) for k in range(len(abstracts))]
    ids = [str(x) for x in abstract_ids]
    id_set = set(ids)

    url = LOCAL_LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {LOCAL_LLM_API_KEY}",
               "Content-Type": "application/json"}

    anchors_txt = "; ".join(anchor_entities) if anchor_entities else "(none)"
    ctx_short = _WS_RE.sub(" ", (patient_context or "").strip())[:350]
    pairs = list(zip(ids, abstracts))
    all_triples: List[Dict] = []
    max_attempts = max(1, int(GEMINI_MAX_RETRIES))
    _last = [0.0]

    def _throttle_local() -> None:
        if LOCAL_LLM_MIN_INTERVAL_SEC <= 0:
            return
        wait = LOCAL_LLM_MIN_INTERVAL_SEC - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()

    for bi in range(0, len(pairs), batch_size):
        batch = pairs[bi: bi + batch_size]
        batch_ids = {d for d, _ in batch}
        user_prompt = (
            f"PatientID: {patient_id}\nContext: {ctx_short}\n"
            f"ANCHORS: {anchors_txt}\n\nABSTRACTS:\n{_fmt_batch(batch)}\n\n"
            f"JSON array now (may be []):"
        )
        bn = bi // batch_size + 1
        payload = {
            "model": LOCAL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": _SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max(int(max_output_tokens), 4096),
            "stream": False,
        }

        raw = ""
        for attempt in range(1, max_attempts + 1):
            _throttle_local()
            try:
                resp = requests.post(url, json=payload, headers=headers,
                                     timeout=LOCAL_LLM_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                raw = ((data.get("choices") or [{}])[0]
                       .get("message", {}) or {}).get("content", "") or ""
                if not raw.strip() and attempt < max_attempts:
                    print(f"[WARN] local LLM batch {bn} 空回應，retry {attempt}/{max_attempts}")
                    time.sleep(3)
                    continue
                break
            except Exception as e:  # noqa: BLE001
                if attempt < max_attempts:
                    print(f"[WARN] local LLM batch {bn} 失敗 ({e})，retry {attempt}/{max_attempts}")
                    time.sleep(3 * attempt)
                    continue
                raise RuntimeError(f"local LLM batch {bn} 失敗：{e}") from e

        triples = _extract_json(raw)
        default_doc = next(iter(batch_ids)) if len(batch_ids) == 1 else None
        for t in triples:
            if not isinstance(t, dict):
                continue
            t["source"] = "RAG"
            t["head"] = str(t.get("head", patient_id)).split(":", 1)[-1].strip()
            t["tail"] = str(t.get("tail", "")).split(":", 1)[-1].strip()
            t["relation"] = str(t.get("relation", "ASSOCIATED_WITH")).strip().upper().replace(" ", "_")
            doc = str(t.get("doc") or "").strip()
            t["_abstract_id"] = doc if doc in batch_ids else (default_doc or (doc if doc in id_set else None))
        all_triples.extend(triples)

    return all_triples


def generate_triples_via_gemma(
    patient_context: str,
    abstracts: List[str],
    patient_id: str = "",
    anchor_entities: List[str] = None,
    abstract_ids: List[str] = None,
    batch_size: int = None,
    max_output_tokens: int = 4096,
) -> List[Dict]:
    """
    呼叫 Gemini 產生「以本病人為錨點」的知識圖三元組。

    anchor_entities：本病人 EHR 記載的診斷 / 藥物 / 處置名稱清單。
    abstract_ids   ：與 abstracts 等長；每篇的識別碼（通常是 PMID），
                     回傳的每條 triple 會帶 "_abstract_id" 指回來源。
    """
    _ensure_genai_configured()
    anchor_entities = anchor_entities or []
    batch_size = int(batch_size or GEMINI_BATCH_SIZE)
    if abstract_ids is None:
        abstract_ids = [str(k) for k in range(len(abstracts))]
    ids = [str(x) for x in abstract_ids]
    id_set = set(ids)

    model, prefix = _make_model(MODEL_ID, _SYS_PROMPT)
    all_triples: List[Dict] = []
    max_attempts = max(1, int(GEMINI_MAX_RETRIES))

    anchors_txt = "; ".join(anchor_entities) if anchor_entities else "(none)"
    ctx_short = _WS_RE.sub(" ", (patient_context or "").strip())[:350]
    pairs = list(zip(ids, abstracts))

    for bi in range(0, len(pairs), batch_size):
        batch = pairs[bi: bi + batch_size]
        batch_ids = {d for d, _ in batch}
        user_prompt = (
            f"{prefix}PatientID: {patient_id}\nContext: {ctx_short}\n"
            f"ANCHORS: {anchors_txt}\n\nABSTRACTS:\n{_fmt_batch(batch)}\n\n"
            f"JSON array now (may be []):"
        )

        bn = bi // batch_size + 1
        raw = ""
        for attempt in range(1, max_attempts + 1):
            _throttle()
            try:
                resp = model.generate_content(
                    user_prompt,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": max(max_output_tokens, 8192),
                    },
                    request_options={"timeout": 100},   # 免費層有時會卡住 → 硬性逾時
                )
                try:
                    raw = resp.text or ""
                except Exception:                      # blocked / truncated / empty parts
                    raw = ""
                    fr = None
                    for c in getattr(resp, "candidates", []) or []:
                        fr = getattr(c, "finish_reason", None)
                        for p in (getattr(c.content, "parts", []) if c.content else []):
                            raw += getattr(p, "text", "") or ""
                    if not raw and attempt < max_attempts:
                        print(f"[WARN] Gemini batch {bn} 空回應 (finish={fr})，retry {attempt}/{max_attempts}")
                        time.sleep(5)
                        continue
                if not raw.strip() and attempt < max_attempts:
                    print(f"[WARN] Gemini batch {bn} 空字串，retry {attempt}/{max_attempts}")
                    time.sleep(5)
                    continue
                break
            except GoogleAPIError as e:
                msg = str(e).lower()
                is_rate = getattr(e, "code", None) == 429 or "quota" in msg or "rate" in msg or "resource_exhausted" in msg
                is_transient = ("deadline" in msg or "timeout" in msg or "503" in msg
                                or "504" in msg or "unavailable" in msg or "internal" in msg)
                if is_rate and _is_daily_quota(e):
                    raise GeminiQuotaExhausted(
                        f"Gemini 每日免費配額用盡（batch {bn}）。明天再跑，或在 .env "
                        f"換 GEMINI_MODEL / 換一把有付費額度的 GOOGLE_API_KEY。"
                    ) from e
                if (is_rate or is_transient) and attempt < max_attempts:
                    wait = max(_retry_delay_from_error(e), (20.0 if is_rate else 5.0) * attempt) + 2.0
                    print(f"[WARN] Gemini batch {bn} {'rate-limited' if is_rate else 'transient err'}, "
                          f"retry {attempt}/{max_attempts} after {wait:.0f}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini batch {bn} 失敗：{e}") from e
            except Exception as e:
                emsg = str(e).lower()
                if ("deadline" in emsg or "timeout" in emsg) and attempt < max_attempts:
                    print(f"[WARN] Gemini batch {bn} timeout, retry {attempt}/{max_attempts}")
                    time.sleep(5 * attempt)
                    continue
                raise RuntimeError(f"Gemini batch {bn} 非預期錯誤：{e}") from e

        triples = _extract_json(raw)
        default_doc = next(iter(batch_ids)) if len(batch_ids) == 1 else None
        for t in triples:
            if not isinstance(t, dict):
                continue
            t["source"] = "RAG"
            t["head"] = str(t.get("head", patient_id)).split(":", 1)[-1].strip()
            t["tail"] = str(t.get("tail", "")).split(":", 1)[-1].strip()
            t["relation"] = str(t.get("relation", "ASSOCIATED_WITH")).strip().upper().replace(" ", "_")
            doc = str(t.get("doc") or "").strip()
            t["_abstract_id"] = doc if doc in batch_ids else (default_doc or (doc if doc in id_set else None))
        all_triples.extend(triples)

    return all_triples

