from pathlib import Path
import os
import json
import re
import time
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

from config import (
    K_UMLS, K_PUBMED, PUBMED_API_KEY, SUPPORT_FILTER_ENABLED,
    UMLS_LINK_SIM_THRESHOLD, PUBMED_MIN_SIM,
)
from embed.encoder import Encoder
from embed.faiss_index import FaissIndex
from embed.umls_concept_index import UMLSConceptIndex
from retrieval.pubmed_client import PubMedClient
from retrieval.umls_client import UMLSClient
from fusion.scoring import cosine_similarity, score_pubmed_hit
from fusion.evidence import assess_triple_support
from fusion.triple_verifier import get_verifier
from kg.triple import Triple
from fusion.triple_generation_hf import (
    generate_triples_via_gemma, generate_triples_via_openai_compat, GeminiQuotaExhausted,
)

# Global encoder (constructed once)
encoder = Encoder()

# entity 欄位 -> KG 節點型別 / 關係
_ENTITY_FIELDS = (
    ("conditions", "Disease", "HAS_DISEASE"),
    ("procedures", "Procedure", "RECEIVED_PROCEDURE"),
    ("drugs", "Drug", "USED_DRUG"),
)


def _rag_generate(
    patient_context: str,
    abstracts: List[str],
    patient_id: str = "",
    anchor_entities: Optional[List[str]] = None,
    abstract_ids: Optional[List[str]] = None,
):
    """文獻三元組生成入口（Gemini）。

    回傳 list  → 成功（可能是空 list，代表沒有可用 triple，要快取）
    回傳 None → 呼叫失敗（限流/錯誤），呼叫端不要快取、稍後重跑可續
    丟 GeminiQuotaExhausted → 每日配額用盡，整條 RAG 中止
    """
    if not abstracts:
        return []
    try:
        from config import TRIPLE_LLM_BACKEND
    except Exception:  # pragma: no cover
        TRIPLE_LLM_BACKEND = "gemini"
    _gen = (generate_triples_via_openai_compat
            if TRIPLE_LLM_BACKEND in ("openai_compat", "local", "ollama", "vllm", "llamacpp")
            else generate_triples_via_gemma)
    try:
        return _gen(
            patient_context=patient_context,
            abstracts=abstracts,
            patient_id=patient_id,
            anchor_entities=anchor_entities or [],
            abstract_ids=abstract_ids,
        )
    except GeminiQuotaExhausted:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] RAG 生成失敗（會停在這批、稍後重跑可續）：{e}")
        return None


# ── RAG triple 正規化 & 病人錨定 ──────────────────────────────
_WS = re.compile(r"\s+")

# 泛人群字眼 → 出現在 head/tail 就丟棄（非個人化知識）
_GENERIC_SUBJECTS = {
    "patient", "patients", "the patient", "this patient", "children", "child",
    "adults", "adult", "individuals", "individual", "cases", "case", "subjects",
    "participants", "the cohort", "cohort", "study population", "population",
    "men", "women", "the elderly", "elderly patients", "humans", "human",
    "people", "the study", "our study",
}

# Gemini 可能吐舊式或變體 relation → 對映到封閉詞彙
_REL_CANON = {
    "HAS_DISEASE": "HAS_DISEASE", "PATIENT_DISEASE": "HAS_DISEASE",
    "HAS_SYMPTOM": "HAS_SYMPTOM", "DISEASE_SYMPTOM": "HAS_SYMPTOM",
    "SYMPTOM_OF": "HAS_SYMPTOM", "PRESENTS_WITH": "HAS_SYMPTOM",
    "HAS_COMPLICATION": "HAS_COMPLICATION", "COMPLICATION_OF": "HAS_COMPLICATION",
    "HAS_RISK_FACTOR": "HAS_RISK_FACTOR", "RISK_FACTOR_FOR": "HAS_RISK_FACTOR",
    "HAS_LAB_ABNORMALITY": "HAS_LAB_ABNORMALITY", "HAS_LAB_RESULT": "HAS_LAB_ABNORMALITY",
    "PATIENT_LABORATORYRESULT": "HAS_LAB_ABNORMALITY", "PATIENT_LABRESULT": "HAS_LAB_ABNORMALITY",
    "TREATED_WITH": "TREATED_WITH", "DISEASE_DRUG": "TREATED_WITH",
    "PATIENT_TREATMENT": "TREATED_WITH", "PATIENT_DRUG": "TREATED_WITH",
    "RECEIVED_TREATMENT": "TREATED_WITH", "MANAGED_WITH": "TREATED_WITH",
    "TREATS": "TREATS", "DRUG_DISEASE": "TREATS",
    "CAUSES_ADVERSE_EFFECT": "CAUSES_ADVERSE_EFFECT", "CAUSES_SIDE_EFFECT": "CAUSES_ADVERSE_EFFECT",
    "DRUG_ADVERSEEFFECT": "CAUSES_ADVERSE_EFFECT", "HAS_ADVERSE_EFFECT": "CAUSES_ADVERSE_EFFECT",
    "ADVERSE_EFFECT": "CAUSES_ADVERSE_EFFECT",
    "CONTRAINDICATED_IN": "CONTRAINDICATED_IN", "INTERACTS_WITH": "INTERACTS_WITH",
    "ASSOCIATED_WITH": "ASSOCIATED_WITH", "TEMPORAL": "ASSOCIATED_WITH",
    "BEFORE": "ASSOCIATED_WITH", "AFTER": "ASSOCIATED_WITH",
}
# 關係的預期 head 型別（用來修正 head/tail 反向）
_REL_HEAD_IS_PATIENT = {"HAS_DISEASE"}
_REL_HEAD_IS_DRUG = {"TREATS", "CAUSES_ADVERSE_EFFECT", "CONTRAINDICATED_IN", "INTERACTS_WITH"}
_REL_HEAD_IS_DISEASE = {"HAS_SYMPTOM", "HAS_COMPLICATION", "HAS_RISK_FACTOR",
                        "HAS_LAB_ABNORMALITY", "TREATED_WITH"}


def _norm(s: str) -> str:
    return _WS.sub(" ", str(s or "").strip().lower())


def _anchor_index(patient_fields: Dict, patient_id: str):
    """回傳 (anchor 名稱清單, 正規化 anchor -> (name, type) dict)。"""
    names, idx = [], {}
    for name, etype, _rel, _code in _iter_named_entities(patient_fields):
        names.append(name)
        idx[_norm(name)] = (name, etype)
    return names, idx


def _is_patient_token(value: str, patient_id: str) -> bool:
    v = _norm(value)
    return v in {"patient", "the patient", "this patient", _norm(patient_id),
                 f"patient:{_norm(patient_id)}"}


def _match_anchor(value: str, anchor_idx: Dict):
    """entity 與某個 anchor 對得上就回傳 (anchor_name, anchor_type)，否則 None。"""
    v = _norm(value)
    if not v:
        return None
    if v in anchor_idx:
        return anchor_idx[v]
    for a_norm, meta in anchor_idx.items():
        if len(a_norm) >= 5 and (a_norm in v or v in a_norm):
            return meta
    return None


def _normalize_rag_triple(t: Dict, patient_id: str, anchor_idx: Dict):
    """把一條 Gemini triple 正規化並做病人錨定。

    回傳 (normalized_dict, reason)。reason == "ok" 才該進圖；
    其餘："generic" / "unanchored" / "empty" / "self"。
    """
    head = str(t.get("head", "")).strip()
    tail = str(t.get("tail", "")).strip()
    rel_raw = str(t.get("relation", "")).strip().upper().replace(" ", "_").replace("-", "_")
    rel = _REL_CANON.get(rel_raw, "ASSOCIATED_WITH")

    if not head or not tail:
        return None, "empty"

    # 泛人群字眼 → 丟
    if _norm(head) in _GENERIC_SUBJECTS or _norm(tail) in _GENERIC_SUBJECTS:
        # 例外：head 是「the patient」且能對到本病人，視為 Patient 節點
        if not (_is_patient_token(head, patient_id) or _is_patient_token(tail, patient_id)):
            return None, "generic"

    # 病人 token 正規化成 patient_id
    if _is_patient_token(head, patient_id):
        head, head_is_patient = patient_id, True
    else:
        head_is_patient = False
    if _is_patient_token(tail, patient_id):
        tail, tail_is_patient = patient_id, True
    else:
        tail_is_patient = False

    if head == tail:
        return None, "self"

    # Patient 一律當 head；若關係詞彙不是 patient-level 就降級成 ASSOCIATED_WITH
    if tail_is_patient and not head_is_patient:
        head, tail = tail, head
        head_is_patient, tail_is_patient = True, False
        if rel not in _REL_HEAD_IS_PATIENT:
            rel = "ASSOCIATED_WITH"

    a_head = _match_anchor(head, anchor_idx)
    a_tail = _match_anchor(tail, anchor_idx)

    if rel in _REL_HEAD_IS_DRUG and a_tail and a_tail[1] == "Drug" and not (a_head and a_head[1] == "Drug"):
        head, tail = tail, head
        a_head, a_tail = a_tail, a_head
    if rel in _REL_HEAD_IS_DISEASE and a_tail and a_tail[1] == "Disease" and not (a_head and a_head[1] == "Disease"):
        head, tail = tail, head
        a_head, a_tail = a_tail, a_head

    # 錨定：至少一端是 Patient 或本病人的 EHR entity
    anchored = head_is_patient or tail_is_patient or bool(a_head) or bool(a_tail)
    if not anchored:
        return None, "unanchored"

    # 用 anchor 的標準寫法取代
    if a_head:
        head = a_head[0]
    if a_tail:
        tail = a_tail[0]

    # 錨定正規化後可能兩端變同一個節點（如 "Hepatitis A" 和 "Hepatitis" 都對到 Hepatitis）
    if _norm(head) == _norm(tail):
        return None, "self"

    norm = {
        "head": head,
        "head_type": "Patient" if head_is_patient else (a_head[1] if a_head else str(t.get("head_type") or "Concept")),
        "relation": rel,
        "tail": tail,
        "tail_type": "Patient" if tail_is_patient else (a_tail[1] if a_tail else str(t.get("tail_type") or "Concept")),
        "evidence_quote": str(t.get("evidence_quote") or "").strip(),
        "source": "RAG",
    }
    return norm, "ok"


def _iter_named_entities(patient_fields: Dict):
    """從 patient_fields 展開出 (name, entity_type, relation, code)，去重。"""
    seen = set()
    for field, etype, relation in _ENTITY_FIELDS:
        for group in patient_fields.get(field, []) or []:
            items = group if isinstance(group, list) else [group]
            for item in items:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                    code = str(item.get("code") or "")
                else:
                    name, code = str(item).strip(), ""
                if not name:
                    continue
                key = (etype, name.lower())
                if key in seen:
                    continue
                seen.add(key)
                yield name, etype, relation, code


def _link_entities_to_umls(patient_fields: Dict, umls_dir: str) -> List[Dict]:
    """逐 entity 連結到 UMLS CUI；未達門檻標記 unlinked（該 entity 仍是病歷事實）。"""
    index = UMLSConceptIndex.shared(umls_dir, encoder=encoder)
    hits: List[Dict] = []
    for name, etype, relation, code in _iter_named_entities(patient_fields):
        best = index.link_best(name, sim_threshold=UMLS_LINK_SIM_THRESHOLD)
        hits.append({
            "query_entity": name,
            "entity_type": etype,
            "relation": relation,
            "code": code,
            "cui": best["cui"] if best else None,
            "umls_name": best["umls_name"] if best else None,
            "score": round(float(best["score"]), 4) if best else 0.0,
            "match": best["match"] if best else "unlinked",
        })
    return hits


def _dedupe_hits(hits: List[Dict], key_fields: Optional[List[str]] = None) -> List[Dict]:
    if not hits:
        return []
    seen = set()
    unique_hits = []
    for hit in hits:
        if key_fields is None:
            key = json.dumps(hit, sort_keys=True, ensure_ascii=False)
        else:
            key = tuple(str(hit.get(field, "")) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique_hits.append(hit)
    return unique_hits


def fuse_and_score_progress(
    patient_id: str,
    *args,
    **kwargs,
) -> Dict[str, List]:
    """
    Backward-compatible API:
      fuse_and_score_progress(patient_id, visit_id, patient_text, patient_fields)
      fuse_and_score_progress(patient_id, visit_id, patient_text, umls_dir, pubmed_email, patient_fields)
    """
    visit_id = kwargs.pop("visit_id", None)
    patient_text = kwargs.pop("patient_text", None)
    umls_dir = kwargs.pop("umls_dir", None)
    pubmed_email = kwargs.pop("pubmed_email", None)
    patient_fields = kwargs.pop("patient_fields", None)
    pbar = kwargs.pop("pbar", None)
    k_umls = kwargs.pop("k_umls", K_UMLS)
    k_pubmed = kwargs.pop("k_pubmed", K_PUBMED)

    if args:
        if len(args) == 3 and not isinstance(args[0], dict):
            visit_id, patient_text, patient_fields = args
            umls_dir = "."
            pubmed_email = "test@example.com"
        elif len(args) == 4 and isinstance(args[2], dict):
            visit_id, patient_text, umls_dir, patient_fields = args
            pubmed_email = "test@example.com"
        elif len(args) >= 5:
            visit_id, patient_text, umls_dir, pubmed_email, patient_fields = args[:5]

    if patient_fields is None and isinstance(umls_dir, dict):
        patient_fields = umls_dir
        umls_dir = "."
        pubmed_email = "test@example.com"
    if patient_text is None:
        patient_text = ""
    if umls_dir is None:
        umls_dir = "."
    if pubmed_email is None:
        pubmed_email = "test@example.com"
    if patient_fields is None:
        patient_fields = {}
    cache_dir = Path("retrieval_cache") / patient_id / visit_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    umls_fp = cache_dir / "umls_links.json"
    pubmed_fp = cache_dir / "pubmed.json"

    if umls_fp.exists() and pubmed_fp.exists():
        umls_hits = json.load(open(umls_fp, encoding="utf-8"))
        pubmed_hits = _dedupe_hits(json.load(open(pubmed_fp, encoding="utf-8")), ["pmid", "title", "abstract"])
    else:
        # (1) Embed
        if pbar: pbar.set_description("embed")
        qvec = encoder.encode([patient_text])[0]

        # (2) UMLS — 逐 entity 連結到 CUI（不再用整段 context 撈 top-k）
        if pbar: pbar.set_description("umls")
        umls_hits = _link_entities_to_umls(patient_fields, umls_dir)

        # ---------- (3) PubMed ----------
        if pbar: pbar.set_description("pubmed")
        client_p = PubMedClient(email=pubmed_email, api_key=PUBMED_API_KEY)

        # 先把所有 name 撈出來
        topics_raw = [
            c["name"]
            for fld in ("conditions", "procedures", "drugs")
            for sub in patient_fields.get(fld, [])
            for c in sub
        ]
        # 過濾掉純 ICD-10 代號（開頭為字母＋數字）
        topics = [t for t in topics_raw if t and t.strip()]
        tqdm.write(f"[DBG-PM] PID={patient_id}  topics_raw={len(topics_raw)}  topics_filtered={len(topics)}")
        if not topics:
            pubmed_hits = []
        else:
            # 依過濾後的 topics 去搜尋 PMID
            all_pmids = []
            for name in topics:
                pmids = client_p.search(name, retmax=k_pubmed)
                all_pmids.extend(pmids)
            # DEBUG ───────────────────────────────
            tqdm.write(f"[DBG-PM] PID={patient_id}  topics={len(topics)}  "
                       f"raw_pmids={len(all_pmids)}")
            # 取唯一值並限制總數
            all_pmids = list(dict.fromkeys(all_pmids))[: k_pubmed * max(len(topics), 1)]
            arts     = client_p.fetch_abstracts(all_pmids)

            abstracts = [art["abstract"] for art in arts]
            abstract_vecs = encoder.encode(abstracts)

            pubmed_hits = []
            dropped_lowsim = 0
            for art, vec in zip(arts, abstract_vecs):
                # ① 相似度計算
                sim   = cosine_similarity(qvec, vec)

                # ①' Phase 2a：與病人 context 太不相關的 abstract 直接丟掉，不送生成
                if sim < PUBMED_MIN_SIM:
                    dropped_lowsim += 1
                    continue

                score = score_pubmed_hit(
                    sim,
                    art.get("year") or datetime.now().year
                )

                # ② 時間戳：優先用 fetch_abstracts 解析到的 timestamp
                ts = art.get("timestamp")
                if not ts:
                    yr = art.get("year")
                    if yr and str(yr).isdigit():    # 只有年份 → 補成 YYYY-01-01
                        ts = f"{yr}-01-01"

                # ③ 收進 pubmed_hits（注意，這行前面不要多縮排）
                pubmed_hits.append({
                    **art,
                    "sim": round(float(sim), 4),
                    "score": score,
                    "timestamp": ts          # ← 這裡就帶進去了
                })
            pubmed_hits.sort(key=lambda x: x['score'], reverse=True)
            pubmed_hits = _dedupe_hits(pubmed_hits, ["pmid", "title", "abstract"])
            # DEBUG ───────────────────────────────
            tqdm.write(f"[DBG-PM] PID={patient_id}  final_hits={len(pubmed_hits)}  "
                       f"dropped_lowsim={dropped_lowsim} (thr={PUBMED_MIN_SIM})")

        # Cache to disk
        json.dump(umls_hits, open(umls_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(pubmed_hits, open(pubmed_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if pbar: pbar.set_description("done")
    return {"umls": umls_hits, "pubmed": pubmed_hits}


def merge_to_triples(
    patient_id: str,
    *args,
    **kwargs,
) -> List[Triple]:
    """
    Backward-compatible API:
      merge_to_triples(patient_id, fused, patient_context)
      merge_to_triples(patient_id, visit_id, fused, patient_context, patient_fields, visit_date=...)
    """
    visit_id = kwargs.pop("visit_id", None)
    fused = kwargs.pop("fused", None)
    patient_context = kwargs.pop("patient_context", None)
    patient_fields = kwargs.pop("patient_fields", None)
    visit_date = kwargs.pop("visit_date", None)
    output_dir = kwargs.pop("output_dir", "triples_output")
    accumulate = kwargs.pop("accumulate", True)

    if args:
        if len(args) == 2 and isinstance(args[0], dict):
            fused, patient_context = args
            visit_id = "default_visit"
        elif len(args) >= 3:
            if isinstance(args[0], dict):
                fused, patient_context = args[0], args[1]
                visit_id = "default_visit"
            else:
                visit_id, fused, patient_context = args[:3]
                if len(args) >= 4:
                    patient_fields = args[3]
                if len(args) >= 5:
                    visit_date = args[4]

    if fused is None:
        fused = {"umls": [], "pubmed": []}
    if patient_context is None:
        patient_context = ""
    if patient_fields is None:
        patient_fields = {}
    if visit_id is None:
        visit_id = "default_visit"
    # 1) 準備輸出與快取資料夾
    pdir      = Path(output_dir) / patient_id
    pdir.mkdir(parents=True, exist_ok=True)
    out_path  = pdir / f"triples_{visit_id}.json"

    cache_dir = Path("retrieval_cache") / patient_id / visit_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_fp   = cache_dir / "last_batch.txt"
    start_i   = int(last_fp.read_text()) if last_fp.exists() else 0

    triples: List[Triple] = []
    rejected_triples: List[Triple] = []
    review_triples: List[Triple] = []

    # ① Patient → Disease / Procedure / Drug（來自 EHR 記錄，為結構性事實；UMLS 只提供 CUI grounding）
    _REL_BY_TYPE = {"Disease": "HAS_DISEASE", "Procedure": "RECEIVED_PROCEDURE", "Drug": "USED_DRUG"}
    umls_hits = fused.get("umls", []) or []
    for rank, hit in enumerate(umls_hits, start=1):
        # 相容舊 schema：舊版是 {cui, name, score}
        entity_name = hit.get("query_entity") or hit.get("name")
        if not entity_name:
            continue
        etype = hit.get("entity_type", "Disease")
        relation = hit.get("relation") or _REL_BY_TYPE.get(etype, "HAS_DISEASE")
        cui = hit.get("cui")
        umls_name = hit.get("umls_name") or hit.get("name")
        link_score = float(hit.get("score") or 0.0)
        linked = bool(cui)
        match = hit.get("match", "linked" if linked else "unlinked")

        candidate = Triple(
            head=patient_id, head_type="Patient",
            relation=relation,
            tail=entity_name, tail_type=etype,
            visit_id=visit_id, visit_date=visit_date,
            timestamp=visit_date,
            source="EHR",
            weight=link_score if linked else None,
            evidence_source="UMLS" if linked else "EHR",
            evidence_id=cui,
            evidence_rank=rank,
            evidence_score=link_score if linked else None,
            evidence_title=umls_name if linked else None,
            evidence_text=(
                f"EHR-recorded {etype.lower()} '{entity_name}'"
                + (f" grounded to UMLS concept '{umls_name}' (CUI {cui}, {match} match)." if linked else ".")
            ),
        )
        # EHR 事實直接進圖；support 欄位僅記錄 UMLS 連結強度供追溯
        candidate.support_score = link_score
        candidate.support_status = "ACCEPTED"
        candidate.support_components = {"umls_link": link_score, "match": match}
        triples.append(candidate)

    # ② RAG via Gemini（文獻三元組）
    from config import GEMINI_BATCH_SIZE
    pubmed_hits = fused.get("pubmed", []) or []
    # 每篇 abstract 的識別碼＝PMID（沒有就用序號）
    def _pid_of(h, k):
        return str(h.get("pmid") or f"idx{k}")
    abstracts    = [h.get("abstract", "") for h in pubmed_hits]
    abstract_ids = [_pid_of(h, k) for k, h in enumerate(pubmed_hits)]
    hit_by_id    = {aid: h for aid, h in zip(abstract_ids, pubmed_hits)}
    best_score   = pubmed_hits[0]["score"] if pubmed_hits else None

    anchor_names, anchor_idx = _anchor_index(patient_fields, patient_id)
    print(f"[DBG-RAG] PID={patient_id}  abstracts={len(abstracts)}  anchors={len(anchor_names)}")
    verifier = get_verifier()
    print(f"[DBG-RAG] verifier = {verifier.model_name if verifier else 'lexical fallback'}")
    rag_drop = {"generic": 0, "unanchored": 0, "empty": 0, "self": 0, "no_evidence": 0}
    unanchored_triples: List[Triple] = []

    b = max(1, int(GEMINI_BATCH_SIZE))
    for i in range(start_i, len(abstracts), b):
        batch_idx = i // b
        cache_fp  = cache_dir / f"rag_{batch_idx}.json"

        if cache_fp.exists():
            rag = json.load(open(cache_fp, "r", encoding="utf-8"))
        else:
            try:
                rag = _rag_generate(
                    patient_context=patient_context,
                    abstracts=abstracts[i:i + b],
                    patient_id=str(patient_id),
                    anchor_entities=anchor_names,
                    abstract_ids=abstract_ids[i:i + b],
                )
            except GeminiQuotaExhausted as e:
                print(f"[STOP] {e}\n       已完成 {batch_idx} 批，重跑會從第 {batch_idx} 批續。")
                break
            if rag is None:
                print(f"[STOP] RAG batch {batch_idx} 未成功，停在此。重跑可續。")
                break
            json.dump(rag, open(cache_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            last_fp.write_text(str(i + b))

        for t_raw in rag:
            if not isinstance(t_raw, dict) or not t_raw.get("head") or not t_raw.get("tail"):
                rag_drop["empty"] += 1
                continue

            # 這條 triple 的來源 abstract
            aid = t_raw.get("_abstract_id")
            current_hit = hit_by_id.get(aid, {})

            t, reason = _normalize_rag_triple(t_raw, str(patient_id), anchor_idx)
            if reason != "ok":
                rag_drop[reason] = rag_drop.get(reason, 0) + 1
                if reason == "unanchored" and t_raw.get("relation"):
                    unanchored_triples.append(Triple(
                        head=str(t_raw.get("head")), head_type=str(t_raw.get("head_type") or "Concept"),
                        relation=str(t_raw.get("relation")).upper(),
                        tail=str(t_raw.get("tail")), tail_type=str(t_raw.get("tail_type") or "Concept"),
                        visit_id=visit_id, visit_date=visit_date, source="RAG",
                        evidence_source="PubMed", evidence_id=current_hit.get("pmid"),
                        evidence_title=current_hit.get("title"),
                        support_status="UNANCHORED",
                    ))
                continue

            evidence_text = current_hit.get("abstract")
            evidence_title = current_hit.get("title")
            if not evidence_text and not evidence_title:
                rag_drop["no_evidence"] += 1
                continue

            ts = current_hit.get("timestamp") or visit_date
            if ts and isinstance(ts, int):
                ts = f"{ts}-01-01"
            elif ts and isinstance(ts, str) and re.fullmatch(r"\d{4}", ts):
                ts = f"{ts}-01-01"
            ts = str(ts) if ts is not None else ""

            candidate = Triple(
                head=t["head"], head_type=t.get("head_type", "Concept"),
                relation=t.get("relation", "ASSOCIATED_WITH"),
                tail=t["tail"], tail_type=t.get("tail_type", "Concept"),
                visit_id=visit_id, visit_date=visit_date, timestamp=ts,
                source="RAG", weight=current_hit.get("score", best_score),
                evidence_source="PubMed",
                evidence_id=current_hit.get("pmid"),
                evidence_score=current_hit.get("score", best_score),
                evidence_title=evidence_title,
                evidence_text=evidence_text,
            )

            if verifier is not None:
                # Phase 2b：triple ↔ sentence entailment（可追溯）
                vr = verifier.verify(
                    candidate,
                    evidence_text=evidence_text,
                    evidence_title=evidence_title,
                    evidence_id=current_hit.get("pmid"),
                )
                status = vr.decision
                candidate.support_score = vr.score
                candidate.support_status = status
                candidate.support_components = {
                    "entail_prob": vr.score,
                    "accept_threshold": verifier.accept_threshold,
                    "review_threshold": verifier.review_threshold,
                }
                candidate.verifier_model = vr.model_name
                candidate.verifier_score = vr.score
                candidate.verifier_decision = status
                candidate.supporting_sentence = vr.supporting_sentence
                candidate.claim_text = vr.claim
            else:
                support = assess_triple_support(
                    candidate,
                    retrieval_score=current_hit.get("score", best_score),
                    patient_context=patient_context,
                    evidence_text=evidence_text,
                    evidence_title=evidence_title,
                    evidence_source="PubMed",
                )
                status = support.status
                candidate.support_score = support.score
                candidate.support_status = status
                candidate.support_components = support.components

            if status == "ACCEPTED" or not SUPPORT_FILTER_ENABLED:
                triples.append(candidate)
            elif status == "REVIEW":
                review_triples.append(candidate)
            else:
                rejected_triples.append(candidate)

    print(f"[DBG-RAG] PID={patient_id}  kept={len(triples)}  review={len(review_triples)}  "
          f"nli_rejected={len(rejected_triples)}  dropped={rag_drop}")

    # ③ 去重＆寫檔
    if accumulate:
        existing = json.load(open(out_path, "r", encoding="utf-8")) if out_path.exists() else []
        merged   = {json.dumps(x, sort_keys=True) for x in existing + [t.__dict__ for t in triples]}
        json.dump([json.loads(s) for s in merged],
                  open(out_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[DEBUG] 已寫入 {out_path}，共 {len(merged)} 條三元組")

        if rejected_triples:
            rejected_out_path = pdir / f"rejected_triples_{visit_id}.json"
            existing_rejected = json.load(open(rejected_out_path, "r", encoding="utf-8")) if rejected_out_path.exists() else []
            rejected_merged = {json.dumps(x, sort_keys=True) for x in existing_rejected + [t.__dict__ for t in rejected_triples]}
            json.dump([json.loads(s) for s in rejected_merged],
                      open(rejected_out_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"[DEBUG] 已寫入 {rejected_out_path}，共 {len(rejected_merged)} 條被過濾三元組")
        if review_triples:
            review_out_path = pdir / f"review_triples_{visit_id}.json"
            existing_review = json.load(open(review_out_path, "r", encoding="utf-8")) if review_out_path.exists() else []
            review_merged = {json.dumps(x, sort_keys=True) for x in existing_review + [t.__dict__ for t in review_triples]}
            json.dump([json.loads(s) for s in review_merged],
                      open(review_out_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"[DEBUG] 已寫入 {review_out_path}，共 {len(review_merged)} 條待人工審核三元組")
        if unanchored_triples:
            un_path = pdir / f"unanchored_triples_{visit_id}.json"
            existing_un = json.load(open(un_path, "r", encoding="utf-8")) if un_path.exists() else []
            un_merged = {json.dumps(x, sort_keys=True) for x in existing_un + [t.__dict__ for t in unanchored_triples]}
            json.dump([json.loads(s) for s in un_merged],
                      open(un_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"[DEBUG] 已寫入 {un_path}，共 {len(un_merged)} 條未錨定到本病人的三元組（一般知識，不進個人圖）")

    return triples
