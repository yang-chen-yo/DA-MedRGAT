# === fusion/kg_builder.py ===
"""
多跳、以病人為中心的文獻證據知識圖建置。

    hop 0 : Patient ──HAS_DISEASE / USED_DRUG / RECEIVED_PROCEDURE──> EHR entity
            （結構性事實；UMLS 提供 CUI grounding）
    hop k : 對「上一層新增的概念」再做 PubMed 檢索 + Gemini 生成 + NLI 驗證，
            連到下一層新概念。每條邊都有 PMID + supporting sentence + entailment 分數。

節點只會出現一次（跨層去重）；每一跳都限制 frontier 大小與每概念抓幾篇，避免爆炸。
斷點續跑：每一跳的 PubMed / RAG 結果都快取在 retrieval_cache/<pid>/<visit>/hop<k>/。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    PUBMED_API_KEY, PUBMED_MIN_SIM,
    KG_MAX_HOPS, KG_FRONTIER_TOPK, KG_HOP_K_PUBMED,
    SUPPORT_FILTER_ENABLED,
)
from kg.triple import Triple
from retrieval.pubmed_client import PubMedClient
from fusion.scoring import cosine_similarity, score_pubmed_hit
from fusion.triple_verifier import get_verifier
from fusion.triple_generation_hf import GeminiQuotaExhausted
from fusion.merger import (
    encoder, _norm, _normalize_rag_triple, _link_entities_to_umls,
    _iter_named_entities, _rag_generate,
)

_REL_BY_TYPE = {"Disease": "HAS_DISEASE", "Procedure": "RECEIVED_PROCEDURE", "Drug": "USED_DRUG"}
# 新概念端的 tail_type → 節點型別
_KEEP_STATUSES = {"ACCEPTED"}


def _load(fp: Path):
    return json.load(open(fp, encoding="utf-8")) if fp.exists() else None


def _dump(fp: Path, obj):
    fp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _retrieve_pubmed(terms: List[str], patient_text: str, client: PubMedClient,
                     used_pmids: set, k_each: int, cache_fp: Path) -> List[Dict]:
    """對一批 term 做 PubMed 檢索 + 相關性過濾；結果快取。"""
    cached = _load(cache_fp)
    if cached is not None:
        return cached

    raw: List[Dict] = []
    for term in terms:
        try:
            pmids = client.search(term, retmax=k_each, sort="relevance")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] PubMed search 失敗 term={term!r}: {e}")
            continue
        pmids = [p for p in pmids if p not in used_pmids]
        if not pmids:
            continue
        try:
            arts = client.fetch_abstracts(pmids)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] PubMed fetch 失敗: {e}")
            continue
        for a in arts:
            a["for_term"] = term
            raw.append(a)

    if not raw:
        _dump(cache_fp, [])
        return []

    # 相關性以「該概念」為主、病人整體 context 為輔（取兩者較大者）
    ctx_vec = encoder.encode([patient_text])[0]
    term_list = sorted({a["for_term"] for a in raw})
    term_vec = {t: v for t, v in zip(term_list, encoder.encode(term_list))}
    vecs = encoder.encode([a.get("abstract", "") for a in raw])
    hits: List[Dict] = []
    for a, v in zip(raw, vecs):
        sim_term = cosine_similarity(term_vec[a["for_term"]], v)
        sim_ctx = cosine_similarity(ctx_vec, v)
        sim = max(sim_term, sim_ctx)
        if sim < PUBMED_MIN_SIM:
            continue
        a["sim"] = round(float(sim), 4)
        a["sim_term"] = round(float(sim_term), 4)
        a["score"] = round(float(score_pubmed_hit(sim, a.get("year") or datetime.now().year)), 4)
        hits.append(a)
    # 每個 pmid 只留一次
    seen, uniq = set(), []
    for h in sorted(hits, key=lambda x: x["score"], reverse=True):
        if h["pmid"] in seen:
            continue
        seen.add(h["pmid"])
        uniq.append(h)
    print(f"[KG]   PubMed: {len(raw)} fetched -> {len(uniq)} kept (sim>={PUBMED_MIN_SIM}); "
          f"per-term kept: " +
          ", ".join(f"{t}:{sum(1 for h in uniq if h['for_term']==t)}" for t in term_list))
    _dump(cache_fp, uniq)
    return uniq


def build_patient_kg(
    patient_id: str,
    visit_id: str,
    patient_context: str,
    patient_fields: Dict,
    visit_date: Optional[str] = None,
    umls_dir: str = ".",
    pubmed_email: str = "test@example.com",
    max_hops: int = None,
    output_dir: str = "triples_output",
) -> List[Triple]:
    max_hops = KG_MAX_HOPS if max_hops is None else max_hops
    pid = str(patient_id)
    cache_root = Path("retrieval_cache") / pid / visit_id
    cache_root.mkdir(parents=True, exist_ok=True)

    verifier = get_verifier()
    print(f"[KG] verifier = {verifier.model_name if verifier else 'lexical fallback'}")
    client = PubMedClient(email=pubmed_email, api_key=PUBMED_API_KEY)

    accepted: List[Triple] = []
    rejected: List[Triple] = []
    review: List[Triple] = []
    contradicted: List[Triple] = []
    unanchored: List[Triple] = []

    # 圖上目前有的節點： norm(name) -> (canonical_name, node_type)
    nodes: Dict[str, Tuple[str, str]] = {_norm(pid): (pid, "Patient")}
    used_pmids: set = set()

    # ---------- hop 0：EHR entity（UMLS grounding）----------
    umls_hits = _link_entities_to_umls(patient_fields, umls_dir)
    _dump(cache_root / "umls_links.json", umls_hits)
    frontier: List[str] = []
    for rank, h in enumerate(umls_hits, 1):
        name = h.get("query_entity")
        if not name:
            continue
        etype = h.get("entity_type", "Disease")
        rel = h.get("relation") or _REL_BY_TYPE.get(etype, "HAS_DISEASE")
        cui, umls_name = h.get("cui"), h.get("umls_name")
        link = float(h.get("score") or 0.0)
        t = Triple(
            head=pid, head_type="Patient", relation=rel, tail=name, tail_type=etype,
            visit_id=visit_id, visit_date=visit_date, timestamp=visit_date, source="EHR",
            weight=link if cui else None,
            evidence_source="UMLS" if cui else "EHR", evidence_id=cui, evidence_rank=rank,
            evidence_score=link if cui else None, evidence_title=umls_name,
            evidence_text=(f"EHR-recorded {etype.lower()} '{name}'"
                           + (f" grounded to UMLS concept '{umls_name}' (CUI {cui})." if cui else ".")),
        )
        t.support_score, t.support_status = link, "ACCEPTED"
        t.support_components = {"umls_link": link, "match": h.get("match", "unlinked")}
        accepted.append(t)
        nkey = _norm(name)
        if nkey not in nodes:
            nodes[nkey] = (name, etype)
            frontier.append(name)

    print(f"[KG] hop0: {len(frontier)} EHR entities")

    # ---------- hop 1..N：文獻擴張 ----------
    for hop in range(1, max_hops + 1):
        if not frontier:
            break
        hop_dir = cache_root / f"hop{hop}"
        hop_dir.mkdir(parents=True, exist_ok=True)

        # hop1：病人所有 EHR entity 都該有文獻（他們是真實臨床事實）
        # hop2+：frontier 可能爆炸，只擴張分數最高的前 K 個
        expand = frontier if hop == 1 else frontier[:KG_FRONTIER_TOPK]
        print(f"[KG] hop{hop}: expanding {len(expand)} concepts -> {expand}")

        hits = _retrieve_pubmed(
            expand, patient_context, client, used_pmids,
            KG_HOP_K_PUBMED, hop_dir / "pubmed.json",
        )
        used_pmids.update(h["pmid"] for h in hits)
        if not hits:
            print(f"[KG] hop{hop}: no relevant PubMed hits, stop")
            break

        anchor_idx = {k: v for k, v in nodes.items()}
        anchor_names = [v[0] for v in nodes.values() if v[1] != "Patient"]
        abstract_ids = [h["pmid"] for h in hits]
        hit_by_id = {h["pmid"]: h for h in hits}

        rag_fp = hop_dir / "rag.json"
        rag = _load(rag_fp)
        if rag is None:
            try:
                rag = _rag_generate(
                    patient_context=patient_context,
                    abstracts=[h["abstract"] for h in hits],
                    patient_id=pid,
                    anchor_entities=anchor_names,
                    abstract_ids=abstract_ids,
                )
            except GeminiQuotaExhausted as e:
                print(f"[STOP] {e}")
                break                       # 不快取、不推進 → 明天/換模型重跑會續這一跳
            if rag is None:
                print(f"[STOP] hop{hop} RAG 未成功，停在此。重跑可續。")
                break
            _dump(rag_fp, rag)

        next_frontier: List[Tuple[str, float]] = []
        drop = {"generic": 0, "unanchored": 0, "self": 0, "empty": 0, "no_evidence": 0}
        for t_raw in rag:
            if not isinstance(t_raw, dict) or not t_raw.get("head") or not t_raw.get("tail"):
                drop["empty"] += 1
                continue
            hit = hit_by_id.get(t_raw.get("_abstract_id"), {})
            norm_t, reason = _normalize_rag_triple(t_raw, pid, anchor_idx)
            if reason != "ok":
                drop[reason] = drop.get(reason, 0) + 1
                if reason == "unanchored":
                    unanchored.append(Triple(
                        head=str(t_raw.get("head")), head_type=str(t_raw.get("head_type") or "Concept"),
                        relation=str(t_raw.get("relation", "")).upper(),
                        tail=str(t_raw.get("tail")), tail_type=str(t_raw.get("tail_type") or "Concept"),
                        visit_id=visit_id, source="RAG", evidence_source="PubMed",
                        evidence_id=hit.get("pmid"), support_status="UNANCHORED"))
                continue

            ev_text, ev_title = hit.get("abstract"), hit.get("title")
            if not ev_text and not ev_title:
                drop["no_evidence"] += 1
                continue

            cand = Triple(
                head=norm_t["head"], head_type=norm_t.get("head_type", "Concept"),
                relation=norm_t.get("relation", "ASSOCIATED_WITH"),
                tail=norm_t["tail"], tail_type=norm_t.get("tail_type", "Concept"),
                visit_id=visit_id, visit_date=visit_date,
                timestamp=hit.get("timestamp") or visit_date, source="RAG",
                weight=hit.get("score"), evidence_source="PubMed",
                evidence_id=hit.get("pmid"), evidence_score=hit.get("score"),
                evidence_title=ev_title, evidence_text=ev_text,
            )
            cand.support_components = {"hop": hop}

            if verifier is not None:
                vr = verifier.verify(cand, evidence_text=ev_text, evidence_title=ev_title,
                                     evidence_id=hit.get("pmid"))
                status = vr.decision
                cand.verifier_model = vr.model_name
                cand.verifier_score = vr.score
                cand.verifier_decision = status
                cand.supporting_sentence = vr.supporting_sentence
                cand.claim_text = vr.claim
                cand.support_score = vr.score
                cand.support_status = status
                cand.support_components.update({"entail_prob": vr.score})
            else:
                status = "REVIEW"
                cand.support_status = status

            if status == "CONTRADICTED":
                contradicted.append(cand)
                continue
            if status == "REVIEW":
                review.append(cand)
                continue
            if status != "ACCEPTED" and SUPPORT_FILTER_ENABLED:
                rejected.append(cand)
                continue

            accepted.append(cand)
            # 新概念端 → 下一層 frontier
            for endp, etype in ((cand.head, cand.head_type), (cand.tail, cand.tail_type)):
                k = _norm(endp)
                if k not in nodes:
                    nodes[k] = (endp, etype or "Concept")
                    next_frontier.append((endp, float(cand.verifier_score or 0.0)))

        print(f"[KG] hop{hop}: +{len(next_frontier)} new concepts  drop={drop}  "
              f"(accepted so far {len(accepted)})")
        next_frontier.sort(key=lambda x: x[1], reverse=True)
        frontier = [n for n, _ in next_frontier]

    # ---------- 寫檔 ----------
    pdir = Path(output_dir) / pid
    pdir.mkdir(parents=True, exist_ok=True)
    stem = f"single_patient_demo_{visit_id}"
    _dump(pdir / f"{stem}.json", [t.__dict__ for t in accepted])
    for name, lst in (("rejected", rejected), ("review", review),
                      ("contradicted", contradicted), ("unanchored", unanchored)):
        if lst:
            _dump(pdir / f"{name}_triples_{visit_id}.json", [t.__dict__ for t in lst])

    print(f"[KG] done: {len(accepted)} accepted / {len(review)} review / "
          f"{len(contradicted)} contradicted / {len(rejected)} rejected / "
          f"{len(unanchored)} unanchored  | nodes={len(nodes)}")
    return accepted
