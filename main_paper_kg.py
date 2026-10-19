#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只重建「論文三個任務實際用到的」病人×就診的知識圖，不碰全 MIMIC-IV。

樣本清單 = exp_data/ccscm_ccsproc/sample/sample_dataset_mimic4_{task}_preprocessed.pkl
          三個任務的 (patient_id, visit_id) 聯集（~1258 對 / ~494 病人）。

已 enrich 的樣本直接從 cache/prep/prep_*.pkl 取（那是修正後的 preprocess 產的）。

因為檢索參數改了（K_PUBMED、PUBMED_MIN_SIM），會先清掉每一對的
retrieval_cache 的 rag_*.json / last_batch.txt / pubmed.json 和 triples_output
的舊檔，強制以新參數重建。umls_links.json 不清（跟檢索參數無關）。

用法：
  python main_paper_kg.py                 # 全部 1258 對
  python main_paper_kg.py --limit 50      # 先跑 50 對測試
"""
import argparse
import glob
import os
import json
import pickle
from pathlib import Path

from tqdm import tqdm

from config import UMLS_DATA_DIR, K_PUBMED, PUBMED_MIN_SIM
from fusion.merger import fuse_and_score_progress, merge_to_triples

TASKS = ["mortality", "readmission", "lenofstay"]
SAMPLE_DIR = "exp_data/ccscm_ccsproc/sample"
PREP_GLOB = "cache/prep/prep_*.pkl"
EMAIL = os.getenv("NCBI_EMAIL", "")


def needed_pairs():
    need = set()
    for t in TASKS:
        p = f"{SAMPLE_DIR}/sample_dataset_mimic4_{t}_preprocessed.pkl"
        for s in pickle.load(open(p, "rb")):
            need.add((str(s["patient_id"]), str(s["visit_id"])))
    return need


def load_enriched():
    by_key = {}
    for f in sorted(glob.glob(PREP_GLOB)):
        for s in pickle.load(open(f, "rb")):
            by_key[(str(s["patient_id"]), str(s["visit_id"]))] = s
    return by_key


def reset_pair(pid, vid):
    cd = Path("retrieval_cache") / pid / vid
    for pat in ("rag_*.json", "last_batch.txt", "pubmed.json"):
        for fp in cd.glob(pat):
            fp.unlink()
    od = Path("triples_output") / pid
    for pat in (f"triples_{vid}.json", f"rejected_triples_{vid}.json",
                f"review_triples_{vid}.json", f"contradicted_triples_{vid}.json",
                f"unanchored_triples_{vid}.json"):
        fp = od / pat
        if fp.exists():
            fp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-reset", action="store_true",
                    help="不清舊快取（斷點續跑時用）")
    ap.add_argument("--resume", action="store_true",
                    help="已經有 triples_<vid>.json 的就跳過（配 --no-reset 用）")
    a = ap.parse_args()

    need = needed_pairs()
    enriched = load_enriched()
    pairs = sorted(k for k in need if k in enriched)
    missing = sorted(need - set(enriched))
    if missing:
        print(f"[WARN] {len(missing)} 對在 prep pkl 裡找不到，會略過：{missing[:5]}")
    if a.limit:
        pairs = pairs[: a.limit]

    print(f"[paper-kg] K_PUBMED={K_PUBMED}  PUBMED_MIN_SIM={PUBMED_MIN_SIM}")
    print(f"[paper-kg] 要處理 {len(pairs)} 對 (patient, visit)")

    for pid, vid in tqdm(pairs, unit="visit"):
        # --resume：pubmed.json 是這次 K_PUBMED 重跑才會重建的，用它判斷「本次跑過了」，
        # 才不會被舊 K=1 留下的 triples_<vid>.json 誤導而整批跳過。
        if a.resume and (Path("retrieval_cache") / pid / vid / "pubmed.json").exists():
            continue
        s = enriched[(pid, vid)]
        visit_date = s.get("visit_date", "")
        fields = {
            "conditions": s["conditions"],
            "procedures": s["procedures"],
            "drugs": s["drugs"],
        }
        ctx = (
            f"PatientID: {pid}; VisitDate: {visit_date}; "
            f"Conditions: {[c['name'] for g in s['conditions'] for c in g]}; "
            f"Drugs: {[d['name'] for g in s['drugs'] for d in g]}; "
            f"Procedures: {[p['name'] for g in s['procedures'] for p in g]}"
        )

        if not a.no_reset:
            reset_pair(pid, vid)

        fused = fuse_and_score_progress(
            patient_id=pid, visit_id=vid, patient_text=ctx,
            umls_dir=UMLS_DATA_DIR, pubmed_email=EMAIL, patient_fields=fields,
        )
        merge_to_triples(
            patient_id=pid, visit_id=vid, fused=fused,
            patient_context=ctx, patient_fields=fields,
            visit_date=visit_date, accumulate=True,
        )

    print("[paper-kg] done")


if __name__ == "__main__":
    main()
