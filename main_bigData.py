#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_batch.py —— 批次載入 → 批次前處理 → 批次 RAG
只改 `.pkl` 快取輸出資料夾：
    cache/raw/raw_000.pkl
    cache/prep/prep_000.pkl
三元組 JSON 路徑由 merge_to_triples 內部決定，**不做修改**。
"""
import os, pickle, gc
from pathlib import Path
from tqdm import tqdm
from pandarallel import pandarallel

from data.loader import load_bigger_mimic4_dataset, load_mappings, enrich_sample_with_names
from data.preprocess import preprocess_samples
from config import sample_bigger_dataset_path, UMLS_DATA_DIR
from fusion.merger import fuse_and_score_progress, merge_to_triples

# ===== 參數 =====
BATCH_SIZE     = 1000
DATASET_NAME   = "mimic4"
TASK_NAME      = "lenofstay"
USE_BIG_SAMPLE = True
PAD_WIDTH      = 3           # 批次編號補零位數，依資料量自行調整

pandarallel.initialize(nb_workers=3)

def main() -> None:
    # 取總病患數
    total = load_bigger_mimic4_dataset(
    False,                         # ← 加這個
    dataset=DATASET_NAME,
    task=TASK_NAME,
    count_only=True
    )
    print(f"總病患數: {total}")

    # 建立快取資料夾
    RAW_DIR  = Path("cache/raw");  RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREP_DIR = Path("cache/prep"); PREP_DIR.mkdir(parents=True, exist_ok=True)

    cond_map, proc_map, drug_map, ndc_map = load_mappings()

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        idx = start // BATCH_SIZE
        idx_str = f"{idx:0{PAD_WIDTH}d}"
        print(f"\n=== 批次 {idx_str}（{start}–{end-1}）===")

        # ---------- 1) RAW ----------
        dataset = load_bigger_mimic4_dataset(
    False,                        # ← 這裡改成 False，不要用舊快取
    dataset=DATASET_NAME,
    task=TASK_NAME,
    offset=start,
    limit=BATCH_SIZE,
    max_samples=start + BATCH_SIZE,   # ← 給它一個上限才能重建
    big_sample=USE_BIG_SAMPLE,
        )
        raw_pkl = RAW_DIR / f"raw_{idx_str}.pkl"
        with raw_pkl.open("wb") as f:
            pickle.dump(dataset, f)

        # ---------- 2) 前處理 ----------
        processed = [
            enrich_sample_with_names(
                s,
                cond_map,           # ICD-診斷
                proc_map,           # ICD-處置
                drug_map,           # 這裡可給 {} 也行
                ndc_map=ndc_map     # ← 必傳，才能對 NDC→藥名
            )
            for s in preprocess_samples(dataset)
        ]
        prep_pkl = PREP_DIR / f"prep_{idx_str}.pkl"
        with prep_pkl.open("wb") as f:
            pickle.dump(processed, f)

        # ---------- 3) UMLS / PubMed / RAG（沿用原邏輯 + 強制重跑 rag） --------
        pbar = tqdm(processed, desc=f"RAG 批次 {idx_str}", unit="patient")
        
        # ---------- 3) UMLS / PubMed / RAG ----------          <-- 迴圈上方已有 pbar = tqdm(...)
        for sample in pbar:
            pid        = sample["patient_id"]
            vid        = sample["visit_id"]
            visit_date = sample.get("visit_date", "")

            # ── 0) 準備 patient_context 文字 ──────────────────
            patient_ctx = (
                f"PatientID: {pid}; VisitDate: {visit_date}; "
                f"Conditions: {[c['name'] for g in sample['conditions'] for c in g]}; "
                f"Drugs: {[d['name'] for g in sample['drugs'] for d in g]}; "
                f"Procedures: {[p['name'] for g in sample['procedures'] for p in g]}"
            )

            # ── 1) 檢查 PubMed 快取；若 pubmed.json 不在就重置 RAG ──────
            cache_dir = Path("retrieval_cache") / pid / vid
            pubmed_fp = cache_dir / "pubmed.json"
            if not pubmed_fp.exists():
                # (a) 刪掉 last_batch.txt  ⇒ 讓 merge_to_triples 從 0 開始
                last_fp = cache_dir / "last_batch.txt"
                if last_fp.exists():
                    last_fp.unlink()
                # (b) 刪掉舊的 triples_*.json ⇒ 重新累積
                out_path = Path("triples_output") / pid / f"triples_{vid}.json"
                if out_path.exists():
                    out_path.unlink()

            # ── 2) 執行（或讀快取的）UMLS + PubMed ───────────────
            fused = fuse_and_score_progress(
                patient_id     = pid,
                visit_id       = vid,
                patient_text   = patient_ctx,
                umls_dir       = UMLS_DATA_DIR,
                pubmed_email   = os.getenv("NCBI_EMAIL", ""),
                patient_fields = {
                    "conditions": sample["conditions"],
                    "procedures": sample["procedures"],
                    "drugs":      sample["drugs"],
                },
                pbar = pbar,
            )

            # ── 3) 產生／合併三元組（累加模式） ─────────────────
            merge_to_triples(
                patient_id      = pid,
                visit_id        = vid,
                fused           = fused,
                patient_context = patient_ctx,
                patient_fields  = {
                    "conditions": sample["conditions"],
                    "procedures": sample["procedures"],
                    "drugs":      sample["drugs"],
                },
                visit_date      = visit_date,
                accumulate      = True,   # 一律累加
            )
        pbar.close()

        # ---------- 4) 釋放記憶體 ----------
        del dataset, processed
        gc.collect()

if __name__ == "__main__":
    main()

