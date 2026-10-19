# data/loader.py
from tqdm import tqdm
import re, functools
import logging
from pyhealth.datasets import MIMIC4Dataset
from simple_icd_10 import get_description
from icd9cms.icd9 import search as icd9_search
from typing import List, Dict, Set,Optional
import pandas as pd, os, pickle, gc 
import numpy as np   
import os, pickle
import csv 
from icdmappings import Mapper
mapper = Mapper() 
from data.task_fn import (
    drug_recommendation_mimic4_fn,
    mortality_prediction_mimic4_fn,
    readmission_prediction_mimic4_fn,
    length_of_stay_prediction_mimic4_fn
)
from config import (                           
    MIMIC4_ROOT,
    sample_dataset_path,
    full_dataset_path,
    sample_bigger_dataset_path,
    CONDITION_MAPPING_FILE,                  
    PROCEDURE_MAPPING_FILE,                  
    DRUG_MAPPING_FILE,                        
)
# ---------- 1) 工具函式 ---------- #
# ===== data/loader.py =====
def _load_id_subset(root: str, max_samples: int) -> Set[int]:
    """
    root 已經指到 '.../hosp'。
    嘗試讀取 patients.csv 或 patients.csv.gz，讀不到就直接報錯。
    """
    csv_plain = os.path.join(root, "patients.csv")
    csv_gzip  = os.path.join(root, "patients.csv.gz")

    if os.path.isfile(csv_plain):
        csv_path = csv_plain
    elif os.path.isfile(csv_gzip):
        csv_path = csv_gzip
    else:
        raise FileNotFoundError(
            f"找不到 {csv_plain} 或 {csv_gzip}，請確認檔案存在且具可讀權限"
        )

    print(f"[INFO] 讀取 patients 檔案：{csv_path}")
    ids = pd.read_csv(csv_path, usecols=["subject_id"], nrows=max_samples)["subject_id"]
    return set(ids.astype(int).tolist())
def _filtered_table(path: str, target_ids: Set[int], usecols: List[str]) -> pd.DataFrame:
    """
    以 chunksize 方式讀取大表，只留下 subject_id ∈ target_ids 的列。
    """
    keep_chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000, low_memory=False):
        chunk = chunk[chunk["subject_id"].isin(target_ids)]
        if not chunk.empty:
            keep_chunks.append(chunk)
    if keep_chunks:
        return pd.concat(keep_chunks, ignore_index=True)
    return pd.DataFrame(columns=usecols)

# ---------- 2) 精簡版 Dataset 建立 ---------- #
def load_bigger_mimic4_dataset(
    load_processed: bool,
    dataset: str,
    task: str,
    dev: bool = True,
    max_samples: int | None = None,
    big_sample: bool = False,
    count_only: bool = False,
    offset: int = 0,                      # ← 新增
    limit: int | None = None,            # ← 新增
) -> List[dict] | int:
    """
    通用：前 max_samples 位病人 → 讀 5 張表
    - mortality/drugrec/lenofstay：仍用 PyHealth 的 task_fn
    - readmission：自行手刻 15 天再入院樣本，避免 PyHealth 因樣本不足丟空
    訓練用的標籤由 build_labels.py 另外產生，這裡的 label 只供參考
    """
        # ── 0. 若只要總筆數 ────────────────────────────────────────
    if count_only:
        ids_csv = os.path.join(MIMIC4_ROOT, "patients.csv")
        if not os.path.isfile(ids_csv):
            ids_csv += ".gz"
        total = sum(1 for _ in pd.read_csv(ids_csv, usecols=["subject_id"]).itertuples())
        return total

    # 內部小工具：依 offset / limit 對 list 做切片 ---------------
    def _batch_slice(lst: list) -> list:
        if offset == 0 and limit is None:
            return lst
        end = None if limit is None else offset + limit
        return lst[offset:end]
    # ---------- 1) 決定快取 ----------
    cache_path = (
        sample_bigger_dataset_path(dataset, task) if big_sample and dev
        else sample_dataset_path(dataset, task)   if dev
        else full_dataset_path(dataset, task)
    )

    if load_processed and os.path.exists(cache_path):
        print(f"[INFO] 讀取快取 {cache_path}")
        data = pickle.load(open(cache_path, "rb"))
        if max_samples is not None:
            data = data[:max_samples]
        return _batch_slice(data)

    assert max_samples is not None, "須提供 max_samples 才能重建資料集"
    print(f"[INFO] 建立縮小版 MIMIC-IV（前 {max_samples} 位病人）…")

    hosp_dir = MIMIC4_ROOT  # config 指向 .../hosp
    ids_csv = os.path.join(hosp_dir, "patients.csv")
    if not os.path.isfile(ids_csv):
        ids_csv += ".gz"

    pid_list = pd.read_csv(ids_csv, usecols=["subject_id"], nrows=max_samples)["subject_id"].tolist()
    target_ids = set(pid_list)

    # ---------- 2) 讀 5 張表，僅留下 target_ids ----------
    def _load(path, usecols):
        dfs = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000, low_memory=False):
            sub = chunk[chunk["subject_id"].isin(target_ids)]
            if not sub.empty:
                dfs.append(sub)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=usecols)

    diag_df = _load(os.path.join(hosp_dir, "diagnoses_icd.csv"),
                    ["subject_id", "hadm_id", "icd_code"])
    proc_df = _load(os.path.join(hosp_dir, "procedures_icd.csv"),
                    ["subject_id", "hadm_id", "icd_code"])
    drug_df = _load(os.path.join(hosp_dir, "prescriptions.csv"),
                    ["subject_id", "hadm_id", "ndc"])
    adm_df  = _load(os.path.join(hosp_dir, "admissions.csv"),
                    ["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"])
    pat_df  = _load(os.path.join(hosp_dir, "patients.csv"),
                    ["subject_id", "anchor_age", "anchor_year", "gender"])

    print(f"  • diagnoses_icd: {len(diag_df):,} rows")
    print(f"  • procedures_icd: {len(proc_df):,} rows")
    print(f"  • prescriptions: {len(drug_df):,} rows")
    print(f"  • admissions: {len(adm_df):,} rows")
    print(f"  • patients: {len(pat_df):,} rows")
    gc.collect()

    # ---------- 3) 如果是 readmission：手刻 15 天再入院樣本 ----------
    if task == "readmission":
        adm_df["admittime"]  = pd.to_datetime(adm_df["admittime"])
        adm_df["dischtime"]  = pd.to_datetime(adm_df["dischtime"])

        # 先按 (subject_id, admittime) 排序
        adm_df = adm_df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

        samples: List[dict] = []
        grouped = adm_df.groupby("subject_id", sort=False)
        for pid, grp in tqdm(grouped, desc="Building readmission samples", unit="patient"):
            grp = grp.sort_values("admittime")
            # 需要「有下一次住院」的 admission 才能建立樣本
            for i in range(len(grp) - 1):
                curr = grp.iloc[i]
                nxt  = grp.iloc[i + 1]

                if pd.isna(curr["dischtime"]) or pd.isna(nxt["admittime"]):
                    continue

                # 與 build_labels.py 相同：下次入院距本次入院 < 15 天
                gap_days = (nxt["admittime"] - curr["admittime"]).days
                label = int(gap_days < 15)

                hadm = curr["hadm_id"]
                cond = diag_df.loc[diag_df["hadm_id"] == hadm, "icd_code"].dropna().astype(str).tolist()
                proc = proc_df.loc[proc_df["hadm_id"] == hadm, "icd_code"].dropna().astype(str).tolist()
                drug = drug_df.loc[drug_df["hadm_id"] == hadm, "ndc"].dropna().astype(str).tolist()

                samples.append({
                    "patient_id": str(pid),
                    "visit_id":   str(hadm),
                    "visit_date": curr["admittime"].strftime("%Y-%m-%d %H:%M:%S"),
                    "conditions": [cond],
                    "procedures": [proc],
                    "drugs":      [drug],
                    "label":      label,
                })

                if max_samples and len(samples) >= max_samples:
                    break
            if max_samples and len(samples) >= max_samples:
                break

        if not samples:
            raise RuntimeError("readmission 樣本為空，可能所有病人都只有一次住院")

        # 存快取並回傳
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"[INFO] 已存快取至 {cache_path}")
        return _batch_slice(samples)

        # ---------- 3‑B) lenofstay：手刻住院天數樣本 ----------
    elif task == "lenofstay":
        adm_df["admittime"] = pd.to_datetime(adm_df["admittime"])
        adm_df["dischtime"] = pd.to_datetime(adm_df["dischtime"])
        adm_df["los_days"]  = (adm_df["dischtime"] - adm_df["admittime"]).dt.total_seconds() / 86400.0
        # 去掉缺失或負值
        adm_df = adm_df[adm_df["los_days"].notnull() & (adm_df["los_days"] >= 0)]

        samples: List[dict] = []
        for _, adm in tqdm(adm_df.iterrows(), desc="Building lenofstay samples", total=len(adm_df)):
            pid  = adm["subject_id"]
            hadm = adm["hadm_id"]

            cond = diag_df.loc[diag_df["hadm_id"] == hadm, "icd_code"].dropna().astype(str).tolist()
            proc = proc_df.loc[proc_df["hadm_id"] == hadm, "icd_code"].dropna().astype(str).tolist()
            drug = drug_df.loc[drug_df["hadm_id"] == hadm, "ndc"].dropna().astype(str).tolist()

            samples.append({
                "patient_id": str(pid),
                "visit_id":   str(hadm),
                "visit_date": adm["admittime"].strftime("%Y-%m-%d %H:%M:%S"),
                "conditions": [cond],
                "procedures": [proc],
                "drugs":      [drug],
                "label":      float(adm["los_days"]),   # 連續值；後續模型可做回歸或離散化
            })
            if max_samples and len(samples) >= max_samples:
                break

        if not samples:
            raise RuntimeError("lenofstay 樣本為空，可能所有紀錄都缺 admittime 或 dischtime")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"[INFO] 已存快取至 {cache_path}")
        return _batch_slice(samples)

        # ---------- 3‑C) drugrec：手刻「條件→推薦藥」樣本 ----------
    elif task == "drugrec":
        # 前置：把 prescriptions 依 hadm_id 聚合成藥物清單
        drug_grp = drug_df.groupby("hadm_id")["ndc"].apply(
            lambda x: [str(code) for code in x.dropna().unique()]
        ).to_dict()

        samples: List[dict] = []
        # 取 admissions 為基礎，只挑有開藥的 hadm_id
        for _, adm in tqdm(adm_df.iterrows(), desc="Building drugrec samples", total=len(adm_df)):
            hadm = adm["hadm_id"]
            if hadm not in drug_grp:
                continue  # 沒開任何藥 → 跳過

            pid  = adm["subject_id"]
            cond = diag_df.loc[diag_df["hadm_id"] == hadm, "icd_code"].dropna().astype(str).tolist()
            proc = proc_df.loc[proc_df["hadm_id"] == hadm, "icd_code"].dropna().astype(str).tolist()
            drug_list = drug_grp[hadm]

            samples.append({
                "patient_id": str(pid),
                "visit_id":   str(hadm),
                "visit_date": adm["admittime"],
                "conditions": [cond],
                "procedures": [proc],
                "drugs":      [drug_list],   # ground‑truth 用藥
                "label":      drug_list,     # 給模型學習的 target
            })
            if max_samples and len(samples) >= max_samples:
                break

        if not samples:
            raise RuntimeError("drugrec 樣本為空，可能這批病人都沒有 prescriptions")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
        print(f"[INFO] 已存快取至 {cache_path}")
        return _batch_slice(samples)



    # ---------- 4) 其他任務：仍用 PyHealth fn ----------
    ds = MIMIC4Dataset(root=MIMIC4_ROOT, tables=[], dev=False, refresh_cache=False)
    ds.tables = {
        "diagnoses_icd": diag_df,
        "procedures_icd": proc_df,
        "prescriptions":  drug_df,
        "admissions":     adm_df,
        "patients":       pat_df,
    }
    # 動態找能用的索引重建方法（不同版本 API 名稱不同）
    for m in ("rebuild_index", "build_index", "_build_index"):
        if hasattr(ds, m):
            getattr(ds, m)()
            break

    if task == "mortality":
        fn = mortality_prediction_mimic4_fn
    elif task == "drugrec":
        fn = drug_recommendation_mimic4_fn
    elif task == "lenofstay":
        fn = length_of_stay_prediction_mimic4_fn
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"[INFO] 轉換為 '{task}' 樣本 …")
    raw_ds = ds.set_task(fn)
    samples: List[dict] = []
    for s in tqdm(raw_ds, desc=f"Building {task}", unit="patient"):
        samples.append(s)
        if max_samples and len(samples) >= max_samples:
            break

    if not samples:
        raise RuntimeError(f"{task} 樣本為空，請檢查 admissions/patients 表是否正確")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(samples, f)
    print(f"[INFO] 已存快取至 {cache_path}")
    return _batch_slice(samples)


    
def load_mimic4_dataset(
    load_processed: bool,
    dataset: str,
    task: str,
    dev: bool = True,
    max_samples: int | None = None,
    big_sample: bool = False,      # ← ★ 新增參數
) -> List[dict]:
    if dev:
        cache_path = (
            sample_bigger_dataset_path(dataset, task)
            if big_sample else sample_dataset_path(dataset, task)
        )
    else:
        cache_path = full_dataset_path(dataset, task)

    # 1) 從快取讀
    if load_processed and os.path.exists(cache_path):
        print(f"[INFO] Loading cached dataset for '{task}' ({'sample' if dev else 'full'}) from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data if max_samples is None else data[:max_samples]
        except (AttributeError, EOFError, ImportError, ModuleNotFoundError, pickle.UnpicklingError, ValueError):
            print(f"[WARN] cache is stale or incompatible for '{task}', rebuilding: {cache_path}")
            try:
                os.remove(cache_path)
            except OSError:
                pass

    # 2) 建立原始 PyHealth Dataset
    ds = MIMIC4Dataset(
        root=MIMIC4_ROOT,
        tables=["diagnoses_icd","procedures_icd","prescriptions"],
        code_mapping={
            "NDC":("ATC",{"target_kwargs":{"level":3}}),
            "ICD9CM":"CCSCM","ICD9PROC":"CCSPROC",
            "ICD10CM":"CCSCM","ICD10PROC":"CCSPROC",
        },
        dev=dev,
        refresh_cache=True
    )

    # 3) 選 task_fn
    if task=="drugrec":
        fn=drug_recommendation_mimic4_fn
    elif task=="mortality":
        fn=mortality_prediction_mimic4_fn
    elif task=="readmission":
        fn=readmission_prediction_mimic4_fn
    elif task=="lenofstay":
        fn=length_of_stay_prediction_mimic4_fn
    else:
        raise ValueError(f"Unknown task: {task}")

    # 4) 逐筆載入並顯示進度
    print(f"[INFO] Building '{task}' samples (limit={max_samples}) …")
    raw_ds = ds.set_task(fn)
    samples: List[dict] = []
    for s in tqdm(raw_ds, desc=f"Loading {task}", unit="patient"):
        samples.append(s)
        if max_samples and len(samples)>=max_samples:
            break

    # 5) full 模式下才存快取（且無限載入）
    if not dev and max_samples is None:
        pickle.dump(samples, open(cache_path,"wb"))
        print(f"[INFO] Saved full dataset for '{task}' to {cache_path}")

    return samples


def load_mappings():
    """
    Build mapping dicts directly from MIMIC-IV v3.1 reference tables.
    """

    # ── 0) 決定 hosp 目錄 ─────────────────────────────
    # 如果 MIMIC4_ROOT 已經是 …/hosp，就直接用；
    # 否則自動在後面補 "hosp"
    hosp_dir = (
        MIMIC4_ROOT
        if MIMIC4_ROOT.rstrip("/").endswith("hosp")
        else os.path.join(MIMIC4_ROOT, "hosp")
    )

    # ── 1) ICD-診斷名稱 ─────────────────────────────
    diag_csv = os.path.join(hosp_dir, "d_icd_diagnoses.csv")
    cond_map = {
        r.icd_code.strip(): r.long_title.strip().lower()
        for r in pd.read_csv(diag_csv,
                             usecols=["icd_code", "long_title"]).itertuples()
    }

    # ── 2) ICD-處置名稱 ─────────────────────────────
    proc_csv = os.path.join(hosp_dir, "d_icd_procedures.csv")
    proc_map = {
        r.icd_code.strip(): r.long_title.strip().lower()
        for r in pd.read_csv(proc_csv,
                             usecols=["icd_code", "long_title"]).itertuples()
    }

    # ── 3) (可選) 舊 ATC-3 對照 ─────────────────────
    drug_map = {}
    with open(DRUG_MAPPING_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("level") == "3.0":
                drug_map[row["code"]] = row["name"].lower()

    # ── 4) prescriptions.csv → NDC→藥名 ─────────────
    ndc_csv = os.path.join(hosp_dir, "prescriptions.csv")
    ndc_map = build_ndc_map(ndc_csv)

    return cond_map, proc_map, drug_map, ndc_map
    # ─────────────────────── 正規式 ─────────────────────── #
ICD10_RE = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
ICD9_RE  = re.compile(r"^(V|E)?\d{3,4}(\.\d+)?$")
NDC_RE   = re.compile(r"^\d{7,11}$")        # 7~11 位純數字

# ─────────────────────── 快取裝飾器 ─────────────────── #
@functools.lru_cache(maxsize=50_000)
def _cached_get_description(code:str)->Optional[str]:
    try:
        return get_description(code)
    except ValueError:
        return None

# ─────────────────────── 基本清洗 ───────────────────── #
def clean_code(raw: str) -> Optional[str]:
    if raw is None:
        return None
    # 只做 strip，不過濾任何 ".0" 或 "0" 等
    return str(raw).strip()

# ─────────────────────── 讀 NDC→名稱 ────────────────── #
# ───── 讀 prescriptions.csv 建 NDC → 藥名 對照表 ─────
def build_ndc_map(pres_csv: str) -> Dict[str, str]:
    """
    讀取 prescriptions.csv，以 ndc 欄為 key，
    取 drug / drug_name_generic / drug_name_poe 任一欄作為 value。
    回傳 dict 並統一轉小寫；ndc 會補齊 11 位數字。
    """
    use_cols = ["ndc"]
    # 先偵測可用的藥名欄
    with open(pres_csv, newline='') as f:
        header = f.readline().rstrip('\n').split(',')
        for c in ("drug", "drug_name_generic", "drug_name_poe"):
            if c in header:
                use_cols.append(c)
                name_col = c
                break
        else:
            raise ValueError("prescriptions.csv 找不到藥名欄位")

    df = pd.read_csv(pres_csv, usecols=use_cols, low_memory=False)
    df = df.dropna(subset=["ndc"])
    df["ndc"] = (
        df["ndc"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)  # 去掉尾端 .0
        .str.zfill(11)                         # 補成 11 位
    )
    return dict(df.groupby("ndc")[name_col].first().str.lower())

# ─────────────────────── 主對映函式 ─────────────────── #
def code_desc(code:str,
              csv_map:Dict[str,str],
              ndc_map:Dict[str,str])->Optional[str]:
    # 1) 本地 CSV / NDC_map
    if code in csv_map:     return csv_map[code].title()
    if code in ndc_map:     return ndc_map[code].title()

    # 2) ICD-10
    if ICD10_RE.match(code):
        if (d:=_cached_get_description(code)): return d.title()

    # 3) ICD-9
    if ICD9_RE.match(code):
        # 3-A 無點 → 補點後也試
        queries={code}
        if '.' not in code and len(code)>=3:
            dotted=code[:3]+'.'+code[3:]
            queries.add(dotted)
        for q in queries:
            res=icd9_search(q.replace('.',''))
            if res and (d:=getattr(res,'description',None) or getattr(res,'short_desc',None)):
                return d.title()
        # 3-B GEMs 轉 ICD-10
        try:
            icd10=mapper.map(list(queries)[0],source='icd9',target='icd10')
            if icd10 and (d:=_cached_get_description(icd10)):
                return d.title()
        except Exception:
            pass

    # 4) NDC（純數字）
    if NDC_RE.match(code):
        if code in ndc_map: return ndc_map[code].title()
        # 去掉前導零後再查一次
        nz=code.lstrip('0')
        if nz in ndc_map:   return ndc_map[nz].title()

    # 5) 無結果
    return None

# ─────────────────────── 样本增豐 ───────────────────── #
# ─────────────────────── 样本增豐 ───────────────────── #
def enrich_sample_with_names(sample: dict,
                             cond_map: Dict[str, str],
                             proc_map: Dict[str, str],
                             drug_map: Dict[str, str],
                             ndc_map: Dict[str, str] = None) -> dict:
    ndc_map = ndc_map or {}

    def _conv(nested, m):
        if not nested:
            return []
        if nested and not isinstance(nested[0], list):
            nested = [nested]
        out = []
        for row in nested:
            new = []
            for c in row:
                code = clean_code(c)
                if code is None:
                    continue
                # 先嘗試用 mapping，找不到就 fallback 用 code 當 name
                desc = code_desc(code, m, ndc_map)
                if desc:
                    new.append({"code": code, "name": desc})
                else:
                    new.append({"code": code, "name": code})
            if new:
                out.append(new)
        return out

    sample["conditions"] = _conv(sample.get("conditions", []), cond_map)
    sample["procedures"] = _conv(sample.get("procedures", []), proc_map)
    sample["drugs"]      = _conv(sample.get("drugs", []), {**drug_map, **ndc_map})
    return sample

