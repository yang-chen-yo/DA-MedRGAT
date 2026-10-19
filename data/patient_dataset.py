#!/usr/bin/env python3
# coding: utf-8
"""
data/patient_dataset.py
-----------------------
將所有三元組依 patient_id 分群 → 每位病人一張子圖 (torch_geometric.data.Data)。

濾除：
  • 該病人缺少目標任務標籤 (y = -1)
  • 子圖無任何節點 (極少見)

回傳：
    dataset : List[Data]  ─ 每筆含
        • x             : [num_nodes] 全域實體 ID (拿去做嵌入)
        • edge_index    : [2,E]
        • edge_type     : [E]
        • (edge_weight) : [E,1]（有開 use_time 時）
        • y             : [1] 或 [k] label
        • patient_id    : str (metadata)
    ent2id  : Dict[str,int]
    rel2id  : Dict[str,int]
    out_dim : int  (binary=1 / multilabel=k / multiclass= n_class)
"""

from pathlib import Path
import itertools, json, re
import torch
from torch_geometric.data import Data
from data.triple_loader import norm_patient
from data.label_loader import load_labels
from data.time_utils import visit_recency_weights


# ────────────────────────────────────────────────
def build_patient_dataset(json_dir: str,
                          pkl_path: str,
                          task: str,
                          use_time: bool = True,
                          use_literature: bool = True):
    # ------------ 讀取所有 JSON → 依病人分群 ------------ #
    # triples_output/<patient_id>/triples_*.json：同一資料夾的三元組都屬於該病人，
    # 包含兩端都不是 Patient 的文獻擴張邊（例如 診斷 → 併發症）
    # use_literature=False 時去掉文獻（RAG）邊，只留 EHR / UMLS 的邊（消融用）
    groups = {}
    for fp in Path(json_dir).rglob("triples_*.json"):
        pid = norm_patient(fp.parent.name)
        triples = json.load(open(fp))
        if not use_literature:
            triples = [t for t in triples if str(t.get("source", "")).upper() != "RAG"]
        groups.setdefault(pid, []).extend(triples)
    for pid, triples in groups.items():
        seen, uniq = set(), []
        for t in triples:
            key = (t["head"], t["relation"], t["tail"], t["visit_id"])
            if key not in seen:
                seen.add(key); uniq.append(t)
        groups[pid] = uniq

    # ------------ 建立全域實體 / 關係映射 ------------ #
    all_tr = list(itertools.chain.from_iterable(groups.values()))
    ents = {t["head"] for t in all_tr} | {t["tail"] for t in all_tr}
    rels = {t["relation"] for t in all_tr}
    ent2id = {e: i for i, e in enumerate(sorted(ents))}
    rel2id = {r: i for i, r in enumerate(sorted(rels))}

    # 病人節點共用同一個 embedding：若每位病人有自己的 embedding，模型會直接
    # 記住「病人 → 標籤」，而測試病人的 embedding 從未被訓練過
    patients = {t["head"] for t in all_tr if t.get("head_type") == "Patient"} | \
               {t["tail"] for t in all_tr if t.get("tail_type") == "Patient"}
    PATIENT = "<PATIENT>"
    ent2id[PATIENT] = len(ent2id)
    shared = {ent2id[p]: ent2id[PATIENT] for p in patients}

    # 只出現在一位病人圖中的實體同理，依實體類型共用一個 embedding。
    # 只用輸入圖統計，沒有用到標籤
    etype, n_pat = {}, {}
    for pid, triples in groups.items():
        for e in {t["head"] for t in triples} | {t["tail"] for t in triples}:
            n_pat[e] = n_pat.get(e, 0) + 1
        for t in triples:
            etype[t["head"]] = t.get("head_type", "")
            etype[t["tail"]] = t.get("tail_type", "")
    for e, n in n_pat.items():
        if n == 1 and e not in patients:
            rare = f"<RARE:{etype[e]}>"
            if rare not in ent2id:
                ent2id[rare] = len(ent2id)
            shared[ent2id[e]] = ent2id[rare]

    # ------------ 讀取全域標籤 → y_all ------------ #
    y_all, label_date = load_labels(pkl_path, ent2id, task)
    out_dim = y_all.shape[1] if y_all.dim() == 2 else 1

    dataset = []
    skip_no_label = skip_neg1 = 0

    # ------------ 為每位病人建子圖 ------------ #
    for pid, triples in groups.items():
        if pid not in ent2id:           # 不應發生
            continue

        y_vec = y_all[ent2id[pid]]
        if (y_vec < 0).any():           # -1 → skip
            skip_neg1 += 1; continue

        # 只保留標籤所屬就診（含）以前的邊，避免把之後就診的資訊放進輸入；
        # label 日期可能只有 YYYY-MM-DD，所以只比到同樣長度，當天的就診仍保留
        cut = label_date[pid]
        triples = [t for t in triples if t["visit_date"][:len(cut)] <= cut]

        # 收集此病人涉及的全域節點
        node_set = {ent2id[t["head"]] for t in triples} | \
                   {ent2id[t["tail"]] for t in triples}
        if len(node_set) == 0:
            skip_no_label += 1; continue

        node_list = sorted(node_set)
        gid2lid = {gid: i for i, gid in enumerate(node_list)}

        # 邊
        src, dst, etype = [], [], []
        for t in triples:
            h = ent2id[t["head"]]; d = ent2id[t["tail"]]
            src.append(gid2lid[h]); dst.append(gid2lid[d])
            etype.append(rel2id[t["relation"]])
        if use_time:
            ew = [[w] for w in visit_recency_weights(triples)]

        data = Data(
            x=torch.tensor([shared.get(g, g) for g in node_list], dtype=torch.long),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_type=torch.tensor(etype, dtype=torch.long),
            y=y_vec.unsqueeze(0) if y_vec.dim()==0 else y_vec.unsqueeze(0),
            patient_id=pid
        )
        if use_time:
            data.edge_weight = torch.tensor(ew, dtype=torch.float)

        dataset.append(data)

    print(f"◎ 病人子圖總數 = {len(dataset)} (跳過 -1 標籤 {skip_neg1}, 空圖 {skip_no_label})")
    return dataset, ent2id, rel2id, out_dim

