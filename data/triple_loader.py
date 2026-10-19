# data/triple_loader.py  🔄  最終版
from pathlib import Path
import itertools, json, re

# ───────── 病人 ID 正規化 ─────────
def norm_patient(s: str) -> str:
    """
    把任何形式 (e.g. "Patient:1003", "patient_1003", "1003") 的病人字串
    統一抽出「純數字 ID」。若抓不到數字，原字串返回 (避免誤刪)。
    """
    m = re.search(r"\d{5,}", s)      # 至少 5 位：避免把 A41 等 ICD 誤抓
    return m.group(0) if m else s

# ───────── 讀檔 ＋ 正規化 ＋ 去重 / 去自迴圈 ─────────
def load_triples(root="triples_output"):
    files = Path(root).rglob("triples_*.json")
    raw_iter = itertools.chain.from_iterable(json.load(open(f)) for f in files)
    triples = []
    loop = 0

    for t in raw_iter:
        h, r, v = t["head"], t["relation"].upper(), t["tail"]

        if t.get("head_type","").lower() == "patient":
            h = norm_patient(h)
        if t.get("tail_type","").lower() == "patient":
            v = norm_patient(v)

        if h == v:
            loop += 1
            continue

        # ✅ 保留完整三元組，包括 visit_date/timestamp 等
        cleaned = dict(t)
        cleaned["head"] = h
        cleaned["relation"] = r
        cleaned["tail"] = v
        cleaned.setdefault("weight", 1.0)
        triples.append(cleaned)

    print(f"✓ load_triples ▶ {len(triples):,} 條 (去掉自迴圈 {loop:,} 條)")
    return triples

# ───────── 建立映射 (不變) ─────────
def build_maps(triples):
    ents, rels = set(), set()
    for t in triples:
        ents.update([t["head"], t["tail"]])
        rels.add(t["relation"])
    entity2id   = {e:i for i, e in enumerate(sorted(ents))}
    relation2id = {r:i for i, r in enumerate(sorted(rels))}
    return entity2id, relation2id

