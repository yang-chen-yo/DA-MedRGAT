#!/usr/bin/env python3
"""
量化評估 verifier —— 用 SciFact（Wadden et al., 2020）這個標準的
「科學 claim × abstract 證據」資料集，完全不需人工標註。

SciFact 每個 claim 對每篇引用 abstract 有 gold label：
    SUPPORT / CONTRADICT / NOINFO(NEI)

我們把 verifier 的 decision 對映：
    ACCEPTED      -> SUPPORT
    CONTRADICTED  -> CONTRADICT
    REJECTED/REVIEW -> NOINFO

輸出：3x3 混淆矩陣、各類別 P/R/F1、macro-F1、accuracy。
把這張表放進 revision 的「5.1 quantitative triple-quality validation」。

用法：
    python test/eval_verifier_scifact.py                 # 跑 dev split
    python test/eval_verifier_scifact.py --split train --limit 300
    python test/eval_verifier_scifact.py --accept 0.6 --contra 0.55   # 掃閾值
"""
import argparse
import io
import json
import os
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.triple_verifier import TripleEvidenceVerifier

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "scifact")
LABELS = ["SUPPORT", "CONTRADICT", "NOINFO"]


def _ensure_data():
    need = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"]
    if all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in need):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_tar = os.path.join(CACHE_DIR, "data.tar.gz")
    if os.path.exists(local_tar):
        raw = open(local_tar, "rb").read()
    else:
        print(f"[scifact] downloading {SCIFACT_URL} ...\n"
              f"          （慢的話可先手動： curl -o {local_tar} {SCIFACT_URL}）")
        raw = urllib.request.urlopen(SCIFACT_URL, timeout=600).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for m in tar.getmembers():
            base = os.path.basename(m.name)
            if base in need:
                with tar.extractfile(m) as fh, open(os.path.join(CACHE_DIR, base), "wb") as out:
                    out.write(fh.read())
    print(f"[scifact] cached to {CACHE_DIR}")


def _load_jsonl(name):
    with open(os.path.join(CACHE_DIR, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_pairs(split, limit):
    corpus = {d["doc_id"]: d for d in _load_jsonl("corpus.jsonl")}
    claims = _load_jsonl(f"claims_{split}.jsonl")
    pairs = []
    for c in claims:
        cited = c.get("cited_doc_ids") or list((c.get("evidence") or {}).keys())
        for did in cited:
            did_i = int(did)
            if did_i not in corpus:
                continue
            ev = (c.get("evidence") or {}).get(str(did)) or (c.get("evidence") or {}).get(did_i)
            if ev:
                labs = {e["label"].upper() for e in ev}
                gold = "CONTRADICT" if "CONTRADICT" in labs else "SUPPORT"
            else:
                gold = "NOINFO"
            doc = corpus[did_i]
            pairs.append({
                "claim": c["claim"],
                "title": doc.get("title", ""),
                "abstract": " ".join(doc.get("abstract", [])),
                "gold": gold,
            })
    if limit:
        pairs = pairs[:limit]
    return pairs


def prf(cm):
    out = {}
    for i, lab in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(3)) - tp
        fn = sum(cm[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[lab] = (p, r, f, sum(cm[i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "train", "all"],
                    help="all = train + dev 一起評估（同一個固定模型、兩批不重疊資料，"
                         "混淆矩陣可直接相加；用來得到單一 n≈1080 的結果）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--accept", type=float, default=None)
    ap.add_argument("--review", type=float, default=None)
    ap.add_argument("--contra", type=float, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    _ensure_data()
    if args.split == "all":
        seen, pairs = set(), []
        for sp in ("train", "dev"):
            for p in build_pairs(sp, 0):
                key = (p["claim"], p["title"], p["abstract"], p["gold"])
                if key not in seen:
                    seen.add(key)
                    pairs.append(p)
        if args.limit:
            pairs = pairs[: args.limit]
    else:
        pairs = build_pairs(args.split, args.limit)
    print(f"[scifact] {len(pairs)} (claim, abstract) pairs  "
          f"gold dist = {dict(Counter(p['gold'] for p in pairs))}")

    kw = {}
    if args.accept is not None: kw["accept_threshold"] = args.accept
    if args.review is not None: kw["review_threshold"] = args.review
    if args.contra is not None: kw["contra_threshold"] = args.contra
    v = TripleEvidenceVerifier(model_name=args.model, **kw) if args.model else TripleEvidenceVerifier(**kw)
    v._ensure_loaded()
    print(f"[scifact] verifier = {v._loaded_name}  "
          f"(accept={v.accept_threshold} review={v.review_threshold} contra={v.contra_threshold})\n")

    dec2gold = {"ACCEPTED": "SUPPORT", "CONTRADICTED": "CONTRADICT",
                "REJECTED": "NOINFO", "REVIEW": "NOINFO"}
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = [[0, 0, 0] for _ in range(3)]
    for k, p in enumerate(pairs):
        res = v.verify_claim(p["claim"], p["abstract"], p["title"])
        pred = dec2gold[res.decision]
        cm[idx[p["gold"]]][idx[pred]] += 1
        if (k + 1) % 50 == 0:
            print(f"  ... {k + 1}/{len(pairs)}")

    print("\nconfusion matrix (rows = gold, cols = pred):")
    print(f"{'':>12}" + "".join(f"{l:>12}" for l in LABELS))
    for i, lab in enumerate(LABELS):
        print(f"{lab:>12}" + "".join(f"{cm[i][j]:>12}" for j in range(3)))

    scores = prf(cm)
    total = sum(sum(r) for r in cm)
    acc = sum(cm[i][i] for i in range(3)) / total if total else 0.0
    macro_f1 = sum(scores[l][2] for l in LABELS) / 3
    print(f"\n{'label':>12}{'P':>8}{'R':>8}{'F1':>8}{'n':>8}")
    for l in LABELS:
        p, r, f, n = scores[l]
        print(f"{l:>12}{p:>8.3f}{r:>8.3f}{f:>8.3f}{n:>8}")
    print(f"\naccuracy = {acc:.3f}   macro-F1 = {macro_f1:.3f}")


if __name__ == "__main__":
    main()
