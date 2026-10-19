#!/usr/bin/env python3
"""
Triple evidence verifier 的量化評估。

輸入 JSONL，每行一個標註樣本：
  {"head", "relation", "tail", "evidence_title", "evidence_text", "gold"}
  gold ∈ {"support", "not"}

用法：
  python test/eval_verifier.py --data test/fixtures/verifier_eval_sample.jsonl
  python test/eval_verifier.py --data <你的標註檔> --sweep

輸出：混淆矩陣、precision / recall / F1（以「support」為正類），
      以及 entailment 機率的門檻掃描，供你決定 VERIFIER_ACCEPT_THRESHOLD。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.triple_verifier import TripleEvidenceVerifier


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None, help="覆寫 VERIFIER_MODEL")
    ap.add_argument("--sweep", action="store_true", help="門檻掃描")
    args = ap.parse_args()

    rows = load_rows(args.data)
    print(f"[eval] {len(rows)} labelled examples from {args.data}")

    v = TripleEvidenceVerifier(model_name=args.model) if args.model else TripleEvidenceVerifier()
    v._ensure_loaded()
    print(f"[eval] model = {v._loaded_name}\n")

    scored = []
    for row in rows:
        res = v.verify(
            {"head": row["head"], "relation": row["relation"], "tail": row["tail"]},
            evidence_text=row.get("evidence_text"),
            evidence_title=row.get("evidence_title"),
        )
        gold_pos = row["gold"].lower() == "support"
        scored.append((gold_pos, res.score, res.decision, row, res))

    # --- 用目前 config 門檻的三分類結果 ---
    print("=== decision @ current thresholds "
          f"(accept={v.accept_threshold}, review={v.review_threshold}) ===")
    tp = fp = fn = tn = review = 0
    for gold_pos, score, decision, row, res in scored:
        pred_pos = decision == "ACCEPTED"
        mark = "ok " if (pred_pos == gold_pos and decision != "REVIEW") else "XX "
        if decision == "REVIEW":
            review += 1
            mark = "?? "
        elif pred_pos and gold_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif not pred_pos and gold_pos:
            fn += 1
        else:
            tn += 1
        print(f" {mark} gold={row['gold']:8s} p={score:.3f} {decision:9s} "
              f"| {res.claim}")
        print(f"      supp: {res.supporting_sentence}")

    p, r, f1 = prf(tp, fp, fn)
    print(f"\n TP={tp} FP={fp} FN={fn} TN={tn} REVIEW={review}")
    print(f" precision={p:.3f} recall={r:.3f} f1={f1:.3f}  (正類 = support, REVIEW 不計入)")

    if args.sweep:
        print("\n=== threshold sweep (二分類: p>=thr → support) ===")
        print(f" {'thr':>5} {'P':>6} {'R':>6} {'F1':>6}")
        thr = 0.10
        while thr <= 0.951:
            tp = fp = fn = 0
            for gold_pos, score, *_ in scored:
                pred_pos = score >= thr
                if pred_pos and gold_pos:
                    tp += 1
                elif pred_pos and not gold_pos:
                    fp += 1
                elif not pred_pos and gold_pos:
                    fn += 1
            p, r, f1 = prf(tp, fp, fn)
            print(f" {thr:5.2f} {p:6.3f} {r:6.3f} {f1:6.3f}")
            thr += 0.05


if __name__ == "__main__":
    main()
