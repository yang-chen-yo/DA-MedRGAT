#!/usr/bin/env python3
"""
把 triples_output/*/triples_*.json 裡所有病人的 accepted triple 合併成一個
list，做四種 negative control（跨全部病人，樣本數才有意義）。全部都不需要
人工/專家標註 —— 「錯」是造出來的時候就定義好的：

  1. entity-swap   : tail 換成同型別的別的實體，用「同一篇」證據重驗。
  2. evidence-swap : triple 不變，換成「別條 triple」的證據重驗。
  3. relation-swap : head/tail 不變，relation 換成別的關係類型，用原證據重驗。
  4. direction-flip: head/tail 對調（只對非對稱關係做），用原證據重驗。

四種都是「保證錯」的假 triple（by construction），所以正確答案永遠是
REJECTED/REVIEW，verifier 說 ACCEPTED 就是 false-accept。false-accept rate
越低代表 verifier 對亂配的假關係越有抵抗力。

用法：
  python test/aggregate_and_eval_negcontrol.py
  python test/aggregate_and_eval_negcontrol.py --glob 'triples_output/*/triples_*.json'
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fusion.triple_verifier import get_verifier, TripleEvidenceVerifier

random.seed(13)

# 跟 fusion/triple_generation_hf.py::DIRECTED_RELATIONS 同一套封閉詞彙。
# SYMMETRIC 的關係頭尾對調後語意可能還是對的，direction-flip 不測這幾種。
ALL_RELATIONS = [
    "HAS_DISEASE", "HAS_SYMPTOM", "HAS_COMPLICATION", "HAS_RISK_FACTOR",
    "HAS_LAB_ABNORMALITY", "TREATED_WITH", "TREATS", "CAUSES_ADVERSE_EFFECT",
    "CONTRAINDICATED_IN", "INTERACTS_WITH", "ASSOCIATED_WITH",
]
SYMMETRIC_RELATIONS = {"ASSOCIATED_WITH", "INTERACTS_WITH"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="triples_output/*/triples_*.json")
    ap.add_argument("--max", type=int, default=0, help="最多取幾條 RAG triple（0=不限）")
    args = ap.parse_args()

    rag = []
    for fp in glob.glob(args.glob):
        try:
            d = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        for t in d:
            if str(t.get("source", "")).upper() == "RAG" and t.get("evidence_text"):
                rag.append(t)
    if args.max:
        random.shuffle(rag)
        rag = rag[: args.max]
    print(f"[agg] {len(rag)} literature (RAG) triples with evidence, from glob={args.glob!r}")
    if not rag:
        return

    v = get_verifier() or TripleEvidenceVerifier()
    v._ensure_loaded()

    tails_by_type = defaultdict(set)
    for t in rag:
        tails_by_type[t.get("tail_type", "Concept")].add(t.get("tail", ""))

    swap_tested = swap_false_accept = 0
    for t in rag:
        pool = list(tails_by_type[t.get("tail_type", "Concept")] - {t.get("tail", "")})
        if not pool:
            continue
        fake = dict(t)
        fake["tail"] = random.choice(pool)
        r = v.verify(fake, evidence_text=t["evidence_text"], evidence_title=t.get("evidence_title"))
        swap_tested += 1
        if r.decision == "ACCEPTED":
            swap_false_accept += 1

    ev_tested = ev_false_accept = 0
    for t in rag:
        others = [o for o in rag if o.get("evidence_id") != t.get("evidence_id")]
        if not others:
            continue
        o = random.choice(others)
        r = v.verify(t, evidence_text=o["evidence_text"], evidence_title=o.get("evidence_title"))
        ev_tested += 1
        if r.decision == "ACCEPTED":
            ev_false_accept += 1

    # ---- 3) relation-swap：head/tail 不變，relation 換成別的關係 ----
    rel_tested = rel_false_accept = 0
    for t in rag:
        orig_rel = str(t.get("relation", "")).upper()
        pool = [r for r in ALL_RELATIONS if r != orig_rel]
        if not pool:
            continue
        fake = dict(t)
        fake["relation"] = random.choice(pool)
        r = v.verify(fake, evidence_text=t["evidence_text"], evidence_title=t.get("evidence_title"))
        rel_tested += 1
        if r.decision == "ACCEPTED":
            rel_false_accept += 1

    # ---- 4) direction-flip：head/tail 對調（只測非對稱關係）----
    dir_tested = dir_false_accept = 0
    for t in rag:
        orig_rel = str(t.get("relation", "")).upper()
        if orig_rel in SYMMETRIC_RELATIONS:
            continue
        head, tail = t.get("head"), t.get("tail")
        if not head or not tail or head == tail:
            continue
        fake = dict(t)
        fake["head"], fake["tail"] = tail, head
        r = v.verify(fake, evidence_text=t["evidence_text"], evidence_title=t.get("evidence_title"))
        dir_tested += 1
        if r.decision == "ACCEPTED":
            dir_false_accept += 1

    print("=== entity-swap ===")
    print(f"    tested={swap_tested}  false-accept={swap_false_accept}  "
          f"rate={swap_false_accept / max(swap_tested,1):.3f}")
    print("=== evidence-swap ===")
    print(f"    tested={ev_tested}  false-accept={ev_false_accept}  "
          f"rate={ev_false_accept / max(ev_tested,1):.3f}")
    print("=== relation-swap（同一 head/tail，relation 亂配） ===")
    print(f"    tested={rel_tested}  false-accept={rel_false_accept}  "
          f"rate={rel_false_accept / max(rel_tested,1):.3f}")
    print("=== direction-flip（head/tail 對調） ===")
    print(f"    tested={dir_tested}  false-accept={dir_false_accept}  "
          f"rate={dir_false_accept / max(dir_tested,1):.3f}")


if __name__ == "__main__":
    main()
