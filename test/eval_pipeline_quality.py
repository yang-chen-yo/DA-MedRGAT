#!/usr/bin/env python3
"""
在「實際產出的病人知識圖」上做量化品質檢驗（不需人工標註）。

三個自動指標：
  1. entity-swap 負樣本   : 把 accepted triple 的 tail 換成同型別的別的實體，
                            用同一篇 abstract 重驗 → verifier 應該不再 ACCEPT。
                            回報 false-accept rate（越低越好）。
  2. evidence-swap 負樣本 : 用「別條 triple 的 abstract」重驗 accepted triple →
                            應該不再 ACCEPT。回報 false-accept rate。
  3. LLM-judge 一致性      : 抽樣 accepted triple，用 Gemini 當 silver 裁判判斷
                            supporting sentence 是否支持 claim，與 verifier 的
                            ACCEPT 決策比對 → Cohen's κ + 一致率。

用法：
  python test/eval_pipeline_quality.py --json triples_output/10000032/single_patient_demo_22595853.json
  python test/eval_pipeline_quality.py --json <file> --judge 30      # 用 30 條做 LLM-judge
  python test/eval_pipeline_quality.py --json <file> --no-judge
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusion.triple_verifier import get_verifier, TripleEvidenceVerifier

random.seed(13)


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("triples", d) if isinstance(d, dict) else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--judge", type=int, default=20, help="LLM-judge 抽樣數（0 = 關閉）")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    triples = load(args.json)
    rag = [t for t in triples if str(t.get("source", "")).upper() == "RAG"
           and t.get("evidence_text")]
    print(f"[eval] {len(triples)} triples, {len(rag)} literature (RAG) triples with evidence\n")
    if not rag:
        print("沒有帶證據的文獻 triple 可評。先跑一次 run_single_patient_fast.py。")
        return

    v = get_verifier() or TripleEvidenceVerifier()
    v._ensure_loaded()

    # tail 依型別分組（做 entity-swap）
    tails_by_type = defaultdict(set)
    for t in rag:
        tails_by_type[t.get("tail_type", "Concept")].add(t.get("tail", ""))

    # ---- 1) entity-swap ----
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

    # ---- 2) evidence-swap ----
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

    print("=== 1) entity-swap 負樣本（tail 換成同型別別的實體） ===")
    print(f"    tested={swap_tested}  false-accept={swap_false_accept}  "
          f"rate={swap_false_accept / max(swap_tested,1):.3f}   (越低越好)")
    print("=== 2) evidence-swap 負樣本（換成別條 triple 的 abstract） ===")
    print(f"    tested={ev_tested}  false-accept={ev_false_accept}  "
          f"rate={ev_false_accept / max(ev_tested,1):.3f}   (越低越好)")

    # ---- 3) LLM-judge 一致性 ----
    if args.no_judge or args.judge <= 0:
        return
    try:
        from fusion.triple_generation_hf import llm_judge_support
        from sklearn.metrics import cohen_kappa_score
    except Exception as e:  # noqa: BLE001
        print(f"\n[judge] 略過（{e}）")
        return

    sample = random.sample(rag, min(args.judge, len(rag)))
    verifier_yes, judge_yes = [], []
    for t in sample:
        claim = t.get("claim_text") or f"{t.get('head')} {t.get('relation','').lower()} {t.get('tail')}"
        sent = t.get("supporting_sentence") or t.get("evidence_title") or ""
        j = llm_judge_support(claim, sent)
        if j is None:
            continue
        verifier_yes.append(1)   # 這些都是 accepted（進了主檔）
        judge_yes.append(1 if j else 0)

    if len(judge_yes) >= 2:
        agree = sum(a == b for a, b in zip(verifier_yes, judge_yes)) / len(judge_yes)
        # verifier 全 1 → κ 需要有變異；用 judge 對「accepted 是否真的 support」的比例
        precision_vs_judge = sum(judge_yes) / len(judge_yes)
        print("\n=== 3) LLM-judge（Gemini 當 silver 裁判） ===")
        print(f"    n={len(judge_yes)}  verifier ACCEPT 中，judge 也說 support 的比例 "
              f"= {precision_vs_judge:.3f}")
        print(f"    （這就是 verifier 的 silver-standard precision；越接近 1 越好）")
        try:
            k = cohen_kappa_score([1] * len(judge_yes), judge_yes)
            print(f"    Cohen's κ (verifier vs judge) = {k:.3f}  "
                  f"（verifier 無變異時 κ 會偏低，precision 那行才是重點）")
        except Exception:
            pass
    else:
        print("\n[judge] 有效樣本不足")


if __name__ == "__main__":
    main()
