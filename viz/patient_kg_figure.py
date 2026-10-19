#!/usr/bin/env python3
"""
把單一病人的知識圖三元組畫成「投稿等級」的靜態圖（PDF + PNG），另存 GraphML 供 Gephi。

版面：
  - 病人節點置中
  - 內圈：EHR 記載的診斷 / 藥物 / 處置（依型別分弧段），實線灰邊 = 結構性事實
  - 外圈：文獻推導出的概念（symptom / adverse effect / complication / ...），
          邊的顏色代表關係、粗細代表 verifier entailment 分數
  - REVIEW 邊用虛線；REJECTED / UNANCHORED 預設不畫

用法：
  python viz/patient_kg_figure.py --json triples_output/10000032/single_patient_demo_22595853.json
  python viz/patient_kg_figure.py --json <file> --out figures/patient_10000032 --show-review
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Circle
import networkx as nx

# ---- 視覺設定 -------------------------------------------------------
NODE_STYLE = {
    "Patient":          dict(color="#1d4e89", text="#ffffff", r=0.055),
    "Disease":          dict(color="#c1443c", text="#ffffff", r=0.044),
    "Drug":             dict(color="#2f7d4f", text="#ffffff", r=0.044),
    "Procedure":        dict(color="#7a5195", text="#ffffff", r=0.044),
    "Symptom":          dict(color="#e8a13a", text="#20160a", r=0.038),
    "AdverseEffect":    dict(color="#8c510a", text="#ffffff", r=0.038),
    "LaboratoryResult": dict(color="#4aa3c4", text="#ffffff", r=0.038),
    "RiskFactor":       dict(color="#c987b8", text="#20160a", r=0.038),
    "Concept":          dict(color="#8b8b8b", text="#ffffff", r=0.036),
}
_HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
EHR_TYPES = ("Disease", "Drug", "Procedure")
# 對稱關係（無方向）；畫成沒有箭頭的線
SYMMETRIC_RELATIONS = {"ASSOCIATED_WITH", "INTERACTS_WITH"}
# 互為反向的關係：正規化成同一條邊，方向統一
INVERSE_RELATIONS = {"TREATS": ("TREATED_WITH", True), "TREATED_WITH": ("TREATED_WITH", False)}
REL_COLOR = {
    "HAS_DISEASE": "#c1443c", "RECEIVED_PROCEDURE": "#7a5195", "USED_DRUG": "#2f7d4f",
    "HAS_SYMPTOM": "#e8a13a", "CAUSES_ADVERSE_EFFECT": "#8c510a",
    "HAS_COMPLICATION": "#d1495b", "HAS_RISK_FACTOR": "#7b4fa3",
    "HAS_LAB_ABNORMALITY": "#4aa3c4", "TREATED_WITH": "#2f7d4f", "TREATS": "#2f7d4f",
    "CONTRAINDICATED_IN": "#7d3c98", "INTERACTS_WITH": "#7d3c98",
    "ASSOCIATED_WITH": "#8b8b8b",
}


def _norm_type(raw: str) -> str:
    v = (raw or "").strip().lower()
    m = {
        "patient": "Patient", "person": "Patient",
        "disease": "Disease", "diagnosis": "Disease", "condition": "Disease",
        "drug": "Drug", "medication": "Drug", "treatment": "Drug",
        "procedure": "Procedure",
        "symptom": "Symptom", "sign": "Symptom",
        "adverseeffect": "AdverseEffect", "adverse_effect": "AdverseEffect",
        "sideeffect": "AdverseEffect",
        "laboratoryresult": "LaboratoryResult", "labresult": "LaboratoryResult",
        "lab": "LaboratoryResult",
        "riskfactor": "RiskFactor", "risk_factor": "RiskFactor",
    }
    return m.get(v, "Concept")


def load_triples(path: str) -> List[Dict]:
    payload = json.load(open(path, encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("triples", [])
    return payload


def load_triples_with_buckets(path: str) -> List[Dict]:
    """把 rejected / review / contradicted / unanchored 的姊妹檔一起讀進來，
       這樣才畫得出「verifier 有在篩」的決策光譜。"""
    triples = list(load_triples(path))
    base = os.path.basename(path)
    # single_patient_demo_<visit>.json  →  <visit>
    for pref in ("single_patient_demo_", "triples_"):
        if base.startswith(pref):
            stamp = base[len(pref):].rsplit(".", 1)[0]
            break
    else:
        return triples
    d = os.path.dirname(path)
    for bucket in ("review", "rejected", "contradicted", "unanchored"):
        fp = os.path.join(d, f"{bucket}_triples_{stamp}.json")
        if os.path.exists(fp):
            triples.extend(load_triples(fp))
    return triples


def wrap(label: str, width: int = 16) -> str:
    words, lines, cur = str(label).split(), [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(l for l in lines[:3] if l.strip()) or str(label).strip()


def build_graph(triples: List[Dict], show_review: bool, show_rejected: bool):
    G = nx.MultiDiGraph()
    patient_id = None
    _edge_bucket: Dict[tuple, list] = {}

    for t in triples:
        status = str(t.get("verifier_decision") or t.get("support_status") or "ACCEPTED").upper()
        src = str(t.get("source") or "").upper()
        if src != "EHR":
            if status == "REJECTED" and not show_rejected:
                continue
            if status in ("REVIEW", "UNANCHORED") and not show_review:
                continue

        h, ta = str(t.get("head") or "").strip(), str(t.get("tail") or "").strip()
        if not h or not ta:
            continue
        ht, tt = _norm_type(t.get("head_type")), _norm_type(t.get("tail_type"))
        if ht == "Patient":
            patient_id = h
        if tt == "Patient":
            patient_id = ta

        for n, nt in ((h, ht), (ta, tt)):
            if not G.has_node(n):
                G.add_node(n, ntype=nt)
            elif nt != "Concept":
                G.nodes[n]["ntype"] = nt

        score = t.get("verifier_score")
        if score is None:
            score = t.get("support_score") or t.get("weight") or 0.0
        # 檢索相關度（來源文獻與該概念的 cosine）；entailment 幾乎都 ~1，沒鑑別度，
        # 所以線寬改用這個當「證據強度」的視覺代理，entailment 仍是納入的門檻。
        relevance = t.get("evidence_score")
        if relevance is None:
            relevance = t.get("weight")

        relation = str(t.get("relation") or "ASSOCIATED_WITH").upper()
        directed = relation not in SYMMETRIC_RELATIONS
        # 反向關係正規化：TREATS(a,b) 與 TREATED_WITH(b,a) 視為同一條邊
        if relation in INVERSE_RELATIONS:
            relation, needs_swap = INVERSE_RELATIONS[relation]
            if needs_swap:
                h, ta = ta, h
        # 對稱關係：端點排序，(a,b) 與 (b,a) 收斂成一條
        eh, et = (h, ta)
        if not directed and eh.strip().lower() > et.strip().lower():
            eh, et = et, eh
        _edge_bucket.setdefault((eh, et, relation), []).append(dict(
            source=src, status=status, directed=directed,
            score=float(score or 0.0),
            relevance=float(relevance) if isinstance(relevance, (int, float)) else None,
            pmid=str(t.get("evidence_id") or ""),
            title=str(t.get("evidence_title") or ""),
            claim=str(t.get("claim_text") or ""),
            sentence=str(t.get("supporting_sentence") or t.get("evidence_text") or "")))

    _DECISION_RANK = {"CONTRADICTED": 4, "ACCEPTED": 3, "REVIEW": 2, "UNANCHORED": 1, "REJECTED": 0}
    for (eh, et, relation), recs in _edge_bucket.items():
        best = max(recs, key=lambda r: (r["relevance"] or 0.0))
        pmids = sorted({r["pmid"] for r in recs if r["pmid"]})
        status = max((r["status"] for r in recs), key=lambda s: _DECISION_RANK.get(s, 0))
        G.add_edge(eh, et,
                   relation=relation, source=best["source"], status=status,
                   score=best["score"], relevance=best["relevance"],
                   directed=recs[0]["directed"],
                   pmid=pmids[0] if pmids else "",
                   pmids=pmids, n_support=len(pmids),
                   title=best["title"], claim=best["claim"], sentence=best["sentence"])
    return G, patient_id


def prune_graph(G: nx.MultiDiGraph, patient_id: str,
                focus: str = "", max_nodes: int = 0, focus_hops: int = 2,
                display_hops: int = 0) -> nx.MultiDiGraph:
    """--focus：只留下某個 anchor 的證據子樹（+ 病人到它的路徑）。
       --max-nodes：超過上限時，反覆砍掉『分數最低的葉節點』直到節點數達標。
       --display-hops：overview 只畫距病人 N 跳內的節點（資料仍是多跳）。"""
    und = G.to_undirected()

    if display_hops and not focus and patient_id and patient_id in und:
        dep = nx.single_source_shortest_path_length(und, patient_id)
        G = G.subgraph([n for n in G.nodes
                        if dep.get(n, 99) <= display_hops]).copy()
        und = G.to_undirected()

    if focus:
        target = None
        fl = focus.strip().lower()
        for n in G.nodes:                       # 容錯：子字串比對
            if fl == n.lower() or fl in n.lower():
                target = n
                break
        if target is None:
            raise SystemExit(f"--focus '{focus}' 找不到對應節點；可用節點例如："
                             + ", ".join(list(G.nodes)[:12]))
        # 以「距 focus 的層數」往外 BFS，只沿層數遞增的邊 → 乾淨的樹（不回捲）
        lvl = {target: 0}
        frontier = [target]
        while frontier:
            nxt = []
            for u in frontier:
                if lvl[u] >= focus_hops:
                    continue
                for v in G.successors(u):
                    if v not in lvl and v != patient_id:
                        lvl[v] = lvl[u] + 1
                        nxt.append(v)
            frontier = nxt
        keep = set(lvl)
        if patient_id:
            keep.add(patient_id)
        G = G.subgraph(keep).copy()
        G.graph["focus"] = target

    if max_nodes and G.number_of_nodes() > max_nodes:
        protected = {patient_id}
        while G.number_of_nodes() > max_nodes:
            leaves = [n for n in G.nodes
                      if n not in protected and G.out_degree(n) == 0
                      and not any(d["source"] == "EHR" for *_, d in G.in_edges(n, data=True))]
            if not leaves:
                break
            def _score(n):
                return min((d["score"] for *_, d in G.in_edges(n, data=True)), default=0.0)
            G.remove_node(min(leaves, key=_score))
    return G


def layered_layout(G: nx.MultiDiGraph, patient_id: str) -> Dict[str, tuple]:
    """--focus 專用：由左至右分層（Patient → anchor → 文獻概念一層層），
       同層節點沿垂直方向均分，子節點盡量對齊父節點的高度。"""
    focus = G.graph.get("focus")
    # col = 距 focus 的最短跳數，每個節點只設一次（cycle-safe）
    col = {focus: 1}
    frontier = [focus]
    while frontier:
        nxt = []
        for u in frontier:
            for v in G.successors(u):
                if v == patient_id or v in col:
                    continue
                col[v] = col[u] + 1
                nxt.append(v)
        frontier = nxt
    if patient_id and patient_id in G:
        col[patient_id] = 0
    for n in G.nodes:
        col.setdefault(n, 1)

    by_col = defaultdict(list)
    for n, c in col.items():
        by_col[c].append(n)

    ncol = max(by_col) + 1
    # 節點太多的欄 → 拆成兩個子欄（zig-zag），整張圖才不會又高又窄
    split = {c: (len(v) > 6) for c, v in by_col.items()}
    eff_k = max((len(v) + 1) // 2 if split[c] else len(v) for c, v in by_col.items())
    dy = max(0.62, min(1.0, 3.6 / max(eff_k - 1, 1)))
    dx = 1.6 if ncol >= 5 else 2.2
    pos = {}
    for c, nodes in sorted(by_col.items()):
        def _py(n):
            ps = [pos[p][1] for p in G.predecessors(n) if p in pos]
            return sum(ps) / len(ps) if ps else 0.0
        nodes = sorted(nodes, key=_py)
        k = len(nodes)
        if split.get(c):
            rows = (k + 1) // 2
            for i, n in enumerate(nodes):
                r, sub = i // 2, i % 2
                y = (r - (rows - 1) / 2) * dy
                pos[n] = (c * dx + (sub - 0.5) * 0.7 * dx, y)
        else:
            for i, n in enumerate(nodes):
                y = (i - (k - 1) / 2) * (dy if k > 1 else 0)
                pos[n] = (c * dx, y)
    return pos


def radial_layout(G: nx.MultiDiGraph, patient_id: str) -> Dict[str, tuple]:
    """多跳環狀版面：環 = 距病人的 BFS 跳數；每個節點掛在上一跳的「母節點」附近。"""
    und = G.to_undirected()
    if patient_id and patient_id in und:
        depth = nx.single_source_shortest_path_length(und, patient_id)
    else:
        depth = {n: 1 for n in G.nodes}
        patient_id = None
    for n in G.nodes:
        depth.setdefault(n, max(depth.values(), default=1) + 1)

    max_d = max(depth.values(), default=0)
    n1 = sum(1 for n in G.nodes if depth.get(n) == 1)
    r1 = max(0.85, 0.16 * n1 / math.pi)
    ring_r = {0: 0.0, 1: r1}
    for d in range(2, max_d + 1):
        ring_r[d] = r1 + 0.66 * (d - 1)

    # 每個 hop-1 節點的「整棵子樹大小」→ 決定它在圓周上分到多寬的扇形
    def subtree_size(root):
        seen, stack, cnt = {root}, [root], 1
        while stack:
            u = stack.pop()
            for v in G.successors(u):
                if v not in seen and depth.get(v, 0) > depth.get(u, 0):
                    seen.add(v); stack.append(v); cnt += 1
        return cnt

    layer1 = [n for n in G.nodes if depth.get(n) == 1]
    n1 = max(len(layer1), 1)
    # 每個 hop-1 節點：保底 40%/n1 的圓弧 + 依子樹大小分配剩下的 60%
    raw = {n: subtree_size(n) ** 0.7 for n in layer1}
    rawtot = sum(raw.values()) or 1.0
    weight = {n: 0.40 / n1 + 0.60 * raw[n] / rawtot for n in layer1}
    # 大 hub 交錯擺放，避免兩個大子樹相鄰互相擠：由大到小輪流插到頭 / 尾
    from collections import deque
    dq = deque()
    for k, n in enumerate(sorted(layer1, key=lambda n: -weight[n])):
        (dq.append if k % 2 else dq.appendleft)(n)
    ring_order = list(dq)
    tot = sum(weight[n] for n in ring_order) or 1.0

    pos: Dict[str, tuple] = {}
    angle: Dict[str, float] = {}
    wedge: Dict[str, tuple] = {}
    if patient_id:
        pos[patient_id] = (0.0, 0.0)
        angle[patient_id] = math.pi / 2

    acc = 0.0
    childless = 0
    for n in ring_order:
        frac = weight[n] / tot
        a0, a1 = acc, acc + frac
        acc = a1
        mid = math.pi / 2 - 2 * math.pi * (a0 + a1) / 2
        angle[n] = mid
        wedge[n] = (math.pi / 2 - 2 * math.pi * a1, math.pi / 2 - 2 * math.pi * a0)
        # 沒有文獻子節點的 EHR 節點：交錯內外半徑，標籤才不會疊在一起
        has_kids = any(depth.get(m) == 2 for m in und.neighbors(n))
        rr = r1
        if not has_kids:
            rr = r1 * (1.0 if childless % 2 == 0 else 1.22)
            childless += 1
        pos[n] = (rr * math.cos(mid), rr * math.sin(mid))

    for d in range(2, max_d + 1):
        by_parent = defaultdict(list)
        for n in (x for x in G.nodes if depth.get(x) == d):
            ps = [p for p in und.neighbors(n) if depth.get(p) == d - 1]
            by_parent[ps[0] if ps else None].append(n)
        for parent, group in by_parent.items():
            lo_a, hi_a = wedge.get(parent, (angle.get(parent, math.pi / 2) - 0.4,
                                            angle.get(parent, math.pi / 2) + 0.4))
            # 留 12% 邊界，子節點在父節點的扇形內均分
            pad = (hi_a - lo_a) * 0.12
            lo_a, hi_a = lo_a + pad, hi_a - pad
            group = sorted(group)
            k = len(group)
            for j, n in enumerate(group):
                a = (lo_a + hi_a) / 2 if k == 1 else lo_a + (hi_a - lo_a) * j / (k - 1)
                angle[n] = a
                wedge[n] = (a - (hi_a - lo_a) / max(k, 1) / 2,
                            a + (hi_a - lo_a) / max(k, 1) / 2)
                rr = ring_r[d] + (j % 2) * 0.16
                pos[n] = (rr * math.cos(a), rr * math.sin(a))

    for n in G.nodes:                       # 保險：漏掉的節點放最外圈
        if n not in pos:
            a = hash(n) % 360 * math.pi / 180
            pos[n] = (1.05 * math.cos(a), 1.05 * math.sin(a))
    return pos


def _clip_sentence(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def render_graph(ax, G, pos, patient_id, *, hide_ehr=False, straight=False,
                 edge_labels=False, label_scale=1.0):
    """把 G 畫到指定的 ax 上（節點 + 邊），回傳 evidence list。legend / 存檔另外處理。"""
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    pad = 0.55
    x0, x1b = min(xs) - pad, max(xs) + pad
    y0, y1b = min(ys) - pad, max(ys) + pad
    ax.set_xlim(x0, x1b)
    ax.set_ylim(y0, y1b)
    ax.set_aspect("equal")
    ax.axis("off")
    nscale = min(1.0, 13.0 / max(len(pos), 1) ** 0.5)

    rels = [d["relevance"] for *_ , d in G.edges(data=True)
            if d["source"] != "EHR" and d.get("relevance") is not None]
    rlo, rhi = (min(rels), max(rels)) if rels else (0.0, 1.0)
    rspan = (rhi - rlo) or 1.0

    def _lit_lw(d):
        r = d.get("relevance")
        if r is None:
            return 2.4
        return 1.0 + 5.5 * (r - rlo) / rspan

    evidence = []
    seen_pair = defaultdict(int)
    curv = 0.05 if straight else 0.10
    for h, ta, d in G.edges(data=True):
        is_ehr = d["source"] == "EHR"
        # 佐證表 / CSV 收錄所有文獻邊（含 REJECTED）；圖上不畫 REJECTED
        if not is_ehr and d.get("pmid"):
            _npm = d.get("n_support", 1)
            evidence.append((len(evidence) + 1, h, d["relation"], ta,
                             "; ".join(d.get("pmids") or [d["pmid"]]),
                             d["score"], d.get("relevance"), d["status"],
                             d.get("title", ""), d.get("sentence", "")))
            tag = f"[{len(evidence)}]{'×' + str(_npm) if _npm > 1 else ''} "
        else:
            tag = ""
        if h not in pos or ta not in pos or h == ta:
            continue
        # hide_ehr 時 EHR 邊不再完全隱藏，改畫成極細的骨架線，讓病人仍是中心
        faint_ehr = is_ehr and hide_ehr
        if d["status"] == "REJECTED":
            continue
        x1, y1 = pos[h]
        x2, y2 = pos[ta]
        if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
            continue
        key = tuple(sorted((h, ta)))
        rad = curv * (seen_pair[key] + 1) * (1 if seen_pair[key] % 2 == 0 else -1)
        seen_pair[key] += 1

        contra = d["status"] == "CONTRADICTED"
        review = d["status"] in ("REVIEW", "UNANCHORED")
        if is_ehr:
            color, lw, ls, alpha = ("#cdd2d8", 0.7, "-", 0.30) if faint_ehr else ("#9aa0a6", 1.3, "-", 0.55)
        elif contra:
            color, lw, ls, alpha = "#b3261e", max(_lit_lw(d), 2.0), "-", 0.85
        elif review:
            color, lw, ls, alpha = "#8a6d1d", max(_lit_lw(d) * 0.8, 1.4), (0, (5, 4)), 0.85
        else:
            color, lw, ls, alpha = REL_COLOR.get(d["relation"], "#8b8b8b"), _lit_lw(d), "-", 0.85
        astyle = "-" if d.get("directed") is False else "-|>"
        arr = FancyArrowPatch(
            (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
            arrowstyle=astyle, mutation_scale=11, lw=lw, color=color,
            linestyle=ls, alpha=alpha, zorder=1 if not is_ehr else 0,
            shrinkA=6, shrinkB=6,
        )
        ax.add_patch(arr)
        # 邊標籤只在決策光譜圖畫（要對應佐證表列號）；其他圖靠顏色就好
        if edge_labels:
            mx = (x1 + x2) / 2 + rad * (y1 - y2) * 0.5
            my = (y1 + y2) / 2 + rad * (x2 - x1) * 0.5
            ax.text(mx, my, tag + d["relation"].replace("_", " ").lower(), fontsize=5.6,
                    ha="center", va="center", color="#5f6368", zorder=3,
                    path_effects=_HALO)

    # --- nodes ---
    for n, (x, y) in pos.items():
        nt = G.nodes[n].get("ntype", "Concept")
        st = NODE_STYLE.get(nt, NODE_STYLE["Concept"])
        r = st["r"] * nscale * (1.6 if n == patient_id else 1.0)
        ax.add_patch(Circle((x, y), r, fc=st["color"], ec="white", lw=1.4, zorder=4))
        label = f"Patient {n}" if n == patient_id else wrap(n, 18)
        if label.strip() and label_scale > 0:
            ax.text(x, y - r - 0.014, label,
                    fontsize=(6.6 if n != patient_id else 9.0) * (0.7 + 0.3 * nscale) * label_scale,
                    ha="center", va="top", color="#1a1a1a", zorder=5,
                    fontweight="bold" if n == patient_id else "normal",
                    path_effects=_HALO)
    return evidence


def _legend_handles(G, hide_ehr):
    rels_present = {dd["relation"] for *_ , dd in G.edges(data=True) if dd["source"] != "EHR"}
    stats = {dd["status"] for *_ , dd in G.edges(data=True) if dd["source"] != "EHR"}
    handles = []
    for nt in ("Disease", "Drug", "Procedure", "Symptom", "AdverseEffect",
               "LaboratoryResult", "RiskFactor"):
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markersize=10,
                       markerfacecolor=NODE_STYLE[nt]["color"], label=nt))
    handles.append(plt.Line2D([0], [0], color="none", label=" "))
    for keys, col, lab in [
        (("HAS_DISEASE", "HAS_COMPLICATION"), "#c1443c", "has disease / complication"),
        (("HAS_SYMPTOM",), "#e8a13a", "has symptom"),
        (("HAS_RISK_FACTOR",), "#7b4fa3", "has risk factor"),
        (("TREATED_WITH", "TREATS", "USED_DRUG"), "#2f7d4f", "treated with / drug"),
        (("CAUSES_ADVERSE_EFFECT",), "#8c510a", "causes adverse effect"),
        (("HAS_LAB_ABNORMALITY",), "#4aa3c4", "lab abnormality"),
    ]:
        if rels_present & set(keys):
            handles.append(plt.Line2D([0], [0], color=col, lw=3, label=lab))
    if rels_present & {"ASSOCIATED_WITH", "INTERACTS_WITH"}:
        handles.append(plt.Line2D([0], [0], color="#8b8b8b", lw=3, label="associated with (no arrowhead)"))
    if not hide_ehr:
        handles.append(plt.Line2D([0], [0], color="#9aa0a6", lw=1.3,
                       label="patient's EHR record (diagnosis / drug / procedure)"))
    handles.append(plt.Line2D([0], [0], color="#555", lw=0.1,
                   label="edge width $\\propto$ relevance of supporting article"))
    if stats & {"REVIEW", "UNANCHORED"}:
        handles.append(plt.Line2D([0], [0], color="#8a6d1d", lw=1.8, ls=(0, (5, 4)), label="held for REVIEW"))
    if "CONTRADICTED" in stats:
        handles.append(plt.Line2D([0], [0], color="#b3261e", lw=2.4, label="CONTRADICTED"))
    return handles


def draw(G, pos, patient_id, title, out_base, dpi, evidence_box=False,
         hide_ehr=False, straight=False, edge_labels=False,
         minimal=False, no_legend=False):
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    pad = 0.55
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    span = max(x1 - x0, y1 - y0)
    scale = min(2.6, 17.0 / span)
    gw, gh = max(7, (x1 - x0) * scale), max(6, (y1 - y0) * scale)

    fig = plt.figure(figsize=(gw, gh))
    ax = fig.add_subplot(1, 1, 1)
    evidence = render_graph(ax, G, pos, patient_id, hide_ehr=hide_ehr,
                            straight=straight, edge_labels=edge_labels)

    handles = _legend_handles(G, hide_ehr)

    if no_legend:
        pass
    elif evidence_box and not minimal:         # focus 圖：legend 放進左上空白處
        ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=7.3,
                  frameon=False, bbox_to_anchor=(0.0, 1.0))
    else:
        ax.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.3,
                  frameon=False, bbox_to_anchor=(0.5, -0.02))

    if title and not minimal:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(f"{out_base}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"[OK] {out_base}.pdf")
    print(f"[OK] {out_base}.png")
    return evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="single patient triples JSON")
    ap.add_argument("--out", default=None, help="output basename (no extension)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--show-review", action="store_true", help="也畫 REVIEW / UNANCHORED 邊")
    ap.add_argument("--show-rejected", action="store_true", help="也畫 REJECTED / CONTRADICTED 邊")
    ap.add_argument("--decision-spectrum", action="store_true",
                    help="連同 review/rejected/contradicted 姊妹檔一起讀，畫出 verifier 決策光譜")
    ap.add_argument("--focus", default="", help="只畫這個 anchor 的證據子樹（如 --focus Hepatitis）")
    ap.add_argument("--focus-hops", type=int, default=2, help="--focus 時從 anchor 往外展開幾跳")
    ap.add_argument("--display-hops", type=int, default=2,
                    help="overview 只畫距病人幾跳內（0=全畫）；資料本身仍是多跳")
    ap.add_argument("--max-nodes", type=int, default=0, help="超過就砍掉分數最低的葉節點")
    ap.add_argument("--hide-ehr", action="store_true",
                    help="只畫文獻邊，隱藏 patient→EHR 骨架（預設會畫）")
    ap.add_argument("--layout", choices=["auto", "radial", "spring"], default="auto",
                    help="auto：節點多就用 spring（力導向，較不糾結），少就用 radial")
    ap.add_argument("--minimal", action="store_true", help="不畫標題、legend 移到下方（合成用）")
    ap.add_argument("--no-legend", action="store_true", help="完全不畫 legend（inset 用）")
    args = ap.parse_args()

    spectrum = args.decision_spectrum or args.show_rejected
    triples = load_triples_with_buckets(args.json) if spectrum else load_triples(args.json)
    if not triples:
        raise SystemExit(f"no triples in {args.json}")

    G, pid = build_graph(triples, args.show_review or spectrum,
                         args.show_rejected or args.decision_spectrum)
    if pid is None:
        # 沒有明確 patient 節點時，取 degree 最高者
        pid = max(G.nodes, key=lambda n: G.degree(n)) if G.number_of_nodes() else None
    if args.focus or args.max_nodes or args.display_hops:
        G = prune_graph(G, pid, args.focus, args.max_nodes, args.focus_hops,
                        args.display_hops)

    hide_ehr = args.hide_ehr and not args.focus       # 預設都畫 EHR 骨架

    # 只留「圖上真的會畫出來」的邊 → 決定佈局與哪些節點要保留
    # EHR 邊即使 hide_ehr 也會畫成細骨架線，所以仍算「可見」，只是佈局權重低
    vis_und = nx.Graph()
    for u, v, dd in G.edges(data=True):
        if u == v or dd["status"] == "REJECTED":
            continue
        is_e = dd["source"] == "EHR"
        w = (0.4 if hide_ehr else 2.4) if is_e else 1.0 + 2.2 * (dd.get("relevance") or 0.4)
        if vis_und.has_edge(u, v):
            vis_und[u][v]["weight"] = max(vis_und[u][v]["weight"], w)
        else:
            vis_und.add_edge(u, v, weight=w)
    keep = set(vis_und.nodes)
    if pid in G:
        keep.add(pid)
    G = G.subgraph(keep).copy()

    if args.focus:
        pos = layered_layout(G, pid)
        use_layout = "layered"
    else:
        use_layout = "spring" if args.layout == "spring" else "radial"
        if use_layout == "spring":
            vis_und = vis_und.subgraph(keep).copy()
            pos = nx.spring_layout(
                vis_und, pos=radial_layout(G, pid),
                fixed=[pid] if pid in vis_und else None,
                k=2.6 / math.sqrt(max(vis_und.number_of_nodes(), 1)),
                weight="weight", iterations=200, seed=7)
        else:
            pos = radial_layout(G, pid)

    base = args.out or os.path.join("figures", os.path.splitext(os.path.basename(args.json))[0])
    if args.title:
        title = args.title
    elif args.focus:
        title = (f"Literature evidence chain for “{G.graph.get('focus', args.focus)}”"
                 f"  (patient {pid})")
    else:
        title = f"Patient-specific literature-evidenced knowledge graph  (patient {pid})"
    evidence = draw(G, pos, pid, title, base, args.dpi, evidence_box=bool(args.focus),
                    hide_ehr=hide_ehr, straight=(use_layout == "spring"),
                    edge_labels=args.decision_spectrum,
                    minimal=args.minimal, no_legend=args.no_legend)

    import csv
    with open(f"{base}_evidence.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "head", "relation", "tail", "pmids", "entailment",
                     "source_relevance", "decision", "title", "supporting_sentence"])
        for row in evidence:
            wr.writerow(row)
    print(f"[OK] {base}_evidence.csv  ({len(evidence)} literature edges)")

    def _tex_escape(s):
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}"), ("{", r"\{"), ("}", r"\}")]:
            s = s.replace(a, b)
        return s

    with open(f"{base}_evidence.tex", "w", encoding="utf-8") as fh:
        fh.write("% auto-generated evidence table — \\input{...} this file inside a table/table* float\n")
        fh.write("\\begin{tabular}{@{}rlcccp{0.42\\linewidth}@{}}\n\\toprule\n")
        fh.write("\\# & Relation & PMID & Entail. & Rel. & Supporting sentence \\\\\n\\midrule\n")
        for n, hh, rel, tt, pmids, sc, rlv, st, _ti, sent in evidence:
            r = f"{_tex_escape(hh)} $\\rightarrow$ {rel.replace('_',' ').lower()} $\\rightarrow$ {_tex_escape(tt)}"
            pm = _tex_escape((pmids or '').split(';')[0].strip())
            rl = f"{rlv:.2f}" if isinstance(rlv, (int, float)) else "--"
            dec = "" if st == "ACCEPTED" else f" \\textit{{({st.lower()})}}"
            fh.write(f"{n} & {r}{dec} & {pm} & {sc:.2f} & {rl} & "
                     f"``{_tex_escape(_clip_sentence(sent, 200))}'' \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[OK] {base}_evidence.tex  (LaTeX table)")

    graphml = f"{base}.graphml"
    H = nx.DiGraph()
    for n, dd in G.nodes(data=True):
        H.add_node(n, **{k: str(v) for k, v in dd.items()})
    for h, ta, dd in G.edges(data=True):
        H.add_edge(h, ta, **{k: str(v) for k, v in dd.items()})
    nx.write_graphml(H, graphml)
    print(f"[OK] {graphml}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


if __name__ == "__main__":
    main()
