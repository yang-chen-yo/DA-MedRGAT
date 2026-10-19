#!/usr/bin/env python3
import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

_VIS_URL = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"
_VIS_CACHE = Path(__file__).with_name(".vis-network-9.1.9.min.js")


def vis_network_js() -> str:
    """回傳 vis-network 原始碼（本地快取；抓不到就退回 CDN <script src>）。"""
    if _VIS_CACHE.exists():
        return _VIS_CACHE.read_text(encoding="utf-8")
    try:
        data = urllib.request.urlopen(_VIS_URL, timeout=15).read().decode("utf-8")
        _VIS_CACHE.write_text(data, encoding="utf-8")
        return data
    except Exception:
        return ""


def load_triples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("triples", [])
    return payload


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))


def short_label(value: str, limit: int = 26) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit - 1] + "…"


def readable_label(raw: str, fallback: str = "Entity") -> str:
    txt = (raw or "").strip()
    if not txt:
        return fallback
    if txt.isdigit():
        return f"Patient {txt}"
    return short_label(txt, 28)


def build_nodes_and_edges(triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_index: Dict[str, Dict[str, Any]] = {}
    _edge_bucket: Dict[tuple, Dict[str, Any]] = {}

    def normalize_kind(raw: str, default: str = "entity") -> str:
        if raw is None:
            return default
        value = str(raw).strip().lower()
        if value in {"patient", "person"}:
            return "patient"
        if value in {"disease", "diagnosis", "condition"}:
            return "disease"
        if value in {"symptom", "sign"}:
            return "symptom"
        if value in {"procedure", "treatment", "medication", "drug"}:
            return "treatment"
        if value in {"adverseeffect", "adverse_effect", "sideeffect"}:
            return "adverseeffect"
        return default

    def ensure_node(node_id: str, label: str, kind: str = "entity"):
        if node_id not in node_index:
            node_index[node_id] = {"id": node_id, "label": label, "kind": kind, "x": 0.0, "y": 0.0}
            nodes.append(node_index[node_id])
        return node_index[node_id]

    for idx, triple in enumerate(triples):
        head = str(triple.get("head") or "unknown")
        tail = str(triple.get("tail") or "unknown")
        relation = str(triple.get("relation") or "REL")
        source = str(triple.get("source") or triple.get("evidence_source") or "")
        evidence_source = str(triple.get("evidence_source") or source or "")
        evidence_id = str(triple.get("evidence_id") or "")
        evidence_text = str(
            triple.get("supporting_sentence")
            or triple.get("evidence_text")
            or triple.get("evidence_title")
            or ""
        )
        title = str(triple.get("evidence_title") or "")
        entail = triple.get("verifier_score")
        try:
            entail = float(entail)
        except (TypeError, ValueError):
            entail = None
        relevance = triple.get("evidence_score")
        if relevance is None:
            relevance = triple.get("weight")
        try:
            relevance = float(relevance)
        except (TypeError, ValueError):
            relevance = None
        decision = str(triple.get("verifier_decision")
                       or triple.get("support_status") or "").upper()
        is_ehr = source.upper() == "EHR"
        if is_ehr:
            decision = decision or "EHR-FACT"

        relation = relation.upper()
        directed = relation not in ("ASSOCIATED_WITH", "INTERACTS_WITH")
        if relation == "TREATS":                       # 正規化反向關係
            relation, head, tail = "TREATED_WITH", tail, head
        eh, et = (head, tail)
        if not directed and eh.lower() > et.lower():
            eh, et = et, eh

        h_kind = normalize_kind(triple.get("head_type"), "patient" if str(head).isdigit() else "entity")
        t_kind = normalize_kind(triple.get("tail_type"), "concept" if not str(tail).isdigit() else "entity")
        ensure_node(eh, readable_label(eh, "Patient"),
                    normalize_kind(triple.get("head_type"), "patient" if str(eh).isdigit() else "entity"))
        ensure_node(et, readable_label(et, "Concept"),
                    normalize_kind(triple.get("tail_type"), "concept" if not str(et).isdigit() else "entity"))

        key = (eh, et, relation)
        bucket = _edge_bucket.setdefault(key, {
            "id": len(_edge_bucket), "source": eh, "target": et, "label": relation,
            "is_ehr": is_ehr, "directed": directed, "pmids": [], "ev_title": title,
            "entail": entail, "relevance": relevance, "decision": decision or "UNKNOWN",
            "sentence": evidence_text[:600],
        })
        if evidence_id and evidence_id not in bucket["pmids"]:
            bucket["pmids"].append(evidence_id)
        # 保留相關度最高的那筆佐證
        if (relevance or 0) > (bucket["relevance"] or 0):
            bucket.update(relevance=relevance, entail=entail, decision=decision or "UNKNOWN",
                          sentence=evidence_text[:600], ev_title=title)

    for b in _edge_bucket.values():
        b["pmid"] = b["pmids"][0] if b["pmids"] else ""
        b["n_support"] = len(b["pmids"])
        edges.append(b)
    return nodes, edges


NODE_PALETTE = {
    "patient":       ("#1d4e89", "#ffffff"),
    "disease":       ("#c1443c", "#ffffff"),
    "treatment":     ("#2f7d4f", "#ffffff"),
    "symptom":       ("#e8a13a", "#20160a"),
    "adverseeffect": ("#8c510a", "#ffffff"),
    "concept":       ("#6b7280", "#ffffff"),
    "entity":        ("#6b7280", "#ffffff"),
}
REL_COLOR = {
    "HAS_DISEASE": "#c1443c", "HAS_SYMPTOM": "#e8a13a", "HAS_COMPLICATION": "#d1495b",
    "HAS_RISK_FACTOR": "#4d7c37", "HAS_LAB_ABNORMALITY": "#4aa3c4",
    "TREATED_WITH": "#2f7d4f", "TREATS": "#2f7d4f", "USED_DRUG": "#2f7d4f",
    "RECEIVED_PROCEDURE": "#7a5195", "CAUSES_ADVERSE_EFFECT": "#8c510a",
    "CONTRAINDICATED_IN": "#7d3c98", "INTERACTS_WITH": "#7d3c98", "ASSOCIATED_WITH": "#8b8b8b",
}


def build_html(triples: List[Dict[str, Any]], title: str) -> str:
    nodes, edges = build_nodes_and_edges(triples)

    rels = [e["relevance"] for e in edges if not e["is_ehr"] and e["relevance"] is not None]
    rlo, rhi = (min(rels), max(rels)) if rels else (0.0, 1.0)
    rspan = (rhi - rlo) or 1.0

    vis_nodes = []
    for n in nodes:
        bg, fg = NODE_PALETTE.get(n["kind"], NODE_PALETTE["concept"])
        vis_nodes.append({
            "id": n["id"], "label": n["label"],
            "color": {"background": bg, "border": "#ffffff",
                      "highlight": {"background": bg, "border": "#111827"}},
            "font": {"color": "#1a1a1a", "size": 20 if n["kind"] == "patient" else 15,
                     "face": "-apple-system, Segoe UI, Roboto, Arial",
                     "strokeWidth": 4, "strokeColor": "#ffffff",
                     "bold": n["kind"] == "patient"},
            "shape": "dot",
            "value": 34 if n["kind"] == "patient" else 13,
        })

    vis_edges = []
    for e in edges:
        if e["is_ehr"]:
            color, width, dashes = "#94a3b8", 1.6, [2, 3]         # 灰點線 = EHR 事實
        else:
            rel = e["relevance"]
            frac = 0.0 if rel is None else (rel - rlo) / rspan
            width = 2.0 + 6.5 * frac
            dec = e["decision"]
            if dec == "CONTRADICTED":
                color, dashes = "#b3261e", [6, 4]
            elif dec in ("REVIEW", "UNANCHORED"):
                color, dashes = "#8a6d1d", [8, 6]
            elif dec == "REJECTED":
                color, dashes = "#9aa0a6", [2, 3]
            else:
                color, dashes = REL_COLOR.get(e["label"].upper(), "#8b8b8b"), False
        rel_txt = e["label"].replace("_", " ").lower()
        tip = f"{e['source']} —{rel_txt}→ {e['target']}"
        if not e["is_ehr"]:
            tip += f"  ·  {e['decision']}  ·  relevance {e['relevance']:.2f}" if isinstance(e["relevance"], float) else f"  ·  {e['decision']}"
        vis_edges.append({
            "id": e["id"], "from": e["source"], "to": e["target"],
            "title": tip, "isEhr": bool(e["is_ehr"]), "hidden": bool(e["is_ehr"]),
            "color": {"color": color, "highlight": "#111827", "hover": "#111827"},
            "width": round(width, 2), "dashes": dashes,
            "arrows": {"to": {"enabled": bool(e.get("directed", True)), "scaleFactor": 0.7}},
            "smooth": {"type": "dynamic"},
        })

    rows = []
    for e in edges:
        if e["is_ehr"]:
            continue
        ent = f"{e['entail']:.2f}" if isinstance(e["entail"], float) else "–"
        rlv = f"{e['relevance']:.2f}" if isinstance(e["relevance"], float) else "–"
        rows.append({
            "id": e["id"], "head": e["source"], "rel": e["label"].replace("_", " ").lower(),
            "tail": e["target"], "pmid": e["pmid"], "entail": ent, "relevance": rlv,
            "decision": e["decision"], "title": e["ev_title"], "sentence": e["sentence"],
        })

    payload = json.dumps({"nodes": vis_nodes, "edges": vis_edges, "rows": rows})
    n_lit = len(rows)
    n_ehr = len(edges) - n_lit
    _js = vis_network_js()
    vis_tag = (f"<script>{_js}</script>" if _js
               else f'<script src="{_VIS_URL}"></script>')

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{safe_text(title)}</title>
  {vis_tag}
  <style>
    html,body{{height:100%;margin:0;background:#f3f4f6;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;}}
    .wrap{{display:flex;height:100vh;width:100vw;overflow:hidden;}}
    .gcol{{flex:1 1 auto;display:flex;flex-direction:column;min-width:0;}}
    #graph{{flex:1 1 auto;background:#fff;border-right:1px solid #e2e8f0;min-height:0;}}
    .side{{flex:0 0 520px;overflow-y:auto;padding:16px;box-sizing:border-box;}}
    h1{{font-size:17px;margin:0 0 4px;}}
    .sub{{color:#64748b;font-size:12px;margin-bottom:10px;}}
    .legend{{display:flex;flex-wrap:wrap;align-items:center;gap:6px 16px;font-size:12px;color:#1f2937;padding:10px 14px;border-bottom:1px solid #e2e8f0;background:#fafafa;}}
    .legend b{{font-weight:600;}}
    .swatch{{display:inline-block;width:26px;height:0;border-top-width:4px;border-top-style:solid;vertical-align:middle;margin-right:5px;}}
    .legend label{{cursor:pointer;user-select:none;}}
    .card{{border:1px solid #e2e8f0;border-radius:10px;padding:11px 13px;margin-bottom:10px;background:#fff;cursor:pointer;}}
    .card.sel{{border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,.15);}}
    .card .claim{{font-weight:600;font-size:14px;margin-bottom:5px;line-height:1.4;}}
    .card .m{{font-size:12px;color:#475569;line-height:1.6;}}
    .card .s{{font-size:13px;color:#1f2937;margin-top:6px;line-height:1.6;}}
    .tag{{display:inline-block;border-radius:999px;padding:2px 8px;font-size:10px;font-weight:700;color:#fff;margin-right:5px;}}
    .ACCEPTED{{background:#16a34a;}} .REVIEW{{background:#b45309;}}
    .REJECTED{{background:#6b7280;}} .CONTRADICTED{{background:#dc2626;}} .UNKNOWN{{background:#6b7280;}}
    a{{color:#2563eb;}}
    h1{{font-size:18px;}} .sub{{font-size:13px;}}
  </style>
</head>
<body>
<div class="wrap">
  <div class="gcol">
    <div class="legend">
      <label><input type="checkbox" id="tglEhr"> <span class="swatch" style="border-top-color:#94a3b8;border-top-style:dotted;"></span>EHR fact ({n_ehr}) — off</label>
      <span><span class="swatch" style="border-top-color:#c1443c;"></span>ACCEPTED ({n_lit}) · <b>width ∝ source relevance</b></span>
      <span><span class="swatch" style="border-top-color:#8a6d1d;border-top-style:dashed;"></span>REVIEW</span>
      <span><span class="swatch" style="border-top-color:#b3261e;border-top-style:dashed;"></span>CONTRADICTED</span>
      <span><span class="swatch" style="border-top-color:#9aa0a6;border-top-style:dotted;"></span>REJECTED</span>
      <span style="color:#64748b;">drag · scroll-zoom · click an edge → its evidence</span>
    </div>
    <div id="graph"></div>
  </div>
  <div class="side">
    <h1>{safe_text(title)}</h1>
    <div class="sub">{len(nodes)} nodes · {n_lit} literature edges (+{n_ehr} EHR facts) · click an edge or card to trace its PubMed evidence</div>
    <div id="cards"></div>
  </div>
</div>
<script>
const DATA = {payload};
const nodes = new vis.DataSet(DATA.nodes);
const edges = new vis.DataSet(DATA.edges);
let network;

function initGraph() {{
  const container = document.getElementById('graph');
  network = new vis.Network(container, {{nodes, edges}}, {{
    physics: {{barnesHut: {{gravitationalConstant: -9000, springLength: 130, springConstant: 0.03, avoidOverlap: 0.4}},
              stabilization: {{iterations: 350}}}},
    interaction: {{hover: true, tooltipDelay: 120}},
    nodes: {{scaling: {{min: 10, max: 34}}, borderWidth: 2, shadow: false}},
    edges: {{selectionWidth: 2.2}},
  }});
  network.once('stabilizationIterationsDone', () => {{
    network.setOptions({{physics: false}});
    network.fit({{animation: false}});
  }});
  network.on('selectEdge', p => {{ if (p.edges.length) focusEdge(p.edges[0]); }});
  network.on('click', p => {{ if (!p.edges.length && !p.nodes.length) Object.values(cardById).forEach(el => el.classList.remove('sel')); }});
  window.addEventListener('resize', () => network.fit({{animation: false}}));
  const tgl = document.getElementById('tglEhr');
  if (tgl) tgl.addEventListener('change', () => {{
    const show = tgl.checked;
    edges.update(DATA.edges.filter(e => e.isEhr).map(e => ({{id: e.id, hidden: !show}})));
    tgl.parentElement.lastChild.textContent = show ? ' EHR fact ({n_ehr}) — on' : ' EHR fact ({n_ehr}) — off';
  }});
}}
if (document.readyState === 'complete') initGraph();
else window.addEventListener('load', initGraph);

const cardBox = document.getElementById('cards');
const cardById = {{}};
DATA.rows.forEach(r => {{
  const c = document.createElement('div');
  c.className = 'card'; c.dataset.id = r.id;
  const pmidLink = r.pmid ? `<a href="https://pubmed.ncbi.nlm.nih.gov/${{r.pmid}}/" target="_blank">PMID ${{r.pmid}}</a>` : 'PMID –';
  c.innerHTML = `<div class="claim"><span class="tag ${{r.decision}}">${{r.decision}}</span>${{r.head}} —${{r.rel}}→ ${{r.tail}}</div>`
    + `<div class="m">${{pmidLink}} · entailment ${{r.entail}} · source relevance ${{r.relevance}}</div>`
    + (r.title ? `<div class="m"><i>${{r.title}}</i></div>` : '')
    + `<div class="s">&ldquo;${{r.sentence}}&rdquo;</div>`;
  c.addEventListener('click', () => {{
    focusEdge(r.id);
    if (network) {{ network.selectEdges([r.id]); network.focus(r.id, {{scale: 1.0, animation: true}}); }}
  }});
  cardBox.appendChild(c);
  cardById[r.id] = c;
}});

function focusEdge(id) {{
  Object.values(cardById).forEach(el => el.classList.remove('sel'));
  const el = cardById[id];
  if (el) {{ el.classList.add('sel'); el.scrollIntoView({{block: 'nearest', behavior: 'smooth'}}); }}
}}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Render one patient evidence graph with edge-level evidence details.")
    parser.add_argument("--json", required=True, help="Path to a triples JSON file")
    parser.add_argument("--out", default=None, help="Where to write the HTML")
    parser.add_argument("--title", default="Single Patient Evidence Graph", help="Graph title")
    args = parser.parse_args()

    json_path = Path(args.json)
    triples = load_triples(str(json_path))
    if not triples:
        raise SystemExit(f"No triples loaded from {json_path}")

    out_path = Path(args.out) if args.out else Path("graphify-out") / f"{json_path.stem}_evidence.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(triples, args.title)
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] wrote evidence graph HTML to {out_path}")


if __name__ == "__main__":
    main()
