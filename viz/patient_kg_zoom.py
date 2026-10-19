#!/usr/bin/env python3
"""
一張圖：左上角小的「完整病人 KG」+ 主圖是某 anchor 的放大細部。
小圖上用紅框標出「主圖畫的是哪一塊」。

用法：
  python viz/patient_kg_zoom.py --json triples_output/10000032/single_patient_demo_22595853.json \
      --focus Hepatitis --out figures/patient_kg_zoom
"""
from __future__ import annotations
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import patient_kg_figure as P


def build(json_path, focus, focus_hops):
    triples = P.load_triples(json_path)           # 只用 accepted，不含 review/rejected
    G0, pid = P.build_graph(triples, False, False)
    if pid is None:
        pid = max(G0.nodes, key=lambda n: G0.degree(n))

    # 完整圖（小圖）：所有跳
    Gf = P.prune_graph(G0.copy(), pid, "", 0, focus_hops, 0)
    pos_full = P.radial_layout(Gf, pid)

    # 放大圖（主圖）：focus anchor 往外 focus_hops 跳
    Gz = P.prune_graph(G0.copy(), pid, focus, 0, focus_hops, 0)
    pos_zoom = P.layered_layout(Gz, pid)
    return pid, Gf, pos_full, Gz, pos_zoom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--focus", required=True)
    ap.add_argument("--focus-hops", type=int, default=2)
    ap.add_argument("--out", default="figures/patient_kg_zoom")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    pid, Gf, pos_full, Gz, pos_zoom = build(a.json, a.focus, a.focus_hops)

    xs = [p[0] for p in pos_zoom.values()]
    ys = [p[1] for p in pos_zoom.values()]
    zw, zh = (max(xs) - min(xs) + 1.1), (max(ys) - min(ys) + 1.1)
    W = 13.0
    H = W * zh / zw
    fig = plt.figure(figsize=(W, H))

    # 主圖：放大細部
    ax = fig.add_axes([0.02, 0.14, 0.96, 0.84])
    P.render_graph(ax, Gz, pos_zoom, pid, hide_ehr=False, straight=False)

    # 小圖：完整病人圖（inset，左上）。節點太多，只畫結構不畫文字，當「地圖」用。
    axi = fig.add_axes([0.0, 0.58, 0.40, 0.42])
    P.render_graph(axi, Gf, pos_full, pid, hide_ehr=False, straight=False, label_scale=0.0)
    axi.set_title("full patient graph", fontsize=9, color="#555", pad=2)
    for s in axi.spines.values():
        s.set_visible(True); s.set_edgecolor("#aaa"); s.set_linewidth(0.7)

    # 紅框：主圖畫的那些節點，在完整圖裡的位置
    zoom_nodes = [n for n in Gz.nodes if n in pos_full and n != pid]
    if zoom_nodes:
        bx = [pos_full[n][0] for n in zoom_nodes]
        by = [pos_full[n][1] for n in zoom_nodes]
        mx, my = 0.06, 0.06
        axi.add_patch(Rectangle((min(bx) - mx, min(by) - my),
                                (max(bx) - min(bx)) + 2 * mx,
                                (max(by) - min(by)) + 2 * my,
                                fill=False, edgecolor="#c1443c", lw=1.8, zorder=10))

    # 共用 legend（底部）
    handles = P._legend_handles(Gz, hide_ehr=False)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.savefig(f"{a.out}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{a.out}.png", dpi=a.dpi // 2, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"[OK] {a.out}.pdf")
    print(f"[OK] {a.out}.png")


if __name__ == "__main__":
    main()
