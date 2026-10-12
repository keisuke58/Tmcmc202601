#!/usr/bin/env python3
"""two_channel_pilot.py の JSON を並べて Markdown の表にする（2026-10-13）。

    python3 tools/two_channel_pilot_table.py docs/revision/generated/two_channel_pilot/improve/*.json

表 1: RMSE（今の MAP の 生きた菌↔φ / 最適化後の 生きた菌↔φψ・全菌↔φ・生きた割合・ψ↔データ）
表 2: ψ_i（日の中央値）とデータの ψ_i^data
"""
import json
import sys
from pathlib import Path

rows = []
for f in sys.argv[1:]:
    d = json.load(open(f))
    a = d.get("args", {})
    rows.append((d["tag"], a.get("psi_data", 0.0), Path(f).stem, d))
rows.sort(key=lambda r: (["CS", "CH", "DS", "DH"].index(r[0]), r[3].get("args", {}).get("version", 1), r[1]))

print("| 条件 | 版 | W(ψ^data) | 再スタート×反復 | −logL(2c) MAP→最適 | 今の MAP: 生きた菌↔φ | 生きた割合 | → 最適: 生きた菌↔φψ | 全菌↔φ | 生きた割合 | ψ↔データ | 箱の端 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for tag, w, name, d in rows:
    m, o, a = d["map"], d["two_channel_opt"], d.get("args", {})
    it = f"{a.get('restarts', '?')}×{a.get('maxiter', '?')}"
    ver = f"v{a.get('version', 1)}"
    nl = f"{m['neg_logL']:.1f}→{o['neg_logL']:.1f}" if "neg_logL" in o else "–"
    edge = ",".join(str(i) for i in d.get("at_edge", [])) or "–"
    print(f"| {tag} | {ver} | {w:g} | {it} | {nl} | {m['viable_vs_phi']:.3f} | {m['live']:.3f} | "
          f"**{o['viable_vs_phipsi']:.3f}** | {o['total_vs_phi']:.3f} | **{o['live']:.3f}** | "
          f"{o.get('psi_vs_data', float('nan')):.3f} | {edge} |")

print()
print("| 条件 | 版 | W | ψ So/An/Vei/Fn/Pg（日の中央値） | データ ψ^data |")
print("|---|---|---|---|---|")
for tag, w, name, d in rows:
    p = "/".join(f"{x:.2f}" for x in d["psi_opt_median_days"])
    q = "/".join(f"{x:.2f}" for x in d["psi_data_median_days"]) if "psi_data_median_days" in d else "–"
    print(f"| {tag} | v{d.get('args', {}).get('version', 1)} | {w:g} | {p} | {q} |")
