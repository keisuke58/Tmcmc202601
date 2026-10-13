"""図・表の道具が共有する「モデルの組成をどのデータと比べるか」（2026-10-13、2 通路の尤度に合わせる）。

estimator（make_log_likelihood_jax_ode）と同じ対応:
  - 2 通路の run（args.two_channel）: 生きた菌の組成 ↔ φψ/Σφψ、全菌の組成 ↔ φ/Σφ
  - それ以外（今までの run）:       生きた菌の組成 ↔ φ/Σφ（全菌の組成も φ/Σφ と比べる）
群集の生きた割合 ↔ Σφψ/Σφ はどちらも同じ。clip も estimator と同じ（φ は [1e-10, 1-1e-10]、ψ は [1e-10, 1]）。
データは run が読んだ場所（EP.data_dir_for(args)、digitised / raw）から読む。
"""

import numpy as np


def is_two_channel(rec):
    return bool(rec["args"].get("two_channel", False))


def fractions(g, two_channel):
    """状態の軌道 g（..., 12 以上）→ {"viable": 生きた菌の組成と比べる予測, "total": φ/Σφ, "live": Σφψ/Σφ}。"""
    g = np.asarray(g)
    phi = np.clip(g[..., 0:5], 1e-10, 1 - 1e-10)
    psi = g[..., 6:11]
    tot = phi / np.maximum(phi.sum(-1, keepdims=True), 1e-12)
    if two_channel:
        pv = phi * np.clip(psi, 1e-10, 1.0)
        via = pv / np.maximum(pv.sum(-1, keepdims=True), 1e-12)
    else:
        via = tot
    raw_phi, raw_psi = g[..., 0:5], g[..., 6:11]
    live = (raw_phi * raw_psi).sum(-1) / np.maximum(raw_phi.sum(-1), 1e-12)
    return {"viable": via, "total": tot, "live": live}


def observations(EP, rec, t_days):
    """run が読んだデータの全菌の組成（行和 1）と生きた割合（観測日 t_days）。"""
    a = rec["args"]
    mc = EP.load_multichannel_data(data_dir=EP.data_dir_for(a) / "experiment_data", condition=a["condition"],
                                   cultivation=a["cultivation"], days_filter=list(np.asarray(t_days).tolist()))
    tot = mc.get("data_total")
    if tot is not None:
        tot = np.asarray(tot, dtype=float)
        tot = tot / tot.sum(1, keepdims=True)
    live = mc.get("data_viability")
    return tot, (None if live is None else np.asarray(live, dtype=float))


def replicate_csv(EP, rec):
    """Fig. 2 の箱ひげに使う生きた菌の組成の反復（run が読んだデータ）。"""
    return EP.data_dir_for(rec["args"]) / "experiment_data" / "fig3_species_distribution_replicates.csv"
