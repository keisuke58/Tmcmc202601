import sys, numpy as np
sys.path.insert(0, "tools")
import holdout_eval as HE, make_paper_figures as MPF
root = "docs/revision/generated/final_runs"
sp = ["So", "An", "Vd", "Fn", "Pg"]
for day in (10, 15):
    for lab, pat in (("holdout", f"CH_p2_2c_raw_a33w_ho{day}_seed*"), ("full", "CH_p2_2c_raw_a33w_seed*")):
        g = MPF.group(root, pat, "CH"); rec = g["best"]["rec"]; a = rec["args"]
        MPF.EP.DATA_DIR = MPF.EP.data_dir_for(a)
        days = MPF.model_days(rec); k = int(np.argmin(np.abs(days - day)))
        rng = np.random.default_rng(0); S = g["pooled"]
        X = MPF.simulate_full(S[rng.choice(len(S), 300, replace=False)], rec)[:, k, :]
        phi, psi = np.clip(X[:, :5], 1e-10, 1), np.clip(X[:, 6:11], 1e-10, 1)
        v = phi * psi; v /= v.sum(1, keepdims=True); t = phi / phi.sum(1, keepdims=True)
        data, t_days, _ = MPF.exp_data(a["condition"], a["cultivation"], {**a, "start_from_day": 1})
        j = int(np.where(np.asarray(t_days) == day)[0][0])
        mc = MPF.EP.load_multichannel_data(data_dir=MPF.EP.DATA_DIR / "experiment_data", condition=a["condition"], cultivation=a["cultivation"], days_filter=list(t_days))
        ov = np.asarray(data[j]); ot = np.asarray(mc["data_total"][j]); ot = ot / ot.sum()
        for nm, P, o in (("viable", v, ov), ("total", t, ot)):
            lo, hi = np.quantile(P, .05, 0), np.quantile(P, .95, 0)
            print(day, lab, nm, " ".join(f"{s}:{o[i]:.3f}[{lo[i]:.3f},{hi[i]:.3f}]{'' if lo[i]<=o[i]<=hi[i] else '*'}" for i, s in enumerate(sp)))
