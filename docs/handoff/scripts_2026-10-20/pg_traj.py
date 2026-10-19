import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "tools")
import make_paper_figures as MPF
import comp_pred as CP
root = Path("docs/revision/generated/final_runs")
for name in ["CH_p2_2c_raw_a33w_seed42", "CH_p2_2c_raw_a33w_ho10_seed42"]:
    d = root / name
    rec = json.load(open(d / "run_record.json")); a = rec["args"]
    MPF.EP.DATA_DIR = MPF.EP.data_dir_for(a)
    th = np.asarray(json.load(open(d / "polish.json"))["theta"])
    g = MPF.simulate_full(th, rec)[0]
    days = MPF.model_days(rec)
    _, t_days, idx = MPF.exp_data(a["condition"], a["cultivation"], {**a, "start_from_day": 1})
    tot, live = CP.observations(MPF.EP, rec, t_days)
    print("==", name, "theta len", len(th))
    print(" day   phi(So,An,Vd,Fn,Pg)            sum_phi  total_frac_Pg  psi_Pg   obs_total_Pg")
    for D in [0, 1, 3, 6, 10, 15, 21]:
        k = int(np.argmin(np.abs(days - D)))
        phi = g[k, 0:5]; psi = g[k, 6:11]
        j = np.where(np.asarray(t_days) == D)[0]
        o = tot[j[0]][4] if len(j) else float("nan")
        print(f" {D:3d}  {np.round(phi,4)}  {phi.sum():.3f}  {phi[4]/phi.sum():.3f}  {psi[4]:.3f}  {o:.3f}")
    names = MPF.PARAM_NAMES if hasattr(MPF, "PARAM_NAMES") else None
    print(" theta:", np.round(th, 3).tolist())
