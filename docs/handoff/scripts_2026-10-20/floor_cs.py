import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, "tools")
import make_paper_figures as MPF, comp_pred as CP
for name in sys.argv[1:]:
    d = Path("docs/revision/generated/final_runs") / name
    rec = json.load(open(d / "run_record.json")); a = rec["args"]
    MPF.EP.DATA_DIR = MPF.EP.data_dir_for(a)
    th = np.asarray(json.load(open(d / "polish.json"))["theta"]) if (d/"polish.json").exists() else np.asarray(json.load(open(d/"theta_MAP.json"))["theta_full"])
    g = MPF.simulate_full(th, rec)[0]; days = MPF.model_days(rec)
    _, t_days, idx = MPF.exp_data(a["condition"], a["cultivation"], {**a, "start_from_day": 1})
    tot, _ = CP.observations(MPF.EP, rec, t_days)
    print("==", name, "phi_init", np.round(rec["phi_init"], 3))
    for j, D in enumerate(t_days):
        k = int(np.argmin(np.abs(days - D))); phi = g[k, :5]
        print(f" day {int(D):2d} model total Fn,Pg {phi[3]/phi.sum():.3f},{phi[4]/phi.sum():.3f}  obs {tot[j][3]:.3f},{tot[j][4]:.3f}")
