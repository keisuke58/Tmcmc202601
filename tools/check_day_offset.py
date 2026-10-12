#!/usr/bin/env python3
"""1 日のずれ（Day 1 の組成を t=0=day 0 に置いている）の影響: 同じ MAP で観測日の対応を t=s·d と t=s·(d−1) で比べる（2026-10-12）。DS・DH の run 名は仮。"""
import sys,json,glob,numpy as np
sys.argv=[sys.argv[0]]; sys.path.insert(0,"data_5species/main")
import jax; jax.config.update("jax_enable_x64",True); jax.config.update("jax_platform_name","cpu")
import jax.numpy as jnp, estimate_paper_jax as E
G={"CS":("Commensal","Static","CS_ult_wide2_sd4_seed42"),"CH":("Commensal","HOBIC","CH_ult_mut150_wide2_noph_sd4_seed7"),
   "DS":("Dysbiotic","Static","DS_p2_wide80_p18k_wide2_seed123"),"DH":("Dysbiotic","HOBIC","DH_p2_nonarrow_w3_noph_p14k_seed123")}
for t,(c,cu,run) in G.items():
    d="docs/revision/generated/final_runs/"+run; rec=json.load(open(d+"/run_record.json")); a=rec["args"]
    th=json.load(open(d+"/theta_MAP.json")); th=jnp.array([th[str(i)] for i in range(20)])
    data,td,_,_,_=E.load_experimental_data(E.DATA_DIR,c,cu,1,normalize=True,use_exp_init=True)
    sc=0.95*a["n_steps"]*a["dt"]/td.max()
    g=np.asarray(E.simulate_0d(th,n_steps=a["n_steps"],dt=a["dt"],phi_init=jnp.array(rec["phi_init"]),K_hill=a["K_hill"],n_hill=a["n_hill"]))
    out=[]
    for off in [0,1]:
        idx=np.round((td-off)*sc/a["dt"]).astype(int); p=np.clip(g[idx],1e-10,1); p/=p.sum(1,keepdims=True)
        out.append(np.sqrt(np.mean((data-p)**2)))
    print(t,"RMSE 今の対応（Day1 を t=0=day0 に置く）%.4f   Day1 を t=0 にずらした対応（同じ θ、再推定なし）%.4f"%tuple(out))
