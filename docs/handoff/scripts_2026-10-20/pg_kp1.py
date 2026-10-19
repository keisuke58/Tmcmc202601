import sys, json, numpy as np
sys.path.insert(0, "data_5species/main")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import hamilton_ode_jax_paper as H
d = "docs/revision/generated/final_runs/CH_p2_2c_raw_a33w_seed42"
rec = json.load(open(f"{d}/run_record.json")); a = rec["args"]
th = jnp.array(json.load(open(f"{d}/polish.json"))["theta"])
def run(th, kp1, phi_init):
    A, b = H.theta_to_matrices(th)
    g0 = H.make_initial_state(jnp.array(phi_init), jnp.ones(5, dtype=jnp.int64), None)
    p = {"dt_h": a["dt"], "Kp1": kp1, "Eta": jnp.ones(5), "EtaPhi": jnp.ones(5), "c": 25.0, "alpha": 0.0,
         "K_hill": jnp.array(a["K_hill"]), "n_hill": jnp.array(a["n_hill"]), "A": A, "b_diag": b,
         "active_mask": jnp.ones(5, dtype=jnp.int64)}
    _, tr = jax.lax.scan(lambda g, _: (H.newton_step(g, p),) * 2, g0, jnp.arange(a["n_steps"]))
    return np.asarray(tr)
n = a["n_steps"]; ks = {"Day~1": n//20, "Day~10": int(n*0.95*10/21), "Day~21": int(n*0.95)}
print("n_steps", n, "dt", a["dt"])
for kp1 in [1e-4, 1e-5, 1e-6]:
    tr = run(th, kp1, rec["phi_init"])
    print(f"Kp1={kp1:g}: ", "  ".join(f"{k}: Pg phi={tr[i,4]:.4f} psi={tr[i,10]:.3f} Fn phi={tr[i,3]:.4f}" for k, i in ks.items()))
# Pg row interactions all strongly negative vs zero interactions
th0 = th.at[14].set(0.0).at[16].set(0.0).at[17].set(0.0).at[18].set(0.0).at[19].set(0.0)
tr = run(th0, 1e-4, rec["phi_init"]); i = ks["Day~10"]
print("Pg row = 0, Kp1=1e-4: Day~10 Pg phi=%.4f psi=%.3f" % (tr[i,4], tr[i,10]))
# barrier magnitude
for phi in [0.001, 0.01, 0.035, 0.1]:
    print(f"barrier term at phi={phi}: {1e-4*(2-4*phi)/((phi-1)**3*phi**3):.3g}")
