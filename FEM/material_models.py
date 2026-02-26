#!/usr/bin/env python3
"""
material_models.py — φ→E 材料マッピングの集約モジュール
=======================================================

DI ベース (Shannon entropy) と φ_Pg ベース (病原菌濃度) の
2つの E(x) マッピングを提供する。

DI モデル (既存)
----------------
  r = clamp(DI / s_DI, 0, 1)
  E = E_max * (1 - r)^n + E_min * r
  問題: So支配 と Pg支配 を区別できない (entropy のみ)

φ_Pg モデル (新規, メカニズムベース)
------------------------------------
  Pg が gingipain (プロテアーゼ) を分泌 → コラーゲン分解 → 剛性低下
  E = E_healthy - (E_healthy - E_degraded) * σ(φ_Pg / φ_crit)
  σ = Hill sigmoid: x^m / (1 + x^m)

  物理的根拠:
  - Pg の LPS・gingipain は濃度依存的に組織破壊
  - 閾値 φ_crit 以下では組織は健全
  - Hill 係数 m で遷移の急峻さを制御

Virulence index モデル (拡張)
-----------------------------
  V(x) = w_Pg * φ_Pg + w_Fn * φ_Fn
  Fn も LPS・外膜小胞で炎症促進 → 重み付き和
  E(V) = E_healthy - ΔE * σ(V / V_crit)

追加モデル (学術的妥当性順)
---------------------------
  1. Simpson DI: Simpson index D = 1 - Σp_i² → DI_simpson = D (dominance-sensitive)
  2. Voigt mixing: E = Σ φ_i E_i (rule of mixtures, variational-compatible)
  3. Gini evenness: Gini coefficient → 1-Gini as dysbiosis proxy
  4. Pielou evenness: J = H/ln(S) → 1-J as dysbiosis proxy
  5. Reuss mixing: 1/E = Σ φ_i/E_i (inverse rule)

使い方
------
  from material_models import (
      compute_E_di, compute_E_phi_pg, compute_E_virulence,
      compute_di_simpson, compute_E_voigt, compute_E_reuss,
      compute_di_gini, compute_di_pielou,
  )

環境: numpy / jax 両対応 (jax はオプショナル)
"""

from __future__ import annotations
import numpy as np

# ── デフォルトパラメータ ──────────────────────────────────────────────────────

# 共通
E_MAX_PA = 1000.0  # E_healthy [Pa] (commensal upper limit)
E_MIN_PA = 10.0  # E_degraded [Pa] (dysbiotic lower limit)

# DI モデル
DI_SCALE = 0.025778  # DI 正規化スケール (historical default)
DI_EXPONENT = 2.0  # 冪乗則指数

# φ_Pg モデル
# 0D Hamilton ODE: commensal φ_Pg ≈ 0.167 (均等), dysbiotic φ_Pg ≈ 0.8 (Pg支配)
# φ_crit は σ(0.167/φ_crit) ≈ 0 (健全) かつ σ(0.8/φ_crit) ≈ 1 (分解) を満たす値
PHI_PG_CRIT = 0.25  # Pg vol.fraction 閾値 (calibrated for 0D ODE output)
HILL_M = 4.0  # Hill 係数 (遷移の急峻さ, m=1: Michaelis, m→∞: step)

# Virulence index モデル
W_PG = 1.0  # Pg 重み
W_FN = 0.3  # Fn 重み (Pg より弱い病原性)
V_CRIT = 0.30  # Virulence 閾値 (calibrated for 0D ODE output)

# Species index (Hamilton ODE order)
IDX_PG = 4  # P. gingivalis = species[4]
IDX_FN = 3  # F. nucleatum  = species[3]

# Voigt/Reuss mixing: species-specific elastic moduli [Pa]
# Ref: Pattem 2018 (oral biofilm 0.55-14 kPa), Ehret 2013 (network), Laspidou 2014
# Pg weakest (gingipain), So/An stronger (EPS producers)
E_SPECIES_PA = np.array([1000.0, 800.0, 600.0, 200.0, 10.0])  # So, An, Vei, Fn, Pg


# ─────────────────────────────────────────────────────────────────────────────
# NumPy implementations
# ─────────────────────────────────────────────────────────────────────────────


def compute_E_di(
    di: np.ndarray,
    e_max: float = E_MAX_PA,
    e_min: float = E_MIN_PA,
    di_scale: float = DI_SCALE,
    exponent: float = DI_EXPONENT,
) -> np.ndarray:
    """
    DI → E(DI) 冪乗則 (既存モデル).

    E = E_max * (1-r)^n + E_min * r,  r = clamp(DI/s, 0, 1)
    """
    r = np.clip(di / di_scale, 0.0, 1.0)
    return e_max * (1.0 - r) ** exponent + e_min * r


def _hill_sigmoid(x: np.ndarray, m: float) -> np.ndarray:
    """Hill sigmoid: x^m / (1 + x^m), 0→0, 1→0.5, ∞→1."""
    xm = np.power(np.clip(x, 0.0, None), m)
    return xm / (1.0 + xm)


def compute_E_phi_pg(
    phi_species: np.ndarray,
    e_max: float = E_MAX_PA,
    e_min: float = E_MIN_PA,
    phi_crit: float = PHI_PG_CRIT,
    hill_m: float = HILL_M,
) -> np.ndarray:
    """
    φ_Pg → E(φ_Pg) Hill sigmoid (メカニズムベース).

    E = E_max - (E_max - E_min) * σ(φ_Pg / φ_crit)
    σ(x) = x^m / (1 + x^m)

    Parameters
    ----------
    phi_species : (..., 5) array — 菌種別 volume fraction
    phi_crit    : float — Pg 閾値 (default 0.05)
    hill_m      : float — Hill 係数 (default 4.0)

    Returns
    -------
    E : (...) array — 局所剛性 [Pa]

    物理的意味:
    - φ_Pg < φ_crit: E ≈ E_max (組織健全)
    - φ_Pg > φ_crit: E → E_min (組織破壊)
    - m が大きいほど遷移が急峻
    """
    phi_pg = np.asarray(phi_species)[..., IDX_PG]
    sig = _hill_sigmoid(phi_pg / phi_crit, hill_m)
    return e_max - (e_max - e_min) * sig


def compute_E_virulence(
    phi_species: np.ndarray,
    e_max: float = E_MAX_PA,
    e_min: float = E_MIN_PA,
    w_pg: float = W_PG,
    w_fn: float = W_FN,
    v_crit: float = V_CRIT,
    hill_m: float = HILL_M,
) -> np.ndarray:
    """
    Virulence index V → E(V) (Pg + Fn 重み付き).

    V(x) = w_Pg * φ_Pg(x) + w_Fn * φ_Fn(x)
    E(V) = E_max - (E_max - E_min) * σ(V / V_crit)

    根拠: Fn は Pg の co-aggregation partner であり、
    外膜小胞 (OMV) + LPS を通じた炎症促進作用を持つ。
    """
    phi = np.asarray(phi_species)
    v = w_pg * phi[..., IDX_PG] + w_fn * phi[..., IDX_FN]
    sig = _hill_sigmoid(v / v_crit, hill_m)
    return e_max - (e_max - e_min) * sig


def compute_di(phi_species: np.ndarray) -> np.ndarray:
    """DI = 1 - H/H_max (Shannon entropy ベース)."""
    phi_sum = np.asarray(phi_species).sum(axis=-1, keepdims=True)
    phi_sum = np.where(phi_sum > 0, phi_sum, 1.0)
    p = np.asarray(phi_species) / phi_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum(axis=-1)
    return 1.0 - H / np.log(5.0)


def compute_di_simpson(phi_species: np.ndarray) -> np.ndarray:
    """
    Simpson-based dysbiosis index.

    Simpson index D = 1 - Σ p_i² (probability of interspecific encounter).
    D_max = 1 - 1/S for uniform (S species). Normalize: DI = 1 - D/D_max
    so that DI=0 for uniform (commensal), DI=1 for single-species (dysbiotic).
    """
    phi_sum = np.asarray(phi_species).sum(axis=-1, keepdims=True)
    phi_sum = np.where(phi_sum > 0, phi_sum, 1.0)
    p = np.asarray(phi_species) / phi_sum
    n = p.shape[-1]
    D = 1.0 - (p * p).sum(axis=-1)  # Simpson diversity
    D_max = 1.0 - 1.0 / n  # max for uniform
    return 1.0 - D / (D_max + 1e-12)  # DI: 0=commensal, 1=dysbiotic


def compute_di_gini(phi_species: np.ndarray) -> np.ndarray:
    """
    Gini-based dysbiosis index.

    Gini coefficient: 0=perfect equality, 1=complete inequality.
    Sort p, compute 2*Σ(i*p_i) - (n+1)/n. DI_gini = Gini (high=unequal=dysbiotic).
    """
    phi_sum = np.asarray(phi_species).sum(axis=-1, keepdims=True)
    phi_sum = np.where(phi_sum > 0, phi_sum, 1.0)
    p = np.asarray(phi_species) / phi_sum
    n = p.shape[-1]
    # Sort each row
    p_sorted = np.sort(p, axis=-1)
    # Gini = (2 * sum(i * x_i) - (n+1)) / n  for sorted x
    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * (p_sorted * idx).sum(axis=-1) - (n + 1)) / n
    return np.clip(gini, 0.0, 1.0)


def compute_di_pielou(phi_species: np.ndarray) -> np.ndarray:
    """
    Pielou evenness-based dysbiosis index.

    J = H / ln(S), S=species count. J=1: perfectly even, J=0: maximally uneven.
    DI_pielou = 1 - J (high=uneven=dysbiotic).
    """
    phi_sum = np.asarray(phi_species).sum(axis=-1, keepdims=True)
    phi_sum = np.where(phi_sum > 0, phi_sum, 1.0)
    p = np.asarray(phi_species) / phi_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log(p), 0.0)
    H = -(p * log_p).sum(axis=-1)
    S = 5.0  # fixed species count
    J = H / (np.log(S) + 1e-12)
    return 1.0 - np.clip(J, 0.0, 1.0)


def compute_E_voigt(
    phi_species: np.ndarray,
    e_species: np.ndarray | None = None,
) -> np.ndarray:
    """
    Voigt (rule of mixtures): E = Σ φ_i E_i.

    Variational-compatible: E is linear in φ. Upper bound for composite modulus.
    Ref: Rule of mixtures, effective medium theory.
    """
    phi_sum = np.asarray(phi_species).sum(axis=-1, keepdims=True)
    phi_sum = np.where(phi_sum > 0, phi_sum, 1.0)
    p = np.asarray(phi_species) / phi_sum
    E_i = e_species if e_species is not None else E_SPECIES_PA
    return (p * E_i).sum(axis=-1)


def compute_E_reuss(
    phi_species: np.ndarray,
    e_species: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reuss (inverse rule): 1/E = Σ φ_i/E_i.

    Lower bound for composite modulus. Series loading analogy.
    """
    phi_sum = np.asarray(phi_species).sum(axis=-1, keepdims=True)
    phi_sum = np.where(phi_sum > 0, phi_sum, 1.0)
    p = np.asarray(phi_species) / phi_sum
    E_i = e_species if e_species is not None else E_SPECIES_PA
    inv_E = (p / (E_i + 1e-12)).sum(axis=-1)
    return 1.0 / (inv_E + 1e-12)


def compute_E_di_simpson(
    phi_species: np.ndarray,
    e_max: float = E_MAX_PA,
    e_min: float = E_MIN_PA,
    di_scale: float = 1.0,
    exponent: float = DI_EXPONENT,
) -> np.ndarray:
    """E from Simpson-based DI (same power-law as Shannon DI)."""
    di = compute_di_simpson(phi_species)
    return compute_E_di(di, e_max, e_min, di_scale, exponent)


def compute_E_di_gini(
    phi_species: np.ndarray,
    e_max: float = E_MAX_PA,
    e_min: float = E_MIN_PA,
    di_scale: float = 1.0,
    exponent: float = DI_EXPONENT,
) -> np.ndarray:
    """E from Gini-based DI (same power-law as Shannon DI)."""
    di = compute_di_gini(phi_species)
    return compute_E_di(di, e_max, e_min, di_scale, exponent)


def compute_E_di_pielou(
    phi_species: np.ndarray,
    e_max: float = E_MAX_PA,
    e_min: float = E_MIN_PA,
    di_scale: float = 1.0,
    exponent: float = DI_EXPONENT,
) -> np.ndarray:
    """E from Pielou evenness-based DI (same power-law as Shannon DI)."""
    di = compute_di_pielou(phi_species)
    return compute_E_di(di, e_max, e_min, di_scale, exponent)


# ─────────────────────────────────────────────────────────────────────────────
# JAX implementations (autodiff 対応)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import jax.numpy as jnp

    def compute_E_di_jax(
        di,
        e_max=E_MAX_PA,
        e_min=E_MIN_PA,
        di_scale=DI_SCALE,
        exponent=DI_EXPONENT,
    ):
        """DI → E(DI) JAX differentiable."""
        r = jnp.clip(di / di_scale, 0, 1)
        return e_max * (1 - r) ** exponent + e_min * r

    def _hill_sigmoid_jax(x, m):
        xm = jnp.power(jnp.clip(x, 0.0, None), m)
        return xm / (1.0 + xm)

    def compute_E_phi_pg_jax(
        phi_species,
        e_max=E_MAX_PA,
        e_min=E_MIN_PA,
        phi_crit=PHI_PG_CRIT,
        hill_m=HILL_M,
    ):
        """φ_Pg → E(φ_Pg) JAX differentiable."""
        phi_pg = phi_species[..., IDX_PG]
        sig = _hill_sigmoid_jax(phi_pg / phi_crit, hill_m)
        return e_max - (e_max - e_min) * sig

    def compute_E_virulence_jax(
        phi_species,
        e_max=E_MAX_PA,
        e_min=E_MIN_PA,
        w_pg=W_PG,
        w_fn=W_FN,
        v_crit=V_CRIT,
        hill_m=HILL_M,
    ):
        """Virulence index → E(V) JAX differentiable."""
        v = w_pg * phi_species[..., IDX_PG] + w_fn * phi_species[..., IDX_FN]
        sig = _hill_sigmoid_jax(v / v_crit, hill_m)
        return e_max - (e_max - e_min) * sig

    def compute_di_jax(phi_species):
        """DI from phi — JAX differentiable."""
        eps = 1e-12
        phi_sum = jnp.sum(phi_species, axis=-1, keepdims=True)
        phi_sum_safe = jnp.where(phi_sum > eps, phi_sum, 1.0)
        p = phi_species / phi_sum_safe
        log_p = jnp.where(p > eps, jnp.log(p), 0.0)
        H = -jnp.sum(p * log_p, axis=-1)
        return 1.0 - H / jnp.log(5.0)

    HAS_JAX = True

except ImportError:
    HAS_JAX = False


# ─────────────────────────────────────────────────────────────────────────────
# 比較ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────


def compute_all_E(
    phi_species: np.ndarray,
    mode: str = "all",
    di_scale: float = DI_SCALE,
) -> dict[str, np.ndarray]:
    """
    全 E(φ) モデルを一括計算する。

    Parameters
    ----------
    phi_species : (N, 5) — 菌種別 volume fraction
    mode : "di", "phi_pg", "virulence", "simpson", "gini", "pielou",
           "voigt", "reuss", or "all"
    di_scale : float — DI 正規化スケール (0D の場合は 1.0)

    Returns
    -------
    dict with keys for each enabled model
    """
    result = {}
    phi = np.asarray(phi_species)

    if mode in ("di", "all"):
        di = compute_di(phi)
        result["E_di"] = compute_E_di(di, di_scale=di_scale)
        result["DI"] = di
    if mode in ("phi_pg", "all"):
        result["E_phi_pg"] = compute_E_phi_pg(phi)
        result["phi_Pg"] = phi[..., IDX_PG]
    if mode in ("virulence", "all"):
        result["E_virulence"] = compute_E_virulence(phi)
        result["V"] = W_PG * phi[..., IDX_PG] + W_FN * phi[..., IDX_FN]
    if mode in ("simpson", "all"):
        result["E_simpson"] = compute_E_di_simpson(phi, di_scale=di_scale)
        result["DI_simpson"] = compute_di_simpson(phi)
    if mode in ("gini", "all"):
        result["E_gini"] = compute_E_di_gini(phi, di_scale=di_scale)
        result["DI_gini"] = compute_di_gini(phi)
    if mode in ("pielou", "all"):
        result["E_pielou"] = compute_E_di_pielou(phi, di_scale=di_scale)
        result["DI_pielou"] = compute_di_pielou(phi)
    if mode in ("voigt", "all"):
        result["E_voigt"] = compute_E_voigt(phi)
    if mode in ("reuss", "all"):
        result["E_reuss"] = compute_E_reuss(phi)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI: モデル曲線の比較図
# ─────────────────────────────────────────────────────────────────────────────


def plot_model_comparison(outdir: str = None) -> str:
    """3つの E(φ) モデルの応答曲線を比較する図を生成。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if outdir is None:
        import os

        outdir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "figures", "material_models"
        )
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- (a) DI model: E vs DI ---
    ax = axes[0]
    di_arr = np.linspace(0, 0.06, 200)
    for n in [1.0, 2.0, 4.0]:
        E = compute_E_di(di_arr, exponent=n)
        ax.plot(di_arr, E, lw=2, label=f"n={n:.0f}")
    ax.axvline(DI_SCALE, color="red", ls="--", lw=1, alpha=0.7, label=f"DI_scale={DI_SCALE:.4f}")
    ax.set_xlabel("DI (Dysbiotic Index)")
    ax.set_ylabel("E [Pa]")
    ax.set_title("(a) DI model: E(DI)\n[entropy-based, species-blind]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- (b) φ_Pg model: E vs φ_Pg ---
    ax = axes[1]
    phi_pg_arr = np.linspace(0, 0.15, 200)
    for m in [2, 4, 8]:
        sig = _hill_sigmoid(phi_pg_arr / PHI_PG_CRIT, m)
        E = E_MAX_PA - (E_MAX_PA - E_MIN_PA) * sig
        ax.plot(phi_pg_arr, E, lw=2, label=f"m={m}")
    ax.axvline(PHI_PG_CRIT, color="red", ls="--", lw=1, alpha=0.7, label=f"φ_crit={PHI_PG_CRIT}")
    ax.set_xlabel("φ_Pg (P. gingivalis fraction)")
    ax.set_ylabel("E [Pa]")
    ax.set_title("(b) φ_Pg model: E(φ_Pg)\n[mechanism-based, Pg-specific]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- (c) Virulence model: E vs V ---
    ax = axes[2]
    v_arr = np.linspace(0, 0.15, 200)
    sig = _hill_sigmoid(v_arr / V_CRIT, HILL_M)
    E_v = E_MAX_PA - (E_MAX_PA - E_MIN_PA) * sig
    ax.plot(v_arr, E_v, lw=2, color="purple", label=f"m={HILL_M:.0f}")
    ax.axvline(V_CRIT, color="red", ls="--", lw=1, alpha=0.7, label=f"V_crit={V_CRIT}")
    # annotate composition
    ax.annotate(
        f"V = {W_PG}·φ_Pg + {W_FN}·φ_Fn",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        fontsize=9,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow"),
    )
    ax.set_xlabel("V (Virulence index)")
    ax.set_ylabel("E [Pa]")
    ax.set_title("(c) Virulence model: E(V)\n[Pg + Fn weighted]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.suptitle(
        "Material Models: DI (entropy) vs φ_Pg (mechanism) vs Virulence (Pg+Fn)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    path = os.path.join(outdir, "material_model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Material model comparison: {path}")
    return path


if __name__ == "__main__":
    plot_model_comparison()
    print("\nModel parameters:")
    print(f"  DI:        s_DI={DI_SCALE}, n={DI_EXPONENT}")
    print(f"  phi_Pg:    φ_crit={PHI_PG_CRIT}, m={HILL_M}")
    print(f"  Virulence: w_Pg={W_PG}, w_Fn={W_FN}, V_crit={V_CRIT}, m={HILL_M}")
    print(f"  Voigt/Reuss: E_species={E_SPECIES_PA}")

    # Sanity check: commensal vs dysbiotic (S. oralis dominated)
    phi_test = np.array(
        [
            [0.20, 0.20, 0.20, 0.20, 0.20],  # commensal: uniform
            [0.80, 0.05, 0.05, 0.05, 0.05],  # dysbiotic: S.oralis dominant
            [0.05, 0.05, 0.05, 0.05, 0.80],  # dysbiotic: Pg dominant
        ]
    )
    res = compute_all_E(phi_test, di_scale=1.0)
    print("\nSanity check (commensal vs dysbiotic):")
    headers = ["E_di", "E_simpson", "E_gini", "E_pielou", "E_voigt", "E_reuss", "E_phi_pg", "E_vir"]
    key_map = {"E_vir": "E_virulence"}  # display name -> result key
    print(f"  {'cond':>12}  " + "  ".join(f"{h:>10}" for h in headers))
    labels = ["commensal", "dysb(So)", "dysb(Pg)"]
    for i in range(3):
        row = []
        for h in headers:
            k = key_map.get(h, h)
            row.append(f"{res[k][i]:10.1f}" if k in res else "       -")
        print(f"  {labels[i]:>12}  " + "  ".join(row))
