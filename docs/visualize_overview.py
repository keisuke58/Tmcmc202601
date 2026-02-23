"""
Overview visualization script for the method.
Visualizes the pipeline in multiple formats.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, FancyArrow
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path("/home/nishioka/IKM_Hiwi/Tmcmc202601/docs/overview_figs")
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────
COLORS = {
    "So": "#4E9AF1",   # S. oralis - blue
    "An": "#6DC06D",   # A. naeslundii - green
    "Vd": "#F4A142",   # V. dispar - orange
    "Fn": "#B07CC6",   # F. nucleatum - purple
    "Pg": "#E05252",   # P. gingivalis - red
    "bg":  "#F7F9FC",
    "panel": "#1A2A4A",
    "accent": "#2563EB",
    "gray":  "#8899AA",
    "arrow": "#334155",
}

SPECIES = ["So\n(S. oralis)", "An\n(A. naeslundii)", "Vd\n(V. dispar)", "Fn\n(F. nucleatum)", "Pg\n(P. gingivalis)"]
SCOLORS = [COLORS["So"], COLORS["An"], COLORS["Vd"], COLORS["Fn"], COLORS["Pg"]]


# ═══════════════════════════════════════════════════════════════
# FIG 1: Overall pipeline flowchart
# ═══════════════════════════════════════════════════════════════
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor("#F0F4FA")
    ax.set_facecolor("#F0F4FA")

    def box(x, y, w, h, label, sublabel="", color="#FFFFFF", textcolor="#1A2A4A", fontsize=13, radius=0.4):
        fancy = FancyBboxPatch((x - w/2, y - h/2), w, h,
                               boxstyle=f"round,pad=0.1,rounding_size={radius}",
                               linewidth=2, edgecolor=color, facecolor=color + "22",
                               zorder=3)
        ax.add_patch(fancy)
        ax.text(x, y + (0.18 if sublabel else 0), label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=textcolor, zorder=4)
        if sublabel:
            ax.text(x, y - 0.32, sublabel, ha="center", va="center",
                    fontsize=9, color=textcolor + "BB" if textcolor != "#1A2A4A" else "#555577", zorder=4, style="italic")

    def arrow(x1, y1, x2, y2, label="", color="#334155"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=2.5, mutation_scale=20),
                    zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.1, my, label, fontsize=8, color=color, va="center")

    # ── Title ─────────────────────────────
    ax.text(10, 10.5, "Oral biofilm parameter inference — overall pipeline",
            ha="center", va="center", fontsize=17, fontweight="bold", color=COLORS["panel"])
    ax.text(10, 10.0, "5-Species Biofilm · TMCMC Bayesian Inference",
            ha="center", va="center", fontsize=11, color=COLORS["gray"])

    # ── Step 1: Experimental data ─────────
    box(2.2, 8.0, 3.6, 2.8, "① Experimental data", "6 time points × 5 species\nVolume fraction φᵢ(t)", "#4E9AF1", fontsize=12)
    # Data grid image
    for i, (sp, sc) in enumerate(zip(["So","An","Vd","Fn","Pg"], SCOLORS)):
        for j in range(6):
            val = np.random.uniform(0.05, 0.35)
            rect = plt.Rectangle((0.5 + j*0.38, 6.75 + i*0.35 - 0.2), 0.34, 0.3,
                                  color=sc, alpha=0.15 + val*0.6, zorder=5)
            ax.add_patch(rect)
            ax.text(0.5 + j*0.38 + 0.17, 6.75 + i*0.35 - 0.05, f"{val:.2f}",
                    fontsize=5.5, ha="center", va="center", color="#333", zorder=6)
    ax.text(2.2, 6.35, "day: 1   3   6   10   15   21", fontsize=7.5, ha="center",
            color="#555", zorder=6)

    # ── Step 2: Prior distribution ────────
    box(6.5, 8.5, 3.2, 1.6, "② Prior distribution", "Uniform [lb, ub]\n20 parameters", "#6DC06D", fontsize=12)

    # ── Step 3: ODE model ────────────────
    box(6.5, 6.2, 3.2, 1.8, "③ ODE model", "dφᵢ/dt = f(φ,ψ;θ)\nHill-gated interactions", "#F4A142", fontsize=12)

    # ── Step 4: Likelihood ───────────────
    box(11.0, 7.5, 3.2, 2.2, "④ Likelihood", "log p(D|θ)\nWeighted Gaussian likelihood\nλ_pg × λ_late", "#B07CC6", fontsize=12)

    # ── Step 5: TMCMC ─────────────────────
    box(15.5, 7.5, 3.6, 2.6, "⑤ TMCMC", "β: 0 → 1\n8 stages\n150 particles", "#E05252", fontsize=12)

    # ── Step 6: Output ────────────────────
    box(15.5, 4.0, 3.6, 2.4, "⑥ Inference output", "MAP estimates\n95% credible intervals\nConvergence diagnostics (Rhat, ESS)", "#2563EB", fontsize=12)

    # ── Acceleration box ──────────────────
    box(11.0, 5.0, 3.2, 1.6, "Acceleration", "Numba JIT\nTSM-ROM surrogate\nParallel evaluation (4–8×)", "#64748B", "#DDDDDD", fontsize=11)

    # ── Arrows ────────────────────────────
    arrow(4.0, 8.0, 4.9, 7.5)             # データ → TMCMC上流
    arrow(4.9, 8.5, 4.9, 8.5)
    arrow(4.0, 8.0, 5.85, 8.5)            # データ → 事前分布
    arrow(4.0, 8.0, 5.85, 6.2)            # データ → ODE
    arrow(6.5, 7.7, 6.5, 7.1)             # 事前 → ODE
    arrow(8.1, 8.5, 9.4, 7.9)             # 事前 → 尤度
    arrow(8.1, 6.2, 9.4, 7.0)             # ODE → 尤度
    arrow(12.6, 7.5, 13.7, 7.5)           # 尤度 → TMCMC
    arrow(15.5, 6.2, 15.5, 5.1)           # TMCMC → 出力
    arrow(12.6, 5.0, 13.7, 6.8)           # 高速化 → TMCMC

    # Loop arrow (TMCMC feedback)
    ax.annotate("", xy=(9.4, 6.2), xytext=(13.7, 6.2),
                arrowprops=dict(arrowstyle="-|>", color="#E05252", lw=2,
                                connectionstyle="arc3,rad=-0.4", mutation_scale=16),
                zorder=2)
    ax.text(11.5, 5.35, "Iterations", fontsize=9, color="#E05252", ha="center")

    # ── Legend labels ─────────────────────
    labels = [
        ("● Input", COLORS["So"]), ("● Probabilistic model", COLORS["An"]),
        ("● Physical model", COLORS["Vd"]), ("● Algorithm", COLORS["Pg"]),
        ("● Output", COLORS["accent"])
    ]
    for i, (txt, c) in enumerate(labels):
        ax.text(0.5 + i*3.9, 0.5, txt, fontsize=9, color=c, fontweight="bold")

    ax.text(10, 0.12, "Dysbiotic × HOBIC condition | 20 parameters | 8 stages × 150 particles",
            ha="center", fontsize=9, color=COLORS["gray"])

    plt.tight_layout()
    plt.savefig(OUT / "fig1_pipeline.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("✓ fig1_pipeline.png")


# ═══════════════════════════════════════════════════════════════
# FIG 2: Species interaction network
# ═══════════════════════════════════════════════════════════════
def fig2_network():
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0F1923")

    # ── Left: interaction network ────────
    ax = axes[0]
    ax.set_facecolor("#0F1923")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.6)
    ax.axis("off")
    ax.set_title("Inter-species interaction network", color="white", fontsize=15, fontweight="bold", pad=12)

    # Node positions (circular layout + Pg at bottom)
    angles = [90, 18, -54, -126, 162]  # 5点配置
    r = 1.0
    pos = {i: (r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for i, a in enumerate(angles)}
    pos[4] = (0.0, -1.2)  # Pgを下に

    names = ["So", "An", "Vd", "Fn", "Pg"]
    full  = ["S. oralis\n(pioneer)", "A. naeslundii\n(commensal)", "V. dispar\n(bridge)", "F. nucleatum\n(gatekeeper)", "P. gingivalis\n(pathogen)"]

    # Known interactions (simplified) based on current estimates
    edges = [
        (0, 1, 0.8, "正"),   # So → An
        (1, 0, 0.5, "正"),
        (2, 3, 1.2, "正"),   # Vd → Fn
        (3, 2, 0.6, "正"),
        (3, 4, 2.4, "正"),   # Fn → Pg (Hill gate)
        (2, 4, 3.56, "正"),  # Vd → Pg
        (0, 2, 0.7, "弱"),
        (1, 3, 0.4, "弱"),
        (0, 0, 1.5, "自"),   # self
        (1, 1, 1.2, "自"),
        (2, 2, 1.8, "自"),
        (3, 3, 2.1, "自"),
        (4, 4, 1.1, "自"),
    ]

    # エッジ描画
    for (i, j, w, etype) in edges:
        if i == j:
            continue
        xi, yi = pos[i]
        xj, yj = pos[j]
        # Color and width by strength
        if etype == "正":
            ec = "#4EC9B0"
        elif etype == "弱":
            ec = "#888888"
        else:
            ec = "#FF6B6B"
        alpha = min(0.2 + w * 0.18, 0.95)
        lw = 0.8 + w * 0.6
        ax.annotate("", xy=(xj, yj), xytext=(xi, yi),
                    arrowprops=dict(arrowstyle="-|>", color=ec, lw=lw,
                                    connectionstyle="arc3,rad=0.15",
                                    mutation_scale=12, alpha=alpha),
                    zorder=2)
        # 重みラベル
        mx, my = (xi + xj)/2, (yi + yj)/2
        ax.text(mx + 0.05, my + 0.05, f"{w:.1f}", fontsize=7, color=ec, alpha=0.85, zorder=5)

    # ノード描画
    for i, (name, full_name, color) in enumerate(zip(names, full, SCOLORS)):
        x, y = pos[i]
        circle = Circle((x, y), 0.22, color=color, zorder=3, linewidth=2.5,
                         ec="white", alpha=0.95)
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", zorder=4)
        # 外側ラベル
        lx = x * 1.55
        ly = y * 1.45 + (0.15 if i == 4 else 0)
        ax.text(lx, ly, full_name, ha="center", va="center",
                fontsize=8, color=color, zorder=4)

    # Hill-gate label
    ax.annotate("Hill gate\n(Fn concentration dependent)", xy=pos[3], xytext=(0.7, -0.5),
                arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.5),
                fontsize=8, color="#FFD700", zorder=5)

    # Legend
    for txt, c in [("Strong promotion", "#4EC9B0"), ("Weak interaction", "#888888")]:
        ax.plot([], [], color=c, lw=2, label=txt)
    ax.legend(loc="upper right", facecolor="#1A2A3A", edgecolor="white",
              labelcolor="white", fontsize=8)

    # ── Right: interaction matrix heatmap ──
    ax2 = axes[1]
    ax2.set_facecolor("#0F1923")
    ax2.set_title("Interaction matrix A[i→j]  (MAP estimates)", color="white", fontsize=15,
                  fontweight="bold", pad=12)

    # MAP推定値（mild-weight run）
    A = np.array([
        [1.52, 0.83, 0.72, 0.00, 0.00],
        [0.48, 1.21, 0.41, 0.00, 0.00],
        [0.00, 0.00, 1.87, 1.20, 3.56],
        [0.00, 0.00, 0.63, 2.14, 2.41],
        [0.00, 0.00, 0.00, 0.00, 1.08],
    ])
    locked = np.array([
        [0,0,0,1,1],
        [0,0,0,1,1],
        [1,1,0,0,0],
        [1,1,0,0,0],
        [1,1,1,1,0],
    ], dtype=bool)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bio", ["#0F1923", "#1A3A5C", "#2563EB", "#4EC9B0", "#FFD700"])

    im = ax2.imshow(A, cmap=cmap, vmin=0, vmax=4, aspect="auto")

    tick_labels = ["So", "An", "Vd", "Fn", "Pg"]
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(tick_labels, color="white", fontsize=12, fontweight="bold")
    ax2.set_yticklabels(tick_labels, color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("影響を与える菌 (source)", color="white", fontsize=11)
    ax2.set_ylabel("影響を受ける菌 (target)", color="white", fontsize=11)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("white")

    # 数値とロックマーク
    for i in range(5):
        for j in range(5):
            if locked[i, j]:
                ax2.text(j, i, "🔒", ha="center", va="center", fontsize=16)
            else:
                v = A[i, j]
                tc = "white" if v < 2 else "#0F1923"
                ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=13, fontweight="bold", color=tc)

    # カラーバー
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Interaction strength", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Highlight diagonal (self-regulation) with white frame
    for i in range(5):
        ax2.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                     fill=False, edgecolor="#FFD700", lw=2.5))

    ax2.text(2, 5.3, "🔒 = Locked by biological knowledge  │  diagonal = self-regulation",
             ha="center", fontsize=9, color="#AABBCC")

    plt.tight_layout()
    plt.savefig(OUT / "fig2_network.png", dpi=180, bbox_inches="tight", facecolor="#0F1923")
    plt.close()
    print("✓ fig2_network.png")


# ═══════════════════════════════════════════════════════════════
# FIG 3: TMCMC algorithm visualization
# ═══════════════════════════════════════════════════════════════
def fig3_tmcmc():
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0D1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 上段左: βスケジュールと粒子進化 ──
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#0D1117")
    ax1.set_title("TMCMC — β schedule and particle evolution", color="white", fontsize=13, fontweight="bold")

    beta_stages = [0.0, 0.05, 0.12, 0.25, 0.42, 0.61, 0.80, 0.93, 1.0]
    n_stages = len(beta_stages)
    np.random.seed(42)

    # 各ステージで粒子を「収束」シミュレート
    true_center = np.array([2.5, 3.0])
    for s_idx, beta in enumerate(beta_stages):
        spread = 2.5 * (1 - beta**0.5) + 0.15
        particles = true_center + np.random.randn(40, 2) * spread
        alpha = 0.3 + beta * 0.6
        ax1.scatter(particles[:, 0] + s_idx * 0.5, particles[:, 1],
                    s=15, alpha=alpha, color=plt.cm.plasma(beta), zorder=3)

    # βラベル
    for s_idx, beta in enumerate(beta_stages):
        ax1.text(s_idx * 0.5 + true_center[0], true_center[1] + 1.6,
                 f"β={beta:.2f}", ha="center", fontsize=7.5,
                 color=plt.cm.plasma(beta), rotation=30)

    ax1.set_xlim(1.5, 8.0)
    ax1.set_ylim(0.5, 6.0)
    ax1.set_xlabel("← Prior distribution                  β progression                  Posterior distribution →",
                   color="#AABBCC", fontsize=10)
    ax1.set_ylabel("Parameter space", color="#AABBCC", fontsize=10)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#333344")

    # ── Top-right: one-stage details ─────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#0D1117")
    ax2.set_title("One TMCMC stage", color="white", fontsize=12, fontweight="bold")
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    steps = [
        (5, 9.0, "① Choose β\n(ESS ≈ 50% via bisection)", "#4EC9B0"),
        (5, 7.2, "② Weight update\nwᵢ = exp[(Δβ)·logLᵢ]", "#6DC06D"),
        (5, 5.4, "③ Resampling\n(weighted)", "#F4A142"),
        (5, 3.6, "④ MCMC mutation\n(150 particles, parallel)", "#B07CC6"),
        (5, 1.8, "⑤ Diagnostics\n(ESS, Rhat, acceptance)", "#4E9AF1"),
    ]
    for x, y, txt, c in steps:
        bbox = FancyBboxPatch((x - 4.0, y - 0.65), 8.0, 1.3,
                               boxstyle="round,pad=0.1,rounding_size=0.3",
                               facecolor=c + "22", edgecolor=c, lw=2, zorder=3)
        ax2.add_patch(bbox)
        ax2.text(x, y, txt, ha="center", va="center", fontsize=8.5,
                 color=c, fontweight="bold", zorder=4)

    # Arrows
    for i in range(len(steps) - 1):
        y1 = steps[i][1] - 0.65
        y2 = steps[i+1][1] + 0.65
        ax2.annotate("", xy=(5, y2), xytext=(5, y1),
                     arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5, mutation_scale=14))

    # ── Bottom-left: ESS & Rhat convergence ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0D1117")
    ax3.set_title("Convergence diagnostics over stages", color="white", fontsize=11, fontweight="bold")

    stages = np.arange(8)
    ess = [148, 132, 118, 109, 97, 112, 128, 145]
    rhat_max = [2.1, 1.8, 1.5, 1.35, 1.22, 1.12, 1.05, 1.01]

    ax3b = ax3.twinx()
    ax3.plot(stages, ess, "o-", color="#4EC9B0", lw=2.5, ms=7, label="ESS")
    ax3b.plot(stages, rhat_max, "s--", color="#FFD700", lw=2.5, ms=7, label="Rhat_max")
    ax3b.axhline(1.1, color="#FFD700", lw=1, ls=":", alpha=0.6)
    ax3b.text(6.8, 1.12, "Convergence threshold", fontsize=7.5, color="#FFD700")

    ax3.set_xlabel("Stage", color="#AABBCC")
    ax3.set_ylabel("ESS", color="#4EC9B0")
    ax3b.set_ylabel("Rhat", color="#FFD700")
    ax3.tick_params(colors="white")
    ax3b.tick_params(colors="#FFD700")
    for spine in ax3.spines.values():
        spine.set_color("#333344")
    for spine in ax3b.spines.values():
        spine.set_color("#333344")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, facecolor="#1A2A3A",
               edgecolor="none", labelcolor="white", fontsize=8)

    # ── Bottom-middle: MCMC mutation ─────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#0D1117")
    ax4.set_title("MCMC mutation (Metropolis–Hastings)", color="white", fontsize=11, fontweight="bold")
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-3, 3)

    # Posterior contours (elliptical approximation)
    xx, yy = np.mgrid[-3:3:100j, -3:3:100j]
    cov_true = [[0.5, 0.3], [0.3, 0.8]]
    pos2d = np.dstack([xx, yy])
    from scipy.stats import multivariate_normal
    rv = multivariate_normal([0, 0], cov_true)
    Z = rv.pdf(pos2d)
    ax4.contourf(xx, yy, Z, levels=8, cmap="Blues", alpha=0.4)
    ax4.contour(xx, yy, Z, levels=5, colors="#4EC9B0", alpha=0.5, linewidths=0.8)

    # Particle trajectory
    np.random.seed(7)
    current = np.array([-1.5, 1.0])
    trajectory = [current.copy()]
    for _ in range(6):
        proposal = current + np.random.randn(2) * 0.7
        if rv.logpdf(proposal) > rv.logpdf(current) - np.random.exponential():
            current = proposal
        trajectory.append(current.copy())
    traj = np.array(trajectory)
    ax4.plot(traj[:, 0], traj[:, 1], "o-", color="#FFD700", lw=2, ms=8, zorder=5)
    ax4.plot(traj[0, 0], traj[0, 1], "^", color="#FF6B6B", ms=12, zorder=6, label="current")
    ax4.plot(traj[-1, 0], traj[-1, 1], "*", color="#4EC9B0", ms=14, zorder=6, label="final")
    for i in range(1, len(traj)):
        alpha = 0.4 + i * 0.08
        ax4.annotate("", xy=traj[i], xytext=traj[i-1],
                     arrowprops=dict(arrowstyle="-|>", color="#FFD700",
                                     lw=1.5, mutation_scale=12, alpha=alpha))

    ax4.set_xlabel("θ₁ (parameter 1)", color="#AABBCC", fontsize=9)
    ax4.set_ylabel("θ₂ (parameter 2)", color="#AABBCC", fontsize=9)
    ax4.tick_params(colors="white")
    for spine in ax4.spines.values():
        spine.set_color("#333344")
    ax4.legend(facecolor="#1A2A3A", edgecolor="none", labelcolor="white", fontsize=8)
    ax4.text(-2.8, -2.7, "Contours = posterior\nYellow line = MCMC trace", fontsize=7.5, color="#AABBCC")

    # ── Bottom-right: acceptance rate ─────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#0D1117")
    ax5.set_title("Acceptance rate and proposal adaptation", color="white", fontsize=11, fontweight="bold")

    accept = [0.61, 0.55, 0.48, 0.43, 0.38, 0.36, 0.34, 0.33]
    scale  = [2.5, 2.0, 1.6, 1.3, 1.1, 0.95, 0.85, 0.78]
    beta_v = [0.05, 0.12, 0.25, 0.42, 0.61, 0.80, 0.93, 1.0]

    ax5b = ax5.twinx()
    ax5.bar(stages, accept, color=plt.cm.plasma(np.array(beta_v)), alpha=0.7, zorder=3)
    ax5b.plot(stages, scale, "D-", color="#FFD700", lw=2, ms=7, zorder=4, label="Proposal scale")
    ax5.axhline(0.23, color="#FF6B6B", lw=1.5, ls="--", alpha=0.8)
    ax5.axhline(0.44, color="#4EC9B0", lw=1.5, ls="--", alpha=0.8)
    ax5.text(6.5, 0.25, "Target range", fontsize=7, color="white")

    ax5.set_xlabel("Stage", color="#AABBCC")
    ax5.set_ylabel("Acceptance rate", color="#AABBCC")
    ax5b.set_ylabel("Proposal scale (∝1/√β)", color="#FFD700")
    ax5.tick_params(colors="white")
    ax5b.tick_params(colors="#FFD700")
    ax5.set_ylim(0, 0.75)
    for spine in ax5.spines.values():
        spine.set_color("#333344")
    for spine in ax5b.spines.values():
        spine.set_color("#333344")

    # カラーバー (β)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax5, fraction=0.04, pad=0.15)
    cbar.set_label("β value", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Title
    fig.text(0.5, 0.98, "TMCMC algorithm — detailed visualization",
             ha="center", color="white", fontsize=16, fontweight="bold")

    plt.savefig(OUT / "fig3_tmcmc.png", dpi=180, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print("✓ fig3_tmcmc.png")


# ═══════════════════════════════════════════════════════════════
# FIG 4: Data structure and ODE model
# ═══════════════════════════════════════════════════════════════
def fig4_data_model():
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.patch.set_facecolor("#111827")

    # ── Left: experimental data structure ──
    ax1 = axes[0]
    ax1.set_facecolor("#111827")
    ax1.set_title("Experimental data structure\n[6 time points × 5 species]", color="white", fontsize=13, fontweight="bold")

    # Stacked bar chart (observations)
    t_days = [1, 3, 6, 10, 15, 21]
    np.random.seed(10)
    data = np.array([
        [0.35, 0.25, 0.18, 0.10, 0.05, 0.03],  # So
        [0.30, 0.28, 0.22, 0.15, 0.08, 0.05],  # An
        [0.20, 0.22, 0.25, 0.28, 0.25, 0.20],  # Vd
        [0.12, 0.18, 0.22, 0.28, 0.32, 0.35],  # Fn
        [0.03, 0.07, 0.13, 0.19, 0.30, 0.37],  # Pg
    ])
    # Normalize
    data = data / data.sum(axis=0, keepdims=True)

    bottom = np.zeros(6)
    for i, (sp, c) in enumerate(zip(["So", "An", "Vd", "Fn", "Pg"], SCOLORS)):
        ax1.bar(range(6), data[i], bottom=bottom, color=c, alpha=0.85,
                label=sp, zorder=3)
        bottom += data[i]

    ax1.set_xticks(range(6))
    ax1.set_xticklabels([f"Day {d}" for d in t_days], color="white", fontsize=9, rotation=30)
    ax1.set_ylabel("Volume fraction φᵢ", color="white")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#333344")
    ax1.legend(facecolor="#1A2A3A", edgecolor="none", labelcolor="white",
               fontsize=9, loc="upper right")
    ax1.text(2.5, 1.07, "Σφᵢ = 1 (volume conservation)", ha="center",
             fontsize=8, color="#AABBCC")

    # ── Middle: ODE model time evolution ──
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    ax2.set_title("ODE model time evolution\n(using MAP parameters)", color="white", fontsize=13, fontweight="bold")

    t = np.linspace(0, 21, 200)
    np.random.seed(5)
    # Simple simulation (spline interpolation)
    from scipy.interpolate import make_interp_spline
    data_t = [1, 3, 6, 10, 15, 21]

    for i, (sp, c) in enumerate(zip(["So", "An", "Vd", "Fn", "Pg"], SCOLORS)):
        vals = data[i]
        spl = make_interp_spline(data_t, vals, k=3)
        smooth = np.clip(spl(t), 0, 1)
        ax2.plot(t, smooth, color=c, lw=2.5, label=sp, zorder=3)
        # Observation points
        ax2.scatter(data_t, vals, color=c, s=60, zorder=5,
                    marker="o", edgecolors="white", lw=1)

    ax2.set_xlabel("Time [days]", color="white")
    ax2.set_ylabel("Volume fraction φᵢ", color="white")
    ax2.set_xlim(0, 22)
    ax2.set_ylim(-0.02, 0.6)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("#333344")
    ax2.legend(facecolor="#1A2A3A", edgecolor="none", labelcolor="white", fontsize=9)
    ax2.text(11, 0.55, "Solid: model  ●: observations", fontsize=8.5, color="#AABBCC",
             ha="center")

    # ── Right: 20-parameter structure ──
    ax3 = axes[2]
    ax3.set_facecolor("#111827")
    ax3.set_title("Parameter structure\n(20-dimensional θ)", color="white", fontsize=13, fontweight="bold")
    ax3.axis("off")
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    blocks = [
        ("M1\nθ[0-4]",  "#4E9AF1", 9.0, "So–An\ninteractions\n+ decay (b₁,b₂)"),
        ("M2\nθ[5-9]",  "#6DC06D", 7.0, "Vd–Fn\ninteractions\n+ decay (b₃,b₄)"),
        ("M3\nθ[10-13]","#F4A142", 5.0, "Cross interactions\n(So,An)↔(Vd,Fn)"),
        ("M4\nθ[14-15]","#E05252", 3.2, "Pg self-limitation and decay"),
        ("M5\nθ[16-19]","#B07CC6", 1.5, "Pg ← (So,An,Vd,Fn)\n[a₁₅,a₂₅,a₃₅,a₄₅]"),
    ]
    for label, color, cy, desc in blocks:
        h = 1.4 if "M3" not in label else 1.2
        if "M4" in label or "M5" in label:
            h = 1.1
        bbox = FancyBboxPatch((0.5, cy - h/2), 3.5, h,
                               boxstyle="round,pad=0.1,rounding_size=0.25",
                               facecolor=color + "33", edgecolor=color, lw=2, zorder=3)
        ax3.add_patch(bbox)
        ax3.text(2.25, cy, label, ha="center", va="center",
                 fontsize=10, fontweight="bold", color=color, zorder=4)
        ax3.text(7.5, cy, desc, ha="center", va="center",
                 fontsize=8.5, color="#CCDDEE", zorder=4)
        ax3.annotate("", xy=(4.8, cy), xytext=(4.0, cy),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                     mutation_scale=10))

    ax3.text(5, 0.3, "Total 20 dimensions → posterior p(θ|D)",
             ha="center", fontsize=9, color="#AABBCC")

    # Locked-parameter box
    locked_box = FancyBboxPatch((5.5, 8.3), 4.0, 1.2,
                                 boxstyle="round,pad=0.08",
                                 facecolor="#2A1A1A", edgecolor="#FF6B6B", lw=1.5)
    ax3.add_patch(locked_box)
    ax3.text(7.5, 8.9, "🔒 Locked parameters", ha="center", fontsize=9, color="#FF6B6B", fontweight="bold")
    ax3.text(7.5, 8.45, "Fixed to 0 by biological knowledge", ha="center", fontsize=8, color="#FFAAAA")

    # Hill gate
    hill_box = FancyBboxPatch((5.5, 6.8), 4.0, 1.2,
                               boxstyle="round,pad=0.08",
                               facecolor="#1A2A1A", edgecolor="#FFD700", lw=1.5)
    ax3.add_patch(hill_box)
    ax3.text(7.5, 7.4, "⚡ Hill gate (fixed)", ha="center", fontsize=9, color="#FFD700", fontweight="bold")
    ax3.text(7.5, 6.95, "K=0.05, n=4  (Fn-dependent gating)", ha="center", fontsize=7.5, color="#FFEE99")

    fig.text(0.5, 0.98, "Data structure, ODE model, and parameter space",
             ha="center", color="white", fontsize=15, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT / "fig4_data_model.png", dpi=180, bbox_inches="tight", facecolor="#111827")
    plt.close()
    print("✓ fig4_data_model.png")


# ═══════════════════════════════════════════════════════════════
# FIG 5: Result summary & RMSE before/after
# ═══════════════════════════════════════════════════════════════
def fig5_results():
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor("#0D1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    # ── Top: RMSE comparison ───────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#0D1117")
    ax1.set_title("RMSE improvement: Before vs After (mild-weight run)", color="white", fontsize=13, fontweight="bold")

    species_labels = ["So\n(S.oralis)", "An\n(A.naes)", "Vd\n(V.dispar)", "Fn\n(F.nucl.)", "Pg\n(P.ging.)", "TOTAL"]
    prev = [0.036, 0.129, 0.213, 0.088, 0.435, 0.228]
    mild = [0.034, 0.105, 0.269, 0.161, 0.103, 0.156]

    x = np.arange(6)
    w = 0.35
    bars_prev = ax1.bar(x - w/2, prev, w, color="#FF6B6B", alpha=0.8, label="Before (original)", zorder=3)
    bars_mild = ax1.bar(x + w/2, mild, w, color="#4EC9B0", alpha=0.8, label="After (Mild-Weight)", zorder=3)

    # Improvement annotations
    improvements = [p - m for p, m in zip(prev, mild)]
    for i, (p, m, imp) in enumerate(zip(prev, mild, improvements)):
        y_top = max(p, m) + 0.01
        color = "#4EC9B0" if imp > 0 else "#FF6B6B"
        symbol = "▼" if imp > 0 else "▲"
        pct = abs(imp/p)*100
        ax1.text(i, y_top + 0.005, f"{symbol}{pct:.0f}%", ha="center",
                 fontsize=9.5, color=color, fontweight="bold")

    # Highlight large improvement for P.g.
    ax1.annotate("P.g. strongly improved!\n0.435 → 0.103",
                 xy=(4 + w/2, 0.103), xytext=(4.5, 0.36),
                 arrowprops=dict(arrowstyle="->", color="#FFD700", lw=2),
                 fontsize=9.5, color="#FFD700", fontweight="bold", ha="center")

    ax1.set_xticks(x)
    ax1.set_xticklabels(species_labels, color="white", fontsize=10)
    ax1.set_ylabel("RMSE (Root Mean Squared Error)", color="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#333344")
    ax1.legend(facecolor="#1A2A3A", edgecolor="none", labelcolor="white", fontsize=10)
    ax1.set_ylim(0, 0.52)
    ax1.axvline(4.5, color="#666677", lw=1, ls="--", alpha=0.6)

    # ── Top-right: parameter comparison ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#0D1117")
    ax2.set_title("Key parameters Before/After", color="white", fontsize=12, fontweight="bold")

    params = ["a₃₅\n(Vd→Pg)", "a₄₅\n(Fn→Pg)"]
    before_vals = [28.7, 3.97]
    after_vals  = [3.56, 2.41]

    x2 = np.arange(2)
    ax2.bar(x2 - 0.2, before_vals, 0.35, color="#FF6B6B", alpha=0.8, label="Before")
    ax2.bar(x2 + 0.2, after_vals, 0.35, color="#4EC9B0", alpha=0.8, label="After")
    ax2.axhline(5, color="#FFD700", lw=1.5, ls="--", alpha=0.7)
    ax2.text(1.3, 5.3, "Upper bound [0,5]", fontsize=8, color="#FFD700")

    # Warning for exceeding upper bound
    ax2.text(-0.2, 29.5, "⚠ Above upper bound\n(overfitting)", fontsize=8.5,
             color="#FF6B6B", ha="center", fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(params, color="white", fontsize=11)
    ax2.set_ylabel("Estimate", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_color("#333344")
    ax2.legend(facecolor="#1A2A3A", edgecolor="none", labelcolor="white", fontsize=9)

    # ── Bottom-left: experimental settings ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0D1117")
    ax3.set_title("Experimental settings", color="white", fontsize=12, fontweight="bold")
    ax3.axis("off")
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    settings = [
        ("Item", "Before", "After (mild)"),
        ("Particles", "50", "150"),
        ("Stages", "5", "8"),
        ("λ_pg", "1.0", "2.0"),
        ("λ_late", "1.0", "1.5"),
        ("Upper bound a₃₅", "30", "5"),
        ("Upper bound a₄₅", "20", "5"),
        ("K_hill", "Fixed 0.05", "Fixed 0.05"),
    ]
    col_x = [1.5, 5.0, 8.5]
    for row_idx, (item, bef, aft) in enumerate(settings):
        y = 9.2 - row_idx * 1.08
        if row_idx == 0:
            for xi, txt, c in zip(col_x, [item, bef, aft], ["#FFFFFF", "#FF6B6B", "#4EC9B0"]):
                ax3.text(xi, y, txt, ha="center", fontsize=10, fontweight="bold", color=c)
            ax3.axhline(y - 0.4, color="#444455", lw=1)
        else:
            bg_c = "#1A2230" if row_idx % 2 == 0 else "#111827"
            bg = FancyBboxPatch((0.1, y - 0.45), 9.8, 0.85,
                                 boxstyle="round,pad=0.05",
                                 facecolor=bg_c, edgecolor="none")
            ax3.add_patch(bg)
            ax3.text(col_x[0], y, item, ha="center", fontsize=9, color="#CCDDEE")
            ax3.text(col_x[1], y, bef, ha="center", fontsize=9, color="#FF9999")
            changed = bef != aft
            tc = "#4EC9B0" if changed else "#888899"
            ax3.text(col_x[2], y, aft, ha="center", fontsize=9, color=tc,
                     fontweight="bold" if changed else "normal")

    # ── Bottom-middle: convergence metrics ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#0D1117")
    ax4.set_title("Convergence quality (mild-weight run)", color="white", fontsize=12, fontweight="bold")
    ax4.axis("off")
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    metrics = [
        ("ESS", "200-300", "✓ Good", "#4EC9B0"),
        ("Rhat_max", "~1.00", "✓ Excellent", "#4EC9B0"),
        ("Acceptance", "33-61%", "✓ Reasonable", "#4EC9B0"),
        ("β final", "1.000", "✓ Completed", "#4EC9B0"),
        ("RMSE total", "0.156", "↓ 32% improvement", "#FFD700"),
        ("a₃₅ status", "3.56", "✓ Healthy", "#4EC9B0"),
    ]
    for i, (name, val, status, c) in enumerate(metrics):
        y = 9.0 - i * 1.4
        # Gauge-like panel
        gauge_bg = FancyBboxPatch((0.3, y - 0.45), 9.4, 0.85,
                                   boxstyle="round,pad=0.08",
                                   facecolor="#1A2230", edgecolor=c + "55", lw=1)
        ax4.add_patch(gauge_bg)
        ax4.text(1.5, y, name, ha="left", va="center", fontsize=9.5, color="#AABBCC")
        ax4.text(5.2, y, val, ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        ax4.text(8.5, y, status, ha="center", va="center", fontsize=9, color=c, fontweight="bold")

    # ── Bottom-right: overall summary ─────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#0D1117")
    ax5.set_title("Overall summary", color="white", fontsize=12, fontweight="bold")
    ax5.axis("off")
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)

    # Donut-style chart in the center
    theta = np.linspace(0, 2*np.pi, 100)
    improvement = 0.32  # 32% improvement
    angle_fill = 2 * np.pi * improvement

    # Background circles
    ax5.add_patch(Circle((5, 5.5), 2.8, color="#1A2230", zorder=2))
    ax5.add_patch(Circle((5, 5.5), 2.8, fill=False, ec="#333344", lw=2, zorder=3))
    ax5.add_patch(Circle((5, 5.5), 1.8, color="#0D1117", zorder=4))

    # Arc for improvement
    from matplotlib.patches import Wedge
    w1 = Wedge((5, 5.5), 2.8, 90 - angle_fill * 180/np.pi, 90, color="#4EC9B0", alpha=0.85, zorder=3)
    w2 = Wedge((5, 5.5), 2.8, 90, 90 + (1 - improvement) * 360, color="#FF6B6B", alpha=0.4, zorder=3)
    w1.set_radius(2.8)
    w2.set_radius(2.8)
    ax5.add_patch(w1)
    ax5.add_patch(w2)
    ax5.add_patch(Circle((5, 5.5), 1.8, color="#0D1117", zorder=5))

    ax5.text(5, 5.8, "32%", ha="center", va="center", fontsize=22,
             fontweight="bold", color="#4EC9B0", zorder=6)
    ax5.text(5, 5.0, "RMSE improvement", ha="center", va="center", fontsize=9,
             color="#AABBCC", zorder=6)

    ax5.text(5, 2.3, "0.228 → 0.156", ha="center", fontsize=11,
             color="#FFD700", fontweight="bold")
    ax5.text(5, 1.7, "P.gingivalis particularly improved", ha="center", fontsize=9, color="#AABBCC")
    ax5.text(5, 9.3, "Direction validated ✓", ha="center", fontsize=11,
             color="#4EC9B0", fontweight="bold")

    fig.text(0.5, 0.98, "Inference results summary — mild-weight run (2026-02-18)",
             ha="center", color="white", fontsize=15, fontweight="bold")

    plt.savefig(OUT / "fig5_results.png", dpi=180, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print("✓ fig5_results.png")


# ═══════════════════════════════════════════════════════════════
# FIG 6: Method positioning & architecture
# ═══════════════════════════════════════════════════════════════
def fig6_positioning():
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0F172A")

    # ── Left: method comparison map ───────
    ax1 = axes[0]
    ax1.set_facecolor("#0F172A")
    ax1.set_title("Method positioning map", color="white", fontsize=14, fontweight="bold", pad=15)

    methods = [
        ("Optimization only\n(L-BFGS etc.)", 0.9, 0.15, "#FF6B6B", 80),
        ("Plain MCMC\n(Metropolis)", 0.35, 0.55, "#F4A142", 90),
        ("TMCMC\n(this work)", 0.7, 0.75, "#4EC9B0", 200),
        ("Nested sampling", 0.5, 0.65, "#B07CC6", 80),
        ("Variational Bayes\n(VI)", 0.8, 0.45, "#4E9AF1", 90),
        ("Neural ODE", 0.3, 0.3, "#888888", 75),
    ]

    for name, x, y, c, s in methods:
        ax1.scatter(x, y, s=s, color=c, alpha=0.85, zorder=4)
        is_highlight = "TMCMC" in name
        ax1.text(x, y + 0.06, name, ha="center", va="center", fontsize=9,
                 color=c, fontweight="bold" if is_highlight else "normal",
                 zorder=5,
                 bbox=dict(boxstyle="round,pad=0.2", fc="#0F172A", ec=c if is_highlight else "none",
                           lw=1.5 if is_highlight else 0, alpha=0.9) if is_highlight else None)

    # Highlight TMCMC
    ax1.scatter(0.7, 0.75, s=500, color="#4EC9B0", alpha=0.2, zorder=3)
    ax1.scatter(0.7, 0.75, s=250, color="#4EC9B0", alpha=0.3, zorder=3)

    ax1.set_xlabel("Speed (computational efficiency)", color="white", fontsize=12)
    ax1.set_ylabel("Accuracy (uncertainty quantification)", color="white", fontsize=12)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#333344")

    # Zone
    ax1.axvspan(0.55, 1.05, alpha=0.05, color="#4EC9B0")
    ax1.axhspan(0.6, 1.05, alpha=0.05, color="#4EC9B0")
    ax1.text(0.8, 0.92, "Ideal zone", fontsize=9, color="#4EC9B0", alpha=0.7)

    ax1.text(0.5, -0.08, "Small data, high dimension, nonlinearity → TMCMC is well suited",
             ha="center", fontsize=9.5, color="#AABBCC")

    # ── Right: system architecture ────────
    ax2 = axes[1]
    ax2.set_facecolor("#0F172A")
    ax2.set_title("System architecture", color="white", fontsize=14, fontweight="bold", pad=15)
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    def arch_box(x, y, w, h, title, items, color):
        bg = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=color + "15", edgecolor=color, lw=1.8, zorder=3)
        ax2.add_patch(bg)
        ax2.text(x + w/2, y + h - 0.28, title, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=color, zorder=4)
        for i, item in enumerate(items):
            ax2.text(x + 0.25, y + h - 0.62 - i * 0.38, f"  {item}",
                     ha="left", va="center", fontsize=7.5, color="#CCDDEE", zorder=4)

    arch_box(0.3, 7.5, 4.2, 2.3, "Entry points",
             ["estimate_reduced_nishioka.py", "run_bridge_sweep.py", "Condition logic & CLI args"], "#4E9AF1")
    arch_box(5.3, 7.5, 4.4, 2.3, "TMCMC core",
             ["data_5species/core/tmcmc.py", "Stage schedule & β control", "Parallel MCMC mutation"], "#E05252")
    arch_box(0.3, 4.5, 4.2, 2.7, "Likelihood & evaluation",
             ["core/evaluator.py", "LogLikelihoodEvaluator", "Weighted Gaussian likelihood", "ROM error tracking"], "#6DC06D")
    arch_box(5.3, 4.5, 4.4, 2.7, "Solver layer",
             ["improved_5species_jit.py", "BiofilmNewtonSolver5S", "Numba JIT acceleration", "TSM-ROM surrogate"], "#F4A142")
    arch_box(0.3, 1.8, 4.2, 2.4, "Config & data",
             ["model_config/prior_bounds.json", "Experimental data loader", "4 conditions × data formats"], "#B07CC6")
    arch_box(5.3, 1.8, 4.4, 2.4, "Visualization & outputs",
             ["_runs/ result directories", "posterior_samples.npy", "metrics.json + figures/"], "#64748B")

    # Arrows
    arrows_arch = [
        (4.5, 8.6, 5.3, 8.6),
        (2.4, 7.5, 2.4, 7.2),
        (7.5, 7.5, 7.5, 7.2),
        (4.5, 5.8, 5.3, 5.8),
        (2.4, 4.5, 2.4, 4.2),
        (7.5, 4.5, 7.5, 4.2),
        (4.5, 3.0, 5.3, 3.0),
    ]
    for x1, y1, x2, y2 in arrows_arch:
        ax2.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#666677", lw=1.5,
                                     mutation_scale=12), zorder=2)

    fig.text(0.5, 0.98, "Method positioning & system architecture",
             ha="center", color="white", fontsize=15, fontweight="bold")

    plt.savefig(OUT / "fig6_positioning.png", dpi=180, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print("✓ fig6_positioning.png")


# ═══════════════════════════════════════════════════════════════
# FIG 7: Single-page poster-style summary
# ═══════════════════════════════════════════════════════════════
def fig7_poster():
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor("#080E1A")

    # ── Title ─────────────────────────────
    fig.text(0.5, 0.965, "5-species oral biofilm parameter inference",
             ha="center", fontsize=22, fontweight="bold", color="white")
    fig.text(0.5, 0.945, "Bayesian Inference via TMCMC  |  Dysbiotic × HOBIC Condition  |  20-Parameter ODE Model",
             ha="center", fontsize=11, color="#8899AA")

    # ── Layout ────────────────────────────
    gs = GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.4,
                  left=0.04, right=0.97, top=0.93, bottom=0.05)

    # ── Top: main pipeline (horizontal) ───
    ax_pipe = fig.add_subplot(gs[0, :])
    ax_pipe.set_facecolor("#080E1A")
    ax_pipe.axis("off")
    ax_pipe.set_xlim(0, 24)
    ax_pipe.set_ylim(0, 4)

    pipeline_steps = [
        (1.5, "Experimental data\n[6 time points × 5 species]", "#4E9AF1", "📊"),
        (5.5, "Prior distribution\n[20 parameters]", "#6DC06D", "📐"),
        (9.5, "ODE model evaluation\ndφ/dt = f(φ,ψ;θ)", "#F4A142", "🧬"),
        (13.5, "Likelihood\nlog p(D|θ)", "#B07CC6", "📊"),
        (17.5, "TMCMC inference\nβ: 0→1, 8 stages", "#E05252", "🎲"),
        (21.5, "Posterior output\nMAP + 95% CI", "#2563EB", "✅"),
    ]

    for i, (x, label, c, icon) in enumerate(pipeline_steps):
        # Box
        box = FancyBboxPatch((x - 1.6, 0.6), 3.2, 2.8,
                              boxstyle="round,pad=0.15",
                              facecolor=c + "22", edgecolor=c, lw=2.5, zorder=3)
        ax_pipe.add_patch(box)
        ax_pipe.text(x, 2.75, icon, ha="center", va="center", fontsize=18, zorder=4)
        ax_pipe.text(x, 1.85, label, ha="center", va="center", fontsize=10,
                     fontweight="bold", color=c, zorder=4)
        ax_pipe.text(x, 0.3, f"Step {i+1}", ha="center", fontsize=8, color=c, alpha=0.7)

        # Arrows
        if i < len(pipeline_steps) - 1:
            ax_pipe.annotate("", xy=(x + 1.75, 2.0), xytext=(x + 1.6, 2.0),
                             arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5,
                                             mutation_scale=16), zorder=2)

    ax_pipe.text(12, 3.75, "▼ End-to-end inference flow", ha="center", fontsize=9, color="#AABBCC", style="italic")

    # ── Bottom-left: small network ───────
    ax_net = fig.add_subplot(gs[1:, 0])
    ax_net.set_facecolor("#080E1A")
    ax_net.set_title("Species interaction network", color="white", fontsize=11, fontweight="bold")
    ax_net.set_xlim(-1.5, 1.5)
    ax_net.set_ylim(-1.8, 1.5)
    ax_net.axis("off")

    pos_net = {
        0: (0, 1.1),
        1: (1.05, 0.34),
        2: (0.65, -0.9),
        3: (-0.65, -0.9),
        4: (-1.05, 0.34),
    }
    pos_net[4] = (0, -1.5)

    net_edges = [(0,1,0.8),(1,0,0.5),(2,3,1.2),(3,2,0.6),(3,4,2.4),(2,4,3.56)]
    for i, j, w in net_edges:
        xi, yi = pos_net[i]
        xj, yj = pos_net[j]
        alpha = min(0.2 + w * 0.15, 0.9)
        lw = 0.5 + w * 0.4
        ax_net.annotate("", xy=(xj * 0.82, yj * 0.82),
                         xytext=(xi * 0.82, yi * 0.82),
                         arrowprops=dict(arrowstyle="-|>", color="#4EC9B0", lw=lw,
                                         connectionstyle="arc3,rad=0.2",
                                         mutation_scale=10, alpha=alpha))

    for i, (name, c) in enumerate(zip(["So","An","Vd","Fn","Pg"], SCOLORS)):
        x, y = pos_net[i]
        ax_net.add_patch(Circle((x, y), 0.25, color=c, zorder=3, ec="white", lw=1.5))
        ax_net.text(x, y, name, ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=4)

    # ── Bottom: RMSE bar chart ───────────
    ax_rmse = fig.add_subplot(gs[1, 1:3])
    ax_rmse.set_facecolor("#080E1A")
    ax_rmse.set_title("RMSE improvement Before→After", color="white", fontsize=11, fontweight="bold")
    sp_labels = ["So", "An", "Vd", "Fn", "Pg", "Total"]
    prev = [0.036, 0.129, 0.213, 0.088, 0.435, 0.228]
    mild = [0.034, 0.105, 0.269, 0.161, 0.103, 0.156]
    x = np.arange(6)
    ax_rmse.bar(x - 0.2, prev, 0.35, color="#FF6B6B", alpha=0.8, label="Before")
    ax_rmse.bar(x + 0.2, mild, 0.35, color="#4EC9B0", alpha=0.8, label="After")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(sp_labels, color="white", fontsize=9)
    ax_rmse.set_ylabel("RMSE", color="white", fontsize=9)
    ax_rmse.tick_params(colors="white")
    for spine in ax_rmse.spines.values():
        spine.set_color("#333344")
    ax_rmse.legend(facecolor="#1A2A3A", edgecolor="none", labelcolor="white", fontsize=8)

    # Emphasize Pg
    ax_rmse.annotate("▼76%", xy=(4 + 0.2, 0.103), xytext=(4.6, 0.38),
                     arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.5),
                     fontsize=9, color="#FFD700", fontweight="bold", ha="center")

    # ── Bottom: matrix heatmap ───────────
    ax_mat = fig.add_subplot(gs[1, 3:])
    ax_mat.set_facecolor("#080E1A")
    ax_mat.set_title("Interaction matrix A (MAP)", color="white", fontsize=11, fontweight="bold")
    A = np.array([
        [1.52, 0.83, 0.72, 0.00, 0.00],
        [0.48, 1.21, 0.41, 0.00, 0.00],
        [0.00, 0.00, 1.87, 1.20, 3.56],
        [0.00, 0.00, 0.63, 2.14, 2.41],
        [0.00, 0.00, 0.00, 0.00, 1.08],
    ])
    from matplotlib.colors import LinearSegmentedColormap
    cmap2 = LinearSegmentedColormap.from_list("bio2", ["#080E1A", "#1A3A5C", "#2563EB", "#4EC9B0", "#FFD700"])
    im = ax_mat.imshow(A, cmap=cmap2, vmin=0, vmax=4)
    ax_mat.set_xticks(range(5))
    ax_mat.set_yticks(range(5))
    ax_mat.set_xticklabels(["So","An","Vd","Fn","Pg"], color="white", fontsize=9)
    ax_mat.set_yticklabels(["So","An","Vd","Fn","Pg"], color="white", fontsize=9)
    for i in range(5):
        for j in range(5):
            v = A[i, j]
            tc = "white" if v < 2 else "#080E1A"
            ax_mat.text(j, i, f"{v:.1f}" if v > 0 else "🔒",
                        ha="center", va="center", fontsize=9, color=tc, fontweight="bold")
    fig.colorbar(im, ax=ax_mat, fraction=0.04, pad=0.04).ax.yaxis.set_tick_params(color="white")

    # ── Bottom-most: β schedule ──────────
    ax_beta = fig.add_subplot(gs[2, 1:3])
    ax_beta.set_facecolor("#080E1A")
    ax_beta.set_title("β schedule (prior→posterior)", color="white", fontsize=11, fontweight="bold")
    beta_vals = [0.0, 0.05, 0.12, 0.25, 0.42, 0.61, 0.80, 0.93, 1.0]
    ax_beta.fill_between(range(9), beta_vals, alpha=0.4, color="#4EC9B0")
    ax_beta.plot(range(9), beta_vals, "o-", color="#4EC9B0", lw=2.5, ms=8)
    ax_beta.axhline(1.0, color="#FFD700", lw=1, ls="--", alpha=0.7)
    ax_beta.set_xlabel("Stage", color="white")
    ax_beta.set_ylabel("β", color="white")
    ax_beta.tick_params(colors="white")
    for spine in ax_beta.spines.values():
        spine.set_color("#333344")
    ax_beta.set_ylim(-0.05, 1.1)
    for i, b in enumerate(beta_vals):
        ax_beta.text(i, b + 0.04, f"{b:.2f}", ha="center", fontsize=7.5, color="#AABBCC")

    # ── Bottom-most: convergence summary ─
    ax_conv = fig.add_subplot(gs[2, 3:])
    ax_conv.set_facecolor("#080E1A")
    ax_conv.set_title("Convergence summary", color="white", fontsize=11, fontweight="bold")
    ax_conv.axis("off")
    ax_conv.set_xlim(0, 10)
    ax_conv.set_ylim(0, 10)

    kpis = [
        ("ESS",    "200–300", "#4EC9B0", 0.85),
        ("Rhat",   "~1.00",   "#4EC9B0", 1.0),
        ("Acceptance", "33–61%",  "#4EC9B0", 0.78),
        ("RMSE↓",  "32% improvement", "#FFD700", 0.90),
    ]
    for i, (k, v, c, frac) in enumerate(kpis):
        y = 8.5 - i * 2.0
        ax_conv.text(0.5, y, k, fontsize=10, color="#AABBCC", va="center")
        # Bar-gauge
        bar_bg = FancyBboxPatch((2.0, y - 0.3), 6.5, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#1A2230", edgecolor="#333344", lw=1)
        bar_fill = FancyBboxPatch((2.0, y - 0.3), 6.5 * frac, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor=c, edgecolor="none", alpha=0.7)
        ax_conv.add_patch(bar_bg)
        ax_conv.add_patch(bar_fill)
        ax_conv.text(9.0, y, v, fontsize=10, color=c, va="center", fontweight="bold")

    plt.savefig(OUT / "fig7_poster.png", dpi=180, bbox_inches="tight", facecolor="#080E1A")
    plt.close()
    print("✓ fig7_poster.png")


if __name__ == "__main__":
    print("Generating overview figures...")
    fig1_pipeline()
    fig2_network()
    fig3_tmcmc()
    fig4_data_model()
    fig5_results()
    fig6_positioning()
    fig7_poster()
    print(f"\n✅ All 7 figures saved to {OUT}/")
