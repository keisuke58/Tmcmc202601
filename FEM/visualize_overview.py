"""
Overview visualization script — English version.
Visualizes overview2602.md content in multiple formats (PNG, PDF, SVG) for posters and slides.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path

# Output: multiple formats in FEM/overview_figs
BASE = Path(__file__).resolve().parent
OUT = BASE / "overview_figs"
OUT.mkdir(parents=True, exist_ok=True)
FORMATS = ["png", "pdf", "svg"]  # poster-ready formats

# ---------------------------------------------------------------------------
# Color palette — white background for all figures (poster/print quality)
# ---------------------------------------------------------------------------
BG = "#FFFFFF"
TEXT = "#1A1A1A"
TEXT_SEC = "#444444"
AXES = "#333333"
LEGEND_BG = "#F5F5F5"
LEGEND_EC = "#CCCCCC"

COLORS = {
    "So": "#4E9AF1",
    "An": "#6DC06D",
    "Vd": "#F4A142",
    "Fn": "#B07CC6",
    "Pg": "#E05252",
    "bg": BG,
    "panel": "#1A2A4A",
    "accent": "#2563EB",
    "gray": "#555555",
    "arrow": "#334155",
}

SPECIES = [
    "So\n(S. oralis)",
    "An\n(A. naeslundii)",
    "Vd\n(V. dispar)",
    "Fn\n(F. nucleatum)",
    "Pg\n(P. gingivalis)",
]
SCOLORS = [COLORS["So"], COLORS["An"], COLORS["Vd"], COLORS["Fn"], COLORS["Pg"]]


def save_fig(fig, name: str) -> None:
    """Save figure in all configured formats (white background, poster-grade DPI)."""
    fig.patch.set_facecolor(BG)
    for fmt in FORMATS:
        path = OUT / f"{name}.{fmt}"
        dpi = 300 if fmt == "png" else 300  # poster/print quality
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    print(f"  {name} [{', '.join(FORMATS)}]")


# ===========================================================================
# FIG 1: Overall pipeline flowchart
# ===========================================================================
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    def box(
        x,
        y,
        w,
        h,
        label,
        sublabel="",
        color="#FFFFFF",
        textcolor="#1A2A4A",
        fontsize=13,
        radius=0.4,
    ):
        fancy = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0.1,rounding_size={radius}",
            linewidth=2,
            edgecolor=color,
            facecolor=color + "22",
            zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(
            x,
            y + (0.18 if sublabel else 0),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=textcolor,
            zorder=4,
        )
        if sublabel:
            ax.text(
                x,
                y - 0.32,
                sublabel,
                ha="center",
                va="center",
                fontsize=9,
                color=textcolor + "BB" if len(textcolor) > 6 else "#555577",
                zorder=4,
                style="italic",
            )

    def arrow(x1, y1, x2, y2, label="", color="#334155"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=20),
            zorder=2,
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, label, fontsize=8, color=color, va="center")

    ax.text(
        10,
        10.5,
        "Oral Biofilm Parameter Estimation — Overall Pipeline",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        10,
        10.0,
        "5-Species Biofilm · TMCMC Bayesian Inference",
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_SEC,
    )

    box(
        2.2,
        8.0,
        3.6,
        2.8,
        "① Experimental data",
        "6 time points × 5 species\nvolume fraction φᵢ(t)",
        "#4E9AF1",
        fontsize=12,
    )
    for i, (sp, sc) in enumerate(zip(["So", "An", "Vd", "Fn", "Pg"], SCOLORS)):
        for j in range(6):
            val = np.random.uniform(0.05, 0.35)
            rect = plt.Rectangle(
                (0.5 + j * 0.38, 6.75 + i * 0.35 - 0.2),
                0.34,
                0.3,
                color=sc,
                alpha=0.15 + val * 0.6,
                zorder=5,
            )
            ax.add_patch(rect)
            ax.text(
                0.5 + j * 0.38 + 0.17,
                6.75 + i * 0.35 - 0.05,
                f"{val:.2f}",
                fontsize=5.5,
                ha="center",
                va="center",
                color="#333",
                zorder=6,
            )
    ax.text(
        2.2, 6.35, "day: 1   3   6  10  15  21", fontsize=7.5, ha="center", color="#555", zorder=6
    )

    box(
        6.5,
        8.5,
        3.2,
        1.6,
        "② Prior distribution",
        "Uniform [lb, ub]\n20 parameters",
        "#6DC06D",
        fontsize=12,
    )
    box(
        6.5,
        6.2,
        3.2,
        1.8,
        "③ ODE model",
        "dφᵢ/dt = f(φ,ψ;θ)\nHill-function gating",
        "#F4A142",
        fontsize=12,
    )
    box(
        11.0,
        7.5,
        3.2,
        2.2,
        "④ Likelihood",
        "log p(D|θ)\nweighted Gaussian\nλ_pg × λ_late",
        "#B07CC6",
        fontsize=12,
    )
    box(15.5, 7.5, 3.6, 2.6, "⑤ TMCMC", "β: 0 → 1\n8 stages\n150 particles", "#E05252", fontsize=12)
    box(
        15.5,
        4.0,
        3.6,
        2.4,
        "⑥ Output",
        "MAP estimate\n95% CI\nConvergence (Rhat, ESS)",
        "#2563EB",
        fontsize=12,
    )
    box(
        11.0,
        5.0,
        3.2,
        1.6,
        "Speedup",
        "Numba JIT\nTSM-ROM\nParallel (4–8×)",
        "#64748B",
        TEXT,
        fontsize=11,
    )

    arrow(4.0, 8.0, 4.9, 7.5)
    arrow(4.9, 8.5, 4.9, 8.5)
    arrow(4.0, 8.0, 5.85, 8.5)
    arrow(4.0, 8.0, 5.85, 6.2)
    arrow(6.5, 7.7, 6.5, 7.1)
    arrow(8.1, 8.5, 9.4, 7.9)
    arrow(8.1, 6.2, 9.4, 7.0)
    arrow(12.6, 7.5, 13.7, 7.5)
    arrow(15.5, 6.2, 15.5, 5.1)
    arrow(12.6, 5.0, 13.7, 6.8)
    ax.annotate(
        "",
        xy=(9.4, 6.2),
        xytext=(13.7, 6.2),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#E05252",
            lw=2,
            connectionstyle="arc3,rad=-0.4",
            mutation_scale=16,
        ),
        zorder=2,
    )
    ax.text(11.5, 5.35, "iterate", fontsize=9, color="#E05252", ha="center")

    for i, (txt, c) in enumerate(
        [
            ("● Input", COLORS["So"]),
            ("● Prob. model", COLORS["An"]),
            ("● ODE model", COLORS["Vd"]),
            ("● Algorithm", COLORS["Pg"]),
            ("● Output", COLORS["accent"]),
        ]
    ):
        ax.text(0.5 + i * 3.9, 0.5, txt, fontsize=9, color=c, fontweight="bold")
    ax.text(
        10,
        0.12,
        "Dysbiotic × HOBIC | 20 parameters | 8 stages × 150 particles",
        ha="center",
        fontsize=9,
        color=TEXT_SEC,
    )

    plt.tight_layout()
    save_fig(plt.gcf(), "fig1_pipeline")
    plt.close()


# ===========================================================================
# FIG 2: Species network & interaction matrix
# ===========================================================================
def fig2_network():
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(BG)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.6)
    ax.axis("off")
    ax.set_title(
        "Inter-species interaction network", color=TEXT, fontsize=15, fontweight="bold", pad=12
    )

    angles = [90, 18, -54, -126, 162]
    r = 1.0
    pos = {i: (r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for i, a in enumerate(angles)}
    pos[4] = (0.0, -1.2)

    names = ["So", "An", "Vd", "Fn", "Pg"]
    full = [
        "S. oralis\n(pioneer)",
        "A. naeslundii\n(commensal)",
        "V. dispar\n(bridge)",
        "F. nucleatum\n(gatekeeper)",
        "P. gingivalis\n(pathogen)",
    ]

    edges = [
        (0, 1, 0.8, "+"),
        (1, 0, 0.5, "+"),
        (2, 3, 1.2, "+"),
        (3, 2, 0.6, "+"),
        (3, 4, 2.4, "+"),
        (2, 4, 3.56, "+"),
        (0, 2, 0.7, "w"),
        (1, 3, 0.4, "w"),
    ]
    for i, j, w, etype in edges:
        xi, yi = pos[i]
        xj, yj = pos[j]
        ec = "#4EC9B0" if etype == "+" else "#888888"
        alpha = min(0.2 + w * 0.18, 0.95)
        lw = 0.8 + w * 0.6
        ax.annotate(
            "",
            xy=(xj, yj),
            xytext=(xi, yi),
            arrowprops=dict(
                arrowstyle="-|>",
                color=ec,
                lw=lw,
                connectionstyle="arc3,rad=0.15",
                mutation_scale=12,
                alpha=alpha,
            ),
            zorder=2,
        )
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        ax.text(mx + 0.05, my + 0.05, f"{w:.1f}", fontsize=7, color=ec, alpha=0.85, zorder=5)

    for i, (name, full_name, color) in enumerate(zip(names, full, SCOLORS)):
        x, y = pos[i]
        ax.add_patch(Circle((x, y), 0.22, color=color, zorder=3, linewidth=2.5, ec=BG, alpha=0.95))
        ax.text(
            x, y, name, ha="center", va="center", fontsize=12, fontweight="bold", color=BG, zorder=4
        )
        lx, ly = x * 1.55, y * 1.45 + (0.15 if i == 4 else 0)
        ax.text(lx, ly, full_name, ha="center", va="center", fontsize=8, color=color, zorder=4)

    ax.annotate(
        "Hill gate\n(Fn-dependent)",
        xy=pos[3],
        xytext=(0.7, -0.5),
        arrowprops=dict(arrowstyle="->", color="#B8860B", lw=1.5),
        fontsize=8,
        color="#8B6914",
        zorder=5,
    )
    for txt, c in [("strong facilitation", "#4EC9B0"), ("weak interaction", "#888888")]:
        ax.plot([], [], color=c, lw=2, label=txt)
    ax.legend(
        loc="upper right", facecolor=LEGEND_BG, edgecolor=LEGEND_EC, labelcolor=TEXT, fontsize=8
    )

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.set_title(
        "Interaction matrix A[i→j] (MAP estimate)",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    A = np.array(
        [
            [1.52, 0.83, 0.72, 0.00, 0.00],
            [0.48, 1.21, 0.41, 0.00, 0.00],
            [0.00, 0.00, 1.87, 1.20, 3.56],
            [0.00, 0.00, 0.63, 2.14, 2.41],
            [0.00, 0.00, 0.00, 0.00, 1.08],
        ]
    )
    locked = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=bool,
    )
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "bio", [BG, "#E8F4FC", "#2563EB", "#4EC9B0", "#DAA520"]
    )
    im = ax2.imshow(A, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    tick_labels = ["So", "An", "Vd", "Fn", "Pg"]
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(tick_labels, color=TEXT, fontsize=12, fontweight="bold")
    ax2.set_yticklabels(tick_labels, color=TEXT, fontsize=12, fontweight="bold")
    ax2.set_xlabel("Source species", color=TEXT, fontsize=11)
    ax2.set_ylabel("Target species", color=TEXT, fontsize=11)
    ax2.tick_params(colors=AXES)
    for spine in ax2.spines.values():
        spine.set_color(AXES)
    for i in range(5):
        for j in range(5):
            if locked[i, j]:
                ax2.text(j, i, "0", ha="center", va="center", fontsize=11)
            else:
                v = A[i, j]
                tc = TEXT if v < 2.5 else BG
                ax2.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=tc,
                )
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Interaction strength", color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=AXES)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    for i in range(5):
        ax2.add_patch(
            plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#B8860B", lw=2.5)
        )
    ax2.text(
        2,
        5.3,
        "0 = fixed by biology  |  diagonal = self-regulation",
        ha="center",
        fontsize=9,
        color=TEXT_SEC,
    )

    plt.tight_layout()
    save_fig(plt.gcf(), "fig2_network")
    plt.close()


# ===========================================================================
# FIG 3: TMCMC algorithm
# ===========================================================================
def fig3_tmcmc():
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(BG)
    ax1.set_title(
        "TMCMC — β schedule and particle evolution", color=TEXT, fontsize=13, fontweight="bold"
    )
    beta_stages = [0.0, 0.05, 0.12, 0.25, 0.42, 0.61, 0.80, 0.93, 1.0]
    np.random.seed(42)
    true_center = np.array([2.5, 3.0])
    for s_idx, beta in enumerate(beta_stages):
        spread = 2.5 * (1 - beta**0.5) + 0.15
        particles = true_center + np.random.randn(40, 2) * spread
        alpha = 0.3 + beta * 0.6
        ax1.scatter(
            particles[:, 0] + s_idx * 0.5,
            particles[:, 1],
            s=15,
            alpha=alpha,
            color=plt.cm.plasma(beta),
            zorder=3,
        )
    for s_idx, beta in enumerate(beta_stages):
        ax1.text(
            s_idx * 0.5 + true_center[0],
            true_center[1] + 1.6,
            f"β={beta:.2f}",
            ha="center",
            fontsize=7.5,
            color=plt.cm.plasma(beta),
            rotation=30,
        )
    ax1.set_xlim(1.5, 8.0)
    ax1.set_ylim(0.5, 6.0)
    ax1.set_xlabel("← Prior    β progression    Posterior →", color=TEXT_SEC, fontsize=10)
    ax1.set_ylabel("Parameter space", color=TEXT_SEC, fontsize=10)
    ax1.tick_params(colors=AXES)
    for spine in ax1.spines.values():
        spine.set_color(AXES)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG)
    ax2.set_title("One stage", color=TEXT, fontsize=12, fontweight="bold")
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    steps = [
        (5, 9.0, "① Set β\n(ESS ≈ 50%)", "#4EC9B0"),
        (5, 7.2, "② Weights\nwᵢ = exp[(Δβ)·logLᵢ]", "#6DC06D"),
        (5, 5.4, "③ Resample\n(by weight)", "#F4A142"),
        (5, 3.6, "④ MCMC mutation\n(150 particles)", "#B07CC6"),
        (5, 1.8, "⑤ Diagnostics\n(ESS, Rhat)", "#4E9AF1"),
    ]
    for x, y, txt, c in steps:
        bbox = FancyBboxPatch(
            (x - 4.0, y - 0.65),
            8.0,
            1.3,
            boxstyle="round,pad=0.1,rounding_size=0.3",
            facecolor=c + "22",
            edgecolor=c,
            lw=2,
            zorder=3,
        )
        ax2.add_patch(bbox)
        ax2.text(
            x, y, txt, ha="center", va="center", fontsize=8.5, color=c, fontweight="bold", zorder=4
        )
    for i in range(len(steps) - 1):
        y1, y2 = steps[i][1] - 0.65, steps[i + 1][1] + 0.65
        ax2.annotate(
            "",
            xy=(5, y2),
            xytext=(5, y1),
            arrowprops=dict(arrowstyle="-|>", color=AXES, lw=1.5, mutation_scale=14),
        )

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)
    ax3.set_title("Convergence metrics", color=TEXT, fontsize=11, fontweight="bold")
    stages = np.arange(8)
    ess = [148, 132, 118, 109, 97, 112, 128, 145]
    rhat_max = [2.1, 1.8, 1.5, 1.35, 1.22, 1.12, 1.05, 1.01]
    ax3b = ax3.twinx()
    ax3.plot(stages, ess, "o-", color="#4EC9B0", lw=2.5, ms=7, label="ESS")
    ax3b.plot(stages, rhat_max, "s--", color="#FFD700", lw=2.5, ms=7, label="Rhat_max")
    ax3b.axhline(1.1, color="#FFD700", lw=1, ls=":", alpha=0.6)
    ax3b.text(6.8, 1.12, "target", fontsize=7.5, color="#B8860B")
    ax3.set_xlabel("Stage", color=TEXT_SEC)
    ax3.set_ylabel("ESS", color="#4EC9B0")
    ax3b.set_ylabel("Rhat", color="#B8860B")
    ax3.tick_params(colors=AXES)
    ax3b.tick_params(colors=AXES)
    for spine in list(ax3.spines.values()) + list(ax3b.spines.values()):
        spine.set_color(AXES)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(
        lines1 + lines2,
        labels1 + labels2,
        facecolor=LEGEND_BG,
        edgecolor=LEGEND_EC,
        labelcolor=TEXT,
        fontsize=8,
    )

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)
    ax4.set_title("MCMC mutation (Metropolis–Hastings)", color=TEXT, fontsize=11, fontweight="bold")
    ax4.set_xlim(-3, 3)
    ax4.set_ylim(-3, 3)
    xx, yy = np.mgrid[-3:3:100j, -3:3:100j]
    from scipy.stats import multivariate_normal

    rv = multivariate_normal([0, 0], [[0.5, 0.3], [0.3, 0.8]])
    Z = rv.pdf(np.dstack([xx, yy]))
    ax4.contourf(xx, yy, Z, levels=8, cmap="Blues", alpha=0.4)
    ax4.contour(xx, yy, Z, levels=5, colors="#4EC9B0", alpha=0.5, linewidths=0.8)
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
    ax4.set_xlabel("θ₁", color=TEXT_SEC, fontsize=9)
    ax4.set_ylabel("θ₂", color=TEXT_SEC, fontsize=9)
    ax4.tick_params(colors=AXES)
    for spine in ax4.spines.values():
        spine.set_color(AXES)
    ax4.legend(facecolor=LEGEND_BG, edgecolor=LEGEND_EC, labelcolor=TEXT, fontsize=8)
    ax4.text(-2.8, -2.7, "Contours = posterior\nYellow = MCMC trace", fontsize=7.5, color=TEXT_SEC)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG)
    ax5.set_title("Acceptance rate & proposal scale", color=TEXT, fontsize=11, fontweight="bold")
    accept = [0.61, 0.55, 0.48, 0.43, 0.38, 0.36, 0.34, 0.33]
    scale = [2.5, 2.0, 1.6, 1.3, 1.1, 0.95, 0.85, 0.78]
    beta_v = [0.05, 0.12, 0.25, 0.42, 0.61, 0.80, 0.93, 1.0]
    ax5b = ax5.twinx()
    ax5.bar(stages, accept, color=plt.cm.plasma(np.array(beta_v)), alpha=0.7, zorder=3)
    ax5b.plot(stages, scale, "D-", color="#FFD700", lw=2, ms=7, zorder=4, label="proposal scale")
    ax5.axhline(0.23, color="#FF6B6B", lw=1.5, ls="--", alpha=0.8)
    ax5.axhline(0.44, color="#4EC9B0", lw=1.5, ls="--", alpha=0.8)
    ax5.set_xlabel("Stage", color=TEXT_SEC)
    ax5.set_ylabel("Acceptance rate", color=TEXT_SEC)
    ax5b.set_ylabel("Proposal scale", color="#B8860B")
    ax5.tick_params(colors=AXES)
    ax5b.tick_params(colors=AXES)
    ax5.set_ylim(0, 0.75)
    for spine in list(ax5.spines.values()) + list(ax5b.spines.values()):
        spine.set_color(AXES)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax5, fraction=0.04, pad=0.15)
    cbar.set_label("β", color=TEXT, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=AXES)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

    fig.text(
        0.5,
        0.98,
        "TMCMC algorithm — detail",
        ha="center",
        color=TEXT,
        fontsize=16,
        fontweight="bold",
    )
    save_fig(plt.gcf(), "fig3_tmcmc")
    plt.close()


# ===========================================================================
# FIG 4: Data structure & ODE model
# ===========================================================================
def fig4_data_model():
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.patch.set_facecolor(BG)

    ax1 = axes[0]
    ax1.set_facecolor(BG)
    ax1.set_title(
        "Data structure\n[6 time points × 5 species]", color=TEXT, fontsize=13, fontweight="bold"
    )
    t_days = [1, 3, 6, 10, 15, 21]
    np.random.seed(10)
    data = np.array(
        [
            [0.35, 0.25, 0.18, 0.10, 0.05, 0.03],
            [0.30, 0.28, 0.22, 0.15, 0.08, 0.05],
            [0.20, 0.22, 0.25, 0.28, 0.25, 0.20],
            [0.12, 0.18, 0.22, 0.28, 0.32, 0.35],
            [0.03, 0.07, 0.13, 0.19, 0.30, 0.37],
        ]
    )
    data = data / data.sum(axis=0, keepdims=True)
    bottom = np.zeros(6)
    for i, (sp, c) in enumerate(zip(["So", "An", "Vd", "Fn", "Pg"], SCOLORS)):
        ax1.bar(range(6), data[i], bottom=bottom, color=c, alpha=0.85, label=sp, zorder=3)
        bottom += data[i]
    ax1.set_xticks(range(6))
    ax1.set_xticklabels([f"Day {d}" for d in t_days], color="white", fontsize=9, rotation=30)
    ax1.set_ylabel("Volume fraction φᵢ", color="white")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_color("#333344")
    ax1.legend(
        facecolor="#1A2A3A", edgecolor="none", labelcolor="white", fontsize=9, loc="upper right"
    )
    ax1.text(2.5, 1.07, "Σφᵢ = 1 (volume conserved)", ha="center", fontsize=8, color="#AABBCC")

    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    ax2.set_title(
        "ODE model trajectory\n(MAP parameters)", color="white", fontsize=13, fontweight="bold"
    )
    t = np.linspace(0, 21, 200)
    from scipy.interpolate import make_interp_spline

    data_t = [1, 3, 6, 10, 15, 21]
    for i, (sp, c) in enumerate(zip(["So", "An", "Vd", "Fn", "Pg"], SCOLORS)):
        vals = data[i]
        spl = make_interp_spline(data_t, vals, k=3)
        smooth = np.clip(spl(t), 0, 1)
        ax2.plot(t, smooth, color=c, lw=2.5, label=sp, zorder=3)
        ax2.scatter(data_t, vals, color=c, s=60, zorder=5, marker="o", edgecolors=BG, lw=1)
    ax2.set_xlabel("Time [days]", color=TEXT)
    ax2.set_ylabel("Volume fraction φᵢ", color=TEXT)
    ax2.set_xlim(0, 22)
    ax2.set_ylim(-0.02, 0.6)
    ax2.tick_params(colors=AXES)
    for spine in ax2.spines.values():
        spine.set_color(AXES)
    ax2.legend(facecolor=LEGEND_BG, edgecolor=LEGEND_EC, labelcolor=TEXT, fontsize=9)
    ax2.text(11, 0.55, "Lines: model  ●: data", fontsize=8.5, color=TEXT_SEC, ha="center")

    ax3 = axes[2]
    ax3.set_facecolor(BG)
    ax3.set_title("Parameter structure\n(20-D θ)", color=TEXT, fontsize=13, fontweight="bold")
    ax3.axis("off")
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    blocks = [
        ("M1 θ[0–4]", "#4E9AF1", 9.0, "So, An interaction\n+ decay b₁,b₂"),
        ("M2 θ[5–9]", "#6DC06D", 7.0, "Vd, Fn interaction\n+ decay b₃,b₄"),
        ("M3 θ[10–13]", "#F4A142", 5.0, "Cross (So,An)↔(Vd,Fn)"),
        ("M4 θ[14–15]", "#E05252", 3.2, "Pg self & decay"),
        ("M5 θ[16–19]", "#B07CC6", 1.5, "Pg ← (So,An,Vd,Fn)"),
    ]
    for label, color, cy, desc in blocks:
        h = 1.4 if "M3" not in label else 1.2
        if "M4" in label or "M5" in label:
            h = 1.1
        bbox = FancyBboxPatch(
            (0.5, cy - h / 2),
            3.5,
            h,
            boxstyle="round,pad=0.1,rounding_size=0.25",
            facecolor=color + "33",
            edgecolor=color,
            lw=2,
            zorder=3,
        )
        ax3.add_patch(bbox)
        ax3.text(
            2.25,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
            zorder=4,
        )
        ax3.text(7.5, cy, desc, ha="center", va="center", fontsize=8.5, color=TEXT_SEC, zorder=4)
        ax3.annotate(
            "",
            xy=(4.8, cy),
            xytext=(4.0, cy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=10),
        )
    ax3.text(5, 0.3, "Total 20-D → posterior p(θ|D)", ha="center", fontsize=9, color=TEXT_SEC)
    locked_box = FancyBboxPatch(
        (5.5, 8.3),
        4.0,
        1.2,
        boxstyle="round,pad=0.08",
        facecolor="#FFF0F0",
        edgecolor="#C44",
        lw=1.5,
    )
    ax3.add_patch(locked_box)
    ax3.text(
        7.5, 8.9, "Locked parameters", ha="center", fontsize=9, color="#C44", fontweight="bold"
    )
    ax3.text(7.5, 8.45, "Fixed to 0 by biology", ha="center", fontsize=8, color="#884444")
    hill_box = FancyBboxPatch(
        (5.5, 6.8),
        4.0,
        1.2,
        boxstyle="round,pad=0.08",
        facecolor="#FFFDE7",
        edgecolor="#B8860B",
        lw=1.5,
    )
    ax3.add_patch(hill_box)
    ax3.text(
        7.5, 7.4, "Hill gate (fixed)", ha="center", fontsize=9, color="#8B6914", fontweight="bold"
    )
    ax3.text(7.5, 6.95, "K=0.05, n=4 (Fn-dependent)", ha="center", fontsize=7.5, color="#666")

    fig.text(
        0.5,
        0.98,
        "Data structure · ODE model · parameter space",
        ha="center",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    save_fig(plt.gcf(), "fig4_data_model")
    plt.close()


# ===========================================================================
# FIG 5: Positioning
# ===========================================================================
def fig5_positioning():
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor(BG)

    ax1 = axes[0]
    ax1.set_facecolor(BG)
    ax1.set_title("Method positioning", color=TEXT, fontsize=14, fontweight="bold", pad=15)
    methods = [
        ("Optimization only\n(L-BFGS)", 0.9, 0.15, "#FF6B6B", 80),
        ("MCMC only\n(Metropolis)", 0.35, 0.55, "#F4A142", 90),
        ("TMCMC\n(this work)", 0.7, 0.75, "#4EC9B0", 200),
        ("Nested Sampling", 0.5, 0.65, "#B07CC6", 80),
        ("Variational Bayes", 0.8, 0.45, "#4E9AF1", 90),
        ("Neural ODE", 0.3, 0.3, "#888888", 75),
    ]
    for name, x, y, c, s in methods:
        ax1.scatter(x, y, s=s, color=c, alpha=0.85, zorder=4)
        is_hl = "TMCMC" in name
        ax1.text(
            x,
            y + 0.06,
            name,
            ha="center",
            va="center",
            fontsize=9,
            color=c,
            fontweight="bold" if is_hl else "normal",
            zorder=5,
            bbox=(
                dict(
                    boxstyle="round,pad=0.2",
                    fc=BG,
                    ec=c if is_hl else "none",
                    lw=1.5 if is_hl else 0,
                    alpha=0.9,
                )
                if is_hl
                else None
            ),
        )
    ax1.scatter(0.7, 0.75, s=500, color="#4EC9B0", alpha=0.2, zorder=3)
    ax1.scatter(0.7, 0.75, s=250, color="#4EC9B0", alpha=0.3, zorder=3)
    ax1.set_xlabel("Speed (efficiency)", color=TEXT, fontsize=12)
    ax1.set_ylabel("Accuracy (uncertainty quantification)", color=TEXT, fontsize=12)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors=AXES)
    for spine in ax1.spines.values():
        spine.set_color(AXES)
    ax1.axvspan(0.55, 1.05, alpha=0.08, color="#4EC9B0")
    ax1.axhspan(0.6, 1.05, alpha=0.08, color="#4EC9B0")
    ax1.text(0.8, 0.92, "sweet spot", fontsize=9, color="#2D8B6F", alpha=0.9)
    ax1.text(
        0.5,
        -0.08,
        "Limited data · high-dim · nonlinear → TMCMC fits well",
        ha="center",
        fontsize=9.5,
        color=TEXT_SEC,
    )

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.set_title("Output overview", color=TEXT, fontsize=14, fontweight="bold", pad=15)
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    items = [
        ("config.json", "Run settings"),
        ("posterior_samples.npy", "Posterior samples"),
        ("parameter_summary.csv", "MAP, mean, 95% CI"),
        ("diagnostics_tables/", "β schedule, ESS, Rhat"),
        ("figures/", "Posterior, fit, convergence"),
    ]
    for i, (fname, desc) in enumerate(items):
        y = 8.5 - i * 1.6
        bbox = FancyBboxPatch(
            (0.5, y - 0.5),
            9.0,
            1.0,
            boxstyle="round,pad=0.1",
            facecolor=LEGEND_BG,
            edgecolor="#4EC9B0",
            lw=1,
            zorder=3,
        )
        ax2.add_patch(bbox)
        ax2.text(
            1.0,
            y,
            fname,
            ha="left",
            va="center",
            fontsize=10,
            color="#2D8B6F",
            fontweight="bold",
            zorder=4,
        )
        ax2.text(5.5, y, desc, ha="center", va="center", fontsize=9, color=TEXT_SEC, zorder=4)
    ax2.text(
        5, 0.5, "_runs/Dysbiotic_HOBIC_YYYYMMDD_XXXX/", ha="center", fontsize=9, color=TEXT_SEC
    )

    fig.text(
        0.5,
        0.98,
        "Positioning & output structure",
        ha="center",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    save_fig(plt.gcf(), "fig5_positioning")
    plt.close()


# ===========================================================================
# FIG 6: Poster — one-page summary (English)
# ===========================================================================
def fig6_poster():
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor("#080E1A")

    fig.text(
        0.5,
        0.965,
        "Quantifying Oral Biofilm Inter-Species Interactions via Bayesian Inference",
        ha="center",
        fontsize=22,
        fontweight="bold",
        color="white",
    )
    fig.text(
        0.5,
        0.945,
        "5-Species ODE Model · TMCMC · Dysbiotic × HOBIC · 20 Parameters · MAP + 95% CI",
        ha="center",
        fontsize=11,
        color="#8899AA",
    )

    gs = GridSpec(
        3, 5, figure=fig, hspace=0.55, wspace=0.4, left=0.04, right=0.97, top=0.93, bottom=0.05
    )

    ax_pipe = fig.add_subplot(gs[0, :])
    ax_pipe.set_facecolor("#080E1A")
    ax_pipe.axis("off")
    ax_pipe.set_xlim(0, 24)
    ax_pipe.set_ylim(0, 4)

    pipeline_steps = [
        (1.5, "Experimental data\n[6 x 5 species]", "#4E9AF1", "1"),
        (5.5, "Prior\n[20 params]", "#6DC06D", "2"),
        (9.5, "ODE model\nd\u03c6/dt = f(\u03c6,\u03c8;\u03b8)", "#F4A142", "3"),
        (13.5, "Likelihood\nlog p(D|\u03b8)", "#B07CC6", "4"),
        (17.5, "TMCMC\n\u03b2: 0\u21921, 8 stages", "#E05252", "5"),
        (21.5, "Posterior\nMAP + 95% CI", "#2563EB", "6"),
    ]
    for i, (x, label, c, icon) in enumerate(pipeline_steps):
        box = FancyBboxPatch(
            (x - 1.6, 0.6),
            3.2,
            2.8,
            boxstyle="round,pad=0.15",
            facecolor=c + "22",
            edgecolor=c,
            lw=2.5,
            zorder=3,
        )
        ax_pipe.add_patch(box)
        ax_pipe.text(x, 2.75, icon, ha="center", va="center", fontsize=18, zorder=4)
        ax_pipe.text(
            x,
            1.85,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=c,
            zorder=4,
        )
        ax_pipe.text(x, 0.3, f"Step {i+1}", ha="center", fontsize=8, color=c, alpha=0.7)
        if i < len(pipeline_steps) - 1:
            ax_pipe.annotate(
                "",
                xy=(x + 1.75, 2.0),
                xytext=(x + 1.6, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=AXES, lw=2.5, mutation_scale=16),
                zorder=2,
            )
    ax_pipe.text(
        12,
        3.75,
        "End-to-end inference flow",
        ha="center",
        fontsize=9,
        color=TEXT_SEC,
        style="italic",
    )

    ax_net = fig.add_subplot(gs[1:, 0])
    ax_net.set_facecolor(BG)
    ax_net.set_title("Species network", color=TEXT, fontsize=11, fontweight="bold")
    ax_net.set_xlim(-1.5, 1.5)
    ax_net.set_ylim(-1.8, 1.5)
    ax_net.axis("off")
    pos_net = {0: (0, 1.1), 1: (1.05, 0.34), 2: (0.65, -0.9), 3: (-0.65, -0.9), 4: (0, -1.5)}
    net_edges = [(0, 1, 0.8), (1, 0, 0.5), (2, 3, 1.2), (3, 2, 0.6), (3, 4, 2.4), (2, 4, 3.56)]
    for i, j, w in net_edges:
        xi, yi = pos_net[i]
        xj, yj = pos_net[j]
        alpha = min(0.2 + w * 0.15, 0.9)
        lw = 0.5 + w * 0.4
        ax_net.annotate(
            "",
            xy=(xj * 0.82, yj * 0.82),
            xytext=(xi * 0.82, yi * 0.82),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#4EC9B0",
                lw=lw,
                connectionstyle="arc3,rad=0.2",
                mutation_scale=10,
                alpha=alpha,
            ),
        )
    for i, (name, c) in enumerate(zip(["So", "An", "Vd", "Fn", "Pg"], SCOLORS)):
        x, y = pos_net[i]
        ax_net.add_patch(Circle((x, y), 0.25, color=c, zorder=3, ec="white", lw=1.5))
        ax_net.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    ax_rmse = fig.add_subplot(gs[1, 1:3])
    ax_rmse.set_facecolor("#080E1A")
    ax_rmse.set_title(
        "RMSE Before → After (example run)", color="white", fontsize=11, fontweight="bold"
    )
    sp_labels = ["So", "An", "Vd", "Fn", "Pg", "Total"]
    prev = [0.036, 0.129, 0.213, 0.088, 0.435, 0.228]
    mild = [0.034, 0.105, 0.269, 0.161, 0.103, 0.156]
    x = np.arange(6)
    ax_rmse.bar(x - 0.2, prev, 0.35, color="#FF6B6B", alpha=0.8, label="Before")
    ax_rmse.bar(x + 0.2, mild, 0.35, color="#4EC9B0", alpha=0.8, label="After")
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(sp_labels, color=TEXT, fontsize=9)
    ax_rmse.set_ylabel("RMSE", color=TEXT, fontsize=9)
    ax_rmse.tick_params(colors=AXES)
    for spine in ax_rmse.spines.values():
        spine.set_color(AXES)
    ax_rmse.legend(facecolor=LEGEND_BG, edgecolor=LEGEND_EC, labelcolor=TEXT, fontsize=8)
    ax_rmse.annotate(
        "▼ 76%",
        xy=(4 + 0.2, 0.103),
        xytext=(4.6, 0.38),
        arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1.5),
        fontsize=9,
        color="#FFD700",
        fontweight="bold",
        ha="center",
    )

    ax_mat = fig.add_subplot(gs[1, 3:])
    ax_mat.set_facecolor(BG)
    ax_mat.set_title("Interaction matrix A (MAP)", color=TEXT, fontsize=11, fontweight="bold")
    A = np.array(
        [
            [1.52, 0.83, 0.72, 0.00, 0.00],
            [0.48, 1.21, 0.41, 0.00, 0.00],
            [0.00, 0.00, 1.87, 1.20, 3.56],
            [0.00, 0.00, 0.63, 2.14, 2.41],
            [0.00, 0.00, 0.00, 0.00, 1.08],
        ]
    )
    from matplotlib.colors import LinearSegmentedColormap

    cmap2 = LinearSegmentedColormap.from_list(
        "bio2", [BG, "#E8F4FC", "#2563EB", "#4EC9B0", "#DAA520"]
    )
    im = ax_mat.imshow(A, cmap=cmap2, vmin=0, vmax=4)
    ax_mat.set_xticks(range(5))
    ax_mat.set_yticks(range(5))
    ax_mat.set_xticklabels(["So", "An", "Vd", "Fn", "Pg"], color=TEXT, fontsize=9)
    ax_mat.set_yticklabels(["So", "An", "Vd", "Fn", "Pg"], color=TEXT, fontsize=9)
    for i in range(5):
        for j in range(5):
            v = A[i, j]
            tc = TEXT if v < 2.5 else BG
            ax_mat.text(
                j,
                i,
                f"{v:.1f}" if v > 0 else "0",
                ha="center",
                va="center",
                fontsize=9,
                color=tc,
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax_mat, fraction=0.04, pad=0.04).ax.yaxis.set_tick_params(color=AXES)
    plt.setp(fig.axes[-1].yaxis.get_ticklabels(), color=TEXT)

    ax_beta = fig.add_subplot(gs[2, 1:3])
    ax_beta.set_facecolor(BG)
    ax_beta.set_title("β schedule (prior → posterior)", color=TEXT, fontsize=11, fontweight="bold")
    beta_vals = [0.0, 0.05, 0.12, 0.25, 0.42, 0.61, 0.80, 0.93, 1.0]
    ax_beta.fill_between(range(9), beta_vals, alpha=0.4, color="#4EC9B0")
    ax_beta.plot(range(9), beta_vals, "o-", color="#4EC9B0", lw=2.5, ms=8)
    ax_beta.axhline(1.0, color="#B8860B", lw=1, ls="--", alpha=0.7)
    ax_beta.set_xlabel("Stage", color=TEXT)
    ax_beta.set_ylabel("β", color=TEXT)
    ax_beta.tick_params(colors=AXES)
    for spine in ax_beta.spines.values():
        spine.set_color(AXES)
    ax_beta.set_ylim(-0.05, 1.1)

    ax_conv = fig.add_subplot(gs[2, 3:])
    ax_conv.set_facecolor(BG)
    ax_conv.set_title("Takeaway", color=TEXT, fontsize=11, fontweight="bold")
    ax_conv.axis("off")
    ax_conv.set_xlim(0, 10)
    ax_conv.set_ylim(0, 10)
    bullets = [
        "What: Bayesian estimation of 5-species interaction matrix (20 parameters).",
        "How: ODE model × TMCMC (temperature schedule) × Numba + ROM + parallel.",
        "Why it works: Suited to limited data, nonlinearity, and high dimension.",
    ]
    for i, line in enumerate(bullets):
        y = 8.5 - i * 2.2
        ax_conv.text(0.5, y, "•", fontsize=14, color="#4EC9B0", va="center")
        ax_conv.text(1.0, y, line, fontsize=9.5, color=TEXT_SEC, va="center", wrap=True)
    ax_conv.text(
        5,
        2.0,
        "Uncertainty quantified: MAP + 95% CI · ESS · Rhat",
        ha="center",
        fontsize=10,
        color="#2D8B6F",
    )

    save_fig(plt.gcf(), "fig6_poster")
    plt.close()


# ===========================================================================
# FIG 7: Combined TMCMC + FEM poster (one page)
# ===========================================================================
def fig7_combined_poster():
    """One-page poster: TMCMC (parameters) + coupling + FEM (3-tooth mechanics)."""
    fig = plt.figure(figsize=(28, 18))
    fig.patch.set_facecolor("#060B12")

    # Title
    fig.text(
        0.5,
        0.97,
        "From Bayesian Parameters to Tooth-Scale Mechanics",
        ha="center",
        fontsize=24,
        fontweight="bold",
        color="white",
    )
    fig.text(
        0.5,
        0.945,
        "TMCMC (5-species ODE) \u2192 DI field \u2192 3-tooth conformal biofilm FEM \u2192 stress & displacement",
        ha="center",
        fontsize=12,
        color="#8899AA",
    )

    gs = GridSpec(
        4, 3, figure=fig, hspace=0.5, wspace=0.35, left=0.03, right=0.98, top=0.92, bottom=0.04
    )

    # ---- Row 0: Three-column flow ----
    ax_tmcmc = fig.add_subplot(gs[0, 0])
    ax_tmcmc.set_facecolor(BG)
    ax_tmcmc.set_title(
        "TMCMC (Parameter estimation)", color="#2D8B6F", fontsize=13, fontweight="bold"
    )
    ax_tmcmc.axis("off")
    ax_tmcmc.set_xlim(0, 10)
    ax_tmcmc.set_ylim(0, 10)
    steps_t = [
        (5, 8.5, "Experiment\n[6 x 5 species]", "#4E9AF1"),
        (5, 6.5, "ODE + TMCMC\n8 st., 150 pt.", "#E05252"),
        (5, 4.5, "MAP + 95% CI\n20 params", "#2563EB"),
    ]
    for x, y, label, c in steps_t:
        b = FancyBboxPatch(
            (x - 2.2, y - 0.6),
            4.4,
            1.2,
            boxstyle="round,pad=0.1",
            facecolor=c + "22",
            edgecolor=c,
            lw=1.5,
            zorder=3,
        )
        ax_tmcmc.add_patch(b)
        ax_tmcmc.text(
            x, y, label, ha="center", va="center", fontsize=9, color=c, fontweight="bold", zorder=4
        )
    for i in range(len(steps_t) - 1):
        ax_tmcmc.annotate(
            "",
            xy=(5, steps_t[i + 1][1] + 0.6),
            xytext=(5, steps_t[i][1] - 0.6),
            arrowprops=dict(arrowstyle="-|>", color=AXES, lw=1.5, mutation_scale=12),
        )
    ax_tmcmc.text(
        5,
        2.2,
        "Output: \u03b8_MAP per condition\n(4 conditions)",
        ha="center",
        fontsize=8,
        color=TEXT_SEC,
    )

    ax_cpl = fig.add_subplot(gs[0, 1])
    ax_cpl.set_facecolor(BG)
    ax_cpl.set_title("Coupling (TMCMC \u2192 FEM)", color="#8B6914", fontsize=13, fontweight="bold")
    ax_cpl.axis("off")
    ax_cpl.set_xlim(0, 10)
    ax_cpl.set_ylim(0, 10)
    cpl_items = [
        "\u03b8_MAP, snapshot \u2192 \u03d5_i(x,y,z)",
        "DI = 1 \u2212 H / log(5)",
        "E_eff(DI) = E_max(1\u2212r)^\u03b1 + E_min\u00b7r",
        "Field CSV \u2192 INP per condition",
    ]
    for i, line in enumerate(cpl_items):
        y = 8.0 - i * 1.8
        ax_cpl.text(
            0.5, y, line, ha="left", va="center", fontsize=9, color=TEXT_SEC, family="monospace"
        )
    ax_cpl.text(5, 1.2, "tmcmc_to_fem_coupling.py", ha="center", fontsize=8, color="#8B6914")

    ax_fem = fig.add_subplot(gs[0, 2])
    ax_fem.set_facecolor(BG)
    ax_fem.set_title("FEM (3-tooth biofilm)", color="#C76B00", fontsize=13, fontweight="bold")
    ax_fem.axis("off")
    ax_fem.set_xlim(0, 10)
    ax_fem.set_ylim(0, 10)
    steps_f = [
        (5, 8.5, "STL T23/T30/T31\nconformal mesh", "#6DC06D"),
        (5, 6.5, "DI bins \u2192 E_eff\nAssembly + Tie", "#B07CC6"),
        (5, 4.5, "Abaqus Static\n1 MPa inward", "#F4A142"),
        (5, 2.5, "ODB \u2192 MISES, U\nFig1\u2013LateFig7", "#64748B"),
    ]
    for x, y, label, c in steps_f:
        b = FancyBboxPatch(
            (x - 2.2, y - 0.5),
            4.4,
            1.0,
            boxstyle="round,pad=0.08",
            facecolor=c + "22",
            edgecolor=c,
            lw=1.5,
            zorder=3,
        )
        ax_fem.add_patch(b)
        ax_fem.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=c,
            fontweight="bold",
            zorder=4,
        )
    for i in range(len(steps_f) - 1):
        ax_fem.annotate(
            "",
            xy=(5, steps_f[i + 1][1] + 0.5),
            xytext=(5, steps_f[i][1] - 0.5),
            arrowprops=dict(arrowstyle="-|>", color=AXES, lw=1.5, mutation_scale=12),
        )
    ax_fem.text(5, 0.8, "82,080 nodes \u00b7 437,472 C3D4", ha="center", fontsize=8, color=TEXT_SEC)

    # ---- Row 1: Key result — displacement ratio ----
    ax_res = fig.add_subplot(gs[1, :])
    ax_res.set_facecolor(BG)
    ax_res.set_title(
        "Key result (late-time, force control): displacement is the biomarker",
        color=TEXT,
        fontsize=12,
        fontweight="bold",
    )
    ax_res.axis("off")
    ax_res.set_xlim(0, 28)
    ax_res.set_ylim(0, 5)

    # Two boxes: DH-baseline vs Commensal
    for idx, (name, e_eff, u_med, color) in enumerate(
        [
            (
                "DH-baseline (dysbiotic)\nE_eff = 0.50 MPa\n|U|_med \u2248 0.36 \u03bcm",
                0.5,
                0.358,
                "#E05252",
            ),
            (
                "Commensal (e.g. static)\nE_eff \u2248 9.85 MPa\n|U|_med \u2248 0.018 \u03bcm",
                9.85,
                0.018,
                "#4EC9B0",
            ),
        ]
    ):
        x0 = 2 + idx * 12
        b = FancyBboxPatch(
            (x0, 1.2),
            10,
            2.8,
            boxstyle="round,pad=0.15",
            facecolor=color + "22",
            edgecolor=color,
            lw=2,
            zorder=3,
        )
        ax_res.add_patch(b)
        ax_res.text(
            x0 + 5, 2.8, name, ha="center", va="center", fontsize=10, color=color, fontweight="bold"
        )
    ax_res.text(
        14,
        2.5,
        "\u2248 19.7\u00d7",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#B8860B",
    )
    ax_res.text(
        14,
        1.4,
        "displacement ratio\n(soft vs stiff biofilm)",
        ha="center",
        fontsize=9,
        color=TEXT_SEC,
    )
    ax_res.text(
        14,
        0.5,
        "MISES identical under force BC; displacement differentiates conditions.",
        ha="center",
        fontsize=9,
        color=TEXT_SEC,
    )

    # ---- Row 2: Summary table ----
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.set_facecolor(BG)
    ax_tbl.set_title("One-page summary", color=TEXT, fontsize=11, fontweight="bold")
    ax_tbl.axis("off")
    ax_tbl.set_xlim(0, 28)
    ax_tbl.set_ylim(0, 6)
    rows = [
        ("Stage", "Input", "Process", "Output"),
        ("TMCMC", "Exp. [6\u00d75]", "ODE + TMCMC (8 st., 150 pt.)", "MAP + 95% CI, 20 params"),
        (
            "Coupling",
            "\u03b8_MAP, snapshot",
            "DI = 1\u2212H/log5, E_eff(DI)",
            "Field CSV, INP per condition",
        ),
        ("FEM", "INP (3 teeth, Tie, 1 MPa)", "Abaqus Static", "ODB \u2192 MISES, U, figures"),
    ]
    col_x = [2, 8, 15, 24]
    for ri, row in enumerate(rows):
        y = 5.2 - ri * 1.2
        for ci, cell in enumerate(row):
            color = "#2D8B6F" if ri == 0 else TEXT_SEC
            weight = "bold" if ri == 0 else "normal"
            ax_tbl.text(
                col_x[ci],
                y,
                cell,
                ha="left" if ci == 0 else "center",
                va="center",
                fontsize=9,
                color=color,
                fontweight=weight,
            )
        if ri == 0:
            ax_tbl.plot([1, 27], [y - 0.25, y - 0.25], color=AXES, lw=1)

    # ---- Row 3: Takeaway bullets ----
    ax_bt = fig.add_subplot(gs[3, :])
    ax_bt.set_facecolor(BG)
    ax_bt.axis("off")
    ax_bt.set_xlim(0, 28)
    ax_bt.set_ylim(0, 3)
    bullets = [
        "What: Bayesian 5-species interaction (TMCMC) + 3-tooth conformal biofilm FEM (Abaqus).",
        "How: ODE\u00d7TMCMC \u2192 MAP \u2192 DI field \u2192 E_eff \u2192 INP \u2192 stress & displacement; 4 conditions.",
        "Finding: Late-time dysbiotic biofilm ~19.7\u00d7 softer \u2192 ~19.7\u00d7 larger displacement under 1 MPa (force control).",
    ]
    for i, line in enumerate(bullets):
        ax_bt.text(0.5, 2.4 - i * 0.9, "\u2022", fontsize=12, color="#2D8B6F", va="center")
        ax_bt.text(1.2, 2.4 - i * 0.9, line, fontsize=9.5, color=TEXT_SEC, va="center")
    ax_bt.text(
        14,
        0.3,
        "Report: biofilm_3tooth_report.pdf  |  Overview: overview_tmcmc_fem_en.md",
        ha="center",
        fontsize=8,
        color=TEXT_SEC,
    )

    save_fig(plt.gcf(), "fig7_combined_tmcmc_fem")
    plt.close()


# ===========================================================================
# FIG 8: TMCMC → FEM coupling — detailed flowchart
# ===========================================================================
def fig8_tmcmc_fem_flow():
    """Detailed flowchart: TMCMC output → DI field → E_eff → INP → Abaqus → results."""
    fig, ax = plt.subplots(figsize=(24, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 6)
    ax.axis("off")

    steps = [
        (2.2, "TMCMC\n\u03b8_MAP", "20 params\n4 conditions", "#2563EB"),
        (5.2, "3D FD / snapshot", "\u03d5_i(x,y,z)\nper condition", "#4E9AF1"),
        (8.2, "DI field", "DI = 1 \u2212 H/log(5)\nH = \u2212\u03a3 p_i ln p_i", "#6DC06D"),
        (
            11.2,
            "E_eff(DI)",
            "E_max(1\u2212r)^\u03b1 + E_min\u00b7r\nr = clip(DI/s_DI, 0, 1)",
            "#F4A142",
        ),
        (14.2, "Field CSV\n+ assembly", "biofilm_3tooth_\n{condition}.inp", "#B07CC6"),
        (17.2, "Abaqus\nStatic", "1 MPa inward\n82k nodes, 437k C3D4", "#E05252"),
        (20.2, "ODB \u2192 CSV", "MISES, |U|\nper tooth (T23/30/31)", "#2D8B6F"),
    ]
    for i, (x, title, sub, c) in enumerate(steps):
        w, h = 1.8, 2.2
        box = FancyBboxPatch(
            (x - w / 2, 2.2),
            w,
            h,
            boxstyle="round,pad=0.08",
            facecolor=c + "22",
            edgecolor=c,
            lw=2,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x,
            3.7,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=c,
            zorder=4,
        )
        ax.text(x, 3.0, sub, ha="center", va="center", fontsize=8, color=TEXT_SEC, zorder=4)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + w / 2 + 0.25, 3.3),
                xytext=(x + w / 2, 3.3),
                arrowprops=dict(arrowstyle="-|>", color=AXES, lw=2, mutation_scale=18),
            )
    ax.text(
        12,
        5.3,
        "TMCMC \u2192 FEM coupling (tmcmc_to_fem_coupling.py)",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        12,
        0.8,
        "Same geometry & load; only DI\u2192E_eff varies by condition \u2192 different displacement (force BC) or MISES (disp BC)",
        ha="center",
        fontsize=9,
        color=TEXT_SEC,
    )
    plt.tight_layout()
    save_fig(plt.gcf(), "fig8_tmcmc_fem_flow")
    plt.close()


# ===========================================================================
# FIG 9: DI → E_eff mapping (formula + curve)
# ===========================================================================
def fig9_coupling_DI_Eeff():
    """DI to effective modulus: formula and E_eff(DI) curve with condition regions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)

    # Left: formula and parameters
    ax1 = axes[0]
    ax1.axis("off")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.text(
        5, 9, "DI \u2192 E_eff mapping", ha="center", fontsize=14, fontweight="bold", color=TEXT
    )
    ax1.text(
        5,
        7.8,
        r"$r = \mathrm{clip}(\mathrm{DI}\,/\,s_{\mathrm{DI}},\ 0,\ 1)$",
        ha="center",
        fontsize=12,
    )
    ax1.text(
        5,
        6.8,
        r"$E_{\mathrm{eff}}(\mathrm{DI}) = E_{\max}(1-r)^\alpha + E_{\min}\cdot r$",
        ha="center",
        fontsize=11,
    )
    params = [
        ("E_max", "10 MPa", "stiff (commensal / healthy)"),
        ("E_min", "0.5 MPa", "soft (dysbiotic)"),
        (r"\alpha", "2", "exponent"),
        ("s_DI", "0.02578", "DI scale (from TMCMC MAP)"),
    ]
    for i, (sym, val, desc) in enumerate(params):
        y = 5.4 - i * 1.0
        ax1.text(1, y, sym + " = " + val, fontsize=10, color=TEXT, family="monospace")
        ax1.text(5.5, y, desc, fontsize=9, color=TEXT_SEC)
    ax1.text(
        5,
        0.8,
        "Low DI \u2192 high E_eff (stiff)\nHigh DI \u2192 low E_eff (soft)",
        ha="center",
        fontsize=9,
        color=TEXT_SEC,
    )

    # Right: E_eff(DI) curve
    ax2 = axes[1]
    E_max, E_min, alpha, s_DI = 10.0, 0.5, 2.0, 0.025778
    di = np.linspace(0, 0.6, 300)
    r = np.clip(di / s_DI, 0, 1)
    E_eff = E_max * (1 - r) ** alpha + E_min * r
    ax2.plot(di, E_eff, color="#2563EB", lw=2.5, label=r"$E_{\mathrm{eff}}(\mathrm{DI})$")
    ax2.axvspan(0, 0.02, alpha=0.12, color="#4EC9B0")
    ax2.axvspan(0.15, 0.6, alpha=0.12, color="#E05252")
    ax2.axhline(E_min, color="#C44", lw=1, ls="--", alpha=0.7)
    ax2.text(0.01, 8.5, "commensal\n(low DI)", fontsize=9, color="#2D8B6F")
    ax2.text(0.35, 1.2, "dysbiotic (late)\nE_eff = E_min", fontsize=9, color="#C44")
    ax2.set_xlabel("DI (Dysbiotic Index)", color=TEXT, fontsize=11)
    ax2.set_ylabel(r"$E_{\mathrm{eff}}$ (MPa)", color=TEXT, fontsize=11)
    ax2.tick_params(colors=AXES)
    for spine in ax2.spines.values():
        spine.set_color(AXES)
    ax2.legend(facecolor=LEGEND_BG, edgecolor=LEGEND_EC, labelcolor=TEXT, fontsize=10)
    ax2.set_xlim(0, 0.6)
    ax2.set_ylim(0, 11)
    plt.tight_layout()
    save_fig(plt.gcf(), "fig9_coupling_DI_Eeff")
    plt.close()


# ===========================================================================
# FIG 10: Four conditions → same FEM → different mechanical outcome
# ===========================================================================
def fig10_coupling_four_conditions():
    """Four conditions: same geometry/load, different \u03b8_MAP \u2192 different DI \u2192 different E_eff \u2192 different |U|."""
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Top: four condition boxes
    conditions = [
        (
            "dh_baseline",
            "Dysbiotic\n(a_{35} high)",
            "DI \u2248 0.51 (late)",
            "E_eff \u2248 0.5 MPa",
            "#E05252",
        ),
        (
            "commensal_static",
            "Commensal\n(balanced)",
            "DI \u2248 0",
            "E_eff \u2248 9.85 MPa",
            "#4EC9B0",
        ),
        ("dysbiotic_static", "Dysbiotic\n(moderate)", "DI low (early)", "E_eff mid", "#F4A142"),
        (
            "commensal_hobic",
            "Commensal\n(HOBIC)",
            "DI \u2248 0",
            "E_eff \u2248 9.85 MPa",
            "#4E9AF1",
        ),
    ]
    for i, (key, label, di_str, e_str, c) in enumerate(conditions):
        x = 2.2 + i * 4.0
        box = FancyBboxPatch(
            (x - 1.5, 4.8),
            3.0,
            2.6,
            boxstyle="round,pad=0.1",
            facecolor=c + "28",
            edgecolor=c,
            lw=1.5,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, 6.9, key, ha="center", fontsize=9, fontweight="bold", color=c)
        ax.text(x, 6.2, label, ha="center", fontsize=9, color=TEXT_SEC)
        ax.text(x, 5.5, di_str, ha="center", fontsize=8, color=TEXT_SEC)
        ax.text(x, 4.95, e_str, ha="center", fontsize=8, color=TEXT_SEC)

    # Arrows from conditions to central "same INP geometry"
    ax.text(
        9,
        6.0,
        "Same 3-tooth geometry\n1 MPa inward, Tie at slit",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=TEXT,
    )
    box_mid = FancyBboxPatch(
        (7, 5.0), 4, 1.6, boxstyle="round,pad=0.08", facecolor=LEGEND_BG, edgecolor=AXES, lw=1
    )
    ax.add_patch(box_mid)
    for i in range(4):
        x = 2.2 + i * 4.0
        ax.annotate(
            "",
            xy=(8.2, 5.8),
            xytext=(x + 1.5, 4.8),
            arrowprops=dict(arrowstyle="-|>", color=AXES, lw=1.2, connectionstyle="arc3,rad=0.2"),
        )
    ax.annotate(
        "", xy=(9, 4.2), xytext=(9, 5.0), arrowprops=dict(arrowstyle="-|>", color=AXES, lw=2)
    )
    ax.text(9, 3.6, "Abaqus Static", ha="center", fontsize=10, fontweight="bold", color=TEXT)
    ax.annotate(
        "", xy=(9, 2.4), xytext=(9, 3.2), arrowprops=dict(arrowstyle="-|>", color=AXES, lw=2)
    )
    ax.text(9, 2.0, "ODB \u2192 MISES, |U|", ha="center", fontsize=10, color=TEXT_SEC)
    ax.text(
        9,
        1.2,
        "Force BC: MISES same; |U| \u2248 19.7\u00d7 larger for DH-baseline (soft)",
        ha="center",
        fontsize=9,
        color="#2D8B6F",
    )
    plt.tight_layout()
    save_fig(plt.gcf(), "fig10_coupling_four_conditions")
    plt.close()


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Generating overview figures (English, multi-format)...")
    print(f"Output: {OUT}\n")
    fig1_pipeline()
    fig2_network()
    fig3_tmcmc()
    fig4_data_model()
    fig5_positioning()
    fig6_poster()
    fig7_combined_poster()
    fig8_tmcmc_fem_flow()
    fig9_coupling_DI_Eeff()
    fig10_coupling_four_conditions()
    print(f"\nDone. Figures saved as PNG, PDF, SVG in:\n  {OUT}")


if __name__ == "__main__":
    main()
