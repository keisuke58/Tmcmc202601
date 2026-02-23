#!/usr/bin/env python3
"""Render all equations as high-quality PNG images using matplotlib mathtext."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['mathtext.fontset'] = 'cm'  # Computer Modern (LaTeX-like)
rcParams['font.family'] = 'serif'

OUT = "/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/docs/_eq_img"
os.makedirs(OUT, exist_ok=True)

def render_eq(filename, latex_str, fontsize=28, figw=10, figh=1.0, color='#2D2D2D'):
    fig, ax = plt.subplots(figsize=(figw, figh))
    fig.patch.set_alpha(0)
    ax.text(0.5, 0.5, latex_str, fontsize=fontsize, ha='center', va='center',
            color=color, transform=ax.transAxes, math_fontfamily='cm')
    ax.axis('off')
    path = os.path.join(OUT, filename)
    fig.savefig(path, dpi=250, bbox_inches='tight', transparent=True, pad_inches=0.08)
    plt.close()
    print(f"  {filename}")

print("Rendering equations...")

# ── 1. Volume fraction ODE ──
render_eq("eq_phi.png",
    r"$\frac{d\phi_i}{dt} = r_i \cdot \phi_i \cdot \psi_i \cdot \left(1 - \sum_j \phi_j\right) "
    r"- d_i \cdot \phi_i \cdot (1 - \psi_i)$",
    fontsize=26, figw=14, figh=1.2)

# ── 2. Survival fraction ODE ──
render_eq("eq_psi.png",
    r"$\frac{d\psi_i}{dt} = (\alpha_i - \beta_i \psi_i)(1 - \psi_i) "
    r"+ \sum_j a_{ij} \cdot \frac{\phi_j \cdot \psi_j}{K + \phi_j}$",
    fontsize=26, figw=14, figh=1.2)

# ── 3. Bayes' theorem ──
render_eq("eq_bayes.png",
    r"$p(\boldsymbol{\theta} \mid \mathbf{D}) = "
    r"\frac{p(\mathbf{D} \mid \boldsymbol{\theta}) \cdot p(\boldsymbol{\theta})}"
    r"{p(\mathbf{D})}$",
    fontsize=36, figw=10, figh=1.5)

# Bayes labeled version
render_eq("eq_bayes_labels.png",
    r"$p(\boldsymbol{\theta} \mid \mathbf{D})"
    r"\;=\; \frac{p(\mathbf{D} \mid \boldsymbol{\theta})"
    r"\;\cdot\; p(\boldsymbol{\theta})}"
    r"{p(\mathbf{D})}$",
    fontsize=34, figw=12, figh=1.8)

# Bayes component labels as separate image
render_eq("eq_bayes_components.png",
    r"$\mathrm{Posterior}\;=\;\frac{\mathrm{Likelihood}\;\times\;\mathrm{Prior}}"
    r"{\mathrm{Evidence}}$",
    fontsize=26, figw=10, figh=1.3, color='#888888')

# ── 4. Likelihood ──
render_eq("eq_likelihood.png",
    r"$p(\mathbf{D} \mid \boldsymbol{\theta}) = "
    r"\prod_{i=1}^{N_s} \prod_{j=1}^{N_t} "
    r"\mathcal{N}\!\left(d_{ij} \;\middle|\; g_i(t_j, \boldsymbol{\theta}),\, \sigma^2\right)$",
    fontsize=26, figw=14, figh=1.3)

# ── 5. Log-likelihood ──
render_eq("eq_loglikelihood.png",
    r"$\ln p(\mathbf{D} \mid \boldsymbol{\theta}) = "
    r"-\frac{1}{2\sigma^2} \sum_{i=1}^{N_s} \sum_{j=1}^{N_t} "
    r"\left[ d_{ij} - g_i(t_j, \boldsymbol{\theta}) \right]^2 + \mathrm{const.}$",
    fontsize=24, figw=14, figh=1.3)

# ── 6. Prior with constraints ──
render_eq("eq_prior.png",
    r"$p(\boldsymbol{\theta}) = p_{\mathrm{free}}(\boldsymbol{\theta}_{\mathrm{free}}) "
    r"\cdot \prod_{k \in \mathcal{L}} \delta(\theta_k)$",
    fontsize=28, figw=12, figh=1.3)

render_eq("eq_prior_uniform.png",
    r"$\boldsymbol{\theta}_{\mathrm{free}} \sim \mathrm{Uniform}[\mathbf{L},\, \mathbf{U}]$",
    fontsize=26, figw=8, figh=0.8)

# ── 7. Tempered distribution ──
render_eq("eq_tempered.png",
    r"$p_j(\boldsymbol{\theta}) \propto "
    r"p(\boldsymbol{\theta}) \cdot p(\mathbf{D} \mid \boldsymbol{\theta})^{\beta_j}$",
    fontsize=30, figw=12, figh=1.3)

render_eq("eq_beta_seq.png",
    r"$0 = \beta_0 < \beta_1 < \beta_2 < \cdots < \beta_m = 1$",
    fontsize=26, figw=12, figh=0.8)

# ── 8. Importance weights ──
render_eq("eq_weights.png",
    r"$w^{(i)} = p(\mathbf{D} \mid \boldsymbol{\theta}^{(i)})^{\Delta\beta}, \quad "
    r"W^{(i)} = \frac{w^{(i)}}{\sum_{k=1}^{N} w^{(k)}}$",
    fontsize=24, figw=14, figh=1.0)

# ── 9. Model evidence ──
render_eq("eq_evidence.png",
    r"$\hat{p}(\mathbf{D}) = \prod_{j=0}^{m-1} \left[ \frac{1}{N} \sum_{i=1}^{N} w_j^{(i)} \right]$",
    fontsize=30, figw=10, figh=1.5)

# ── 10. Bayes factor ──
render_eq("eq_bayes_factor.png",
    r"$\mathrm{BF}_{12} = \frac{p(\mathbf{D} \mid \mathcal{M}_1)}"
    r"{p(\mathbf{D} \mid \mathcal{M}_2)}$",
    fontsize=30, figw=8, figh=1.5)

# ── 11. Covariance adaptation ──
render_eq("eq_covariance.png",
    r"$\boldsymbol{\Sigma}_j = \beta_{\mathrm{scale}}^2 \cdot "
    r"\mathrm{Cov}_{\mathbf{W}}(\boldsymbol{\theta})$",
    fontsize=26, figw=10, figh=1.0)

# ── 12. Interaction matrix (rendered as multi-row) ──
def render_matrix(filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_alpha(0)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Title
    ax.text(5, 5.5, r"$\mathbf{A}\;=\;$", fontsize=28, ha='right', va='center',
            color='#2D2D2D', math_fontfamily='cm')

    # Matrix bracket (left)
    ax.plot([5.1, 5.3, 5.3, 5.1], [0.5, 0.5, 5.5, 5.5], color='#2D2D2D', lw=2.5)
    # Matrix bracket (right)
    ax.plot([9.7, 9.5, 9.5, 9.7], [0.5, 0.5, 5.5, 5.5], color='#2D2D2D', lw=2.5)

    # Species labels top
    species = ['S.o', 'A.n', 'Vei', 'F.n', 'P.g']
    cols = [5.8, 6.6, 7.3, 8.05, 8.9]
    for sp, cx in zip(species, cols):
        ax.text(cx, 5.8, sp, fontsize=11, ha='center', va='center',
                color='#0072CE', style='italic')

    rows = [
        [r'$\theta_0$',  r'$\theta_5$',  r'$\theta_6$',  r'$\theta_7$',  r'$\theta_8$'],
        [r'$\theta_5$',  r'$\theta_1$',  r'$\theta_9$',  r'$\theta_{10}$',r'$\theta_{11}$'],
        [r'$\theta_6$',  r'$\theta_9$',  r'$\theta_2$',  r'$\theta_{12}$',r'$\theta_{13}$'],
        [r'$\theta_7$',  r'$\theta_{10}$',r'$\theta_{12}$',r'$\theta_3$', r'$\theta_{14}$'],
        [r'$\theta_8$',  r'$\theta_{11}$',r'$\theta_{13}$',r'$\theta_{14}$',r'$\theta_4$'],
    ]
    locked = {(0,2),(1,2),(1,3),(0,4),(1,4), (2,0),(2,1),(3,1),(4,0),(4,1)}  # symmetric
    lock_idx = {(0,2),(2,0), (1,2),(2,1), (1,3),(3,1), (0,4),(4,0), (1,4),(4,1)}
    y_pos = [4.8, 4.0, 3.2, 2.4, 1.6]

    for ri, (row, ry) in enumerate(zip(rows, y_pos)):
        # Row label
        ax.text(5.2, ry, species[ri], fontsize=11, ha='right', va='center',
                color='#0072CE', style='italic')
        for ci, (cell, cx) in enumerate(zip(row, cols)):
            is_locked = (ri, ci) in lock_idx
            c = '#E83E3E' if is_locked else '#2D2D2D'
            fw = 'bold' if ri == ci else 'normal'
            ax.text(cx, ry, cell, fontsize=16, ha='center', va='center',
                    color=c, math_fontfamily='cm')
            if is_locked:
                ax.text(cx, ry - 0.35, r'$= 0$', fontsize=9, ha='center',
                        va='center', color='#E83E3E', math_fontfamily='cm')

    path = os.path.join(OUT, filename)
    fig.savefig(path, dpi=250, bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close()
    print(f"  {filename}")

render_matrix("eq_matrix_A.png")

# ── 13. Decay vector ──
render_eq("eq_decay.png",
    r"$\mathbf{b} = [\theta_{15},\; \theta_{16},\; \theta_{17},\; \theta_{18},\; \theta_{19}]$",
    fontsize=24, figw=10, figh=0.8)

# ── 14. Symmetry constraint ──
render_eq("eq_symmetry.png",
    r"$A_{ij} = A_{ji} \quad \Longrightarrow \quad "
    r"30 \;\rightarrow\; 20 \;\mathrm{parameters}$",
    fontsize=26, figw=12, figh=1.0)

# ── 15. Parameter reduction summary ──
render_eq("eq_reduction.png",
    r"$30 \;\overset{A_{ij}=A_{ji}}{\longrightarrow}\; 20 "
    r"\;\overset{\mathrm{lock\;5}}{\longrightarrow}\; 15 \;\mathrm{free\;parameters}$",
    fontsize=28, figw=14, figh=1.5)

# ── 16. Marginal likelihood integral ──
render_eq("eq_marginal.png",
    r"$p(\mathbf{D}) = \int p(\mathbf{D} \mid \boldsymbol{\theta}) \, "
    r"p(\boldsymbol{\theta}) \, d\boldsymbol{\theta}$",
    fontsize=28, figw=10, figh=1.3)

# ── 17. CoV target ──
render_eq("eq_cov_target.png",
    r"$\mathrm{CoV}(\mathbf{w}) = \frac{\mathrm{std}(\mathbf{w})}{\mathrm{mean}(\mathbf{w})} = 1.0"
    r"\quad \Longrightarrow \quad \mathrm{ESS} \approx \frac{N}{2}$",
    fontsize=24, figw=14, figh=1.0)

# ── 18. MH acceptance ──
render_eq("eq_mh_accept.png",
    r"$\alpha = \min\!\left(1,\; \frac{p(\mathbf{D}\mid\boldsymbol{\theta}^*)^{\beta_j} "
    r"\cdot p(\boldsymbol{\theta}^*)}"
    r"{p(\mathbf{D}\mid\boldsymbol{\theta}^{(i)})^{\beta_j} "
    r"\cdot p(\boldsymbol{\theta}^{(i)})}\right)$",
    fontsize=22, figw=14, figh=1.3)

print(f"\nAll equations rendered to: {OUT}")
print(f"Total: {len(os.listdir(OUT))} images")
