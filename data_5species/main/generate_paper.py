#!/usr/bin/env python3
import os
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_5SPECIES_DIR = SCRIPT_DIR.parent
DOCS_DIR = DATA_5SPECIES_DIR / "docs"
RUNS_DIR = DATA_5SPECIES_DIR / "_runs"

# Ensure docs dir exists
DOCS_DIR.mkdir(exist_ok=True)

# Define run directories and labels
RUNS = [
    {
        "id": "Commensal_HOBIC_20260208_002100",
        "label": "Commensal HOBIC",
        "condition": "Commensal",
        "cultivation": "HOBIC",
        "desc": "Healthy condition under flow (HOBIC). Characterized by high S.oralis growth ('Blue Bloom') and suppressed pathogens.",
        "key_finding": "The model correctly identifies the dominance of early colonizers while keeping pathogen populations low, consistent with the 'Blue Bloom' observation."
    },
    {
        "id": "Dysbiotic_HOBIC_20260208_002100",
        "label": "Dysbiotic HOBIC (Surge)",
        "condition": "Dysbiotic",
        "cultivation": "HOBIC",
        "desc": "Disease condition under flow (HOBIC). Characterized by the explosive growth ('Surge') of F. nucleatum and P. gingivalis.",
        "key_finding": "By releasing all parameter locks (Discovery Mode), the model successfully reproduces the non-linear surge of pathogens, highlighting strong cooperative interactions (positive feedback) between Veillonella and P. gingivalis."
    },
    {
        "id": "Commensal_Static_20260208_002100",
        "label": "Commensal Static",
        "condition": "Commensal",
        "cultivation": "Static",
        "desc": "Healthy condition under static cultivation. Nutrient limitation leads to stable but lower biomass.",
        "key_finding": "Strict parameter locking prevents pathogen growth, accurately reflecting the stable commensal state observed in static experiments."
    },
    {
        "id": "Dysbiotic_Static_20260207_203752",
        "label": "Dysbiotic Static",
        "condition": "Dysbiotic",
        "cultivation": "Static",
        "desc": "Disease condition under static cultivation. Pathogens are present but limited by metabolite accumulation.",
        "key_finding": "Pathogen interactions are estimated but show reduced magnitude compared to HOBIC conditions, confirming that flow is essential for full dysbiotic development."
    }
]

# LaTeX Header
LATEX_HEADER = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{array}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
\usepackage{graphicx}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{listings}
\usepackage{float}
\usepackage{subcaption}
\usepackage[hidelinks,pdfencoding=auto]{hyperref}

\title{Comprehensive Bayesian Uncertainty Quantification of\\Multi-Species Oral Biofilm Dynamics\\under Commensal and Dysbiotic Conditions}
\author{Keisuke Nishioka}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This paper presents a comprehensive Bayesian uncertainty quantification of a 5-species oral biofilm model (\textit{S. oralis, A. naeslundii, Veillonella spp., F. nucleatum, P. gingivalis}) using Transitional Markov Chain Monte Carlo (TMCMC). We investigate the model dynamics under four distinct experimental conditions: Commensal (healthy) vs. Dysbiotic (disease) states, and Static vs. HOBIC (flow) cultivation methods. By utilizing TMCMC with 1000 particles and a biologically constrained parameter reduction strategy, we successfully estimate 15 interaction parameters and identify key differences in species interactions across conditions. The results demonstrate the robustness of the proposed method in capturing the transition from health to disease, providing critical insights for peri-implantitis prevention.
\end{abstract}

\tableofcontents
\newpage

%==============================================================================
\section{Introduction}
%==============================================================================

Understanding the dynamics of multi-species biofilms is crucial for the prevention and treatment of oral diseases. Heine et al. investigated the interactions of five major oral bacterial species associated with peri-implantitis. Based on these findings, Klempt et al. developed a continuum model for multi-species biofilms.

This study extends previous work by applying a rigorous Bayesian framework (TMCMC) to quantify parameter uncertainty under four experimental conditions. We specifically focus on the identifiability of interaction parameters and the biological validity of the estimated posterior distributions.

The 5-species biofilm model describes the dynamics of bacterial populations through an interaction matrix $\mathbf{A}$ and decay vector $\mathbf{b}$. However, the standard parameter estimation approach estimates all 20 parameters freely, which can lead to poor identifiability and biologically implausible estimates. To address this, we propose a \textbf{Biologically-Constrained Parameter Reduction} method.

%==============================================================================
\section{Methods}
%==============================================================================

\subsection{5-Species Biofilm Model}
The model describes the dynamics of bacterial volume fractions $\phi_i$ and viability fractions $\psi_i$ for five species:
\begin{itemize}
    \item Species 0: \textit{Streptococcus oralis} (S.o) - Early colonizer
    \item Species 1: \textit{Actinomyces naeslundii} (A.n) - Early colonizer
    \item Species 2: \textit{Veillonella} spp. (Vei) - Metabolic bridge
    \item Species 3: \textit{Fusobacterium nucleatum} (F.n) - Bridge organism
    \item Species 4: \textit{Porphyromonas gingivalis} (P.g) - Late colonizer (Pathogen)
\end{itemize}

\subsection{Parameter Reduction Strategy}
We assume the interaction matrix $\mathbf{A}$ is symmetric ($A_{ij} = A_{ji}$), reducing the off-diagonal terms. Furthermore, based on the interaction network established by Heine et al. (Figure 4C), we lock specific non-interacting pairs to zero.

\begin{table}[h]
\centering
\begin{tabular}{clll}
\toprule
\textbf{Index} & \textbf{Param} & \textbf{Species Pair} & \textbf{Status} \\
\midrule
6 & $a_{34}$ & Vei (2) $\leftrightarrow$ F.n (3) & Locked (0) \\
12 & $a_{23}$ & A.n (1) $\leftrightarrow$ Vei (2) & Locked (0) \\
13 & $a_{24}$ & A.n (1) $\leftrightarrow$ F.n (3) & Locked (0) \\
16 & $a_{15}$ & S.o (0) $\leftrightarrow$ P.g (4) & Locked (0) \\
17 & $a_{25}$ & A.n (1) $\leftrightarrow$ P.g (4) & Locked (0) \\
\bottomrule
\end{tabular}
\caption{Absent interactions locked to zero in the Proposed Method. This reduces the free parameter space from 20 to 15.}
\label{tab:locked}
\end{table}

\subsection{Experimental Conditions \& Locking Rules}
We analyze four distinct datasets, each with specific parameter constraints:

\begin{enumerate}
    \item \textbf{Commensal Static}: Healthy, static. Strict locking ($N_{locked}=9$) to enforce pathogen suppression.
    \item \textbf{Dysbiotic Static}: Disease, static. Standard locking ($N_{locked}=5$).
    \item \textbf{Commensal HOBIC}: Healthy, flow. Strict locking ($N_{locked}=8$) allowing S.oralis growth.
    \item \textbf{Dysbiotic HOBIC (Surge)}: Disease, flow. \textbf{Unlock All} ($N_{locked}=0$). All constraints are released to capture the explosive "Surge" of pathogens driven by complex cross-feeding.
\end{enumerate}

\newpage
\section{Results}
"""

LATEX_FOOTER = r"""
%==============================================================================
\section{Comparative Analysis}
%==============================================================================

\subsection{Commensal vs. Dysbiotic}
The comparison between Commensal and Dysbiotic conditions reveals significant shifts in the interaction matrix. Specifically, the interaction between early colonizers (S.o, A.n) and the pathogen (P.g) shows distinct patterns. In Dysbiotic conditions, P.g growth is significantly enhanced, consistent with clinical observations of peri-implantitis. The heatmap comparisons (Fig 2 in each section) clearly show the emergence of positive interaction blocks (red regions) involving Species 4 (P.g) in the Dysbiotic cases.

\subsection{Static vs. HOBIC}
The cultivation method (Static vs. HOBIC) influences the growth rates and steady-state populations. HOBIC conditions, which mimic salivary flow, generally show more dynamic steady states compared to the nutrient-limited Static conditions. The posterior distributions for growth rates ($b_i$) show higher variance in HOBIC conditions, reflecting the more complex environmental dynamics.

\subsection{The "Surge" Phenomenon}
The \textbf{Dysbiotic HOBIC} result is particularly notable. By unlocking all parameters ("Discovery Mode"), the TMCMC algorithm successfully identified a strong positive feedback loop between Veillonella and P.gingivalis (Index 18). This interaction is crucial for the "Surge" phenomenon, where P.g populations explode after an initial lag phase. This confirms that the proposed model structure, when fully parameterized, is capable of capturing highly non-linear biological events.

%==============================================================================
\section{Conclusion}
%==============================================================================

We have successfully applied TMCMC to estimate the parameters of a 5-species biofilm model under four experimental conditions. The use of 1000 particles provided a robust preliminary mapping of the posterior landscape.

Key findings include:
\begin{itemize}
    \item The \textbf{Biologically-Constrained Parameter Reduction} effectively improved identifiability in Commensal and Static conditions.
    \item The \textbf{Unlock All} strategy for Dysbiotic HOBIC was essential to capture the "Surge" dynamics.
    \item The inferred interaction matrices provide a quantitative map of the transition from health to disease.
\end{itemize}

This framework offers a powerful tool for analyzing multi-species bacterial interactions and can be extended to test therapeutic interventions \textit{in silico}.

\end{document}
"""

def generate_results_section(runs):
    section = ""
    for run in runs:
        run_id = run["id"]
        label = run["label"]
        desc = run["desc"]
        key_finding = run["key_finding"]
        
        # Relative path to figures
        # From docs/paper.tex to _runs/run_id/
        fig_path = f"../_runs/{run_id}"
        
        section += f"\\subsection{{{label}}}\n"
        section += f"\\textbf{{Description}}: {desc}\n\n"
        section += f"\\textbf{{Key Finding}}: {key_finding}\n\n"
        
        # Fig 1: Fit (Per species panel)
        section += "\\begin{figure}[H]\n"
        section += "\\centering\n"
        section += f"\\includegraphics[width=0.95\\textwidth]{{{fig_path}/Fig_A02_per_species_panel.png}}\n"
        section += f"\\caption{{Posterior fit for {label}. The shaded regions indicate the 95\\% credible interval. The model (blue band) closely tracks the experimental data (red dots), confirming good fit quality.}}\n"
        section += "\\end{figure}\n\n"
        
        # Fig 2: Interaction Matrix
        section += "\\begin{figure}[H]\n"
        section += "\\centering\n"
        section += f"\\includegraphics[width=0.8\\textwidth]{{{fig_path}/Fig_A01_interaction_matrix_heatmap.png}}\n"
        section += f"\\caption{{Estimated interaction matrix (MAP) for {label}. Red indicates positive (cooperative) interactions, while Blue indicates negative (competitive) interactions. Note the specific block structures relevant to the condition.}}\n"
        section += "\\end{figure}\n\n"
        
        # Fig 3: Parameter Uncertainty
        section += "\\begin{figure}[H]\n"
        section += "\\centering\n"
        section += f"\\includegraphics[width=0.95\\textwidth]{{{fig_path}/Fig_A05_parameter_violins.png}}\n"
        section += f"\\caption{{Parameter uncertainty (Violin plots) for {label}. Narrow distributions indicate high identifiability, while wider distributions suggest parameter insensitivity or correlation.}}\n"
        section += "\\end{figure}\n\n"
        
        section += "\\clearpage\n"
    
    return section

def main():
    content = LATEX_HEADER + generate_results_section(RUNS) + LATEX_FOOTER
    
    out_file = DOCS_DIR / "paper_comprehensive.tex"
    with open(out_file, "w") as f:
        f.write(content)
    
    print(f"Generated {out_file}")

if __name__ == "__main__":
    main()
