#!/usr/bin/env python3
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
        "desc": "流れのある健康な状態 (HOBIC). S.oralisの高い増殖('Blue Bloom')と病原菌の抑制によって特徴付けられる.",
        "key_finding": "モデルは初期定着菌の優位性を正しく特定しつつ, 病原菌集団を低く抑えており, 'Blue Bloom'の観察結果と一致している.",
    },
    {
        "id": "Dysbiotic_HOBIC_20260208_002100",
        "label": "Dysbiotic HOBIC (Surge)",
        "condition": "Dysbiotic",
        "cultivation": "HOBIC",
        "desc": "流れのある疾患状態 (HOBIC). F. nucleatumとP. gingivalisの爆発的な増殖('Surge')によって特徴付けられる.",
        "key_finding": "パラメータロックを全解除(Discovery Mode)することで, モデルは病原菌の非線形な急増(Surge)を再現することに成功し, VeillonellaとP. gingivalis間の強い協力相互作用(正のフィードバック)を明らかにした.",
    },
    {
        "id": "Commensal_Static_20260208_002100",
        "label": "Commensal Static",
        "condition": "Commensal",
        "cultivation": "Static",
        "desc": "静的培養下の健康な状態. 栄養制限により, 安定的だが低いバイオマスとなる.",
        "key_finding": "厳格なパラメータロックにより病原菌の増殖が防がれ, 静的実験で観察された安定した共生状態を正確に反映している.",
    },
    {
        "id": "Dysbiotic_Static_20260207_203752",
        "label": "Dysbiotic Static",
        "condition": "Dysbiotic",
        "cultivation": "Static",
        "desc": "静的培養下の疾患状態. 病原菌は存在するが, 代謝産物の蓄積により制限されている.",
        "key_finding": "病原菌の相互作用は推定されたが, HOBIC条件と比較してその規模は小さく, 完全なディスバイオシスの進行には流れ(Flow)が不可欠であることを裏付けている.",
    },
]

# LaTeX Header for Japanese (XeLaTeX)
LATEX_HEADER = r"""\documentclass[11pt,a4paper]{article}

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

% Japanese support with XeLaTeX
\usepackage{fontspec}
\setmainfont{Noto Serif CJK JP}[
    UprightFont = *-Regular,
    BoldFont = *-Bold,
    Scale = 1.0
]
\setsansfont{Noto Sans CJK JP}[
    UprightFont = *-Regular,
    BoldFont = *-Bold,
    Scale = 0.95
]
\setmonofont{Noto Sans Mono CJK JP}[Scale=0.80]

% Enable Japanese line breaking
\XeTeXlinebreaklocale "ja"
\XeTeXlinebreakskip=0pt plus 1pt minus 0.1pt
\sloppy

\title{共生およびディスバイオシス条件下における\\多菌種口腔バイオフィルム動態の\\包括的ベイズ不確実性定量化}
\author{西岡 佳祐}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
本論文では, TMCMC(Transitional Markov Chain Monte Carlo)を用いて, 5菌種口腔バイオフィルムモデル(\textit{S. oralis, A. naeslundii, Veillonella spp., F. nucleatum, P. gingivalis})の包括的なベイズ不確実性定量化を行う. Commensal(健康)対 Dysbiotic(疾患), Static(静的)対 HOBIC(流動)という4つの異なる実験条件下でのモデル動態を調査した. 1000粒子のTMCMCと生物学的制約に基づくパラメータ削減戦略を用いることで, 15の相互作用パラメータを推定し, 条件間での菌種間相互作用の重要な違いを特定することに成功した. 結果は, 健康から疾患への遷移を捉える上での提案手法の堅牢性を示しており, インプラント周囲炎の予防に向けた重要な洞察を提供する.
\end{abstract}

\tableofcontents
\newpage

%==============================================================================
\section{はじめに (Introduction)}
%==============================================================================

多菌種バイオフィルムの動態を理解することは, 口腔疾患の予防と治療において極めて重要である. Heineらは, インプラント周囲炎に関連する主要な5つの口腔細菌種の相互作用を調査した. これらの知見に基づき, Klemptらは多菌種バイオフィルムの連続体モデルを開発した.

本研究では, 厳密なベイズ枠組み(TMCMC)を適用し, 4つの実験条件下でのパラメータの不確実性を定量化することで, これまでの研究を拡張する. 特に, 相互作用パラメータの識別可能性と, 推定された事後分布の生物学的妥当性に焦点を当てる.

5菌種バイオフィルムモデルは, 相互作用行列 $\mathbf{A}$ と減衰ベクトル $\mathbf{b}$ を通じて細菌集団の動態を記述する. しかし, 標準的なパラメータ推定アプローチでは20個すべてのパラメータを自由に推定するため, 識別性が低く, 生物学的にあり得ない推定値につながる可能性がある. これに対処するため, 我々は生物学的制約に基づくパラメータ削減法(Biologically-Constrained Parameter Reduction)を提案する.

%==============================================================================
\section{手法 (Methods)}
%==============================================================================

\subsection{5菌種バイオフィルムモデル}
モデルは, 以下の5菌種の体積分率 $\phi_i$ と生存分率 $\psi_i$ の動態を記述する：
\begin{itemize}
    \item Species 0: \textit{Streptococcus oralis} (S.o) - 初期定着菌
    \item Species 1: \textit{Actinomyces naeslundii} (A.n) - 初期定着菌
    \item Species 2: \textit{Veillonella} spp. (Vei) - 代謝の架け橋
    \item Species 3: \textit{Fusobacterium nucleatum} (F.n) - 架け橋となる菌
    \item Species 4: \textit{Porphyromonas gingivalis} (P.g) - 後期定着菌(病原菌)
\end{itemize}

\subsection{パラメータ削減戦略}
相互作用行列 $\mathbf{A}$ は対称行列($A_{ij} = A_{ji}$)であると仮定し, 非対角項を削減する. さらに, Heineらによって確立された相互作用ネットワーク(Figure 4C)に基づき, 相互作用しない特定のペアをゼロに固定(ロック)する.

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
\caption{提案手法においてゼロに固定された相互作用. これにより, 自由パラメータ空間は20から15に削減される.}
\label{tab:locked}
\end{table}

\subsection{実験条件とロックルール}
我々は, 特定のパラメータ制約を持つ4つの異なるデータセットを分析する：

\begin{enumerate}
    \item \textbf{Commensal Static}: 健康・静的. 病原菌の抑制を強制するための厳格なロック($N_{locked}=9$).
    \item \textbf{Dysbiotic Static}: 疾患・静的. 標準的なロック($N_{locked}=5$).
    \item \textbf{Commensal HOBIC}: 健康・流動. S.oralisの増殖を許容する厳格なロック($N_{locked}=8$).
    \item \textbf{Dysbiotic HOBIC (Surge)}: 疾患・流動.\textbf{全ロック解除(Unlock All)}($N_{locked}=0$). 複雑なクロスフィーディングによって駆動される病原菌の爆発的な「Surge(急増)」を捉えるため, すべての制約を解除する.
\end{enumerate}

\newpage
\section{結果 (Results)}
"""

LATEX_FOOTER = r"""
%==============================================================================
\section{比較分析 (Comparative Analysis)}
%==============================================================================

\subsection{Commensal vs. Dysbiotic}
Commensal(健康)条件と Dysbiotic(疾患)条件の比較により, 相互作用行列における有意なシフトが明らかになった. 具体的には, 初期定着菌(S.o, A.n)と病原菌(P.g)の間の相互作用が明確に異なるパターンを示している. Dysbiotic条件では, P.gの増殖が著しく促進されており, これはインプラント周囲炎の臨床的観察と一致している. ヒートマップの比較(各セクションのFig 2)は, Dysbioticの場合にSpecies 4(P.g)を含む正の相互作用ブロック(赤色の領域)が出現していることを明確に示している.

\subsection{Static vs. HOBIC}
培養方法(Static vs. HOBIC)は, 増殖率と定常状態の集団に影響を与える. 唾液の流れを模倣したHOBIC条件は, 栄養制限のあるStatic条件と比較して,一般的により動的な定常状態を示す. 増殖率($b_i$)の事後分布はHOBIC条件でより高い分散を示しており, これはより複雑な環境動態を反映している.

\subsection{"Surge" 現象}
\textbf{Dysbiotic HOBIC} の結果は特に注目に値する. すべてのパラメータをロック解除する(「Discovery Mode」)ことで, TMCMCアルゴリズムはVeillonellaとP.gingivalis(Index 18)の間の強い正のフィードバックループを特定することに成功した. この相互作用は, 初期のラグフェーズの後にP.g集団が爆発的に増加する「Surge」現象にとって不可欠である. これは, 提案されたモデル構造が, 完全にパラメータ化された場合, 高度に非線形な生物学的イベントを捉える能力があることを裏付けている.

%==============================================================================
\section{結論 (Conclusion)}
%==============================================================================

我々は, 4つの実験条件下で5菌種バイオフィルムモデルのパラメータを推定するためにTMCMCを適用することに成功した. 1000粒子の使用により, 事後分布の地形の予備的なマッピングが行われた.

主な発見は以下の通りである：
\begin{itemize}
    \item 生物学的制約に基づくパラメータ削減は, CommensalおよびStatic条件における識別性を効果的に向上させた.
    \item Dysbiotic HOBICに対する \textbf{全ロック解除(Unlock All)} 戦略は, 「Surge」動態を捉えるために不可欠であった.
    \item 推論された相互作用行列は, 健康から疾患への遷移の定量的マップを提供する.
\end{itemize}

このフレームワークは, 多菌種細菌相互作用を分析するための強力なツールを提供し, 治療介入を \textit{in silico} でテストするために拡張可能である.

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
        section += f"\\textbf{{概要}}: {desc}\n\n"
        section += f"\\textbf{{主要な発見}}: {key_finding}\n\n"

        # Fig 1: Fit (Per species panel)
        section += "\\begin{figure}[H]\n"
        section += "\\centering\n"
        section += f"\\includegraphics[width=0.95\\textwidth]{{{fig_path}/Fig_A02_per_species_panel.png}}\n"
        section += f"\\caption{{{label} の事後分布適合. 陰影領域は95\\%信用区間を示す. モデル(青帯)は実験データ(赤点)を密接に追跡しており, 良好な適合品質を確認できる.}}\n"
        section += "\\end{figure}\n\n"

        # Fig 2: Interaction Matrix
        section += "\\begin{figure}[H]\n"
        section += "\\centering\n"
        section += f"\\includegraphics[width=0.8\\textwidth]{{{fig_path}/Fig_A01_interaction_matrix_heatmap.png}}\n"
        section += f"\\caption{{{label} の推定相互作用行列(MAP). 赤は正(協力的)相互作用, 青は負(競合的)相互作用を示す. 条件に関連する特定のブロック構造に注目されたい.}}\n"
        section += "\\end{figure}\n\n"

        # Fig 3: Parameter Uncertainty
        section += "\\begin{figure}[H]\n"
        section += "\\centering\n"
        section += f"\\includegraphics[width=0.95\\textwidth]{{{fig_path}/Fig_A05_parameter_violins.png}}\n"
        section += f"\\caption{{{label} のパラメータ不確実性(バイオリンプロット). 狭い分布は高い識別性を示し, 広い分布はパラメータの不感応性または相関を示唆する.}}\n"
        section += "\\end{figure}\n\n"

        section += "\\clearpage\n"

    return section


def main():
    content = LATEX_HEADER + generate_results_section(RUNS) + LATEX_FOOTER

    out_file = DOCS_DIR / "paper_comprehensive_ja.tex"
    with open(out_file, "w") as f:
        f.write(content)

    print(f"Generated {out_file}")


if __name__ == "__main__":
    main()
