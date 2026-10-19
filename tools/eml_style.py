"""EML の図の共通スタイル（2026-10-12）。

すべての図で同じ:
  - 文字は usetex・Latin Modern、最終サイズで 7–8 pt（図は最終の幅 MM で作り、\\includegraphics で拡大縮小しない）
  - 条件の色は 1 組だけ（Okabe–Ito、色覚の違いに強い）: CS 青・CH 緑・DS 橙・DH 赤
  - 符号つき係数は RdBu_r、0 が中央、上下対称の範囲
  - パネル記号 "(a)" は太字、軸の左上の外に同じ位置で
  - 軸の線は細く、グリッドは無しか薄く
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE_MM, DOUBLE_MM = 90.0, 190.0

COND = ["CS", "CH", "DS", "DH"]
COND_COL = {"CS": "#0072B2", "CH": "#009E73", "DS": "#E69F00", "DH": "#D55E00"}
COND_NAME = {"CS": "commensal static", "CH": "commensal HOBIC",
             "DS": "dysbiotic static", "DH": "dysbiotic HOBIC"}
INK, MUTE, FAINT = "#222222", "#6B6B6B", "#BDBDBD"
DIVERGING = "RdBu_r"

_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}"


def apply():
    """rcParams を設定する。LaTeX が無ければ通常の serif に落とす。"""
    base = {
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
        "lines.linewidth": 1.0, "grid.linewidth": 0.3, "grid.color": "#DDDDDD",
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "hatch.linewidth": 0.5, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "standard", "pdf.fonttype": 42, "figure.facecolor": "white",
    }
    mpl.rcParams.update(base)
    try:
        mpl.rcParams.update({"text.usetex": True, "text.latex.preamble": _PREAMBLE})
        f = plt.figure()
        f.text(0, 0, r"$a_{11}$ \textbf{(a)}")
        f.savefig(os.devnull, format="pdf")
        plt.close(f)
    except Exception:  # noqa: BLE001
        mpl.rcParams.update({"text.usetex": False})
    return mpl.rcParams["text.usetex"]


THESIS_MM = 160.0  # 修論の本文幅（2026-10-19）
_PREAMBLE_THESIS = r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}"  # 修論の本文と同じ Latin Modern（10-19 夜に Times から変更）


def apply_thesis():
    """修論用（2026-10-19）: Latin Modern（修論の本文・ほかの図と同じ）、印刷サイズで目盛り 8 pt・軸ラベル 9 pt・見出し 10 pt。"""
    apply()
    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    if mpl.rcParams["text.usetex"]:
        mpl.rcParams["text.latex.preamble"] = _PREAMBLE_THESIS
    else:
        mpl.rcParams["font.serif"] = ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"]
    return mpl.rcParams["text.usetex"]


def panel_label(ax, s, dx_mm=-1.0, dy_mm=1.2, fig=None):
    """太字の "(a)" を軸の左上の外（軸の左端から dx_mm、上端から dy_mm）に置く。"""
    fig = fig or ax.figure
    from matplotlib.transforms import ScaledTranslation
    tr = ax.transAxes + ScaledTranslation(dx_mm * MM, dy_mm * MM, fig.dpi_scale_trans)
    t = rf"\textbf{{({s})}}" if mpl.rcParams["text.usetex"] else f"({s})"
    return ax.text(0, 1, t, transform=tr, ha="right", va="bottom", fontsize=8,
                   fontweight="bold")


def save(fig, path_noext, **kw):
    """PDF（ベクタ）と PNG（確認用）を書く。"""
    fig.savefig(path_noext + ".pdf", **kw)
    fig.savefig(path_noext + ".png", dpi=300, **kw)
    print("  -> " + path_noext + ".{pdf,png}")
