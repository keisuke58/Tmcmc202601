#!/usr/bin/env python3
"""a45 = 0 モデルと a45 自由モデルの Bayes 因子（査読対策・2026-10-08f の (b)）。

ln BF = ln Z(a45 自由) - ln Z(a45 = 0)。TMCMC は段ごとの重みから ln Z をそのまま出すので
run_record.json の `log_evidence` を使う。

Bayes 因子は「同じデータ・同じ尤度・a45 以外は同じ事前分布」でしか意味を持たないので、
比較の前に run の config を突き合わせて、食い違っていたら数字を出さずに止める。
mutation 回数の違いだけは止めない（サンプラーの設定であってモデルの違いではない）が、
ln Z の精度に効くので警告として出す。

使い方:
    python3 tools/bayes_factor_a45.py
    python3 tools/bayes_factor_a45.py --runs data_5species/main/_runs/paper_gateoff
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

A45 = 19
SEEDS = (42, 7, 123)

# 条件 -> (a45 自由のベースラインの RUNTAG, a45=0 の RUNTAG)
PAIRS = {
    "DH": ("mut80", "a45zero"),
    "DS": ("wide80", "a45zero"),
    "CH": ("mut120", "a45zero"),
    "CS": ("mut120", "a45zero"),
}

# モデル・データが同じであることを担保する鍵。ここが違えば ln Z は比較できない
MUST_MATCH = (
    "condition",
    "cultivation",
    "sigma_obs",
    "phi_init",
    "psi_fixed",
    "n_hill",
    "K_hill",
    "multichannel",
    "use_student_t",
    "lambda_rare",
)


def load(run_dir: str) -> dict | None:
    """config.json を読む。

    run_record.json は condition / n_hill / K_hill / n_mutation_steps / n_particles 等を
    持たないので、そちらを読むと照合が黙って素通りする。config.json はその上位集合。
    """
    try:
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except OSError:
        return None


def close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def check_pair(full: dict, zero: dict) -> tuple[list[str], list[str]]:
    """(止める理由, 警告) を返す。"""
    stop, warn = [], []

    for key in MUST_MATCH:
        if key not in full or key not in zero:
            # 鍵が無いまま素通りさせると照合していないのに通ったように見える
            stop.append(f"{key} が記録されていないので照合できない")
        elif not close(full[key], zero[key]):
            stop.append(f"{key} が違う（{full[key]!r} vs {zero[key]!r}）")

    for key in ("n_mutation_steps", "n_particles"):
        if key not in full or key not in zero:
            stop.append(f"{key} が記録されていないので照合できない")

    fd_full, fd_zero = full.get("free_dims"), zero.get("free_dims")
    if fd_full is None or fd_zero is None:
        stop.append("free_dims が記録されていない")
    else:
        diff = set(fd_full) ^ set(fd_zero)
        if diff != {A45}:
            stop.append(f"自由次元の差が a45 だけではない（差: {sorted(diff)}）")
        if A45 in fd_zero:
            stop.append("a45=0 のはずの run で a45 が自由になっている")

    pb_full, pb_zero = full.get("prior_bounds_final"), zero.get("prior_bounds_final")
    if pb_full and pb_zero:
        for i, (lo_hi_f, lo_hi_z) in enumerate(zip(pb_full, pb_zero)):
            if i != A45 and not close(lo_hi_f, lo_hi_z):
                stop.append(f"事前分布の箱が θ[{i}] で違う（{lo_hi_f} vs {lo_hi_z}）")

    for r, tag in ((full, "自由"), (zero, "a45=0")):
        if not close(r.get("beta_final"), 1.0):
            stop.append(f"{tag} の beta_final が 1 でない（{r.get('beta_final')}）")
        if r.get("logL_consistent") is False:
            stop.append(f"{tag} の logL が粒子と不整合")

    # Bayes 因子は余分なパラメータの事前分布の幅に比例して自由モデルを罰する
    # （Lindley-Bartlett）。a45 の箱を広げると ln BF はその分だけ下がるので、
    # 幅を明示しないと査読で「箱の取り方次第」と言われる。
    if pb_full and len(pb_full) > A45:
        lo, hi = pb_full[A45]
        warn.append(
            f"自由モデルの a45 の箱は [{lo:g}, {hi:g}]（幅 {hi - lo:g}）。"
            f"幅を 2 倍にすると ln BF は約 {math.log(2):.2f} 下がる"
        )

    if full.get("n_mutation_steps") != zero.get("n_mutation_steps"):
        warn.append(
            f"mutation 回数が違う（自由 {full.get('n_mutation_steps')} vs "
            f"a45=0 {zero.get('n_mutation_steps')}）。ln Z の精度に効く"
        )
    if full.get("n_particles") != zero.get("n_particles"):
        warn.append(
            f"粒子数が違う（自由 {full.get('n_particles')} vs a45=0 {zero.get('n_particles')}）"
        )
    return stop, warn


def interpret(ln_bf: float) -> str:
    """Kass & Raftery (1995) の目安。ln BF > 0 は a45 自由を支持。"""
    a = abs(ln_bf)
    if a < 1.0:
        strength = "差とは言えない"
    elif a < 3.0:
        strength = "弱い"
    elif a < 5.0:
        strength = "強い"
    else:
        strength = "決定的"
    if a < 1.0:
        return strength
    return f"{strength}（{'a45 自由' if ln_bf > 0 else 'a45=0'} を支持）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data_5species/main/_runs/paper_gateoff")
    args = ap.parse_args()

    print("ln BF = ln Z(a45 自由) - ln Z(a45 = 0)   正なら a45 自由を支持\n")
    exit_code = 0

    for tag, (full_rt, zero_rt) in PAIRS.items():
        print(f"[{tag}]  自由={full_rt}  a45=0={zero_rt}")
        ln_bfs = []
        for seed in SEEDS:
            d_full = os.path.join(args.runs, f"{tag}_pilot_{full_rt}_seed{seed}")
            d_zero = os.path.join(args.runs, f"{tag}_pilot_{zero_rt}_seed{seed}")
            full, zero = load(d_full), load(d_zero)
            missing = [
                os.path.basename(d) for d, r in ((d_full, full), (d_zero, zero)) if r is None
            ]
            if missing:
                print(f"  seed{seed:<4} 未完了: {', '.join(missing)}")
                continue

            stop, warn = check_pair(full, zero)
            if stop:
                print(f"  seed{seed:<4} 比較できない:")
                for s in stop:
                    print(f"           - {s}")
                exit_code = 1
                continue
            for w in warn:
                print(f"  seed{seed:<4} 警告: {w}")

            ln_bf = full["log_evidence"] - zero["log_evidence"]
            ln_bfs.append(ln_bf)
            print(
                f"  seed{seed:<4} lnZ 自由 {full['log_evidence']:8.3f}  "
                f"a45=0 {zero['log_evidence']:8.3f}  ln BF {ln_bf:+7.3f}  {interpret(ln_bf)}"
            )

        if len(ln_bfs) == len(SEEDS):
            lo, hi = min(ln_bfs), max(ln_bfs)
            mean = sum(ln_bfs) / len(ln_bfs)
            print(f"  → 3 seed: 平均 {mean:+.3f}、幅 {hi - lo:.3f}  {interpret(mean)}")
            if hi - lo > 2.0:
                print("     警告: seed 間の幅が 2 を超える。ln Z の推定が安定していない")
                exit_code = 1
        else:
            print(f"  → {len(ln_bfs)}/{len(SEEDS)} seed のみ。全 seed 揃うまで解釈しない")
        print()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
