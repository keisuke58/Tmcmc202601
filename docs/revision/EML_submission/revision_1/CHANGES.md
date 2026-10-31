# revision_1：修正版（投稿後の修正をまとめる所）

- 投稿版（2026-10-31 投稿）の記録は `../submission_source/`（変更しない）。
- このフォルダは投稿版をコピーして、投稿後に見つけた誤りを直したもの。査読コメントが来たらここに続けて直し、修正稿としてそのまま出す。
- 本文はフラット版（補足への参照は S1・S4 などの番号を直書き、図は Figure_1〜4.pdf）。補足 `EML_supplementary.pdf` は `../EML_supplementary.tex` からビルドしたもの。
  補足の番号（節・表）が変わったら、本文の直書きの番号も合わせること。
- 作業用の原稿 `../EML_manuscript_nishioka.tex` とは参照の書き方と図のパス以外は同じ（2026-10-31 に確認）。

## 投稿後の修正一覧（回答書の "Additional changes" に使う）

| # | 場所 | 投稿版 | 修正版 | 結果への影響 |
|---|---|---|---|---|
| 1 | 本文 Eq. (4)（ψ の式）、補足の同じ式 | 末尾に `+γ` | `γ` なし（拘束 C は ψ を含まない。計算コードにも γ はない） | なし（誤植） |
| 2 | 本文 Eq. (2)（散逸 Δ）＋直後の文、補足 | φ₀ の項なし | `+½ φ̇₀²`（φ₀ の単位の散逸。コードの φ₀ の式 0=γ+φ̇₀ がここから導ける）＋説明 1 文 | なし（実装をそのまま書いた） |
| 3 | 本文 Eq. (5)（尤度）と直後の文 | σ_ik（時刻ごと） | σ_i（各日の IQR/1.35、下限 0.05、Day 1 を含む 6 日で平均）。最終 run の sigma_obs と小数 4 桁まで一致 | なし（書き方の誤り） |

回答書の文案（英語、必要なら使う）：
> In addition to the changes requested by the reviewers, we corrected three notational errors found after submission:
> (i) the Lagrange multiplier γ was erroneously included in the evolution equation for ψ_i (Eq. 4); the constraint does not
> contain ψ_i, and the implementation never used this term; (ii) the dissipation Δ now includes the unit dissipation of the
> void fraction φ₀ that the implementation uses to regularise the system, from which the equation 0 = γ + φ̇₀ follows;
> (iii) the observation standard deviations are per species, σ_i (averaged over the six measurement days), not per time
> point. None of these changes affects the computations, results or figures.

## 修正稿を出す前に
- **元論文の確認（2026-10-31、arXiv 2509.01274 の LaTeX ソース）**：
  - 拘束は f = Σ_{l=0}^n φ_l − 1（Eq. Constraint）で、**ψ を含まない**。ほかに拘束はない → δ_ψ C = 0 なので、元論文 Eq. (17)（EvolEq2）の +γ は導出からは出ない＝**元論文の誤植**。
  - 元論文の散逸 Δ^s も η は種 1..n だけで φ₀ の項がない。式は φ の n 本＋ψ の n 本＋拘束 1 本＝2n+1 本、未知数は φ₀・φ_i・ψ_i・γ の 2n+2 個で 1 本足りない（φ₀ で変分すると γ=0 になってしまう）。
    本文は「φ₀ は拘束で決まる」とだけ書いている。こちらのコードは φ₀ に単位の散逸を入れて 0=γ+φ̇₀（＋バリア項）で閉じている（`hamilton_ode_jax_paper.py` の Q[5]）→ 修正版の ½φ̇₀² はこれを書いたもの。
  - コードの ψ の残差（同ファイル active_psi）にも γ はない。
- **Klempt さんの Mathematica（`data_5species/monospecies/Biofilm-TSM-Mathematica-2025-01-17-4species - changing.nb`）も確認（2026-10-31）**：
  - 拘束は `γ (Sum[φ[[i]],{i,4}] + φ0 - 1.)`、φ₀ の散逸は別に `Δ0 = ½ φ0dot²`。
  - 式は記号微分で作っている：`Qψ = D[H, ψ] + D[Δrd, ψdot]`（→ **γ なし**）、`Qφ0 = D[H, φ0] + D[Δ0, φ0dot]`（→ **γ + φ̇₀ ＋バリア**）。
  - つまり元論文の**実装そのものが修正版と同じ式**。論文の +γ と φ₀ の書き漏れは、論文に書き写すときの誤り。こちらの JAX 実装とも一致。
  - （参考）第 2 の散逸項の粘性は実装では別の行列 ηφ。論文・原稿はどちらも η（=1）と書いている。値が同じなので結果は同じ。
- #1・#2 は元論文（Klempt ら、arXiv:2509.01274、Eq. (17)）と書き方が違うが、元論文の Mathematica 実装と一致する（上）。Klempt さん／Junker 先生には確認ではなく報告で足りる。
- 図・数値は変えていない。
- **全体点検は修正稿のときにまとめて行う**（2026-10-31 決定）。それまでは見つけた誤りをこの表に足していくだけ。
