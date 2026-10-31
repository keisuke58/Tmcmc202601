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
- #1・#2 は元論文（Klempt ら、arXiv:2509.01274、Eq. (17)）と書き方が違う（元論文には +γ があり、φ₀ の項はない）。Klempt さん／Junker 先生に一言確認する。
- 図・数値は変えていない。
