# GPU 側報告 2026-10-11（11i / 11k §3.2: 今ある一番いい run で図を作った）

copaam の CPU で `tools/make_paper_figures.py` を 11i のコマンドどおりに実行（`--skip knockout ident`、所要約 8 分）。
glob は 11i の指定どおりで、どれも完走済みの 3 seed にちょうど一致した（差し替えなし）。
**DS・DH は p2 の仮。** DH の `nonarrow_w3_noph_p14k` は判定 4 FAIL（ult 未投入）なので、DH の列は特に仮。

## RMSE（paper_numbers.json の phase2、照合値と一致）

| 条件 | Phase 2 の群 | MAP の run | RMSE | 照合値 | Phase 1 の群 | Phase 1 RMSE | 生存率 RMSE |
|---|---|---|---|---|---|---|---|
| CS | `CS_ult_wide2_sd4_seed*` | `CS_ult_wide2_sd4_seed42` | 0.1794 | 0.179（seed42） | `CS_p1_mut120*` | 0.1141 | 0.084 |
| CH | `CH_ult_mut150_wide2_noph_sd4_seed*` | `CH_ult_mut150_wide2_noph_sd4_seed7` | 0.0756 | 0.0756（seed7） | `CH_p1_mut150*` | 0.0711 | 0.104 |
| DS | `DS_p2_wide80_p18k_wide2_seed*` | `DS_p2_wide80_p18k_wide2_seed123` | 0.0272 | 0.02〜0.03 | `DS_p1_wide80_p3k*` | 0.0155 | 0.080 |
| DH | `DH_p2_nonarrow_w3_noph_p14k_seed*` | `DH_p2_nonarrow_w3_noph_p14k_seed123` | 0.0833 | 0.08〜0.09 | `DH_p1_mut80*` | 0.0536 | 0.110 |

RMSE は estimator の値（`rmse_estimator`）とも1e-15 の桁まで一致。

## 図（`docs/revision/EML_submission/figures/provisional_2026-10-11/`）

- `heatmap_A_4cond.pdf`
- `paper_fig2_phi_transposed.pdf`
- `paper_fig2_phi_transposed.png`
- `paper_posterior_violin_sharey.pdf`
- `phase1_vs_phase2_map.pdf`
- `umap_A_matrix_3d.pdf`

数値: `docs/revision/EML_submission/figures/generated/paper_numbers.json`・`tables.tex`（provisional。`docs/revision/generated/` の本番は触っていない）。
BMB_submission/figures は上書きしていない。

## 図 2 を見て気づいたこと（解釈はしていない）

- CS の So と Vei で、MAP の実線が事後予測の 50%・80% の帯の外にある（So は帯が 0.3〜0.45 で MAP は 0.28 前後、Vei は MAP が帯より上）。
  MAP の seed（42）と、帯を作る 3 seed を合わせた粒子とで、別の領域を見ている可能性がある。CH・DS・DH では MAP は帯の中。

