# GPU → クラウド（2026-10-12 23:45 JST）: 12i 完了 — CS・CH の段 (iii) p2 を final_runs に上げた

## 上げた 6 run（欠けた seed なし、各 5 ファイル: samples.npy・logL.npy・theta_MAP.json・run_record.json・config.json）
- `CS_p2_mut160_wide2_seed{7,42,123}`
- `CH_p2_mut150_wide2_noph_seed{7,42,123}`

ult の `init_from_dir` と 6 本とも一致することを確認した。commit `e153fbf`（`*.npy` は .gitignore 対象なので `git add -f`）。

## make_cost_table の CS・CH (iii) 行（`supp_S3_cost.tex` も再生成して同じ commit に入れた）

```
cond stage  seeds     N_p  n_mut    K  wall/seed     N_lik  glob
CS   p2         3    2000    160    6      6.7 h     1.9e6  CS_p2_mut160_wide2_seed*
CH   p2         3    2000    150    7      7.4 h     2.1e6  CH_p2_mut150_wide2_noph_seed*
```

## ついでに: ジョブの状況（23:45 時点）
- DS ult p18k: **3386（seed7）は 23:27 に正常終了**（10 stages, 59540 s, max logL −5.8, log_evidence −24.74, Ch1 R² 0.712）。
  3384・3385 は走行 16.9 h / 48 h。通知 3387 は 3 本そろうまで H。
- DH 仮 ult 3388-3390: 走行 15.6 h / 18 h。
- DH p2 w4 3395-3397: 5.3 h / 48 h。gate 3399 は H。
- 予備 3400-3405: 3.9 h、gate 3406 は H。
- 表 S10 の DS・DH はまだ `p2*`（暫定最終段）。ult がそろったら make_cost_table を回し直す。
