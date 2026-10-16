# クラウド側 → GPU 側（2026-10-16j）: CS ult の磨き（3 本）、final_theta_MAP の規則

## 決まったこと（ユーザー決定 10/16）
- `final_theta_MAP/<C>.json` は「3 seed の ult を磨いたあとの max logL が最大の点」。磨いていなければ粒子の最良値（磨いたら差し替え）。
- CH.json = `CH_ult_2c_raw_a33w_seed7` の磨いた θ（−25.87、クラウドの `--rescore` で 25.866 を再現）。
- CS.json = `CS_ult_2c_raw_seed123` の粒子の最良値（−32.86）。**CS ult は未研磨**（gate の報告で「polish.json なし」）。

## GPU 側でやること
1. CS ult の 3 seed を、CH と同じ設定で磨く（`TOP_K=3 MAXITER=3000`、ftol 既定 1e-7、`WRITE_RUN=1`）。
   `CS_ult_2c_raw_seed{42,7,123}`。DS・DH・CH 回収のあとでよい（急がない、空いたノードで）。
2. 結果は今までどおり `gpu_<日付>_polish_CS_ult_2c_raw_seed*_k3.{md,json}` を push。16g のとおり旧・新・大きい方を並べる（CS ult は旧が無いので新だけ）。
3. 磨いた値が粒子の最良値より高い seed があれば、クラウドで CS.json を差し替える（GPU 側は何もしなくてよい）。
