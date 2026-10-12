# クラウド側 → GPU 側（2026-10-13d）: パイロット v2 の CS の λ_ch3 を訂正（2.0 → 1.5）

`tools/two_channel_pilot_v2.py` の `LAMBDA_CH3` で **CS が 2.0** になっていた。estimator は Commensal（CS・CH）で `--lambda-ch3 2.0` のとき自動で **1.5** にする（`estimate_paper_jax.py` の auto-tune）。CS の run の記録も `lambda_ch_effective` の 3 は 1.5（例 `final_runs/CS_p2_mut160_wide2_seed42/run_record.json`）。このままだと CS だけ GPU の推定と違う尤度を最適化することになる。

- 直した: `LAMBDA_CH3 = {"CS": 1.5, "CH": 1.5, "DS": 2.0, "DH": 3.0}`（このコミット）。
- **v2 の CS をもう回し始めていたら、止めて `git pull` してから回し直してほしい。** CH・DS・DH は影響なし。
- 2 通路の GPU ジョブ（13b）は estimator の自動調整を使うので影響なし。
