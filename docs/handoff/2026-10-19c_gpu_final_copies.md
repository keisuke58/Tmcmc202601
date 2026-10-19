# 2026-10-19c（クラウド側）: 残りのコピー 2 つで論文用の GPU 作業は完了。1 日外し p2 の「FAIL」は 1 seed のため（解釈してよい）

## クラウドで済んだこと（10/19）
- DS の最終 MAP を seed42 の磨いた点（−96.279）に差し替え（3 seed の磨き −96.28 / −96.32 / −96.42 の最大）。`final_theta_MAP/DS.json`・`phi_init.json`。
- ベイズ因子の表 S8: DS +7.46（確定）、DH +2.07。本文の「暫定」を外す。
- 回収テスト CS・DS を評価（CH と同じ `recovery_evaluate.py`）。判定 (a)〜(e) は 3 条件とも全 PASS。分類の一致 CH 15/15、CS 14/15、DS 14/15。補足に反映中。

## GPU 側へのお願い（これで最後）
1. **CH の 1 日外し p2（`CH_p2_2c_raw_a33w_ho10_seed42`・`CH_p2_2c_raw_a33w_ho15_seed42`）を final_runs にコピー**（5 ファイル。磨いていれば polish.json も）。
   自動通知の「FAIL を含む群」は、**1 seed の群では判定 3・4（seed 間の幅）が評価できず FAIL 扱いになるだけ**。判定 0〜2b（移動 11.1 回/次元）は通っているので、予測の確認には使える（19b の想定どおり seed 42 の 1 系列）。
   コピーが来たらクラウドで `holdout_eval.py`（Day 10・15）を流す。
2. **DH の回収テスト（3656〜3658）が終わったら** `docs/revision/generated/recovery/DH_syn_n1_r0/` と `DH_syn_n1_r0_seed{42,7,123}/` に、CS・DS と同じ形で（samples.npy・logL.npy が入ったことを `git status` で確認してから push）。
3. 新しい投入は不要。上の 2 つで論文に使う計算は全部そろう。ジョブが全部終わったら、`data_5species/main/_runs/paper_gateoff/` と `_runs/recovery/` は消さずに残す（Zenodo の最終確認で参照するかもしれない）。

## 記録
- 13g・13f-4 は見送り（19b）。査読で求められたら検討。
