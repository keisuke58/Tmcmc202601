# 2026-10-18b（クラウド側）: 回収テストの出力を final_runs の隣にコピーしてほしい（CH は今すぐ、残りは終わり次第）

## お願い
`tools/recovery_evaluate.py` はクラウドで動かす（CPU、数分）。入力は条件ごとに「合成データ（真値つき）」と「回収 run（3 seed）」。次の形で置いて push してほしい:

```
docs/revision/generated/recovery/<TAG>_syn_n1_r0/synthetic.json            # 合成データ（真値 θ・箱・source_glob・sha）
docs/revision/generated/recovery/<TAG>_syn_n1_r0_seed{42,7,123}/            # 回収 run（final_runs と同じ 5 ファイル＋ recovery_record.json があれば）
    samples.npy  logL.npy  theta_MAP.json  run_record.json  config.json  [recovery_record.json]
```
- **CH（3614〜3616、10/17 終了）を先に**。`_runs/recovery/CH_syn_n1_r0*` から。
- CS（3659〜3661）・DS（3653〜3655）・DH（3656〜3658）は終わり次第、同じ形で。
- 合成データは生データの集計から作った値（実測の反復値そのものは入っていない）なので git に入れてよい。念のため `synthetic.json` に反復ごとの実測値が無いことを確認して。

## ついでに
- `CH_p1_2c_raw_seed42` と `CH_p2_2c_raw_a33w_seed7` の polish.json（3612・3592）が final_runs にまだ無い。終わっていたらコピー。
  （表 S2 は handoff の磨きの報告も走査するので値は既に反映されている。final_runs にも置いておきたいだけ）
- DS ult の磨き（3662〜3664）が終わったら、3 seed の polish.json を final_runs にコピー → クラウドで DS.json を差し替える。
