# A 行列の条件間比較（Commensal/Dysbiotic × Static/HOBIC）

このメモでは、4 条件で推定した有効相互作用行列 A の違いを定量評価した結果を整理する。

対象条件とファイル:

- Commensal HOBIC: `_runs/Commensal_HOBIC_20260208_002100/theta_MAP.json`
- Dysbiotic HOBIC: `_runs/Dysbiotic_HOBIC_20260208_002100/theta_MAP.json`
- Commensal Static: `_runs/Commensal_Static_20260208_002100/theta_MAP.json`
- Dysbiotic Static: `_runs/Dysbiotic_Static_20260207_203752/theta_MAP.json`

`analyze_A_matrices.py` で各条件の `theta_MAP` から A を再構成し、

- 条件間の相関係数
- Frobenius ノルムに基づく相対差
- 各条件の A のノルムと Veillonella–P. gingivalis ブロック

を計算した。

## 1. 条件間の相関係数と相対差

ペアごとの比較結果:

| cond1             | cond2             | corr(A1, A2) | rel_F_diff = \|\|A1 − A2\|\|_F / max(\|\|A1\|\|_F, \|\|A2\|\|_F) |
|-------------------|-------------------|--------------|---------------------------------------------------------------------|
| Commensal_HOBIC   | Dysbiotic_HOBIC   | −0.240       | 0.713                                                               |
| Commensal_HOBIC   | Commensal_Static  | +0.474       | 0.418                                                               |
| Commensal_HOBIC   | Dysbiotic_Static  | −0.068       | 0.623                                                               |
| Dysbiotic_HOBIC   | Commensal_Static  | −0.224       | 0.643                                                               |
| Dysbiotic_HOBIC   | Dysbiotic_Static  | +0.457       | 0.518                                                               |
| Commensal_Static  | Dysbiotic_Static  | −0.109       | 0.572                                                               |

解釈:

- 同じ「健康グループ」「病的グループ」内のペア
  - Commensal_HOBIC vs Commensal_Static, Dysbiotic_HOBIC vs Dysbiotic_Static は corr ≈ 0.47 と中程度の正の相関。
  - ただし rel_F_diff ≈ 0.42–0.52 と、ノルムの 4–5 割程度の差も存在する。
  - 同じ状態内では相互作用パターンの骨格はある程度共通だが、強さや一部ブロックは大きく変化している。
- 健康 vs 病的をまたぐペア
  - corr は −0.24 ～ +0.47 の範囲で、0 に近い値も多い。
  - rel_F_diff は 0.57–0.71 と大きめで、条件が変わると effective A としてはかなり別物になっている。

したがって、A 行列は 4 条件で「ほぼ同じ」ではなく、健康/病的と培養法の違いを反映してかなり異なる有効相互作用となっている。ただし、同一グループ内（健康 vs 病的）では中程度の相関があり、完全にバラバラではない。

## 2. 各条件の A のノルムと Vei–P. gingivalis ブロック

各条件の A の Frobenius ノルムと、Veillonella–P. gingivalis ブロック A[2,4]（対称性により A[4,2] と同じ）の値:

| condition         | \|\|A\|\|_F | A[2,4] (Vei–P.g) |
|-------------------|------------:|-----------------:|
| Commensal_HOBIC   |       8.446 |            +2.09 |
| Dysbiotic_HOBIC   |       6.970 |            +0.79 |
| Commensal_Static  |       8.890 |            +1.37 |
| Dysbiotic_Static  |      10.089 |            +2.03 |

解釈:

- A のノルム
  - Static 条件（Commensal_Static, Dysbiotic_Static）の方が HOBIC よりも \|\|A\|\|_F が大きい。
  - 静置条件では有効な自己・相互作用の絶対値がやや強く、HOBIC では流れの効果により相互作用が緩和されている可能性がある。
- Vei–P.g ブロック
  - Vei–P.g の有効協調は Commensal_HOBIC と Dysbiotic_Static で大きく、Dysbiotic_HOBIC は中程度の値に留まる。
  - 単純に「Dysbiotic_HOBIC だけ Vei–P.g が極端に強い」というパターンではない。
  - サージ現象の再現は、Vei–P.g 項の絶対値だけでなく、他のブロック（他菌種との相互作用や自己成長項）、b ベクトル、初期条件などとの組み合わせで決まっていることを示唆する。

## 3. 総合評価と今後の課題

- 4 条件の A 行列は、effective parameter として条件ごとにかなり異なっており、健康 vs 病的、Static vs HOBIC の違いを反映している。
- 同一グループ内（Commensal, Dysbiotic）では中程度の相関が見られ、完全に別物ではない一方で、ペアごとの相対差は 0.4–0.7 と大きく、環境に応じて有効相互作用が強く変化している。
- Vei–P.g ブロックはサージに重要な相互作用であるが、その値は条件ごとに非単調に変化しており、終盤のサージ振る舞いは A の 0 次近似だけでは十分に説明できない可能性が高い。
- Dysbiotic HOBIC の終盤の P. gingivalis の立ち上がりを完全に再現するには、A の状態依存性（pH や代謝産物に応じた動的変化）やレジームシフトを導入する拡張が必要であると考えられる。

