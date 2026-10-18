# 2026-10-19a（クラウド側）: 回収テストの samples.npy・logL.npy が commit に入っていない（.gitignore の *.npy）

- 38578f0 には `recovery/CH_syn_n1_r0_seed*/` の json 4 つしか入っていない。`samples.npy`・`logL.npy` は `.gitignore` の `*.npy` で除かれていた
  （final_runs だけ `!` で例外になっていた）。このコミットで `!docs/revision/generated/recovery/**/*.npy` を足した。
- **お願い**: pull してから `git add docs/revision/generated/recovery/CH_syn_n1_r0_seed*/samples.npy docs/revision/generated/recovery/CH_syn_n1_r0_seed*/logL.npy` → commit → push。
  CS・DS・DH の回収テストをコピーするときも、npy が入ったことを `git status` で確かめてから push。
- 評価（recovery_evaluate.py）はクラウドで npy が届いてから流す。報告 §1 の要約（被覆 0.53、同定 6 のうち真値が区間に 2、分類一致 15/15）は手元で再現してから使う。
