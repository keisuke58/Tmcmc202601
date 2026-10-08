# GPU 側報告 2026-10-09 03:5x（CH ult は依存ジョブで自動投入に仕掛けた。ユーザーは就寝）

ユーザーから「CH p2 が終わったら ult を投げて」と指示があったが、**Claude のセッションに依存する
投入では端末を閉じた時点で止まる**ので、Torque の依存ジョブにした（ジョブの寿命は qsub で担保する、
というこのリポジトリの方針どおり）。

## 仕掛けたもの: `tools/gate_submit_ult.sh`（3320、`depend=afterany:3307:3308:3309`）

CH p2 の 3 本が終わると起きて、

1. 前段 `CH_p2_mut150_wide_noph_seed*` が **3 seed そろっているか**を見る（walltime / crash で落ちた seed はディレクトリが無い）
2. `check_paper_runs.py --glob` で判定する
3. **「全群 PASS」かつ 3/3 のときだけ** ult を 3 seed 投入し、通知ジョブを連結する
4. FAIL または seed 欠けなら**投入せず**、判定の全文を `docs/handoff/gpu_<日付>_gate_CH_ult.md` に書いて push する
   （FAIL の段からは進めない、`docs/paper_gateoff_pipeline.md` §5）

どちらの道でも結果は handoff に出て push されるので、こちら（GPU 側 Claude）が起きていなくても読めます。

投入する内容（09a §1・08r のとおり。`qstat -f` の `Variable_List` を全文で確認済み）:

```
STAGE=ult TAG=CH SEED=$S N_MUT=150 RUNTAG=mut150_wide_noph_sd4
OVERRIDE='1:-5:2.5;11:-5:1;16:-5:1'  PREV=_runs/paper_gateoff/CH_p2_mut150_wide_noph_seed$S
-l nodes=1:ppn=1:gpus=1:stuttgart02  -l walltime=12:00:00
```

- `ULT_NSIGMA` は既定 4、`N_PART` は既定 5000 なので渡していない。
- ノードは **stuttgart02**（CH p2 自身が空ける 3 枚をそのまま使う）。他ユーザーに取られていたら Q で待つだけで、run は失われない。
- walltime **12h**: 前回の CH ult sd4 は 4090 で 2.1h だったが、N_MUT 80→150 で約 1.9 倍、かつ 3090 なので 5〜6h 見込み。08s の walltime 不足を踏まえて余裕を取った。
- 枠: ult 3 本は CH p2 が空ける 3 枠に入るので、上限 15 を超えない。`gate_ult` 自体は GPU を掴まない。

## 他の待ち

- CH の `a45zero_m120` 2 本（seed 42 / 123）は**まだ投入していない**。ult に枠を譲る指示（09a §0）どおり、
  DH p2（3304-3306、05:30〜07:30 見込み）が終わって枠が空いてからになる。
  **これは依存ジョブにしていない**（どのジョブの後に空くかが確定しないため）。朝に投入する。
- 走行は 15 本（3298-3300 DS p2 / 3304-3306 DH p2 / 3307-3309 CH p2 / 3310-3312 CS p2 / 3317-3319 CS a45zero）。
  それぞれに通知ジョブ 3313-3316 が付いている。
- DS / DH / CS の p2 についても同じ gate を仕掛けるかは、朝にユーザーの判断を聞く（今回は CH の指示だけだったため）。
