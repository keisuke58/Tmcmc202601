# クラウド側 → GPU 側（2026-10-15、14e）: CS の ult を今の箱のまま投入する（a33 の端の印は例外、ユーザー決定）

[gpu_2026-10-15_gate_CS_ult.md](gpu_2026-10-15_gate_CS_ult.md) で gate が止めた CS の ult を、**箱を広げずに**投入する（ユーザー決定 2026-10-15）。
理由: [gpu_2026-10-15_cs_a33_flat.md](gpu_2026-10-15_cs_a33_flat.md) のとおり、a33（θ[5]）は −5 より下で尤度が平らで、
[−15, 20] に広げても下端に 22〜24% 積もる（広げても新しい下端に積もるだけ）。論文では a33 を**片側の上限**（95% で a33 < 約 −5）として書く
（本文の限界の節に記載済み）。端の印の規則（広げて回し直す）の**例外は CS の a33 だけ**。ほかの成分に印が出たら、今までどおり止める。

## やること
- `tools/gate_submit_ult.sh` に `EDGE_ALLOW`（カンマ区切りの成分名、その成分の ← の行を数えない）を足した（このコミット）。
- CS の gate を前回（3553〜3557 の形、polish は GPU、8h）と同じ設定で、**`EDGE_ALLOW=a33` を足して**投げ直す。
  `TWO_CHANNEL=1 DATA_SOURCE=raw PREV_RUNTAG=2c_raw ULT_RUNTAG=2c_raw`、OVERRIDE は CS の p2 と同じ（13p）、ult は前回の CS と同じ（5000×150、18h）。
  p2 の polish は済んでいる（gpu_2026-10-15_polish_cs_p2_2c_raw_s*）ので、polish を飛ばせるなら飛ばしてよい。
- gate の報告に `EDGE_ALLOW=a33 の印は数えない` の行が出て、印の数が 0 になっていることを確認。
- ult が全 PASS なら、15d と同じ形で `CS_{pilot,p1,p2,ult}_2c_raw_seed*` を final_runs に入れて push。
