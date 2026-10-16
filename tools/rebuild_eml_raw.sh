#!/usr/bin/env bash
# 生データ・2 通路の run から、EML の図・表・数値を一括で作り直す（2026-10-13）。
# raw の ult が通ったら、これ 1 本で投稿用の図と補足の表がそろう。
#
#   R=<run の置き場所> bash tools/rebuild_eml_raw.sh            # 既定 R = docs/revision/generated/final_runs
#   SPEC=<条件 → 段 → glob の JSON>  F=<出力先>  STAGE=ult|p2（Fig. 2 などに使う段）
#   SKIP="knockout ph"  のように空白区切りで飛ばせる（ph は生データの pH が届くまで既定で飛ばす）
#
# 組成とモデルの対応は estimator の尤度と同じ（tools/comp_pred.py）: 2 通路の run では生きた菌の組成 ↔ φψ/Σφψ、
# 全菌の組成 ↔ φ/Σφ。データは各 run が読んだ場所（EP.data_dir_for）から読む。
# 出力の図・表は生データの集計を含む → 共著者の了承の範囲（原稿用）で使い、生データの表そのものは載せない。
set -euo pipefail
cd "$(dirname "$0")/.."
R="${R:-docs/revision/generated/final_runs}"
SPEC="${SPEC:-docs/revision/EML_submission/figures/raw/spec_raw.json}"
F="${F:-docs/revision/EML_submission/figures/raw}"
STAGE="${STAGE:-ult}"
SKIP="${SKIP:-ph}"
# run 名の型（{tag} を条件に置き換える）。spec の glob と同じものにする
PAT="${PAT:-{tag\}_${STAGE}_2c_raw*_seed*}"  # CH は 2c_raw_a33w（14b）も拾う
P1PAT="${P1PAT:-{tag\}_p1_2c_raw_seed*}"
mkdir -p "$F/supp"
skip() { [[ " $SKIP " == *" $1 "* ]]; }
glob_of() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])" "$SPEC" "$1" "$2"; }
echo "=== R=$R  SPEC=$SPEC  F=$F  STAGE=$STAGE  PAT=$PAT  SKIP='$SKIP'"

# 0. 判定 (a)–(e) と、run ごとの数値（2 通路の内訳つき）
skip check   || python3 tools/check_paper_runs.py "$R" | tee "$F/check_paper_runs.txt"
skip eval    || python3 tools/eval_paper_runs.py "$R" --glob "${PAT//\{tag\}/*}" | tee "$F/eval_paper_runs.txt"
skip metrics || python3 tools/fit_metrics_2c.py "$R" $(for t in CS CH DS DH; do glob_of $t "$STAGE"; done) --out "$F/fit_metrics_2c"
skip psi     || python3 tools/psi_table_runs.py "$R" --pattern "$PAT" --out "$F/psi_table" --tex "$F/supp/supp_S_psi2c.tex"

# 1. Fig. 2（生きた菌 ↔ φψ/Σφψ、全菌 ↔ φ/Σφ）と事後のまとめ（paper_numbers.json・tables.tex）
skip fig2 || python3 tools/make_paper_figures.py "$R" \
    --p2-glob "$PAT" --p1-glob "$P1PAT" \
    --out-dir "$F" --skip ident knockout umap

# 2. ノックアウト（Fn・a45=0、DS・DH）
if ! skip knockout; then
  python3 tools/knockout_fn.py "$R" --glob "$(glob_of DH "$STAGE")" --n-samples 200 --json "$F/knockout_DH.json"
  python3 tools/knockout_fn.py "$R" --glob "$(glob_of DS "$STAGE")" --n-samples 200 --json "$F/knockout_DS.json"
fi

# 3. pH（推定には使わない。生データの pH が届いてから: SKIP から ph を外す）
skip ph || python3 tools/ph_check.py --root "$R" --glob-CH "$(glob_of CH "$STAGE")" --glob-DH "$(glob_of DH "$STAGE")" \
    --n-post 200 --out "$F/ph_check"

# 4. Fig. 3（同定性）・Fig. 4（周辺事後）・Fig. 5（確認）
skip figs56 || python3 tools/make_eml_figs56.py --root "$R" --spec "$SPEC" --fig-dir "$F"

# 5. 補足の表（S1 箱・S2 run・S3 ベイズ因子・S4 同定性、コスト、しきい値、R̂、Fisher）
skip supp   || python3 tools/make_supplementary_tables.py "$R" --spec "$SPEC" --out-dir "$F/supp"
skip cost   || python3 tools/make_cost_table.py "$R" --spec "$SPEC" --gpu-model 'RTX\,4090' --out "$F/supp/supp_S3_cost.tex" \
    --ult-glob-DS "$(glob_of DS ult)" --ult-glob-DH "$(glob_of DH ult)"
skip thresh || python3 tools/ident_threshold_table.py "$R" "$SPEC" "$F/supp/supp_S4_thresh.tex"
skip rhat   || python3 tools/rhat_seeds.py "$R" "$SPEC" "$F/rhat_seeds.json"
skip fisher || python3 tools/fisher_ident.py --root "$R" --spec "$SPEC" --out "$F/fisher_ident.json"

# 6. 補足の図: seed ごとの MAP、成長応力
if ! skip seedmaps; then
  python3 - "$SPEC" "$STAGE" "$F/_seed_prefix.json" <<'PY'
import json, sys
sp = json.load(open(sys.argv[1]))
json.dump({t: v[sys.argv[2]].rstrip("*") for t, v in sp.items() if sys.argv[2] in v}, open(sys.argv[3], "w"))
PY
  python3 tools/plot_seed_maps.py --root "$R" --spec "$F/_seed_prefix.json" --stage-label "$STAGE" --n-post 200 --out "$F/seed_maps_4cond"
fi
skip growth || python3 tools/growth_stress_check.py "$R" --p2-glob "$PAT" --out "$F/growth_stress"
echo "=== done -> $F"
