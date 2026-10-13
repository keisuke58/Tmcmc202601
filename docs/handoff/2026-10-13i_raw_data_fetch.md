# クラウド側 → GPU 側（2026-10-13i）: 生データ 2 つを GPU のサーバーに置く（ユーザー指示）

13h の生データ（qPCR の種の割合・LIVE/DEAD の画像ごとの値）は **git には入れない**（public リポジトリ、MHH のデータは再配布しない約束）。
コミット b2751e9 にだけ入っているので、そこからサーバー上のファイルとして取り出す。**取り出したファイルは絶対に git add しない**（`.gitignore` 済み）。

```bash
cd ~/Tmcmc202601 && git pull
mkdir -p data_5species/experiment_data/raw_heine
for f in qpcr_species_distribution.xlsx livedead_HOBIC_Static.xlsx; do
  git show b2751e9:data_5species/experiment_data/raw_heine/$f > data_5species/experiment_data/raw_heine/$f
done
python3 tools/convert_raw_heine.py      # → raw_heine/qpcr_long.csv・livedead_long.csv・volume_long.csv
python3 tools/build_raw_dataset.py      # → data_5species/raw_dataset/experiment_data/（推定が読む形）
git status --short | grep -E 'raw_heine|raw_dataset'   # 何も出ないこと（ignore されている）
```

確認: `convert_raw_heine.py` が qpcr_long 2955 行・livedead_long 670 行・volume_long 715 行、`build_raw_dataset.py` が replicates 1470 / 1485 行・membrane 24 行・boxplot 24 行を出すこと（クラウドと同じ）。

置いたら報告だけ。**raw での投入（`DATA_SOURCE=raw`）はまだしない**（ユーザーの判断待ち。クラウドで raw のパイロットを CPU で回している）。
