# experiment_data — 生データの置き場（中身は git に入れない）

- `raw_heine/`: Heine（MHH）提供の生データ（qPCR・LIVE/DEAD・体積）。**gitignore、再配布しない**。
  置き方は `docs/handoff/2026-10-13i_raw_data_fetch.md`。
- 推定が読む形への変換: `python3 tools/convert_raw_heine.py && python3 tools/build_raw_dataset.py`
  → `data_5species/raw_dataset/experiment_data/`（これも gitignore）。
- 論文の図から読み取った以前のデータは `data_5species/digitised_dataset/`（新しい推定には使わない）。
