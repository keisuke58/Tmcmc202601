# Figure Regression Baselines

このディレクトリにベースライン画像を配置すると、`figure-regression.yml` がピクセル差分で変更を検出します。

## 使い方

1. 期待する図を `figure_test_output.png` として保存
2. PR でプロット関連コードを変更すると、自動で比較が実行される
3. 差分が大きい場合に warning が表示される

## ベースラインの更新

意図した変更の場合は、新しい図でベースラインを上書きしてコミットしてください。
