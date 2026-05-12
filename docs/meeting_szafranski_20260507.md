# Progress Meeting with Szafranski — 2026-05-07

## 概要
multi-species oral biofilm モデリングのさらなる発展と検証方法についての議論。

---

## 現状の確認

- **モデル**: Hamiltonianルールベースアプローチ + Lotka–Volterra モデルとの比較
- **パラメータ情報源**: Dieckow npjB&M 論文 + KEGG 関連リソース → interaction network
- **将来計画**: genome-wide metabolic model (e.g., FBA) の導入

---

## 要確認事項（Szymon からの指摘）

### α値生成プロセスの不透明さ
- データベース由来の正/負相互作用をどう集約してモデルパラメータに変換しているか不明確
- 同一ペアに正と負の両方の証拠がある場合の処理方法
- 方向性（A→B vs B→A）の扱い（Hamilton rule vs LV で異なる）
- **Action**: ドキュメントに明示的に説明を追加する

---

## 科学的内容: 乳酸–ショ糖–バイオフィルム

### シナリオ
1. Streptococci がショ糖を発酵 → 乳酸 + H⁺ を産生
2. Veillonella/Negativicutes が乳酸を利用（cross-feeding）
3. 乳酸濃度 + 低 pH が蓄積すると → 乳酸産生菌・乳酸利用菌両方に阻害的

### 重要な問い
- 乳酸の持続濃度が「cross-feeding が有益か」vs「環境が non-permissive になるか」を決定する
- Veillonella が Streptococci 支配の環境で生存できるか
- 関連スケッチ・メモは WhatsApp で別送予定

---

## データ比較手法

### 現状
- 主に相関（patient data との比較）

### 提案
- **多変量解析**: 組成データを適切な距離行列・非類似度行列に変換
- → 実験・臨床・シミュレーションプロファイル間の構造を探索
- 組成データは **compositional data transformation** が必要（e.g., CLR, ALR）

---

## 実験的検証

### 可能な readout
- Total growth, planktonic growth, biofilm growth, 組成分率
- 組成分析にはシーケンシングが必要

### 将来の実験系
- 嫌気チャンバー内プレートリーダー → 長時間・情報豊富な growth curve 取得可能
- 抗生物質による perturbation 実験

### テスト可能な仮説
> Negativicutes ギルドが欠如すると → Bacilli/Streptococci が優占し、コミュニティ多様性が低下する。
> Negativicutes 存在時は Bacilli/Streptococci のバランサーとして機能する。

**Perturbation 実験**: Negativicutes をコミュニティ開始時に除く、もしくは抗生物質でターゲット → 同様の効果が得られるか検証

---

## Action Items

| # | タスク | 担当 |
|---|------|------|
| 1 | 現行 PDF に「モデル説明セクション」を追加（目的・達成済み・予期外結果・実験との関係・仮定一覧） | Keisuke |
| 2 | DB由来の正/負関係の集約方法を文書化 | Keisuke |
| 3 | Hamilton rule と LV における方向性の扱いを明確化 | Keisuke |
| 4 | 入力パラメータ → 出力パラメータへの変換過程を示す中間データシートを作成 | Keisuke |
| 5 | 実験・シミュレーション両方の compositional data table を準備（sample factors + feature table） | Keisuke |
| 6 | 多変量解析に適したデータ形式で上記テーブルを整備 | Keisuke |
| 7 | WhatsApp でショ糖–乳酸–バイオフィルムのスケッチ・メモを送付 | Szymon |
| 8 | SimCom (Radek/Room Drum) の最新結果をレビューしモデル戦略に反映 | Szymon |
| 9 | CLSM データ（covered area, biofilm volume 等）と in vitro 系の関係を整理 | Szymon |

---

## 「モデル説明セクション」の構成（Action Item 1）

Szymon の要求に基づき、PDF に追加すべき項目:

1. **What the model aims to achieve** — モデルの目的
2. **What has already been achieved** — 達成済みの成果
3. **Which results were expected** — 予期された結果
4. **Which results were unexpected** — 予期しなかった結果
5. **How the model relates to experimental/clinical data** — データとの対応関係
6. **Assumptions currently included** — 現在の仮定一覧
7. **Assumptions that remain uncertain** — 不確かな仮定（要追加検討）
