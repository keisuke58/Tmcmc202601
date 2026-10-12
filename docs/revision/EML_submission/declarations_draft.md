# 投稿用の申告（下書き、2026-10-12）

Elsevier（Extreme Mechanics Letters）の投稿に要る 2 つ。**中身を本人が確認してから**原稿の末尾（謝辞の前後）に入れる。

---

## 1. 生成 AI の使用の申告（必須）

Elsevier の決まった書式（見出しと文の形は変えない。[ ] の部分だけ埋める）:

> **Declaration of generative AI and AI-assisted technologies in the writing process**
>
> During the preparation of this work the author(s) used [NAME TOOL / SERVICE] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication.

### 案（実際の使い方に合わせて選ぶ）

**案 A（おすすめ・正直で広め）**
> During the preparation of this work the authors used Claude (Anthropic) in order to improve the language of the manuscript, to assist in writing and checking analysis and plotting scripts, and to check the consistency of numbers between the text and the computed results. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

**案 B（文章の推敲だけに絞る場合）**
> During the preparation of this work the authors used Claude (Anthropic) in order to improve the language and readability of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

### 確認してほしいこと
- **どこに使ったか**を正直に書く。文章の推敲・コード・図のスクリプト・数値の照合に使っていれば、案 A。
- Elsevier の方針では、**図や画像そのものを生成 AI で作る**ことは原則として認められない（データから描いた図をスクリプトで作るのは、ここでいう「画像の生成」には当たらない、という整理）。図はすべてデータとスクリプトから描いているので問題ない。
- 研究のアイデア・結論・解釈は著者が決めた、という立場で書く（申告文の「take full responsibility」）。
- **共著者全員に関わる**ので、共著者メールで一言確認してもよい（「AI の使用の申告を案 A で入れる」）。
- 参照: https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals

---

## 2. 利益相反の申告（Declaration of competing interest）

Elsevier は投稿時に **Declaration of Interest のツール**（https://declarations.elsevier.com/ ）で作った文書（.docx）を求める。全員に利益相反がなければ、ツールで「なし」を選ぶと次の定型文になる:

> **Declaration of competing interest**
>
> The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### 確認してほしいこと
- **共著者全員について**「なし」でよいか。資金（DFG SIIRI・ERC Gen-TSM）は謝辞・資金の欄に書いてあり、利益相反には当たらない。
- 企業との共同研究・特許・コンサルなどがある人がいれば、その人の分だけ書く（MHH の先生方は臨床・企業との関係がありうるので、共著者メールの確認項目に入れておくと安全）。
- ツールで .docx を作り、投稿時にアップロードする（本人の作業、5 分）。

---

## 原稿への入れ方（確認後）

`EML_manuscript_nishioka.tex` の Funding・Acknowledgements の近くに、見出し付きで 2 節を足す:

```latex
\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Declaration of generative AI and AI-assisted technologies in the writing process}
During the preparation of this work the authors used Claude (Anthropic) in order to ... (案 A または B)
```
