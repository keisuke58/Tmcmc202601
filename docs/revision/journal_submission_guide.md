# 投稿先 4 誌の投稿方法（2026-10-07 調べ）

出所は各誌の公式ページの検索抜粋（公式ページは bot 判定・403 で直接は開けなかった）。**「未確認」は投稿前にブラウザで確認する。**
Meisam 氏の優先順（原文で確認済み）: 1 Bulletin of Mathematical Biology（2.3）→ 2 **Journal of Biological Dynamics**（2.7）→ 3 J. Theor. Biol.（1.9）→ 4 IJNMBE（2.1）。

| | 1. Bull. Math. Biol.（Springer） | 2. J. Biol. Dyn.（Taylor & Francis） | 3. J. Theor. Biol.（Elsevier） | 4. IJNMBE（Wiley） |
|---|---|---|---|---|
| IF | 2.3（JCR 2025） | 2.7（Meisam 氏の表） | 1.9（未確認） | 2.1（未確認） |
| 投稿システム | Springer Nature アカウント（EM か Snapp かは未確認） | Taylor & Francis（ScholarOne／Submission portal、未確認） | Elsevier（EM、未確認） | Wiley Research Exchange（submission.wiley.com/journal/cnm） |
| Abstract／keywords | 150–250 語／4–6 | 未確認 | ≤250／1–7 | ≤400（＋図付き要旨 250）／≤7 |
| 長さの上限 | 明記なし（未確認） | 未確認 | 明記なし（未確認） | 見つからず（未確認） |
| LaTeX | sn-jnl 推奨、Word 可。**通し行番号・ページ番号必須** | T&F の LaTeX テンプレート（未確認） | elsarticle（必須ではない） | Wiley NJD 推奨 |
| 追加物 | 参考文献の前に **Declarations**（利益相反・資金・倫理・データ／コードの可用性・著者貢献） | 未確認 | **Highlights 3–5 行（各 ≤85 字）必須**、Graphical abstract 任意、CRediT 必須 | 図付き要旨、**Novelty File（≤100 語）**、**コードとデータを Data Files でアップロード** |
| APC（定価） | €2,590 | **USD 1,680（Gold OA、全論文）** | $2,580 | $4,250 |
| DEAL（LUH・MHH） | 対象（Springer Nature） | **対象外**（T&F は DEAL なし。LUH・TIB の個別契約は要確認） | 対象（Elsevier、LUH は 2024-01 から） | 対象（Wiley） |
| プレプリント | 可（投稿時に DOI 申告） | 未確認 | 可（DOI を引用） | 可 |
| 最初の判断 | 中央値 8 日 | 未確認 | 7 日 | 未確認 |

- **2 番は Journal of Biological Dynamics**（Meisam 氏の原文で確認）。生物の動的モデルの数理が専門で分野はよく合う。Gold OA で APC USD 1,680、**Taylor & Francis は DEAL の対象外**なので費用の扱いは要確認。詳細な投稿要件は BMB で落ちたときに調べる
- **DEAL**: 責任著者（corresponding author）の所属を LUH か MHH にすると OA 費用が 0。**Gmail ではなく大学のアドレスで投稿する**
  - 責任著者のアドレス（2026-10-07 決定）: **keisuke.nishioka@stud.uni-hannover.de**
  - 要確認: DEAL の資格確認は所属とメールのドメインで行うが、**学生のアドレス（stud.uni-hannover.de）が対象として通るか**は出版社と TIB しだい。受理後の OA 選択の画面で「LUH の対象」と出なければ、TIB（OA 担当）に問い合わせるか、責任著者を職員アドレスを持つ共著者にする
- **BMB の門前払いの基準**: 「生物学の理解に実質的な前進があるか、生物学にはっきり使える新しい数理手法か」。GPU の速さだけの論文は危ない →
  Abstract と Introduction は**生物学的な結果（Fn–Pg と Pg の後期増加の予測）を先に**書く

## BMB 投稿のチェックリスト

1. sn-jnl テンプレート、通し行番号、ページ番号。.tex・.cls/.bst・図・コンパイル済み PDF をアップロード
2. Abstract 150–250 語（未定義の略語なし）、keywords 4–6
3. Declarations 節: Funding、Competing interests、Ethics（in vitro・ヒト／動物なし → not applicable）、Data availability、Code availability（GitHub＋Zenodo DOI）、Author contributions
4. プレプリントを出すなら投稿時に DOI
5. 責任著者は大学のアドレス（DEAL）
6. Cover letter（生物学的な貢献と BMB に合う理由）。推薦査読者を求められたときの候補
7. 最初の判断は 8 日前後（門前払いかどうかがすぐ分かる）
8. ✅ ORCID 取得済み: **0009-0001-9360-1336**（2026-10-07、LUH アカウントと連携、Editorial Manager (bmab) に認可）。原稿の責任著者の脚注に記載済み
