# Zotero セットアップガイド

## 1. Zotero の起動

```bash
flatpak run org.zotero.Zotero
```

## 2. Better BibTeX のインストール（GitHub から取得済み）

1. Zotero を起動
2. **ツール** → **アドオン** → 歯車アイコン ⚙️ → **ファイルからアドオンをインストール**
3. 以下のファイルを選択:
   ```
   /home/nishioka/IKM_Hiwi/docs/zotero_setup/zotero-better-bibtex-8.0.38.xpi
   ```
4. Zotero を再起動

### Better BibTeX の主な機能

- **引用キー自動生成**: `auth.lower + shorttitle(3,3) + year` 形式
- **BibLaTeX 対応**: `.bib` ファイルの自動生成・更新
- **Keep updated**: ライブラリ変更時に `.bib` を自動同期

## 3. Zotero Connector（ブラウザ拡張）

Web ページから文献を直接 Zotero に保存するには、ブラウザ拡張をインストール:

| ブラウザ | インストール先 |
|----------|----------------|
| Chrome | [Chrome ウェブストア](https://chromewebstore.google.com/detail/zotero-connector/ekhagklcjbdpajgpjgmbionohlpdbjgc) |
| Firefox | [Firefox Add-ons](https://addons.mozilla.org/firefox/addon/zotero-connector/) |
| Edge | [Edge Add-ons](https://microsoftedge.microsoft.com/addons/detail/zotero-connector/fgehngbhnnjadehjfajalnjgbljibdda) |

インストール後、ブラウザのツールバーに Zotero アイコンが表示されます。

## 4. 推奨設定（Better BibTeX）

1. Zotero の **編集** → **環境設定** → **Better BibTeX**
2. **Citation Keys** タブ:
   - 形式: `[auth:lower][year]` またはデフォルトのまま
3. **Export** タブ:
   - 「Keep updated」で `.bib` を常に最新に

## 5. よく使うショートカット

- `Ctrl+Shift+A`: 文献を追加
- ブラウザで Zotero アイコンクリック: 現在のページを保存
