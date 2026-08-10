# CLI リファレンス

[README](../README.md) に載っている `add` / `list` / `search` 以外のコマンドと、
その挙動の細部をまとめる。

すべてのコマンドは `--db PATH` を取る。指定がなければ `$HASHLINE_DB`、
それも無ければ `~/.local/share/hashline/hashline.db` を使う。

## 検索

```bash
hashline search QUERY [--limit N] [--semantic]
```

- 結果は BM25 でランク付けし、良い順に表示する。表示されるスコアは
  `-bm25()` なので **大きいほうが良い**。ノート数が少ないうちはスコアが
  `0.00` になることがある。BM25 は半数近くのノートに出現する語に重みを
  与えないためで、異常ではない。
- trigram インデックスは 3 文字未満のクエリに答えられないため、その場合は
  部分一致の走査にフォールバックする。結果は新しい順で、スコアは `0.00`。
- クエリ全体をリテラルとして扱う。`#`、`-`、`*`、`"` は演算子ではなく
  そのまま検索対象になる。
- `--semantic` は意味によるランキングを混ぜる。`ml` エクストラと
  インデックス済みのベクトルが要る。詳細は
  [意味検索](semantic-search.md)。

## インポート

```bash
hashline import PATH... [--mode line|heading|outline] [--tag NAME] [--dry-run]
```

`--mode` がファイルのどの単位をノートにするかを決める。

- `--mode line`（既定）は空行以外の 1 行を 1 ノートにする。階層は作らない。
- `--mode heading` は Markdown の 1 セクション（見出しとその下の行）を
  1 ノートにし、**見出しレベルで入れ子にする**。`#` の下の `##` はその
  返信になり、次の `##` は兄弟になる。レベルは数えるのではなく順位付け
  するので、`#` から `###` に飛ぶ文書は 3 段ではなく 2 段として扱う。
  最初の見出しより前の本文は独立した根として残り、フェンス付きコード
  ブロック内の `#` はセクションを開始しない。
- `--mode outline` は箇条書きのインデントで入れ子にする。`hashline export`
  が出力する形式と対応する。
- `--tag NAME` は繰り返し指定でき、その実行で作られる全ノートに付く。
  タグとして保存されるだけで **本文は書かれたとおりに保たれる**ため、
  `--tag` で付けた名前は全文検索の結果には出てこない。
- ディレクトリは `.md` / `.markdown` / `.txt` を再帰的にたどる。
  コマンドラインで直接名前を挙げたファイルは拡張子によらず読む。
- `--dry-run` は何がインポートされるかを報告するだけで、書き込まない。

## 読書モード

文献を「読んでいる最中」としてピン留めすると、その間に取ったノートに
自動で引用キーのタグとページ番号が付く。

```bash
# .bib から文献を取り込む
hashline bib import library.bib [--replace]
hashline bib list
hashline bib show smith2020

# 読書モードを開始する
hashline read start smith2020 [--tag NAME]
hashline read status

# ページ番号付きでノートを取る（コンテキストから自動で引き継がれる）
hashline add "Trigram indexing is fast for short queries #notes" --page 12-15

# 引用キーで絞り込む
hashline list --citekey smith2020

# 読書モードを終える
hashline read stop
```

知っておくとよいこと:

- `read status` が報告するのはピン留めされた **文献** である。`pin` で
  タグだけをピン留めしている場合、`read status` は何もピン留めされて
  いないと答える。そちらは `pin --show` で見る。
- 読書コンテキスト下で取ったノートには、読書タグ（既定 `#reading`、
  `--tag NAME` で変更可）と文献の引用キータグ（例 `#smith2020`）の
  両方が付く。
- `--page` は自由な文字列（`"42"`、`"12-15"`、`"xii"`、`"第3章"`）である。
  したがってページ順の並べ替えや範囲検索は無い。
- 文献をピン留めせずに `--page` を渡すとエラーになる。黙って無視はしない。
- `.bib` の値にある LaTeX エスケープは書かれたまま保存する。`{\"o}` は
  `ö` に変換しない。
- BibTeX パーサは読めないエントリを飛ばして報告する。インポート全体を
  失敗させない。

## タグのピン留め

`add` のたびに同じタグを打ち直さずに済ませる。

```bash
hashline pin research sqlite   # 以降のノートに付けるタグを指定
hashline pin --show            # 現在ピン留めされているタグ
hashline add "インデックス性能を調べている"   # #research と #sqlite が付く
hashline pin --clear
```

`import --tag` と同じく本文には触れないので、ピン留めしたタグ名は
全文検索の結果には出てこない。

## 返信・スレッド・アウトライン

ノートは既存のノートへの返信にでき、スレッドや木構造を作る。

```bash
hashline reply 1 "実装についての補足"          # ノート ID に返信する
hashline thread 1                              # 深さでインデントして表示
hashline list --roots-only                     # 返信を隠したタイムライン
hashline rm 1 [--recursive]                    # 削除
hashline export [--tag X] [--citekey Y] [--root ID] [-o FILE]
hashline import PATH --mode outline            # アウトラインを読み戻す
```

- `rm` は返信のあるノートを拒否する。スレッドごと消すなら `--recursive`。
- `export` は選択範囲の外に親がいるノートを根に繰り上げるので、絞り込みが
  返信を消してしまうことはない。
- 親の付け替えは無い。組み替えたいときは削除して入れ直す。

## 統計

手元のノートに対する集計ビュー。どれだけ書いたか、どのタグが動いているか、
どの文献を読んだか、スレッドがどんな形をしているか。

```bash
hashline stats                          # 総数と日付の範囲
hashline stats --activity --freq D      # 日ごとのノート数
hashline stats --tags --freq W --top 5  # 上位 5 タグを週ごとに
hashline stats --reading                # ノートのある文献ごとに 1 行
hashline stats --threads                # スレッドの根ごとに 1 行
hashline stats --tags --csv tags.csv    # 同じフレームをファイルに
```

```
                                  title  note_count              first_note_at               last_note_at        pages
citekey
smith2020  A Survey of Trigram Indexing           2 2026-08-10 10:56:15.940450 2026-08-10 10:56:16.030477  [12-15, 40]
```

ページは書かれたまま（`12-15`、`xii`、`第3章`）扱う。ページ参照は計算対象
ではないので、集めるだけで数値には解釈しない。画面ではリストとして表示し、
`--csv` では `;` で連結して 1 列として読めるようにする。

タイムスタンプは読み手のタイムゾーンで表示する。ただしリサンプルした
バケットは UTC のまま置く。バケットは瞬間ではないので、UTC の 1 日を
ずらすと 9 時間先の読み手には `09:00` というラベルになってしまうためである。
`--csv` は全体を UTC で書く。CSV を読むのはプログラムだからである。

同じビューは Web UI の `/stats` にもあり、ビュー・期間・タグ数を
ドロップダウンで選べる。

## アップグレード

現行リリースでは、このプロジェクトで初めてのスキーマ移行を行う。既存の
データベースは次に開いたときにその場で更新され、以降は古い版の `hashline`
では開けなくなる。

アップグレード前にデータベースファイルを控えておくとよい。場所は
`$HASHLINE_DB`、既定では `~/.local/share/hashline/hashline.db`。

```bash
cp "${HASHLINE_DB:-$HOME/.local/share/hashline/hashline.db}" ~/hashline.db.bak
```
