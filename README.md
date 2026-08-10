# hashline

[![CI](https://github.com/Ikuta-N/hashline/actions/workflows/ci.yml/badge.svg)](https://github.com/Ikuta-N/hashline/actions/workflows/ci.yml)

思いついたことを **一行** で書き捨て、あとから引き出すためのメモアプリ。
タイトルもフォルダもエディタも要らず、本文に書いた `#hashtag` がそのまま
タグになる。取り出し方はタグ・全文検索・意味検索の 3 通り。

データはすべて手元の SQLite ファイル 1 つに入る。どこにも送信しない。

![hashline のセッション: ノートの記録、一覧、全文検索、そして最後に、
クエリと 1 文字も共有しないノートを見つける意味検索](docs/cli-session.png)

## できること

- **一行入れて、一件出す。** 記録も取り出しもコマンド 1 つ。
- **タグは本文から取る。** ノートに `#sqlite` と書けばタグが付く。
- **日本語で引ける全文検索。** FTS5 の trigram トークナイザを使うので、
  分かち書きなしに `全文検索` が一致する。
- **意味で引ける検索（任意）。** 「眠れない」で「昨日は寝不足で頭が回らな
  かった」を見つける。
- **まとめて取り込む。** `.md` / `.txt` のディレクトリを指すと、各行あるいは
  Markdown の各セクションがノートになる。
- **読書モードと引用。** 文献をピン留めして読むと、記録したノートに引用
  キーのタグとページ番号が自動で付く。
- **返信とスレッド。** ノートに返信して木構造を作り、Markdown アウトライン
  として書き出せる。
- **Web UI。** CLI と同じデータベースをブラウザからも扱える。

## インストール

[uv](https://docs.astral.sh/uv/) があれば 1 行で入る（Python 3.12 は uv が
用意する）。

```bash
uv tool install git+https://github.com/Ikuta-N/hashline.git
```

これで `hashline` コマンドが使えるようになる。意味検索も使うなら

```bash
uv tool install "hashline[ml] @ git+https://github.com/Ikuta-N/hashline.git"
```

## 使ってみる

```bash
# 記録する。add と打たなくてよい
hashline 今日は寝不足だった
hashline "FTS5 の bm25 を調べた #sqlite #検索"

# 読み返す（新しい順）
hashline list
hashline list --tag sqlite

# 全文検索（一致度の高い順）
hashline search bm25
```

```
    2  2026-08-10 16:56  FTS5 の bm25 を調べた #sqlite #検索  [sqlite, 検索]
    1  2026-08-10 16:56  今日は寝不足だった
```

コマンド名でないテキスト——日本語を含む、空白を含む、`#` で始まる——は
ノートとして解釈される。打ち間違い（`hashline serach`）が黙ってノートに
ならないよう、ASCII の単語 1 つはコマンド扱いのままにしてある。明示したい
ときは従来どおり `hashline add "..."` と書ける。

データベースの場所は `--db PATH` で指定する。指定がなければ `$HASHLINE_DB`、
それも無ければ `~/.local/share/hashline/hashline.db` を使う。

ほかに、まとめて取り込む `import`、読書モードの `bib` / `read`、タグの
ピン留め `pin`、返信の `reply` / `thread`、書き出しの `export`、集計の
`stats` がある。→ [CLI リファレンス](docs/cli.md)

## Web UI

引数なしで起動する。

```bash
hashline
# http://127.0.0.1:8000

hashline serve --port 9000   # ポートを変えるなら
```

![hashline の Web UI: コンテキスト帯、複数行の入力欄、semantic トグル、
そして章・節・項の入れ子として取り込まれた Markdown 文書](docs/web-ui.png)

CLI と同じデータベース（`--db` も `$HASHLINE_DB` も尊重する）に対して、
記録・絞り込み・打ちながらの検索ができる。CLI との対応表は
[設計メモ](docs/design.md#web-ui-と-cli-の対応)。

> **注意:** `/import` と `/bib/import` の `path` 欄は、サーバを動かしている
> マシンのファイルシステムを直接読む。このアプリをネットワークに公開しない
> こと。

## 意味検索（任意）

文字ではなく意味でノートを引く。入れなくてもアプリは問題なく動く。

```bash
# torch（CPU ビルド）と sentence-transformers が入る
uv tool install "hashline[ml] @ git+https://github.com/Ikuta-N/hashline.git"

hashline index                # 未埋め込みのノートを埋め込む
hashline search 眠れない --semantic
```

```
0.0164      5  2026-08-09 22:45  昨日は寝不足で頭が回らなかった #日記  [日記]
0.0161      3  2026-08-09 22:45  朝の散歩を習慣にしたい #日記          [日記]
```

どちらのノートにも「眠れない」という文字列は無い。Web UI では検索ボックス
横の **semantic** トグルにあたり、埋め込みはサーバが自動で行う。
仕組みと保存形式は [意味検索](docs/semantic-search.md)。

## 開発

```bash
git clone https://github.com/Ikuta-N/hashline.git
cd hashline
uv sync --dev          # 開発ツール。ml エクストラは入らない

uv run ruff check .
uv run mypy src
uv run hashline --help # 作業ツリーのまま動かす
```

### テスト

```bash
uv run pytest                                     # 既定のスイート
uv run pytest --cov=src --cov-report=term-missing # カバレッジ付き
uv run pytest -m slow                             # モデルを要するテストのみ
```

`slow` を付けたテストは埋め込みモデルのダウンロードを要するため既定では
除外され、CI でも実行しない。`tests/fixtures/` のインポート用データは
すべて合成で、実在のノートディレクトリはテストのどこからも参照しない。

CI は push と pull request のたびに `ruff check .`、`mypy src`、
`pytest --cov` を実行する。

コードの構成と設計判断は [設計メモ](docs/design.md) にまとめてある。

## ライセンス

MIT。[LICENSE](LICENSE) を参照。

同梱している htmx 2.0.4（`src/hashline/web/static/htmx.min.js`）は 0BSD で
配布されており、帰属表示の義務は無い。
