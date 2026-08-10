# 設計メモ

## モジュール構成

```
src/hashline/
  models.py     全レイヤが共有するデータクラス
  tags.py       #tag の抽出（純粋関数）
  store.py      SQLite リポジトリ。web にも CLI にも依存しない
  importer.py   文書 -> ノート下書き（純粋関数。ファイル I/O なし）
  bib.py        BibTeX の解析（純粋関数。ファイル I/O なし）
  outline.py    Markdown アウトラインの構築と描画（純粋関数）
  analytics.py  ストア上の DataFrame。pandas は遅延 import
  files.py      アダプタが共有するファイル読み込み
  cli.py        Typer アダプタ
  schema.sql    テーブル、インデックス、FTS5 インデックスと同期トリガ
  web/app.py       FastAPI + HTMX アダプタ
  ml/search.py     意味検索のランキング計算（純粋 numpy）
  ml/embed.py      任意エクストラ `ml` の裏にある埋め込みバックエンド
  ml/hybrid.py     ストアに対する意味検索。両アダプタで共有する
  ml/protocols.py  Embedder プロトコル。numpy 以外に依存しない
tests/
```

コア（`models`、`tags`、`store`、`importer`、`bib`、`outline`）は標準
ライブラリと numpy だけで書かれた素の Python である。`analytics` と
`ml/hybrid` はその上に乗り、ストアを読んで値を返す。CLI と Web UI は
それら全体に対する薄いアダプタで、ファイルシステム I/O は `files.py` と
`cli.py`（アダプタ層）が持つ。

## Web UI と CLI の対応

Web UI は CLI と同等の機能を提供する。

| CLI コマンド | Web UI のルート |
|---|---|
| `hashline list` | `GET /` |
| `hashline add` | `POST /notes` |
| `hashline rm` | `POST /notes/{id}/delete` |
| `hashline reply` | `POST /notes`（parent_id 付き） |
| `hashline thread` | `GET /`（root で絞り込み） |
| `hashline search` | `GET /`（q 付き） |
| `hashline pin` | `POST /context/pin`, `POST /context/clear_tags` |
| `hashline pin --clear`, `hashline read stop` | `POST /context/clear`（両方を解除） |
| `hashline read` | `POST /context/read`, `POST /context/clear_read` |
| `hashline bib list` | `GET /bib` |
| `hashline bib show` | `GET /bib/{citekey}` |
| `hashline import` | `POST /import`（ブラウザからのアップロードも可） |
| `hashline bib import` | `POST /bib/import`（ブラウザからのアップロードも可） |
| `hashline export` | `GET /export`, `GET /export/download` |
| `hashline index` | 起動時と書き込みのたびに自動で走る |
| `hashline tags` | *CLI のみ* |
| `hashline search --semantic` | 検索ボックス横の **semantic** トグル |
| `hashline stats` | `GET /stats` |

タグでの絞り込みはブラウザでもできる（`/?tag=NAME`。どの操作もフィルタを
持ち回る）が、**使っているタグの一覧は CLI のみ**である。検索ボックスの
下にあったタグのチップはタグが増えるほど際限なく伸びたので、ページング
するのではなく取り除いた。

意図的な違いが 1 つある。`hashline read start` はピン留めされたタグを
置き換えるが、`POST /context/read` は読書タグをそこに足す。コンテキスト帯
はピン留めしたタグとピン留めした文献を、それぞれ解除ボタン付きで横並びに
表示している。片方を始めたときに隣の列が空になるのは筋が通らないためである。

htmx は `src/hashline/web/static/` に同梱してあるので、ページは CDN を
必要とせずオフラインで動く。同梱しているのは htmx 2.0.4（`htmx.min.js`、
約 50 KB）で、ライセンスは 0BSD。表示義務のある帰属表示は無い。更新が
必要になったらファイルを手で差し替える。

## pandas を集計だけに使い、保存に使わない理由

この層を作るきっかけになった問いは、hashline はノートの管理を SQL では
なく pandas でやるべきではないか、というものだった。議論ではなく計測した。

| | |
|---|---|
| `import hashline.cli` — Typer 込みでアプリ全体 | **約 40 ms** |
| `import pandas` 単体 | **約 230 ms** |
| `hashline list` の実行全体 | 0.06〜0.07 秒 |

1 台のマシンで `python -X importtime` と `time` を使い、キャッシュが温まった
状態で 5 回の中央値をとった。import の値は 37〜41 ms と 208〜238 ms の幅で、
プロセスツリーが冷えた状態での初回 import は 447 ms だった。再現するには:

```bash
uv run python -X importtime -c "import pandas" 2>&1 | tail -1
uv run python -X importtime -c "import hashline.cli" 2>&1 | tail -1
```

日常的な経路に pandas を置くと、一行のノートを取るコストが数倍になる。
しかも「1 行挿入して数行読む」という作業は DataFrame が得意とするもの
ではない。SQLite を置き換えれば、FTS5 インデックスと BM25 ランキング
（pandas に転置インデックスは無い）、単一行の原子的な書き込み、CLI と
Web サーバからの同時アクセス、外部キーのカスケードも失う。

そこで保存と取得には手を入れず、pandas は SQL より明確に優れている
ところ——グループ化・リサンプル・ピボット——にだけ使い、必要とする関数の
**内側**で import する。どちらかのアダプタを import したあとに `pandas`
が `sys.modules` に無いことをテストで表明している。この設計と、あらゆる
コマンドを黙って遅くするトップレベル import との間に立っているのは、その
テストだけだからである。
