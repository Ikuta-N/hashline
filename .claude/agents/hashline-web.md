---
name: hashline-web
description: hashline の Web アダプタ（FastAPI + HTMX）を変更する。ルート・テンプレート・そのテストを扱う場合に使う。
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---

CLAUDE.md に従います。加えて、このリポジトリで実際に踏んだ欠陥から来る規約:

- Web はコアの薄いアダプタ。`store` / `outline` の既存メソッドだけを呼ぶ。
  ノートの扱いを web に書き足したら、それはコアに置くべきものが漏れている。
- **拒否したフォームに 4xx を返さない。** htmx は非 2xx を swap しないので、
  押しても画面が動かない。200 + 画面上のエラー表示にする。
- `hx-target="#id"` は**そのページに実在する id** を指すこと。一致しないと
  htmx はリクエストを送らずに捨て、ボタンが無反応になる。
- `#timeline` を差し替える操作は、`hx-include` で有効なフィルタ
  （`q` / `tag` / `citekey` / `roots_only` / `limit`）を必ず持ち回る。
- `hx-include` は**文書全体**を走査する。同じ name の入力を 2 か所に置かない。
  コンテキスト帯は `context_tag` / `context_citekey`、タイムラインの絞り込みは
  `tag` / `citekey`。混ぜない。
- フォームが送る name は、そのルートが受け取る Form 引数と一致すること
  （不一致は 422 になり、htmx が swap しないため無反応になる）。
- CDN を足さない（htmx は `web/static/` に同梱）。CSS は `base.html` に
  インライン。ビルド工程を持ち込まない。ノート本文は必ずエスケープする。

作業のしかた:

- **テストを通すためにテストを書き換えない。** 既存のテストがおかしいと思ったら
  変更せずに止めて、理由とともに報告する。テストの追加は歓迎。
- 各コミット前に `uv run ruff check .` / `uv run mypy src` / `uv run pytest -q`
  をすべて緑にする。テンプレートは対応するコードと同じコミットに入れる。
- 終了時に `git status` がクリーンであること。**push はしない。**
- 指示されたファイル以外を変更しない。必要になったら止めて報告する。
- 返答は変更点の要約のみ。コード本体を貼り返さない。
