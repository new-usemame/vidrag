# tiktok-rag

Search a TikTok DM archive by what the videos say and show. Ask "places to eat in brooklyn" or "dog catching a frisbee"
and get the matching videos from your chat, with their NAS paths and TikTok links.

It indexes videos that another tool has already downloaded to a NAS as yt-dlp series folders (`TikTok DMs/Season
*/…info.json + .mp4`), plus the chat's order and sender from a DM poller's `chat_index.json`.

## What it builds per video

| Step | What | Where it runs |
|---|---|---|
| `update` | captions, hashtags, author, music, share dates → SQLite FTS5 | local |
| `enrich` | public post page: place tag, TikTok keywords, stickers, TikTok's own subtitles | local (plain HTTP, rate-limited) |
| `transcribe` | Whisper large-v3-turbo (whisper.cpp) | local, CPU-capped container |
| `diarize` | speaker turns (pyannote-3.0 + TitaNet via sherpa-onnx) | local, CPU-capped container |
| `ocr` | text on photo-post slides (RapidOCR) | local, CPU-capped container |
| `watch` | **optional cloud step:** Gemini watches a 1-fps 480p re-encode and describes it | Google Gemini API |
| `embed` | EmbeddingGemma-300m vectors (fastembed, ONNX) | local |

`query` fuses keyword ranking (BM25 with concept groups) and semantic ranking by weighted reciprocal rank. A one-word
query with a handful of exact hits (a handle, a dish, a venue) pins those first. `pipeline` drains every step, newest
video first, and is meant to be started after each poll. `check` fails loudly if the index shrinks or empties.

On a test archive of about a thousand videos, EmbeddingGemma beat nomic-embed clearly. Gemini's descriptions added
a few more points of nDCG@10. The cross-encoder rerankers we tried scored lower and took most of a minute per query on
a 4-core CPU.

## Setup

Needs Python 3.10+ (stdlib only) and Docker on the server. Build the images from `asr/`:

```sh
docker build -t tiktok-rag-asr:3 -f asr/Dockerfile asr
docker build -t tiktok-rag-embed:2 -f asr/Dockerfile.embed2 asr
```

Site settings go in `~/.config/tiktok-rag/config.env` (`KEY=VALUE`; the environment wins):

```sh
TIKTOK_RAG_NAS=/path/to/media            # holds "TikTok DMs/" and "TikTok Saved/"
TIKTOK_RAG_CONTACT=sam                   # the other person in the chat: `--from sam` (or `--from them`)
TIKTOK_RAG_CHAT_INDEX=~/dm-poller/state/chat_index.json
TIKTOK_RAG_FLOOR_MIN_DM=100              # `check` fails below this many DM videos
```

Put a Gemini key in `~/.config/tiktok-rag/gemini.env` (`GEMINI_API_KEY=…`, mode 0600) only if you want the cloud step.
Without a key, `watch` is skipped and everything stays on your machine. `touch ~/tiktok-rag/GEMINI_OFF` pauses it.
Billing, auth and quota errors pause it on their own.

Models for whisper.cpp and sherpa-onnx go in `~/tiktok-rag/models/` (see `WHISPER_MODEL`, `asr/asr.py`).

## Use

```sh
tiktok-rag update && tiktok-rag pipeline        # index, then enrich everything
tiktok-rag query "places to eat in brooklyn"
tiktok-rag query "pasta" --from them --since 2026-06 --newest
tiktok-rag query "cat" --source fav --json
tiktok-rag recent 20
tiktok-rag stats
```

`bin-tiktok-rag` runs the same commands from a laptop over ssh (`TIKTOK_RAG_SSH=user@server`).

`favs/favs.py` optionally mirrors your own Favorites tab into a MeTube folder. It reads the page the way a person
scrolls it, through an already logged-in browser.

## Tests

```sh
python3 -m unittest -v test_tiktok_rag
```

## Evaluating search

`eval/` scores retrieval on your own judged queries (`queries.json`, `qrels.json`). Those files describe your archive
and are git-ignored. `eval/score_prod.py` scores the production `query`. `eval/unjudged.py` lists the top results that
nobody has graded yet, since unjudged results count as misses and make a real gain look like a dip.

## License

All rights reserved. The code is public to read, not to copy or reuse. See [LICENSE](LICENSE).
