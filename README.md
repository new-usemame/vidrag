# vidrag

Search your downloaded videos by what they say and show. Ask "dog catching a frisbee" or "pizza in brooklyn" and get
the matching videos, with the file on disk and a link to the original page, from a CLI or a small HTTP API.

It indexes what yt-dlp based downloaders (MeTube, ytdl-sub, plain yt-dlp) already leave on disk:
`<Series>/Season */<name>.info.json` beside each video. That's the same layout Jellyfin and Plex read. Point it at a
YouTube tree, a TikTok chat archive, or anything else yt-dlp can download. Everything runs on your own server; the
only optional cloud step (Gemini watching a video) is off unless you turn it on for a collection and give it a key.

## What it builds per video

| Step | What | Where it runs |
|---|---|---|
| `update` | title, description, tags, chapters, subtitle files beside the video → SQLite FTS5 | local |
| `embed` | EmbeddingGemma-300m vectors (fastembed, ONNX) | local, CPU-capped container |
| `enrich` | the platform's public page; the TikTok plugin reads place tag, keywords, stickers and captions | local (plain HTTP, rate-limited) |
| `transcribe` | Whisper large-v3-turbo (whisper.cpp) | local, CPU-capped container |
| `diarize` | speaker turns (pyannote-3.0 + TitaNet via sherpa-onnx) | local, CPU-capped container |
| `ocr` | text on photo-post slides (RapidOCR) | local, CPU-capped container |
| `watch` | **optional cloud step:** Gemini watches a 1-fps 480p re-encode and describes it | Google Gemini API |

`update` and `embed` run for every collection. The other steps run only for the collections that list them, so a
large YouTube tree can stay metadata-only while a small chat archive gets everything.

Search fuses a keyword ranking (BM25 with concept groups) and a semantic ranking by weighted reciprocal rank. A
one-word query with a handful of exact hits (a handle, a dish, a venue) pins those first. `pipeline` drains every
step, newest video first, and re-scans collections whose `update_every` has passed. Start it after each download run.
`check` fails loudly (exit 2) if a collection's index shrinks or empties. A collection you remove from the config keeps
its rows until `vidrag update --prune`.

On a judged set of about a thousand short videos, EmbeddingGemma beat nomic-embed clearly and Gemini's descriptions
added a few more points of nDCG@10. The cross-encoder rerankers we tried scored lower and took most of a minute per
query on a 4-core CPU.

## Setup

Needs Python 3.11+ (stdlib only) and Docker on the server. Build the images from `asr/`:

```sh
docker build -t vidrag-asr:3 -f asr/Dockerfile asr
docker build -t vidrag-embed:2 -f asr/Dockerfile.embed2 asr
```

Copy `config.example.toml` to `~/.config/vidrag/config.toml` and list your collections. Models for whisper.cpp and
sherpa-onnx go in `~/vidrag/models/` (see `asr/asr.py`). For the cloud step, put `GEMINI_API_KEY=…` in the key file
(mode 0600) and add `"watch"` to a collection's steps; `touch ~/vidrag/GEMINI_OFF` pauses it, and billing, auth and
quota errors pause it on their own.

```sh
vidrag update && vidrag embed     # index everything, then the vectors
vidrag pipeline                   # run each collection's steps, newest first
```

## Use it (people and agents)

```sh
vidrag query "places to eat in brooklyn"
vidrag query "how does a cassette adapter work" -c youtube --series "Technology Connections"
vidrag query "pasta" -c tiktok --from them --since 2026-06 --newest
vidrag recent 20 -c youtube --json
vidrag show <id>                  # every field, full transcript (JSON)
vidrag collections                # what can be searched, with series and counts (JSON)
vidrag guide                      # a one-screen guide to paste into an agent's context
```

Text output is readable as is; `--json` gives the same results as objects. Each result carries `id`, `collection`,
`series`, `title`, `author`, `caption`, `date`, `path` (the file), `url` (the original page), `seen` (what the video
shows, if watched), `snippet` (the words that matched) and `match` (`meaning` = found by the semantic leg alone).

`vidrag-remote` runs the same commands from a laptop over ssh (`ssh = "user@server"` in the laptop's config).

## HTTP API

`vidrag serve` answers the same searches over HTTP with the embedding model kept loaded, so a semantic query takes a
fraction of a second instead of a container start. Run it inside the embed image, which already has the model runtime:

```sh
head -c 24 /dev/urandom | base64 > ~/.config/vidrag/api-token && chmod 600 ~/.config/vidrag/api-token
docker run -d --name vidrag-api --restart unless-stopped --cpus=2 --memory=3g --cpu-shares=128 \
  --user "$(id -u):$(id -g)" -e HOME="$HOME" -v "$HOME/.config/vidrag:$HOME/.config/vidrag:ro" \
  -v "$HOME/vidrag:$HOME/vidrag" -v "$HOME/vidrag/models/fastembed:/models/fastembed" \
  -p 127.0.0.1:8790:8790 vidrag-embed:2 python "$HOME/vidrag/vidrag.py" serve --listen 0.0.0.0:8790
```

```sh
T=$(cat ~/.config/vidrag/api-token)
curl -s -H "Authorization: Bearer $T" 'http://127.0.0.1:8790/v1/search?q=frisbee+dog&limit=5'
curl -s -H "Authorization: Bearer $T" -d '{"q": "sourdough", "collection": "youtube", "since": "2024"}' \
  http://127.0.0.1:8790/v1/search
```

| Route | What |
|---|---|
| `GET /v1/search?q=…` or `POST /v1/search {"q": …}` | search; filters `collection`, `series`, `author`, `since`, `until`, `from`, `source`, `limit`, `mode`, `newest`, `all` |
| `GET /v1/recent?n=…` | newest first, same filters |
| `GET /v1/videos/{id}` | every field, full transcript |
| `GET /v1/collections` | collections, their series and counts |
| `GET /v1/stats` | coverage and floor verdicts |
| `GET /v1/health` | liveness and floors (no token) |
| `GET /openapi.json` | the spec, for tools that import one (no token) |

Every route except health, the spec and `/` needs `Authorization: Bearer <token>`. The server refuses to listen on a
non-loopback address without a token file. Mount any file a plugin reads (the TikTok chat index, say) at the same path
inside the container, or `/v1/health` reports that collection's floor check as failed. Set `[api] url` and the CLI
borrows the server's loaded model too; when the server is down it falls back to a one-shot container.

## TikTok chat archives

The `tiktok` plugin adds what a DM archive needs on top of the generic index: the chat's order and who sent each
video (from a DM poller's `chat_index.json`), share dates, your Favorites (`favs/favs.py` mirrors the Favorites tab
into a MeTube folder, reading the page the way a person scrolls it through an already logged-in browser), the public
post page, and a floor check that fails when chat videos go missing. The `tiktok-rag` command is that collection with
the older single-archive defaults (`--source dm`) and field names.

## Tests

```sh
python3 -m unittest -v test_vidrag
```

## Evaluating search

`eval/` scores retrieval on your own judged queries (`queries.json`, `qrels.json`). Those files describe your archive
and are git-ignored. `eval/score_prod.py` scores the production query. `eval/unjudged.py` lists the top results that
nobody has graded yet, since unjudged results count as misses and make a real gain look like a dip.

## License

All rights reserved. The code is public to read, not to copy or reuse. See [LICENSE](LICENSE).
