# vidrag

Search your downloaded videos by what they say and show. Ask "dog catching a frisbee" or "pizza in brooklyn" and get
the matching videos, with the file on disk and a link to the original page, from a CLI or a small HTTP API.

It indexes what yt-dlp based downloaders (MeTube, ytdl-sub, plain yt-dlp) already leave on disk:
`<Series>/Season */<name>.info.json` beside each video. That's the same layout Jellyfin and Plex read. Point it at a
YouTube tree, a TikTok chat archive, or anything else yt-dlp can download. Everything runs on your own server; the
optional cloud steps (Luna or Gemini describing a video, an LLM reranking search results) are off unless you give them a key.

## What it builds per video

| Step | What | Where it runs |
|---|---|---|
| `update` | title, description, tags, chapters, subtitle files beside the video → SQLite FTS5 | local |
| `embed` | EmbeddingGemma-300m vectors (fastembed, ONNX) | local, CPU-capped container |
| `enrich` | the platform's public page; the TikTok plugin reads place tag, keywords, stickers and captions | local (plain HTTP, rate-limited) |
| `transcribe` | Whisper large-v3-turbo (whisper.cpp) | local, CPU-capped container |
| `diarize` | speaker turns (pyannote-3.0 + TitaNet via sherpa-onnx) | local, CPU-capped container |
| `ocr` | text on photo-post slides (RapidOCR) | local, CPU-capped container |
| `watch` | **optional cloud step:** Luna describes timestamped frames, Whisper speech and the caption; Gemini video/audio is also available | OpenRouter ZDR (default), or Google Gemini API |

At query time, an optional `[rerank]` model (any OpenRouter chat model with a zero-data-retention endpoint) reads
short text cards of the local search's candidates: it reorders the top 30, and for `--sort newest|oldest` it decides
which candidates match at all, so the answer is every match by date rather than the top few. `--local` skips it.

`update` and `embed` run for every collection. The other steps run only for the collections that list them, so a
large YouTube tree can stay metadata-only while a small chat archive gets everything.

Search fuses a keyword ranking (BM25 with concept groups) and a semantic ranking by weighted reciprocal rank. A
one-word query with a handful of exact hits (a handle, a dish, a venue) pins those first. `pipeline` drains every
step, newest video first, and re-scans collections whose `update_every` has passed. Start it after each download run.
`check` fails loudly (exit 2) if a collection's index shrinks or empties. A collection you remove from the config keeps
its rows until `vidrag update --prune`.

On a judged set of about a thousand short videos, EmbeddingGemma beat nomic-embed clearly and Gemini's descriptions
added a few more points of nDCG@10. The local cross-encoder rerankers we tried scored lower and took most of a minute
per query on a 4-core CPU. An LLM reranking the top 30 over OpenRouter raised nDCG@10 from 0.77 to about 0.92 on the
same set (gemini-3.8-flash at low effort: 2.4 s, about 0.6 cents a query; gpt-6-luna 0.91 at a fifth of the cost).
For sorted queries, gpt-6-luna at medium effort grading the candidate pool kept 95% of the clearly relevant judged
videos, 54% of the partly relevant and 2% of the irrelevant (about 0.6 cents and 15 s for 160 candidates); no local
cosine cutoff came close (0.12 from the best hit kept about 75% of the relevant at 70% precision).

## Setup

Needs Python 3.11+ (stdlib only) and Docker on the server. Build the images from `asr/`:

```sh
docker build -t vidrag-asr:3 -f asr/Dockerfile asr
docker build -t vidrag-embed:2 -f asr/Dockerfile.embed2 asr
```

Copy `config.example.toml` to `~/.config/vidrag/config.toml` and list your collections. Models for whisper.cpp and
sherpa-onnx go in `~/vidrag/models/` (see `asr/asr.py`). Add `"watch"` to a collection's steps to enable the cloud
watcher. The default `[watch] backend = "luna"` uses `openai/gpt-6-luna`, with `effort = "low"` and `max_frames = 50`.
Put `OPENROUTER_API_KEY=…` in `[rerank].key_file` (mode 0600); the `OPENROUTER_API_KEY` environment variable takes
precedence, including for a backfill with a different key. Watch works even when the reranker model is disabled.
Every Luna request requires zero data retention and denies data collection, with routing pinned to Azure and
fallbacks disabled.

Luna receives one request per video: up to 50 timestamped JPEG frames across the whole video, the uploader's caption,
and up to roughly 6,000 characters of Whisper speech (the start and ending of long transcripts). Short clips get
one frame per second; clips over 50 seconds get evenly spaced samples through the last second. Frames are 480 px tall,
with ffmpeg MJPEG quantizer 16 for compact previews. If the collection includes `transcribe`, watch waits for its
`.whisper.txt`, including an empty file for no speech. A `.whisper.fail` still waits: resolve transcription and retry
it before watching. Photo posts retain the same behavior as Gemini: their downloaded video is watched if it exists.

Luna writes `<series>/.transcripts/<id>.luna.json`, including the frame count and actual `usage.cost` charge. The index
prefers a successful Luna description and falls back to the existing `.gemini.json` description after an error or
refusal. Gemini sidecars are preserved. `vidrag stats` reports both backends' sidecar counts, Gemini's estimated spend,
and Luna's reported `cost_usd`, refusal count and count of sidecars with unknown charges.

`touch ~/vidrag/GEMINI_OFF` pauses either watcher. Billing, auth and quota errors (401, 402, 403, 429) also pause it
through `data/.gemini-hold`: six hours, or 30 minutes for 429. Remove the hold after fixing the key or balance to
resume sooner. Content-policy refusals are saved with `refused = true` and never retried automatically; other saved
errors retry after a week. Transient transport, extraction or invalid-output failures write no sidecar, and three
consecutive transient failures stop the run. Temporary frames are cleaned up under `data/.watch-tmp`.

For a backfill, `vidrag watch --workers 4` overlaps API calls while keeping frame extraction at no more than two
containers, each capped at 2 CPUs and 1 GB with low CPU shares. The default and pipeline use one worker. When a hold,
off switch or failure streak stops a concurrent run, already-started requests finish and queued work stops. Run
`vidrag update && vidrag embed` afterward to refresh search.

To switch back, set `[watch] backend = "gemini"`, leave `[watch] model` empty (or set a Gemini model explicitly), and
put `GEMINI_API_KEY=…` in `[gemini].key_file`. The original inline 1-fps 480p video/audio path, prompt, schema and
recitation retry remain available. `[gemini] model` still supplies its default. Switching watchers changes which
sidecars count as pending; existing Gemini descriptions remain available as the index fallback.

```sh
vidrag update && vidrag embed     # index everything, then the vectors
vidrag pipeline                   # run each collection's steps, newest first
```

## Use it (people and agents)

```sh
vidrag query "places to eat in brooklyn"
vidrag query "how does a cassette adapter work" -c youtube --series "Technology Connections"
vidrag query "pasta" -c tiktok --from them --since 2026-06 --sort newest   # every match, newest first
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
| `GET /v1/search?q=…` or `POST /v1/search {"q": …}` | search; filters `collection`, `series`, `author`, `since`, `until`, `from`, `source`, `limit`, `mode`, `all`; `sort=newest\|oldest` returns every match by date (`newest=1` is the same); `local=1` skips the LLM step |
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
