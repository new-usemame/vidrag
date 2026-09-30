#!/usr/bin/env python3
"""tiktok-rag: a local search index over a TikTok DM archive on a NAS.

Runs on the media server, stdlib only. Everything is local except `watch` (Gemini describes each video; see that
section). Site settings (paths, the contact's name, floor sizes) come from the environment or from
~/.config/tiktok-rag/config.env (KEY=VALUE lines; the environment wins).
Sources (all read-only):
  - NAS series dirs  $TIKTOK_RAG_NAS/TikTok DMs|TikTok Saved/Season */*.info.json  (+ .mp4 beside it)
  - chat index       $TIKTOK_RAG_CHAT_INDEX ({cards: [[id, isMyself], ...]}, oldest first)
  - MeTube history   <TikTok DMs>/.metube/completed.json        (download timestamp per id)
  - transcripts      <series>/.transcripts/<id>.*               (phase 2; indexed when present)

Index: SQLite + FTS5 at $TIKTOK_RAG_DB (default ~/tiktok-rag/data/index.db). `update` is incremental
(re-reads only files whose mtime changed) and ends with the floor check, so a shrunken or empty index
exits non-zero instead of passing silently.

Share dates: the chat index has order but no timestamps. A video that entered the chat after the
backfill (chat position >= BACKFILL_INDEX_SIZE) was downloaded within one 15-min poll of being shared,
so its MeTube timestamp is the share time ("poll" precision). Older videos get a lower bound: the
latest upload date among it and every video before it in the chat ("est" precision, shown as ">=").
"""
import argparse
import glob
import json
import math
import os
import re
import sqlite3
import sys
from contextlib import closing
import time
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()


def load_config(path=Path(os.environ.get("TIKTOK_RAG_CONFIG", HOME / ".config/tiktok-rag/config.env"))):
    """KEY=VALUE lines -> os.environ defaults, so site settings live outside the code."""
    try:
        for ln in path.read_text().splitlines():
            k, sep, v = ln.strip().partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


load_config()
NAS = Path(os.environ.get("TIKTOK_RAG_NAS", "/srv/media"))
# The other person in the chat: their videos get this sender label and `--from <name>` (or `--from them`).
CONTACT = os.environ.get("TIKTOK_RAG_CONTACT", "them").strip().lower() or "them"
SERIES = {"dm": "TikTok DMs", "saved": "TikTok Saved"}
CHAT_INDEX = Path(os.environ.get("TIKTOK_RAG_CHAT_INDEX", HOME / "dm-poller/state/chat_index.json"))
DB = Path(os.environ.get("TIKTOK_RAG_DB", HOME / "tiktok-rag/data/index.db"))
# Your own TikTok Favorites, read by favs/favs.py: {"items": {id: {fav_ts, ...}}}. Those files sit in "TikTok Saved";
# the index labels them source 'fav' and dates them by favorite time (a lower bound, like "est").
FAVS_INDEX = Path(os.environ.get("TIKTOK_RAG_FAVS_INDEX", DB.parent / "favs_index.json"))
# How many videos the chat index held when the backfill began; everything after that position was picked up by a
# live poll, so its download time is its share time.
BACKFILL_INDEX_SIZE = int(os.environ.get("TIKTOK_RAG_BACKFILL_INDEX_SIZE", "0"))
# Floor: fail when DM rows drop below this absolute minimum, below the last high-water mark minus slack,
# or when more chat videos lack a row than the known-unrecoverable allowance.
FLOOR_MIN_DM = int(os.environ.get("TIKTOK_RAG_FLOOR_MIN_DM", "1"))
FLOOR_SLACK = int(os.environ.get("TIKTOK_RAG_FLOOR_SLACK", "3"))
MAX_MISSING = int(os.environ.get("TIKTOK_RAG_MAX_MISSING", "15"))

HASHTAG = re.compile(r"#([^\s#.,!?;:()\[\]{}\"'’]+)")
STOP = set("""a an and are as at be but by for from has have i in is it its of on or so that the this to was
were what when where which who will with you your me my she her he his they them their we our us
send sent sends sending tiktok tiktoks video videos things thing stuff some any all""".split()) | {CONTACT}

SCHEMA = """
create table if not exists videos(
  id text primary key, source text, sender text, chat_pos integer,
  share_ts integer, share_precision text, upload_ts integer,
  author text, author_name text, caption text, hashtags text, music text, duration integer,
  is_photo integer, location text, nas_path text, info_path text, info_mtime real, tr_mtime real,
  transcript_tiktok text, transcript_whisper text, ocr text, keywords text, topics text,
  transcript_speakers text, n_speakers integer, first_seen integer, visual text);
create virtual table if not exists fts using fts5(
  id unindexed, caption, hashtags, author, music, location, keywords, transcript, ocr, visual,
  tokenize = "porter unicode61 remove_diacritics 2");
create table if not exists meta(k text primary key, v text);
"""
SCHEMA_VERSION = "4"


def connect():
    """The DB is derived from files on the NAS, so a schema change just rebuilds it from scratch."""
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        old = hw = None
        try:
            with closing(sqlite3.connect(DB)) as c:
                old = c.execute("select v from meta where k='schema'").fetchone()
                hw = c.execute("select v from meta where k='high_water_dm'").fetchone()
        except sqlite3.Error:
            pass
        if not old or old[0] != SCHEMA_VERSION:
            DB.unlink()
            db = sqlite3.connect(DB)
            db.executescript(SCHEMA)
            db.execute("insert into meta values('schema', ?)", (SCHEMA_VERSION,))
            if hw:  # keep the floor's memory across a rebuild
                db.execute("insert into meta values('high_water_dm', ?)", (hw[0],))
            db.commit()
            db.close()
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA + f"insert or ignore into meta values('schema', '{SCHEMA_VERSION}');")
    return db


def meta_get(db, k, default=None):
    r = db.execute("select v from meta where k=?", (k,)).fetchone()
    return r[0] if r else default


def meta_set(db, k, v):
    db.execute("insert into meta(k,v) values(?,?) on conflict(k) do update set v=excluded.v", (k, str(v)))


def load_chat():
    """-> (complete, [(id, mine)]) oldest first."""
    d = json.loads(CHAT_INDEX.read_text())
    return bool(d.get("complete")), [(str(c[0]), c[1]) for c in d.get("cards", [])]


def load_metube_times():
    """id -> download unix time, from every series' MeTube history (ns timestamps)."""
    out = {}
    for f in NAS.glob("*/.metube/completed.json"):
        try:
            items = json.loads(f.read_text()).get("items", [])
        except (OSError, ValueError):
            continue
        for it in items:
            info = it.get("info") or {}
            if info.get("status") == "finished" and info.get("id") and info.get("timestamp"):
                out[str(info["id"])] = int(info["timestamp"]) // 1_000_000_000
    return out


TRANSCRIPT_SUFFIXES = (".tiktok.vtt", ".whisper.txt", ".whisper.json", ".turns.json", ".ocr.txt", ".page.json",
                       ".gemini.json")


def transcript_mtime(series_dir, vid):
    m = 0.0
    for suf in TRANSCRIPT_SUFFIXES:
        try:
            m = max(m, (series_dir / ".transcripts" / f"{vid}{suf}").stat().st_mtime)
        except OSError:
            pass
    return m


def read_sidecars(series_dir, vid):
    """Enrichment sidecars in <series>/.transcripts/ -> dict of index fields (missing files -> empty)."""
    tdir = series_dir / ".transcripts"
    out = {"tiktok": "", "whisper": "", "ocr": "", "location": "", "keywords": "", "topics": "", "stickers": "",
           "speakers": "", "n_speakers": None, "visual": ""}
    wj, tj = tdir / f"{vid}.whisper.json", tdir / f"{vid}.turns.json"
    if wj.exists() and tj.exists():
        try:
            out["speakers"], out["n_speakers"] = speaker_transcript(json.loads(wj.read_text(errors="replace")),
                                                                    json.loads(tj.read_text()))
        except (ValueError, KeyError, TypeError):
            pass
    p = tdir / f"{vid}.tiktok.vtt"
    if p.exists():
        out["tiktok"] = vtt_to_text(p.read_text(errors="replace"))
    p = tdir / f"{vid}.whisper.txt"
    if p.exists():
        out["whisper"] = p.read_text(errors="replace").strip()
    p = tdir / f"{vid}.ocr.txt"
    if p.exists():
        out["ocr"] = p.read_text(errors="replace").strip()
    p = tdir / f"{vid}.page.json"
    if p.exists():
        try:
            pg = json.loads(p.read_text())
        except ValueError:
            pg = {}
        poi = pg.get("poi") or {}
        out["location"] = ", ".join(x for x in (poi.get("name"), poi.get("address"), poi.get("category")) if x)
        out["keywords"] = " · ".join(pg.get("suggestedWords") or [])
        out["topics"] = ", ".join(dict.fromkeys(pg.get("labels") or []))
        out["stickers"] = " ".join(pg.get("stickers") or [])
    p = tdir / f"{vid}.gemini.json"
    if p.exists():
        try:
            out["visual"] = json.loads(p.read_text()).get("description") or ""
        except ValueError:
            pass
    return out


def speaker_transcript(whisper, turns):
    """Whisper segments + diarization turns -> ("S1: … / S2: …" text, n_speakers). Each segment goes to the
    speaker it overlaps most; a segment in a gap keeps the previous speaker. One speaker -> plain text."""
    tr = turns.get("turns") or []
    n = turns.get("n_speakers") or 0
    lines, prev = [], None
    for sg in whisper.get("transcription") or []:
        text = (sg.get("text") or "").strip()
        if not text:
            continue
        a, b = sg["offsets"]["from"] / 1000, sg["offsets"]["to"] / 1000
        best, spk = 0.0, prev
        for s, e, k in tr:
            ov = min(e, b) - max(s, a)
            if ov > best:
                best, spk = ov, k
        spk = 0 if spk is None else spk
        if lines and spk == prev:
            lines[-1] += " " + text
        else:
            lines.append(f"S{spk + 1}: {text}")
        prev = spk
    if n <= 1:
        return " ".join(l.split(": ", 1)[1] for l in lines), n
    return "\n".join(lines), n


def vtt_to_text(vtt):
    lines, prev = [], None
    for ln in vtt.splitlines():
        ln = ln.strip()
        if not ln or ln == "WEBVTT" or "-->" in ln or ln.isdigit() or ln.startswith(("NOTE", "Kind:", "Language:")):
            continue
        ln = re.sub(r"<[^>]+>", "", ln)
        if ln != prev:
            lines.append(ln)
        prev = ln
    return " ".join(lines)


def parse_info(path):
    d = json.loads(path.read_text())
    desc = d.get("description") or d.get("title") or ""
    tags = [t.lower() for t in HASHTAG.findall(desc)]
    for t in d.get("tags") or []:
        if t and t.lower() not in tags:
            tags.append(t.lower())
    music = " - ".join(x for x in (d.get("track"), d.get("artist")) if x)
    base = str(path)[: -len(".info.json")]
    mp4 = base + ".mp4"
    upload_ts = d.get("timestamp")
    if not upload_ts and d.get("upload_date"):
        upload_ts = int(datetime.strptime(d["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp())
    vid = str(d.get("id"))
    return {
        "id": vid, "author": d.get("uploader") or d.get("uploader_id") or "", "author_name": d.get("channel") or "",
        "caption": desc.strip(), "hashtags": " ".join(tags), "music": music, "duration": d.get("duration"),
        "upload_ts": upload_ts, "nas_path": mp4 if os.path.exists(mp4) else base + ".mp4",
        "is_photo": 1 if (path.parent.parent / ".photos" / vid).is_dir() else 0,
        "location": ", ".join(x for x in (d.get("location"),) if x),
    }


def update(args):
    db = connect()
    now = int(time.time())
    complete, cards = load_chat()
    dl = load_metube_times()
    known = {r["info_path"]: r for r in db.execute("select id, info_path, info_mtime, tr_mtime, first_seen from videos")}
    seen_files, changed = set(), 0
    for source, series in SERIES.items():
        sdir = NAS / series
        for p in sorted(sdir.glob("Season */*.info.json")):
            try:
                mtime = p.stat().st_mtime
                prev = known.get(str(p))
                if prev and prev["info_mtime"] == mtime and prev["tr_mtime"] == transcript_mtime(sdir, prev["id"]):
                    seen_files.add(prev["id"])  # unchanged: skip the NFS read + JSON parse
                    continue
                info = parse_info(p)
            except (OSError, ValueError) as e:
                print(f"warn: unreadable {p}: {e}", file=sys.stderr)
                continue
            vid = info["id"]
            if vid in seen_files:  # same id in both series: the first (DMs) wins
                continue
            seen_files.add(vid)
            sc = read_sidecars(sdir, vid)
            first_seen = prev["first_seen"] if prev and prev["first_seen"] else now
            db.execute("""insert into videos(id,source,upload_ts,author,author_name,caption,hashtags,music,
                duration,is_photo,location,nas_path,info_path,info_mtime,tr_mtime,transcript_tiktok,transcript_whisper,
                ocr,keywords,topics,transcript_speakers,n_speakers,first_seen,visual)
                values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                on conflict(id) do update set source=excluded.source,
                upload_ts=excluded.upload_ts,author=excluded.author,author_name=excluded.author_name,caption=excluded.caption,
                hashtags=excluded.hashtags,music=excluded.music,duration=excluded.duration,is_photo=excluded.is_photo,
                location=excluded.location,nas_path=excluded.nas_path,info_path=excluded.info_path,info_mtime=excluded.info_mtime,
                tr_mtime=excluded.tr_mtime,transcript_tiktok=excluded.transcript_tiktok,transcript_whisper=excluded.transcript_whisper,
                ocr=excluded.ocr,keywords=excluded.keywords,topics=excluded.topics,
                transcript_speakers=excluded.transcript_speakers,n_speakers=excluded.n_speakers,visual=excluded.visual""",
                       (vid, source, info["upload_ts"], info["author"], info["author_name"],
                        info["caption"], info["hashtags"], info["music"], info["duration"], info["is_photo"],
                        sc["location"] or info["location"], info["nas_path"], str(p), mtime, transcript_mtime(sdir, vid),
                        sc["tiktok"], sc["whisper"], " ".join(x for x in (sc["stickers"], sc["ocr"]) if x),
                        sc["keywords"], sc["topics"], sc["speakers"], sc["n_speakers"], first_seen, sc["visual"]))
            changed += 1
    # Chat membership, order and sender are re-applied to every row each run (cheap, and the chat index moves).
    db.execute("update videos set chat_pos=null, sender=null")
    for pos, (vid, is_me) in enumerate(cards):
        db.execute("update videos set chat_pos=?, source='dm', sender=? where id=?",
                   (pos, None if is_me is None else ("me" if is_me else CONTACT), vid))
    favs = load_favs()
    for vid in favs:
        db.execute("update videos set source='fav' where id=? and chat_pos is null", (vid,))
    # Rows whose files vanished from the NAS are dropped (the floor check catches a mass vanish).
    gone = [r[0] for r in db.execute("select id from videos") if r[0] not in seen_files]
    for vid in gone:
        db.execute("delete from videos where id=?", (vid,))
    assign_share_dates(db, dl, favs)
    rebuild_fts(db)
    meta_set(db, "updated", now)
    meta_set(db, "chat_complete", int(complete))
    # The floor is checked inside the transaction: a failing run (NAS unmounted, files vanished) is rolled
    # back, so the last good index stays queryable while the failure is still reported (exit 2).
    ok, msg = floor_check(db, cards)
    if ok:
        n_dm = db.execute("select count(*) from videos where source='dm'").fetchone()[0]
        meta_set(db, "high_water_dm", max(n_dm, int(meta_get(db, "high_water_dm", 0))))
        db.commit()
    else:
        db.rollback()
        msg += " — update rolled back, index unchanged"
    if not args.quiet or not ok:
        print(f"indexed {changed} files, dropped {len(gone)}; {msg}", file=sys.stderr if not ok else sys.stdout)
    return 0 if ok else 2


def load_favs():
    try:
        return json.loads(FAVS_INDEX.read_text()).get("items", {})
    except (OSError, ValueError):
        return {}


def assign_share_dates(db, dl, favs=None):
    favs = favs or {}
    rows = db.execute("select id, chat_pos, upload_ts, first_seen, source from videos order by chat_pos").fetchall()
    running = 0
    for r in rows:
        if r["chat_pos"] is None and (favs.get(r["id"]) or {}).get("fav_ts"):
            ts, prec = favs[r["id"]]["fav_ts"], "est"
        elif r["chat_pos"] is None:
            ts, prec = dl.get(r["id"]) or r["first_seen"], "download"
        else:
            running = max(running, r["upload_ts"] or 0)
            if r["chat_pos"] >= BACKFILL_INDEX_SIZE and r["id"] in dl:
                ts, prec = min(dl[r["id"]], r["first_seen"] or dl[r["id"]]), "poll"
            else:
                ts, prec = running or None, "est"
        db.execute("update videos set share_ts=?, share_precision=? where id=?", (ts, prec, r["id"]))


def rebuild_fts(db):
    db.execute("delete from fts")
    db.execute("""insert into fts(id,caption,hashtags,author,music,location,keywords,transcript,ocr,visual)
        select id, caption, hashtags, author || ' ' || author_name, music, location,
               trim(coalesce(keywords,'') || ' ' || coalesce(topics,'')),
               trim(coalesce(transcript_whisper,'') || ' ' || coalesce(transcript_tiktok,'')), coalesce(ocr,''),
               coalesce(visual,'')
        from videos""")


def floor_check(db, cards=None):
    if cards is None:
        _, cards = load_chat()
    n_dm = db.execute("select count(*) from videos where source='dm'").fetchone()[0]
    hw = int(meta_get(db, "high_water_dm", 0))
    have = {r[0] for r in db.execute("select id from videos where source='dm'")}
    missing = [vid for vid, _ in cards if vid not in have]
    problems = []
    if n_dm < FLOOR_MIN_DM:
        problems.append(f"DM rows {n_dm} < floor {FLOOR_MIN_DM}")
    if hw and n_dm < hw - FLOOR_SLACK:
        problems.append(f"DM rows {n_dm} shrank below high-water {hw} - {FLOOR_SLACK}")
    if len(missing) > MAX_MISSING:
        problems.append(f"{len(missing)} chat videos have no indexed file (allowance {MAX_MISSING})")
    msg = f"dm={n_dm} high_water={hw} chat={len(cards)} chat_missing={len(missing)}"
    if problems:
        return False, "FLOOR FAIL: " + "; ".join(problems) + f" ({msg})"
    return True, "floor ok (" + msg + ")"


# ---------------------------------------------------------------- enrichment (phase 2a)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
RETRY_ERROR_AFTER = 7 * 86400  # an unavailable post (deleted/private/blocked) is retried weekly, not per poll


def http_get(url, referer=None, timeout=30):
    import urllib.request
    h = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
    if referer:
        h["Referer"] = referer
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def page_item(vid):
    """Public post page (no login) -> itemStruct dict. Raises on an unavailable post."""
    html = http_get(f"https://www.tiktok.com/@_/video/{vid}")
    m = re.search(rb'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise RuntimeError("no rehydration data (challenge page?)")
    detail = json.loads(m.group(1))["__DEFAULT_SCOPE__"].get("webapp.video-detail") or {}
    if detail.get("statusCode") not in (0, None):
        raise RuntimeError(f"post unavailable: statusCode {detail.get('statusCode')} {detail.get('statusMsg', '')}")
    return detail["itemInfo"]["itemStruct"]


def page_record(it):
    """The fields worth keeping from an itemStruct (small; the raw page is not stored)."""
    poi = it.get("poi") or {}
    stickers = []
    for st in it.get("stickersOnItem") or []:
        stickers += [t for t in st.get("stickerText") or [] if t]
    subs = it.get("video", {}).get("subtitleInfos") or []
    return {
        "id": it.get("id"), "fetched": int(time.time()), "createTime": it.get("createTime"),
        "poi": {k: poi.get(k) for k in ("name", "address", "city", "category", "fatherPoiName", "id")} if poi else None,
        "locationCreated": it.get("locationCreated"), "textLanguage": it.get("textLanguage"),
        "suggestedWords": it.get("suggestedWords") or [], "labels": it.get("diversificationLabels") or [],
        "stickers": stickers,
        "subtitles": [{"lang": x.get("LanguageCodeName"), "format": x.get("Format"), "source": x.get("Source")} for x in subs],
    }, subs


def pick_subtitle(subs):
    """Prefer English; else the video's own-language ASR; else anything webvtt."""
    vtt = [x for x in subs if (x.get("Format") or "").lower() == "webvtt" and x.get("Url")]
    for pref in (lambda x: (x.get("LanguageCodeName") or "").startswith("eng"),
                 lambda x: x.get("Source") == "ASR", lambda x: True):
        for x in vtt:
            if pref(x):
                return x
    return None


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode())
    os.replace(tmp, path)


def enrich_one(tdir, vid):
    """Fetch one post page; write <id>.page.json (always) and <id>.tiktok.vtt (when TikTok has captions)."""
    try:
        it = page_item(vid)
    except Exception as e:  # noqa: BLE001 — recorded, retried weekly
        atomic_write(tdir / f"{vid}.page.json", json.dumps({"id": vid, "fetched": int(time.time()), "error": str(e)[:300]}))
        return "error", str(e)[:120]
    rec, subs = page_record(it)
    sub = pick_subtitle(subs)
    if sub:
        try:
            vtt = http_get(sub["Url"], referer="https://www.tiktok.com/")
            if vtt.lstrip().startswith(b"WEBVTT"):
                atomic_write(tdir / f"{vid}.tiktok.vtt", vtt)
                rec["subtitle_saved"] = sub.get("LanguageCodeName")
            else:
                rec["subtitle_error"] = "not webvtt"
        except Exception as e:  # noqa: BLE001
            rec["subtitle_error"] = str(e)[:200]
    atomic_write(tdir / f"{vid}.page.json", json.dumps(rec, ensure_ascii=False))
    return "ok", (rec.get("poi") or {}).get("name") or ""


def cmd_enrich(args):
    db = connect()
    now = time.time()
    todo = []
    rows = db.execute("select id, info_path, source from videos order by chat_pos is null, chat_pos desc").fetchall()
    for r in rows:
        tdir = Path(r["info_path"]).parent.parent / ".transcripts"
        pj = tdir / f"{r['id']}.page.json"
        if pj.exists():
            try:
                old = json.loads(pj.read_text())
            except ValueError:
                old = {"error": "corrupt", "fetched": 0}
            if "error" not in old or now - old.get("fetched", 0) < RETRY_ERROR_AFTER:
                continue
        todo.append((tdir, r["id"]))
    todo = todo[: args.max] if args.max else todo
    if not args.quiet:
        print(f"enrich: {len(todo)} to fetch (delay {args.delay}s)")
    ok = err = streak = 0
    for i, (tdir, vid) in enumerate(todo):
        if i:
            time.sleep(args.delay)
        status, note = enrich_one(tdir, vid)
        if status == "ok":
            ok, streak = ok + 1, 0
        else:
            err, streak = err + 1, streak + 1
        if not args.quiet:
            print(f"  {vid} {status} {note}", flush=True)
        if streak >= 5:  # blocked or offline: stop instead of hammering
            print(f"enrich: 5 errors in a row, stopping ({ok} ok, {err} errors)", file=sys.stderr)
            return 3
    if todo and not args.quiet:
        print(f"enrich: {ok} ok, {err} errors")
    return 0



# ---------------------------------------------------------------- whisper transcription (phase 2b)

WHISPER_IMAGE = os.environ.get("TIKTOK_RAG_WHISPER_IMAGE", "ghcr.io/ggml-org/whisper.cpp:main")
MODELS = Path(os.environ.get("TIKTOK_RAG_MODELS", HOME / "tiktok-rag/models"))
WHISPER_MODEL = os.environ.get("TIKTOK_RAG_WHISPER_MODEL", "ggml-large-v3-turbo-q5_0.bin")
VAD_MODEL = "ggml-silero-v5.1.2.bin"
WHISPER_CPUS = os.environ.get("TIKTOK_RAG_WHISPER_CPUS", "4")


def whisper_one(mp4, tdir, vid):
    """Transcribe one video in a CPU/memory-capped, low-share container; writes <id>.whisper.json + .txt.
    No speech (VAD found nothing) still writes an empty .txt so the video is not retried every run."""
    import subprocess
    tdir.mkdir(parents=True, exist_ok=True)
    tmp = f"{vid}.whisper.tmp"
    script = ("ffmpeg -loglevel error -y -i /in.mp4 -ar 16000 -ac 1 -f wav /tmp/a.wav && "
              f"whisper-cli -m /models/{WHISPER_MODEL} --vad -vm /models/{VAD_MODEL} -f /tmp/a.wav "
              f"-t {WHISPER_CPUS} -l auto -oj -of /out/{tmp} -np")
    cmd = ["docker", "run", "--rm", f"--cpus={WHISPER_CPUS}", "--memory=3g", "--cpu-shares=128",
           "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{mp4}:/in.mp4:ro", "-v", f"{MODELS}:/models:ro",
           "-v", f"{tdir}:/out", "--entrypoint", "sh", WHISPER_IMAGE, "-c", script]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    out = tdir / f"{tmp}.json"
    if r.returncode != 0 or not out.exists():
        out.unlink(missing_ok=True)
        err = ((r.stderr or r.stdout).strip().splitlines() or ["no output"])[-1][:300]
        atomic_write(tdir / f"{vid}.whisper.fail", err + "\n")  # skipped from now on; delete to retry
        return False, err
    d = json.loads(out.read_text(errors="replace"))
    segs = d.get("transcription") or []
    text = " ".join(x.get("text", "").strip() for x in segs).strip()
    lang = (d.get("result") or {}).get("language")
    os.replace(out, tdir / f"{vid}.whisper.json")
    atomic_write(tdir / f"{vid}.whisper.txt", text + ("\n" if text else ""))
    return True, f"{lang} {len(text)} chars"


ASR_IMAGE = os.environ.get("TIKTOK_RAG_ASR_IMAGE", "tiktok-rag-asr:3")
EMBED_IMAGE = os.environ.get("TIKTOK_RAG_EMBED_IMAGE", "tiktok-rag-embed:2")
# EmbeddingGemma-300m won the bake-off (eval/eval_retrieval.py) against nomic-embed and Qwen3-Embedding-0.6B; every
# reranker tried was worse AND took most of a minute per query.
EMBED_MODEL = os.environ.get("TIKTOK_RAG_EMBED_MODEL", "google/embeddinggemma-300m")
EMBED_MODEL_PATH = os.environ.get("TIKTOK_RAG_EMBED_MODEL_PATH", "/models/fastembed/flat/embeddinggemma-300m")
# Keyword ranking's weight in the fusion. Pure Gemma scored a little higher on the judged set, but the judged
# queries under-represent exact names (a creator's handle, a dish, a venue) where the keyword leg is the safety net.
KW_WEIGHT = float(os.environ.get("TIKTOK_RAG_KW_WEIGHT", "0.25"))
LOOKUP_MAX = int(os.environ.get("TIKTOK_RAG_LOOKUP_MAX", "5"))  # see the lookup rule in cmd_query


def embed_run(mode, payload, timeout):
    import subprocess
    (MODELS / "fastembed").mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "-i", "--cpus=4", "--memory=3g", "--cpu-shares=256",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "THREADS=4",
           "-e", f"EMBED_MODEL={EMBED_MODEL}", *(["-e", f"MODEL_PATH={EMBED_MODEL_PATH}"] if EMBED_MODEL_PATH else []),
           "-v", f"{MODELS / 'fastembed'}:/models/fastembed", "-v", f"{DB.parent}:/data",
           "-v", f"{Path(__file__).resolve().parent / 'asr'}:/asr:ro", EMBED_IMAGE, "python", "/asr/embed.py", mode]
    p = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        err = [l for l in p.stderr.strip().splitlines() if "it/s" not in l and "Fetching" not in l]
        raise RuntimeError(f"exit {p.returncode}{' (OOM-killed?)' if p.returncode == 137 else ''}: "
                           + (err[-1][:300] if err else ""))
    return json.loads(p.stdout.strip().splitlines()[-1])


def embed_query(text, k=200):
    if not (DB.parent / "vectors.npz").exists():
        return None
    try:
        return embed_run("query", {"q": text, "k": k}, timeout=120)
    except Exception as e:  # noqa: BLE001 — degrade to keyword search, never fail the query
        print(f"semantic leg failed: {e}", file=sys.stderr)
        return None


def doc_text(r):
    parts = [r["caption"], "#" + r["hashtags"].replace(" ", " #") if r["hashtags"] else "", r["location"],
             r["keywords"], r["topics"], r["visual"], r["ocr"], r["transcript_whisper"] or r["transcript_tiktok"]]
    return "\n".join(p for p in parts if p)


def cmd_embed(args):
    db = connect()
    docs = [{"id": r["id"], "text": doc_text(r)} for r in db.execute("select * from videos")]
    r = embed_run("index", {"docs": docs}, timeout=3 * 3600)
    if not args.quiet:
        print(f"embed: {r['embedded']} (re)embedded, {r['total']} vectors")
    return 0


def cmd_diarize(args):
    """Speaker turns for every video lacking <id>.turns.json, one container per series (models load once)."""
    import subprocess
    db = connect()
    rows = db.execute("select id, info_path, nas_path from videos order by chat_pos is null, chat_pos desc").fetchall()
    by_series = {}
    for r in rows:
        tdir = Path(r["info_path"]).parent.parent / ".transcripts"
        if not (tdir / f"{r['id']}.turns.json").exists() and os.path.exists(r["nas_path"]):
            by_series.setdefault(tdir, []).append({"id": r["id"], "path": r["nas_path"]})
    n = sum(len(v) for v in by_series.values())
    if not args.quiet:
        print(f"diarize: {n} pending", flush=True)
    left = args.max or n
    for tdir, jobs in by_series.items():
        jobs = jobs[:left]
        if not jobs:
            break
        left -= len(jobs)
        tdir.mkdir(parents=True, exist_ok=True)
        cmd = ["docker", "run", "--rm", "-i", f"--cpus={WHISPER_CPUS}", "--memory=4g", "--cpu-shares=128",
               "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{NAS}:{NAS}:ro", "-v", f"{MODELS}:/models:ro",
               "-v", f"{tdir}:/out", "-v", f"{Path(__file__).resolve().parent / 'asr'}:/asr:ro",
               "-e", f"THREADS={WHISPER_CPUS}", "-e", "MODE=diarize", ASR_IMAGE, "python", "/asr/asr.py"]
        p = subprocess.run(cmd, input=json.dumps(jobs), capture_output=True, text=True, timeout=6 * 3600)
        lines = [l for l in p.stdout.splitlines() if not l.startswith("/project")]
        if not args.quiet or p.returncode:
            print("\n".join(lines), flush=True)
        if p.returncode:
            print(f"diarize: container exit {p.returncode}: {p.stderr.strip()[-300:]}", file=sys.stderr)
            return 3
    return 0


def cmd_ocr(args):
    """RapidOCR (PaddleOCR ONNX) over photo-post slides in <series>/.photos/<id>/ -> <id>.ocr.txt, one line per
    slide. Benchmarked 2026-09-28: Tesseract returned noise on TikTok photo backgrounds; RapidOCR ~1.7 s/slide."""
    import subprocess
    db = connect()
    by_series = {}
    for r in db.execute("select id, info_path from videos where is_photo=1 order by chat_pos is null, chat_pos desc"):
        series = Path(r["info_path"]).parent.parent
        if not (series / ".transcripts" / f"{r['id']}.ocr.txt").exists():
            by_series.setdefault(series, []).append({"id": r["id"], "dir": str(series / ".photos" / r["id"])})
    for series, jobs in by_series.items():
        jobs = jobs[: args.max] if args.max else jobs
        tdir = series / ".transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        cmd = ["docker", "run", "--rm", "-i", f"--cpus={WHISPER_CPUS}", "--memory=3g", "--cpu-shares=128",
               "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-v", f"{NAS}:{NAS}:ro",
               "-v", f"{tdir}:/out", "-v", f"{Path(__file__).resolve().parent / 'asr'}:/asr:ro",
               ASR_IMAGE, "python", "/asr/ocr.py"]
        p = subprocess.run(cmd, input=json.dumps(jobs), capture_output=True, text=True, timeout=3 * 3600)
        if not args.quiet or p.returncode:
            print(p.stdout.strip(), flush=True)
        if p.returncode:
            print(f"ocr: container exit {p.returncode}: {p.stderr.strip()[-300:]}", file=sys.stderr)
            return 3
    return 0


def cmd_transcribe(args):
    db = connect()
    rows = db.execute("select id, info_path, nas_path from videos order by chat_pos is null, chat_pos desc").fetchall()
    todo = [(Path(r["info_path"]).parent.parent / ".transcripts", r["id"], r["nas_path"]) for r in rows
            if not (Path(r["info_path"]).parent.parent / ".transcripts" / f"{r['id']}.whisper.txt").exists()
            and not (Path(r["info_path"]).parent.parent / ".transcripts" / f"{r['id']}.whisper.fail").exists()
            and os.path.exists(r["nas_path"])]
    todo = todo[: args.max] if args.max else todo
    if not args.quiet:
        print(f"transcribe: {len(todo)} pending, model {WHISPER_MODEL}, {WHISPER_CPUS} cpus", flush=True)
    start, done, fails = time.time(), 0, 0
    for tdir, vid, mp4 in todo:
        if args.budget and time.time() - start > args.budget:
            break
        t0 = time.time()
        ok, note = whisper_one(mp4, tdir, vid)
        done += ok
        fails += not ok
        if not args.quiet or not ok:
            print(f"  {vid} {'ok' if ok else 'FAIL'} {note} ({time.time() - t0:.0f}s)", flush=True)
        if fails >= 3 and not done:
            print("transcribe: 3 failures and no success, stopping", file=sys.stderr)
            return 3
    if not args.quiet:
        print(f"transcribe: {done} done, {fails} failed, {time.time() - start:.0f}s")
    return 0


# ---------------------------------------------------------------- Gemini watches each video (the one cloud step)
# Design choice: Gemini may watch the videos; everything else stays local. Nothing else is sent: one
# 1-fps 480p re-encode + audio + the caption per video, over the stateless generateContent endpoint (the Interactions
# API stores requests 55 days by default). Paid-tier key, so Google does not train on it. Off: touch ~/tiktok-rag/GEMINI_OFF.

GEMINI_MODEL = os.environ.get("TIKTOK_RAG_GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_KEY_FILE = Path(os.environ.get("TIKTOK_RAG_GEMINI_KEY_FILE", HOME / ".config/tiktok-rag/gemini.env"))
GEMINI_OFF = HOME / "tiktok-rag/GEMINI_OFF"
GEMINI_PRICE = (0.30, 2.50)  # $/M input, output (thinking included), gemini-3.5-flash-lite standard, 2026-09-29
INLINE_MAX = 14 * 2**20  # generateContent caps a request at 20 MB and base64 adds a third
GEMINI_PROMPT = """You are indexing a TikTok video for a private search engine. Watch the video and listen to the audio.
The uploader's caption is: {caption}
Describe what someone would remember and search for to find this video again. Use English. Name a place, person,
brand or dish only if it is shown, said or captioned; otherwise describe it ("a ramen shop", "a woman cooking")."""
GEMINI_SCHEMA = {"type": "object", "required": ["summary", "tags"], "properties": {
    "summary": {"type": "string", "description": "2-3 sentences: what happens and what the video is about"},
    "genre": {"type": "string", "description": "e.g. recipe, restaurant review, comedy skit, travel vlog, news"},
    "places": {"type": "array", "items": {"type": "string"}, "description": "named venues, neighborhoods, cities"},
    "food": {"type": "array", "items": {"type": "string"}, "description": "dishes, drinks, ingredients shown"},
    "people": {"type": "array", "items": {"type": "string"}, "description": "who appears and what they do"},
    "things": {"type": "array", "items": {"type": "string"}, "description": "notable objects, animals, products"},
    "on_screen_text": {"type": "array", "items": {"type": "string"}, "description": "key overlays, signs, menus"},
    "tags": {"type": "array", "items": {"type": "string"}, "description": "10-20 search keywords"}}}


class GeminiHold(Exception):
    """Billing, auth or quota: every later call would fail the same way, so the whole step pauses."""


def gemini_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    try:
        for ln in GEMINI_KEY_FILE.read_text().splitlines():
            if ln.startswith("GEMINI_API_KEY="):
                return ln.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def gemini_hold():
    """-> reason while a pause from an earlier billing/quota error is in force, else None."""
    try:
        h = json.loads((DB.parent / ".gemini-hold").read_text())
    except (OSError, ValueError):
        return None
    return h.get("reason") if time.time() < h.get("until", 0) else None


def gemini_ready():
    return bool(gemini_key()) and not GEMINI_OFF.exists() and not gemini_hold()


def gemini_post(body, key):
    """POST generateContent -> response JSON. HTTP errors: GeminiHold for billing/auth/quota, RuntimeError otherwise."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(body).encode(), headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        msg = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
        if e.code in (401, 402, 403, 429):
            raise GeminiHold(msg) from None
        raise RuntimeError(msg) from None


def watch_clip(mp4, out):
    """1 fps (Gemini samples 1 fps anyway), 480p, mono audio: a minute of video is ~0.6 MB instead of ~10 MB.
    Too big for one inline request -> once more at 0.5 fps / 360p."""
    import subprocess
    out.parent.mkdir(parents=True, exist_ok=True)
    for vf, crf, ab in (("fps=1,scale=-2:480", 32, "32k"), ("fps=0.5,scale=-2:360", 36, "24k")):
        cmd = ["docker", "run", "--rm", "--cpus=2", "--memory=1g", "--cpu-shares=128",
               "--user", f"{os.getuid()}:{os.getgid()}", "-v", f"{mp4}:/in.mp4:ro", "-v", f"{out.parent}:/out",
               "--entrypoint", "ffmpeg", WHISPER_IMAGE, "-loglevel", "error", "-y", "-i", "/in.mp4", "-vf", vf,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-c:a", "aac", "-b:a", ab, "-ac", "1",
               f"/out/{out.name}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode:
            raise RuntimeError(f"ffmpeg exit {r.returncode}: {r.stderr.strip()[-200:]}")
        if out.stat().st_size <= INLINE_MAX:
            return out.read_bytes()
    raise RuntimeError(f"re-encode still {out.stat().st_size >> 20} MB")


def describe(d):
    """Gemini's JSON -> one searchable paragraph (the index's `visual` field)."""
    parts = [d.get("summary", "").strip(), d.get("genre", "").strip()]
    for k, label in (("places", "Places"), ("food", "Food"), ("people", "People"), ("things", "Things"),
                     ("on_screen_text", "On screen"), ("tags", "Tags")):
        items = [str(x).strip() for x in d.get(k) or [] if str(x).strip()]
        if items:
            parts.append(f"{label}: " + ", ".join(items))
    return "\n".join(p for p in parts if p)


def watch_one(mp4, tdir, vid, caption, key):
    """Gemini watches one video -> <id>.gemini.json {description, raw, usage} or {error} (retried weekly)."""
    import base64
    tmp = DB.parent / ".watch-tmp" / f"{vid}.mp4"
    try:
        clip = watch_clip(mp4, tmp)
    finally:
        tmp.unlink(missing_ok=True)
    body = {"contents": [{"parts": [
        {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(clip).decode()}},
        {"text": GEMINI_PROMPT.format(caption=" ".join((caption or "(none)").split())[:1000])}]}],
        "generationConfig": {"mediaResolution": "MEDIA_RESOLUTION_LOW", "responseMimeType": "application/json",
                             "responseJsonSchema": GEMINI_SCHEMA, "thinkingConfig": {"thinkingLevel": "LOW"}}}
    r = gemini_post(body, key)
    rec = {"id": vid, "model": r.get("modelVersion") or GEMINI_MODEL, "fetched": int(time.time()),
           "clip_bytes": len(clip), "usage": r.get("usageMetadata") or {}}
    cands = r.get("candidates") or []
    text = "".join(p.get("text", "") for p in ((cands[0].get("content") or {}).get("parts") or [])) if cands else ""
    try:
        raw = json.loads(text)
        rec.update(raw=raw, description=describe(raw))
    except ValueError:
        why = (r.get("promptFeedback") or {}).get("blockReason") or (cands[0].get("finishReason") if cands else "empty")
        rec["error"] = f"no JSON: {why}"
    atomic_write(tdir / f"{vid}.gemini.json", json.dumps(rec, ensure_ascii=False))
    return "error" not in rec, rec.get("error") or f"{len(rec['description'])} chars"


def watch_pending(db):
    """-> [(tdir, id, mp4, caption)] newest first: no sidecar yet, or an error sidecar older than a week."""
    now, out = time.time(), []
    for r in db.execute("select id, info_path, nas_path, caption from videos order by chat_pos is null, chat_pos desc"):
        tdir = Path(r["info_path"]).parent.parent / ".transcripts"
        p = tdir / f"{r['id']}.gemini.json"
        if p.exists():
            try:
                old = json.loads(p.read_text())
            except ValueError:
                old = {"error": "corrupt", "fetched": 0}
            if "error" not in old or now - old.get("fetched", 0) < RETRY_ERROR_AFTER:
                continue
        if os.path.exists(r["nas_path"]):
            out.append((tdir, r["id"], r["nas_path"], r["caption"]))
    return out


def cmd_watch(args):
    key = gemini_key()
    if not key or GEMINI_OFF.exists() or gemini_hold():
        if not args.quiet:
            print(f"watch: skipped ({'no key' if not key else 'GEMINI_OFF' if GEMINI_OFF.exists() else gemini_hold()})")
        return 0
    todo = watch_pending(connect())
    todo = todo[: args.max] if args.max else todo
    if not args.quiet:
        print(f"watch: {len(todo)} pending, model {GEMINI_MODEL}", flush=True)
    ok = err = streak = 0
    for tdir, vid, mp4, caption in todo:
        if OFF.exists() or GEMINI_OFF.exists():
            break
        t0 = time.time()
        try:
            good, note = watch_one(mp4, tdir, vid, caption, key)
        except GeminiHold as e:
            # 402 no credits / 401-403 key / 429 quota: pause the step (6 h; 30 min for quota) instead of retrying.
            wait = 1800 if "HTTP 429" in str(e) else 6 * 3600
            atomic_write(DB.parent / ".gemini-hold", json.dumps({"until": int(time.time()) + wait, "reason": str(e)[:300]}))
            print(f"watch: paused {wait // 60} min: {str(e)[:200]}", file=sys.stderr)
            return 3
        except Exception as e:  # noqa: BLE001 — transient (5xx, network, ffmpeg): no sidecar, retried next pass
            good, note = None, str(e)[:200]
        ok += bool(good)
        err += not good
        streak = 0 if good is not None else streak + 1
        if not args.quiet or good is None:
            print(f"  {vid} {'ok' if good else 'error' if good is False else 'FAIL'} {note} ({time.time() - t0:.0f}s)",
                  flush=True)
        if streak >= 3:
            print("watch: 3 transient failures in a row, stopping", file=sys.stderr)
            return 3
    if todo and not args.quiet:
        print(f"watch: {ok} ok, {err} errors")
    return 0



# ---------------------------------------------------------------- queries

def parse_since(s):
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    raise SystemExit(f"bad date {s!r}: use YYYY, YYYY-MM or YYYY-MM-DD")


# Concept groups: a query word in a group also matches its siblings. Stop-gap until phase-3 embeddings;
# keep it small and literal (multi-word entries are phrases).
# The weight scales a group's IDF: "place/spot/things to do" says what KIND of answer is wanted, not what
# the video is about, so it only breaks ties.
ALIASES = [
    (1.0, {"nyc", "new york", "newyork", "newyorkcity", "manhattan", "brooklyn", "queens", "bronx", "harlem",
     "soho", "williamsburg", "nycfood", "nyceats", "thingstodoinnyc"}),
    (1.0, {"eat", "eats", "food", "foodie", "restaurant", "restaurants", "dinner", "lunch", "brunch", "breakfast",
     "cafe", "bakery", "pizza", "bar", "dessert", "nycfood", "nyceats", "foodtok", "eating"}),
    (0.25, {"place", "places", "spot", "spots", "visit", "things to do", "thingstodo", "hidden gem", "hiddengem"}),
    (1.0, {"recipe", "recipes", "cook", "cooking", "easyrecipe", "dinnerideas"}),
]
# Expansion is one-way: a generic word ("food", "nyc") matches its specific siblings, but a specific word the user
# typed is matched literally. Otherwise "pizza nyc" means "any food + nyc" and every NYC food video ties with the
# actual pizza one (2026-09-29 review), and "brooklyn" returns Manhattan.
SPECIFIC = {"manhattan", "brooklyn", "queens", "bronx", "harlem", "soho", "williamsburg", "cafe", "bakery", "pizza",
            "bar", "dessert", "brunch", "breakfast", "lunch", "dinner"}


def fts_terms(text):
    """Natural language -> [(weight, FTS5 OR-expression)] per concept. Returns [] for raw FTS syntax."""
    if re.search(r'"|\bAND\b|\bOR\b|\bNEAR\b|\*', text):
        return []
    low = " ".join(re.findall(r"\w+", text.lower()))
    # A multi-word alias ("new york", "things to do") is ONE concept; split, "new" + "york" would outvote "pizza".
    phrases = [p for _, al in ALIASES for p in al if " " in p and re.search(rf"\b{p}\b", low)]
    for p in phrases:
        low = re.sub(rf"\b{p}\b", " ", low)
    words = phrases + ([t for t in low.split() if t not in STOP] or ([] if phrases else low.split()))
    groups, used = [], set()
    for w in words:
        if w in used:
            continue
        g, weight = {w}, 1.0
        for wt, al in ALIASES:
            if w in al and w not in SPECIFIC:
                g |= al
                weight = min(weight, wt)
        used |= g
        # music is left out: "original sound - The New York Times" is not a video about New York
        groups.append((weight, "{caption hashtags author location keywords transcript ocr visual} : (" + " OR ".join(f'"{t}"' for t in sorted(g)) + ")"))
    return groups


def filters(args, alias="v"):
    where, params = [], []
    if args.since:
        where.append(f"{alias}.share_ts >= ?"); params.append(parse_since(args.since))
    if args.until:
        where.append(f"{alias}.share_ts < ?"); params.append(parse_since(args.until))
    if args.sender and args.sender != "any":
        where.append(f"{alias}.sender = ?"); params.append(CONTACT if args.sender == "them" else args.sender)
    if args.source != "all":
        where.append(f"{alias}.source = ?"); params.append(args.source)
    return where, params


def fmt_date(ts, prec=None):
    if not ts:
        return "?"
    d = datetime.fromtimestamp(ts, timezone.utc)
    if prec == "est":
        return ">=" + d.strftime("%Y-%m-%d")
    return d.strftime("%Y-%m-%d %H:%MZ") if prec == "poll" else d.strftime("%Y-%m-%d")


def emit(rows, args, snippet_col=None):
    out = []
    for r in rows:
        rec = {k: r[k] for k in ("id", "source", "sender", "author", "caption", "hashtags", "music", "location",
                                  "topics", "nas_path", "share_precision", "is_photo")}
        rec["shared"] = fmt_date(r["share_ts"], r["share_precision"])
        rec["uploaded"] = fmt_date(r["upload_ts"])
        rec["url"] = f"https://www.tiktok.com/@{r['author'] or '_'}/video/{r['id']}"
        rec["snippet"] = " ".join((r[snippet_col] if snippet_col and r[snippet_col] else (r["caption"] or "")[:200]).split())
        rec["seen"] = (r["visual"] or "").split("\n")[0]  # Gemini's one-line summary of what the video shows
        rec["has_transcript"] = bool(r["transcript_whisper"] or r["transcript_tiktok"])
        # "meaning": only the semantic leg found it (no query word in the video) — a guess, so say so
        rec["match"] = "meaning" if snippet_col and not r[snippet_col] else "words"
        out.append(rec)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    if not out:
        print("(no results)")
    for i, rec in enumerate(out, 1):
        who = rec["sender"] or {"fav": "my favorites"}.get(rec["source"], rec["source"])
        print(f"{i:>2}. [{rec['shared']}] from {who} · @{rec['author']} · uploaded {rec['uploaded']}"
              f"{' · photo post' if rec['is_photo'] else ''}{' · ≈ related by meaning only' if rec['match'] == 'meaning' else ''}")
        if rec["location"]:
            print(f"    📍 {rec['location']}")
        cap = " ".join((rec["caption"] or "").split())
        print(f"    {cap[:220]}{'…' if len(cap) > 220 else ''}")
        plain = re.sub(r"[\[\]…]", "", rec["snippet"]).strip()
        if rec["seen"]:
            print(f"    seen: {rec['seen'][:220]}{'…' if len(rec['seen']) > 220 else ''}")
        if snippet_col and plain and plain[:40] not in cap and plain[:40] not in rec["seen"]:
            print(f"    match: {rec['snippet']}")
        print(f"    {rec['nas_path']}")
        print(f"    {rec['url']}")


def cmd_query(args):
    """Rank by how many query concepts a video matches, then by bm25 relevance (or by date with --newest)."""
    db = connect()
    where, params = filters(args)
    groups = fts_terms(args.text) or [(1.0, args.text)]
    whole = " OR ".join(f"({g})" for _, g in groups)
    sql = f"""select v.*, snippet(fts, -1, '[', ']', '…', 24) as snip, bm25(fts, 1.0, 2.0, 1.5, 0.5, 2.0, 1.2, 0.8, 0.8, 0.6) as score
              from fts join videos v on v.id = fts.id where fts match ? {''.join(' and ' + w for w in where)}"""
    try:
        rows = db.execute(sql, [whole, *params]).fetchall()
        # Coverage score: each matched concept adds its IDF, so a rare concept ("nyc") outweighs common
        # ones ("eat", "place") — a video about NYC food beats a non-NYC video that mentions both.
        n_docs = max(1, db.execute("select count(*) from fts").fetchone()[0])
        hits, n_hit = {}, {}
        for weight, g in groups:
            ids = [vid for (vid,) in db.execute("select id from fts where fts match ?", (g,))]
            idf = weight * math.log(1 + n_docs / max(1, len(ids)))
            for vid in ids:
                hits[vid] = hits.get(vid, 0) + idf
                n_hit[vid] = n_hit.get(vid, 0) + 1
    except sqlite3.OperationalError as e:
        raise SystemExit(f"query error: {e} (fts query was {whole!r})")
    if args.newest:
        rows.sort(key=lambda r: (-round(hits.get(r["id"], 0), 1), -(r["share_ts"] or 0)))
    else:
        rows.sort(key=lambda r: (-round(hits.get(r["id"], 0), 1), r["score"]))
    if args.all_terms:
        rows = [r for r in rows if n_hit.get(r["id"], 0) == len(groups)]
    if args.mode == "keyword" or args.all_terms:
        emit(rows[: args.limit], args, "snip")
        return 0
    # Semantic leg: cosine over local embeddings, fused with the keyword ranking by weighted reciprocal rank (k=60,
    # keyword x KW_WEIGHT),
    # so a video that says "Katz's on the Lower East Side" answers "nyc places to eat" with no shared word.
    sem = embed_query(args.text, k=200)
    if sem is None:
        print("(semantic search unavailable — keyword results only)", file=sys.stderr)
        emit(rows[: args.limit], args, "snip")
        return 0
    allowed = {r[0] for r in db.execute(
        f"select v.id from videos v {'where ' + ' and '.join(where) if where else ''}", params)}
    kw_rank = [r["id"] for r in rows]
    sem_rank = [i for i, _ in sem if i in allowed]
    if args.mode == "semantic":
        fused = sem_rank
    else:
        score = {}
        for ranking, w in ((kw_rank, KW_WEIGHT), (sem_rank, 1.0)):
            for pos, vid in enumerate(ranking):
                score[vid] = score.get(vid, 0) + w / (60 + pos)
        fused = sorted(score, key=lambda v: -score[v])
        # Lookup rule: a one-concept query that matches only a handful of videos (a handle, a dish, a venue) names
        # something; those exact hits go first. At KW_WEIGHT a keyword-only hit would otherwise rank ~180th behind the
        # semantic list. Multi-concept queries are descriptions ("healthy breakfast ideas") where few videos contain
        # every word by chance; firing there cost nDCG on the judged set, so they are left to fusion.
        exact = [r["id"] for r in rows if n_hit.get(r["id"], 0) == len(groups)]
        if len(groups) == 1 and 0 < len(exact) <= LOOKUP_MAX:
            fused = exact + [v for v in fused if v not in exact]
    by_id = {r["id"]: r for r in rows}
    out = []
    for vid in fused[: args.limit]:
        out.append(by_id.get(vid) or db.execute("select v.*, '' as snip from videos v where id=?", (vid,)).fetchone())
    if args.newest:
        out.sort(key=lambda r: -(r["share_ts"] or 0))
    emit(out, args, "snip")
    return 0


def cmd_recent(args):
    db = connect()
    where, params = filters(args)
    # Chat position is the true order for DMs; fall back to share time for Saved.
    sql = f"""select v.* from videos v {'where ' + ' and '.join(where) if where else ''}
              order by coalesce(v.chat_pos, -1) desc, v.share_ts desc limit ?"""
    if args.source in ("saved", "fav"):
        sql = sql.replace("coalesce(v.chat_pos, -1) desc, ", "")
    emit(db.execute(sql, [*params, args.n]).fetchall(), args)
    return 0


def cmd_show(args):
    db = connect()
    r = db.execute("select * from videos where id=?", (args.id,)).fetchone()
    if not r:
        print(f"no such id {args.id}", file=sys.stderr)
        return 1
    d = dict(r)
    d["shared"] = fmt_date(d["share_ts"], d["share_precision"])
    d["uploaded"] = fmt_date(d["upload_ts"])
    print(json.dumps(d, ensure_ascii=False, indent=1))
    return 0


def cmd_stats(args):
    db = connect()
    ok, msg = floor_check(db)
    q = lambda s: db.execute(s).fetchone()[0]
    upd = meta_get(db, "updated")
    print(json.dumps({
        "db": str(DB), "updated": fmt_date(int(upd), "poll") if upd else None,
        "total": q("select count(*) from videos"),
        "dm": q("select count(*) from videos where source='dm'"),
        "saved": q("select count(*) from videos where source='saved'"),
        "favorites": q("select count(*) from videos where source='fav'"),
        f"from_{CONTACT}": db.execute("select count(*) from videos where sender=?", (CONTACT,)).fetchone()[0],
        "from_me": q("select count(*) from videos where sender='me'"),
        "photo_posts": q("select count(*) from videos where is_photo=1"),
        "with_tiktok_transcript": q("select count(*) from videos where transcript_tiktok<>''"),
        "with_whisper_transcript": q("select count(*) from videos where transcript_whisper<>''"),
        "with_speaker_turns": q("select count(*) from videos where n_speakers is not null"),
        "multi_speaker": q("select count(*) from videos where n_speakers >= 2"),
        "with_ocr_or_stickers": q("select count(*) from videos where coalesce(ocr,'')<>''"),
        "watched_by_gemini": q("select count(*) from videos where coalesce(visual,'')<>''"),
        "gemini": gemini_stats(db),
        "with_location": q("select count(*) from videos where coalesce(location,'')<>''"),
        "with_keywords": q("select count(*) from videos where coalesce(keywords,'')<>''"),
        "share_precision": dict(db.execute("select share_precision, count(*) from videos group by 1").fetchall()),
        "floor": msg,
    }, indent=1))
    return 0 if ok else 2


def gemini_stats(db):
    """Tokens and estimated spend so far, summed from the sidecars' usageMetadata."""
    n = tin = tout = 0
    for r in db.execute("select id, info_path from videos"):
        try:
            u = json.loads((Path(r["info_path"]).parent.parent / ".transcripts" / f"{r['id']}.gemini.json").read_text())
        except (OSError, ValueError):
            continue
        u = u.get("usage") or {}
        n += 1
        tin += u.get("promptTokenCount", 0)
        tout += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return {"sidecars": n, "input_tokens": tin, "output_tokens": tout,
            "est_usd": round(tin / 1e6 * GEMINI_PRICE[0] + tout / 1e6 * GEMINI_PRICE[1], 2),
            "ready": gemini_ready(), "hold": gemini_hold(), "off": GEMINI_OFF.exists()}


def cmd_check(args):
    ok, msg = floor_check(connect())
    print(msg, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


OFF = HOME / "tiktok-rag/OFF"


def pending_counts(db):
    """(enrich, diarize, transcribe, ocr, watch) still to do, from sidecar existence (a few stats per video).
    watch counts only while Gemini can run (key, no GEMINI_OFF, no billing pause), so a pause cannot spin the loop."""
    e = d = t = o = 0
    for r in db.execute("select id, info_path, nas_path, is_photo from videos"):
        td = Path(r["info_path"]).parent.parent / ".transcripts"
        o += bool(r["is_photo"]) and not (td / f"{r['id']}.ocr.txt").exists()
        pj = td / f"{r['id']}.page.json"
        if not pj.exists():
            e += 1
        if not os.path.exists(r["nas_path"]):
            continue
        d += not (td / f"{r['id']}.turns.json").exists()
        t += not ((td / f"{r['id']}.whisper.txt").exists() or (td / f"{r['id']}.whisper.fail").exists())
    g = len(watch_pending(db)) if gemini_ready() else 0
    return e, d, t, o, g


def cmd_pipeline(args):
    """Drain every enrichment step, newest videos first, so a fresh DM is fully processed within one Whisper
    video's time: each pass = enrich <=20, diarize <=20, transcribe 1, Gemini-watch <=10, re-index. Start it detached
    after each clean poll (single instance); stops when nothing is pending or when ~/tiktok-rag/OFF exists."""
    ns = argparse.Namespace
    passes = 0
    while not OFF.exists():
        db = connect()
        e, d, t, o, g = pending_counts(db)
        db.close()
        if not (e or d or t or o or g):
            break
        if o:
            cmd_ocr(ns(max=10, quiet=True))
        if e:
            cmd_enrich(ns(max=20, delay=args.delay, quiet=True))
        if d:
            cmd_diarize(ns(max=20, quiet=True))
        if t:
            cmd_transcribe(ns(max=1, budget=0, quiet=True))
        if g:
            lock = single_instance("watch")  # a hand-started `watch` backlog run holds it: leave that run be
            if lock:
                cmd_watch(ns(max=10, quiet=True))
                lock.close()
            elif not (e or d or t or o):
                # Only Gemini work is left and that run owns it: stop instead of re-indexing in a loop until it ends
                # (that once spun for thousands of idle passes). The next poll restarts the pipeline for anything left.
                update(ns(quiet=True))
                break
        update(ns(quiet=True))
        passes += 1
        if passes % 10 == 0:
            try:
                cmd_embed(ns(quiet=True))
            except Exception as ex:  # noqa: BLE001
                print(f"embed failed: {ex}", file=sys.stderr)
        if not args.quiet:
            print(f"{datetime.now(timezone.utc):%FT%TZ} pass {passes}: pending enrich={e} diarize={d} whisper={t} ocr={o} watch={g}",
                  flush=True)
    if passes:
        try:
            cmd_embed(ns(quiet=True))
        except Exception as ex:  # noqa: BLE001
            print(f"embed failed: {ex}", file=sys.stderr)
        print(f"{datetime.now(timezone.utc):%FT%TZ} pipeline done after {passes} passes"
              f"{' (OFF switch)' if OFF.exists() else ''}", flush=True)
    return 0


def single_instance(name):
    """Non-blocking lock so a per-poll run never overlaps the background backlog run of the same step."""
    import fcntl
    f = open(DB.parent / f".{name}.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tiktok-rag", description="Search the TikTok DM archive (local index).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--since", help="share date >= YYYY[-MM[-DD]]")
        p.add_argument("--until", help="share date < YYYY[-MM[-DD]]")
        p.add_argument("--from", dest="sender", choices=list(dict.fromkeys([CONTACT, "them", "me", "any"])), default="any")
        p.add_argument("--source", choices=["dm", "saved", "fav", "all"], default="dm")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("query", help="search captions, hashtags, author, music, transcripts, OCR and Gemini's video descriptions")
    p.add_argument("text")
    p.add_argument("-n", "--limit", type=int, default=15)
    p.add_argument("--newest", action="store_true", help="newest first among the top matches")
    p.add_argument("--all", dest="all_terms", action="store_true", help="only videos matching every query concept (keyword)")
    p.add_argument("--mode", choices=["hybrid", "keyword", "semantic"], default="hybrid",
                   help="hybrid (default) fuses keyword and local-embedding rankings")
    common(p)
    p.set_defaults(func=cmd_query)
    p = sub.add_parser("recent", help="the N most recently shared videos (chat order)")
    p.add_argument("n", type=int, nargs="?", default=10)
    common(p)
    p.set_defaults(func=cmd_recent)
    p = sub.add_parser("show", help="every indexed field of one video")
    p.add_argument("id")
    p.set_defaults(func=cmd_show)
    p = sub.add_parser("update", help="incremental re-index from the NAS; exits 2 on floor failure")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=update)
    p = sub.add_parser("enrich", help="fetch public post pages: place tag, keywords, stickers, TikTok subtitles")
    p.add_argument("--max", type=int, default=0, help="at most N videos this run (0 = all pending)")
    p.add_argument("--delay", type=float, default=5.0, help="seconds between videos")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_enrich)
    p = sub.add_parser("transcribe", help="Whisper (whisper.cpp, local, CPU-capped) transcripts, newest first")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--budget", type=float, default=0, help="stop starting new videos after S seconds")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_transcribe)
    p = sub.add_parser("diarize", help="speaker turns (pyannote + TitaNet via sherpa-onnx, local, CPU-capped)")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_diarize)
    p = sub.add_parser("pipeline", help="drain enrich + diarize + whisper, newest first (start it detached after each poll)")
    p.add_argument("--delay", type=float, default=6.0, help="seconds between post-page fetches")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_pipeline)
    p = sub.add_parser("ocr", help="RapidOCR over photo-post slides (local, CPU-capped)")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_ocr)
    p = sub.add_parser("watch", help="Gemini watches each video once (the one cloud step); off: touch ~/tiktok-rag/GEMINI_OFF")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_watch)
    p = sub.add_parser("embed", help="(re)embed changed videos with the local embedding model")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=cmd_embed)
    sub.add_parser("stats", help="counts, coverage and the floor verdict").set_defaults(func=cmd_stats)
    sub.add_parser("check", help="floor assertion only; exit 2 if the index is empty or shrunk").set_defaults(func=cmd_check)
    args = ap.parse_args(argv)
    if args.cmd in ("transcribe", "diarize", "enrich", "pipeline", "embed", "ocr", "watch"):
        DB.parent.mkdir(parents=True, exist_ok=True)
        lock = single_instance(args.cmd)
        if lock is None:
            if not getattr(args, "quiet", False):
                print(f"{args.cmd}: another run holds the lock, skipping")
            return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
