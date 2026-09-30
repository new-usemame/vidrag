#!/usr/bin/env python3
"""vidrag: search downloaded videos by what they say and show.

Indexes yt-dlp downloads on disk (<Series>/Season */<name>.info.json with the video beside it, the layout MeTube,
ytdl-sub and Jellyfin share) into SQLite FTS5 plus local embeddings, and answers natural-language queries from a CLI
or a small HTTP API. Runs on the media server, stdlib only; the heavy steps run in CPU-capped containers.

Collections, steps and paths come from ~/.config/vidrag/config.toml ($VIDRAG_CONFIG; see config.example.toml).
A collection is a set of series folders plus the optional steps it gets, all local unless noted:
  enrich (the platform's public page; TikTok: place tag, keywords, stickers, captions) · ocr (photo-post slides)
  · diarize (speaker turns) · transcribe (Whisper) · watch (the one cloud step: Gemini describes the video).
Step outputs are sidecars beside the videos in <series>/.transcripts/<id>.*. The index is derived from them and the
.info.json files, so deleting it and running `update` rebuilds it. `update` ends with a floor check per collection,
so a shrunken or empty index exits non-zero instead of passing silently.
"""
import argparse
import importlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import tomllib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent


def expand(p):
    return Path(os.path.expanduser(str(p)))


def load_config():
    path = expand(os.environ.get("VIDRAG_CONFIG") or HOME / ".config/vidrag/config.toml")
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"vidrag: bad config {path}: {e}")


CFG = load_config()
STATE = expand(CFG.get("home", HOME / "vidrag"))  # off-switches live here
DATA = expand(CFG.get("data", STATE / "data"))  # index.db, vectors.npz, locks, logs
DB = DATA / "index.db"
MODELS = expand(CFG.get("models", STATE / "models"))
OFF = STATE / "OFF"  # stops `pipeline` and every step it runs
GEMINI_OFF = STATE / "GEMINI_OFF"  # stops only the cloud step

HASHTAG = re.compile(r"#([^\s#.,!?;:()\[\]{}\"'’]+)")
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
SIDECAR_SUFFIXES = (".tiktok.vtt", ".whisper.txt", ".whisper.json", ".turns.json", ".ocr.txt", ".page.json",
                    ".gemini.json")
STEPS = ("enrich", "ocr", "diarize", "transcribe", "watch")


class Collection:
    """One [collections.<name>] table: where its series folders are and which steps its videos get."""

    def __init__(self, name, c):
        self.name = name
        self.paths = [expand(p) for p in c.get("paths", [])]
        self.exclude = set(c.get("exclude", []))
        self.kind = c.get("kind", "")  # a word for prompts and help text: "TikTok", "YouTube"
        self.steps = [s for s in c.get("steps", []) if s in STEPS]
        self.floor_min = int(c.get("floor_min", 1))
        self.floor_slack = int(c.get("floor_slack", 3))
        self.update_every = int(c.get("update_every", 0))  # seconds; `pipeline` re-scans when the index is older
        self.titles = bool(c.get("titles", True))
        self.plugin = None
        if c.get("plugin"):
            mod = importlib.import_module(f"vidrag_{c['plugin']}")
            self.plugin = mod.Plugin(sys.modules[__name__], self, c.get(c["plugin"], {}))

    def series_dirs(self):
        """Each path is a series folder (it has Season */) or a folder of series folders."""
        out = []
        for p in self.paths:
            try:
                if next(p.glob("Season *"), None) or next(p.glob("*.info.json"), None):
                    out.append(p)
                else:
                    out += sorted(d for d in p.iterdir()
                                  if d.is_dir() and not d.name.startswith(".") and d.name not in self.exclude)
            except OSError:
                pass
        return out


COLLS = {name: Collection(name, c) for name, c in (CFG.get("collections") or {}).items()}
DEFAULT_SCOPE = [c for c in CFG.get("default_collections", []) if c in COLLS] or list(COLLS)
CONTACTS = {c.plugin.contact for c in COLLS.values() if getattr(c.plugin, "contact", None)}
STOP = set("""a an and are as at be but by for from has have i in is it its of on or so that the this to was
were what when where which who will with you your me my she her he his they them their we our us
send sent sends sending tiktok tiktoks youtube video videos things thing stuff some any all""".split()) | CONTACTS

SCHEMA = """
create table if not exists videos(
  id text primary key, collection text, series text, source text, sender text, chat_pos integer,
  date_ts integer, date_precision text, upload_ts integer,
  title text, author text, author_name text, caption text, hashtags text, music text, duration integer,
  is_photo integer, location text, path text, url text, info_path text, info_mtime real, tr_mtime real,
  transcript_subs text, transcript_whisper text, ocr text, keywords text, topics text,
  transcript_speakers text, n_speakers integer, first_seen integer, visual text);
create index if not exists videos_collection on videos(collection);
create virtual table if not exists fts using fts5(
  caption, hashtags, author, music, location, keywords, transcript, ocr, visual, title,
  tokenize = "porter unicode61 remove_diacritics 2");
create table if not exists meta(k text primary key, v text);
create table if not exists first_seen(id text primary key, ts integer);
"""
SCHEMA_VERSION = "5"  # fts rowid = videos.rowid


def connect(readonly=False):
    """The DB is derived from files on disk, so a schema change just rebuilds it from scratch (keeping the floor's
    high-water marks)."""
    if readonly:
        db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=15, check_same_thread=False)
        db.row_factory = sqlite3.Row
        return db
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        old, keep, seen = None, [], []
        try:
            with closing(sqlite3.connect(DB, timeout=30)) as c:
                old = c.execute("select v from meta where k='schema'").fetchone()
                keep = c.execute("select k, v from meta where k like 'high_water%'").fetchall()
                seen = c.execute("select id, first_seen from videos where first_seen is not null").fetchall()
                seen += c.execute("select id, ts from first_seen").fetchall()
        except sqlite3.OperationalError as e:
            if "locked" in str(e) or "busy" in str(e):
                raise  # another writer: never mistake a busy DB for a stale one
        except sqlite3.DatabaseError:
            pass  # not a database / no tables yet: rebuilt below
        if not old or old[0] != SCHEMA_VERSION:
            DB.unlink()
            db = sqlite3.connect(DB)
            db.executescript(SCHEMA)
            db.execute("insert into meta values('schema', ?)", (SCHEMA_VERSION,))
            db.executemany("insert into meta values(?, ?)", keep)
            # when each video first appeared survives the rebuild (share dates fall back on it)
            db.executemany("insert into first_seen values(?, ?) on conflict(id) do update set ts=min(ts, excluded.ts)",
                           seen)
            db.commit()
            db.close()
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA + f"insert or ignore into meta values('schema', '{SCHEMA_VERSION}');")
    return db


def meta_get(db, k, default=None):
    r = db.execute("select v from meta where k=?", (k,)).fetchone()
    return r[0] if r else default


def meta_set(db, k, v):
    db.execute("insert into meta(k,v) values(?,?) on conflict(k) do update set v=excluded.v", (k, str(v)))


def qmarks(xs):
    return ",".join("?" * len(xs))


def series_of(info_path):
    """<series>/Season X/<name>.info.json -> <series>; a flat <series>/<name>.info.json -> <series>."""
    p = Path(info_path).parent
    return p.parent if p.name.startswith("Season ") else p


def tdir_of(info_path):
    return series_of(info_path) / ".transcripts"


# ---------------------------------------------------------------- index

def sidecar_mtimes(series_dir):
    """id -> newest sidecar mtime, from one directory listing (a stat per video would be thousands over NFS)."""
    out = {}
    try:
        with os.scandir(series_dir / ".transcripts") as it:
            for e in it:
                vid, dot, rest = e.name.partition(".")
                if dot and "." + rest in SIDECAR_SUFFIXES:
                    out[vid] = max(out.get(vid, 0.0), e.stat().st_mtime)
    except OSError:
        pass
    return out


def read_sidecars(tdir, vid):
    """Enrichment sidecars in <series>/.transcripts/ -> dict of index fields (missing files -> empty)."""
    out = {"subs": "", "whisper": "", "ocr": "", "location": "", "keywords": "", "topics": "", "stickers": "",
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
        out["subs"] = vtt_to_text(p.read_text(errors="replace"))
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
    """WebVTT or SRT -> plain text, cue timing and repeated lines dropped."""
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


def pick_subs(names):
    """Subtitle files beside a video -> the one to index: English first, then whatever there is."""
    for pref in (lambda n: ".en." in n or ".en-" in n, lambda n: True):
        for n in names:
            if pref(n):
                return n
    return None


def parse_info(path, siblings, coll):
    """<name>.info.json (+ the names of the files beside it) -> index fields."""
    d = json.loads(path.read_text())
    if d.get("_type") in ("playlist", "multi_video", "channel"):
        return None
    desc = d.get("description") or d.get("title") or ""
    tags = [t.lower() for t in HASHTAG.findall(desc)]
    for t in d.get("tags") or []:
        if t and t.lower() not in tags:
            tags.append(t.lower())
    base = path.name[: -len(".info.json")]
    video = next((base + e for e in VIDEO_EXTS if base + e in siblings), base + ".mp4")
    upload_ts = d.get("timestamp")
    if not upload_ts and d.get("upload_date"):
        upload_ts = int(datetime.strptime(d["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp())
    title = (d.get("title") or "").strip() if coll.titles else ""
    subs = pick_subs(sorted(n for n in siblings if n.startswith(base + ".") and n.endswith((".vtt", ".srt"))))
    info = {
        "id": str(d.get("id")), "title": title if title and title != desc.strip() else "",
        "author": d.get("uploader") or d.get("uploader_id") or "", "author_name": d.get("channel") or "",
        "caption": desc.strip(), "hashtags": " ".join(tags),
        "music": " - ".join(x for x in (d.get("track"), d.get("artist")) if x), "duration": d.get("duration"),
        "upload_ts": upload_ts, "path": str(path.parent / video), "url": d.get("webpage_url") or "",
        "location": ", ".join(x for x in (d.get("location"),) if x),
        "chapters": " · ".join(c["title"] for c in d.get("chapters") or [] if c.get("title")),
        "categories": ", ".join(d.get("categories") or []),
        "subs": vtt_to_text((path.parent / subs).read_text(errors="replace")) if subs else "",
    }
    if coll.plugin and hasattr(coll.plugin, "parse"):
        coll.plugin.parse(info, d)
    return info


def scan(coll):
    """-> [(series_dir, info_path, siblings)] for every .info.json in the collection, in path order."""
    out = []
    for sdir in coll.series_dirs():
        for season in [sdir, *sorted(sdir.glob("Season *"))]:
            try:
                names = set(os.listdir(season))
            except OSError:
                continue
            for n in sorted(names):
                if n.endswith(".info.json"):
                    out.append((sdir, season / n, names))
    return out


def update(args):
    """Incremental re-index of the named collections (all by default); one transaction per collection."""
    db = connect()
    now = int(time.time())
    names = getattr(args, "collection", None) or list(COLLS)
    if getattr(args, "prune", False):  # rows of collections no longer in the config (only when asked)
        for n in [n for (n,) in db.execute("select distinct collection from videos") if n not in COLLS]:
            db.execute("delete from fts where rowid in (select rowid from videos where collection=?)", (n,))
            db.execute("delete from videos where collection=?", (n,))
            print(f"pruned collection {n}")
        db.commit()
    rc, msgs = 0, []
    for name in names:
        lock = wait_lock(f"update-{name}", getattr(args, "wait", 240))
        if lock is None:  # the other run is doing this very work; its result stands
            ok, msg = True, f"skipped: another update of {name} is still running"
        else:
            with lock:
                ok, msg = update_collection(db, COLLS[name], now)
        msgs.append(msg if len(names) == 1 else f"{name}: {msg}")
        rc = rc or (0 if ok else 2)
    if not rc:
        meta_set(db, "updated", now)
        db.commit()
    if not args.quiet or rc:
        print("\n".join(msgs), file=sys.stderr if rc else sys.stdout)
    return rc


def subs_mtimes(season, names):
    """base name -> newest mtime of the subtitle files beside that video (<base>.<lang>.vtt / <base>.srt)."""
    out = {}
    for n in names:
        if n.endswith((".vtt", ".srt")):
            stem = n[:-4]
            try:
                m = (season / n).stat().st_mtime
            except OSError:
                continue
            for base in {stem, stem.rsplit(".", 1)[0]}:
                out[base] = max(out.get(base, 0.0), m)
    return out


def update_collection(db, coll, now):
    """Scan and parse with no write lock held (a big tree over NFS takes minutes), then apply the changes in one
    short transaction, so the chat's per-poll update never waits behind a large collection's scan."""
    name = coll.name
    known = {r["info_path"]: r for r in db.execute(
        "select id, info_path, info_mtime, tr_mtime, first_seen from videos where collection=?", (name,))}
    owner = dict(db.execute("select id, collection from videos where collection<>?", (name,)).fetchall())
    first = dict(db.execute("select id, ts from first_seen").fetchall())
    seen, rows = set(), []
    tr_cache, photos, subs_cache = {}, {}, {}
    for sdir, p, siblings in scan(coll):
        if sdir not in tr_cache:
            tr_cache[sdir] = sidecar_mtimes(sdir)
            try:
                photos[sdir] = set(os.listdir(sdir / ".photos"))
            except OSError:
                photos[sdir] = set()
        if p.parent not in subs_cache:
            subs_cache[p.parent] = subs_mtimes(p.parent, siblings)
        try:
            mtime = p.stat().st_mtime
            prev = known.get(str(p))
            base = p.name[: -len(".info.json")]
            tr_m = max(tr_cache[sdir].get(prev["id"] if prev else "", 0.0), subs_cache[p.parent].get(base, 0.0))
            if prev and prev["info_mtime"] == mtime and prev["tr_mtime"] == tr_m:
                seen.add(prev["id"])  # unchanged: skip the read + JSON parse
                continue
            info = parse_info(p, siblings, coll)
        except (OSError, ValueError) as e:
            print(f"warn: unreadable {p}: {e}", file=sys.stderr)
            continue
        if info is None:  # a playlist or channel description, not a video
            continue
        vid = info["id"]
        if vid in seen or vid in owner:  # same id twice: the first path (or collection) wins
            continue
        seen.add(vid)
        sc = read_sidecars(sdir / ".transcripts", vid)
        tr_m = max(tr_cache[sdir].get(vid, 0.0), subs_cache[p.parent].get(base, 0.0))
        first_seen = (prev["first_seen"] if prev and prev["first_seen"] else None) or first.get(vid) or now
        source = coll.plugin.source_for(sdir) if coll.plugin and hasattr(coll.plugin, "source_for") else None
        rows.append((vid, name, sdir.name, source, info["title"], info["upload_ts"], info["upload_ts"], "upload",
                     info["author"], info["author_name"], info["caption"], info["hashtags"], info["music"],
                     info["duration"], 1 if vid in photos[sdir] else 0, sc["location"] or info["location"],
                     info["path"], info["url"], str(p), mtime, tr_m,
                     sc["subs"] or info["subs"], sc["whisper"], " ".join(x for x in (sc["stickers"], sc["ocr"]) if x),
                     sc["keywords"] or info["chapters"], sc["topics"] or info["categories"], sc["speakers"],
                     sc["n_speakers"], first_seen, sc["visual"]))
    changed = [r[0] for r in rows]
    for row in rows:
        db.execute("""insert into videos(id,collection,series,source,title,upload_ts,date_ts,date_precision,author,
            author_name,caption,hashtags,music,duration,is_photo,location,path,url,info_path,info_mtime,tr_mtime,
            transcript_subs,transcript_whisper,ocr,keywords,topics,transcript_speakers,n_speakers,first_seen,visual)
            values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            on conflict(id) do update set collection=excluded.collection,series=excluded.series,
            source=excluded.source,title=excluded.title,upload_ts=excluded.upload_ts,date_ts=excluded.date_ts,
            date_precision=excluded.date_precision,author=excluded.author,author_name=excluded.author_name,
            caption=excluded.caption,hashtags=excluded.hashtags,music=excluded.music,duration=excluded.duration,
            is_photo=excluded.is_photo,location=excluded.location,path=excluded.path,url=excluded.url,
            info_path=excluded.info_path,info_mtime=excluded.info_mtime,tr_mtime=excluded.tr_mtime,
            transcript_subs=excluded.transcript_subs,transcript_whisper=excluded.transcript_whisper,ocr=excluded.ocr,
            keywords=excluded.keywords,topics=excluded.topics,transcript_speakers=excluded.transcript_speakers,
            n_speakers=excluded.n_speakers,visual=excluded.visual""", row)
        db.execute("insert or ignore into first_seen values(?, ?)", (row[0], row[-2]))
    # Rows whose files vanished are dropped (the floor check catches a mass vanish).
    gone = [r[0] for r in db.execute("select id from videos where collection=?", (name,)) if r[0] not in seen]
    fts_sync(db, changed, gone)
    for i in range(0, len(gone), 500):
        db.execute(f"delete from videos where id in ({qmarks(gone[i:i + 500])})", gone[i:i + 500])
    if coll.plugin and hasattr(coll.plugin, "after_scan"):
        coll.plugin.after_scan(db)
    meta_set(db, f"updated:{name}", now)
    # The floor is checked inside the transaction: a failing run (NAS unmounted, files vanished) is rolled back, so
    # the last good index stays queryable while the failure is still reported.
    ok, msg = floor_check(db, coll)
    if ok:
        bump_high_water(db, coll)
        db.commit()
    else:
        db.rollback()
        msg += " — update rolled back, index unchanged"
    return ok, f"indexed {len(changed)} files, dropped {len(gone)}; {msg}"


FTS_COLS = "caption, hashtags, author, music, location, keywords, transcript, ocr, visual, title"
FTS_SELECT = """caption, hashtags, author || ' ' || author_name, music, location,
    trim(coalesce(keywords,'') || ' ' || coalesce(topics,'')),
    trim(coalesce(transcript_whisper,'') || ' ' || coalesce(transcript_subs,'')), coalesce(ocr,''),
    coalesce(visual,''), coalesce(title,'')"""


def fts_sync(db, changed, gone):
    """Re-index only the rows that changed (the fts rowid is the videos rowid, so this is a keyed delete + insert)."""
    for ids in (changed, gone):
        for i in range(0, len(ids), 500):
            db.execute(f"delete from fts where rowid in (select rowid from videos where id in ({qmarks(ids[i:i + 500])}))",
                       ids[i:i + 500])
    for i in range(0, len(changed), 500):
        db.execute(f"insert into fts(rowid, {FTS_COLS}) select rowid, {FTS_SELECT} from videos "
                   f"where id in ({qmarks(changed[i:i + 500])})", changed[i:i + 500])


def floor_check(db, coll):
    if coll.plugin and hasattr(coll.plugin, "floor"):
        return coll.plugin.floor(db)
    n = db.execute("select count(*) from videos where collection=?", (coll.name,)).fetchone()[0]
    hw = int(meta_get(db, f"high_water:{coll.name}", 0))
    problems = []
    if n < coll.floor_min:
        problems.append(f"rows {n} < floor {coll.floor_min}")
    if hw and n < hw - coll.floor_slack:
        problems.append(f"rows {n} shrank below high-water {hw} - {coll.floor_slack}")
    msg = f"rows={n} high_water={hw}"
    return (False, "FLOOR FAIL: " + "; ".join(problems) + f" ({msg})") if problems else (True, f"floor ok ({msg})")


def bump_high_water(db, coll):
    if coll.plugin and hasattr(coll.plugin, "bump_high_water"):
        return coll.plugin.bump_high_water(db)
    n = db.execute("select count(*) from videos where collection=?", (coll.name,)).fetchone()[0]
    meta_set(db, f"high_water:{coll.name}", max(n, int(meta_get(db, f"high_water:{coll.name}", 0))))


# ---------------------------------------------------------------- shared helpers for steps

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
RETRY_ERROR_AFTER = 7 * 86400  # an unavailable item (deleted/private/blocked) is retried weekly, not per poll


def http_get(url, referer=None, timeout=30):
    import urllib.request
    h = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
    if referer:
        h["Referer"] = referer
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode())
    os.replace(tmp, path)


def step_colls(step):
    return [c.name for c in COLLS.values() if step in c.steps]


def step_rows(db, step, cols="id, info_path, path, caption, collection, is_photo"):
    """Rows of the collections that have `step`, newest first (chat order, then date)."""
    names = step_colls(step)
    if not names:
        return []
    return db.execute(f"select {cols} from videos where collection in ({qmarks(names)}) "
                      "order by chat_pos is null, chat_pos desc, date_ts desc", names).fetchall()


def step_mounts(step):
    """-v args exposing the step's collection folders read-only at the same paths inside a container."""
    out = []
    for n in step_colls(step):
        for p in COLLS[n].paths:
            out += ["-v", f"{p}:{p}:ro"]
    return out


def cmd_enrich(args):
    """Each plugin's page fetch (TikTok: public post page) for the collections with the enrich step."""
    rc = 0
    for c in COLLS.values():
        if "enrich" in c.steps and c.plugin and hasattr(c.plugin, "enrich"):
            rc = c.plugin.enrich(connect(), args) or rc
    return rc


# ---------------------------------------------------------------- whisper transcription

_A = CFG.get("asr", {})
WHISPER_IMAGE = _A.get("whisper_image", "ghcr.io/ggml-org/whisper.cpp:main")
WHISPER_MODEL = _A.get("whisper_model", "ggml-large-v3-turbo-q5_0.bin")
VAD_MODEL = "ggml-silero-v5.1.2.bin"
WHISPER_CPUS = str(_A.get("cpus", 4))
ASR_IMAGE = _A.get("image", "vidrag-asr:3")


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


def transcribe_pending(db):
    return [(tdir_of(r["info_path"]), r["id"], r["path"]) for r in step_rows(db, "transcribe")
            if not (tdir_of(r["info_path"]) / f"{r['id']}.whisper.txt").exists()
            and not (tdir_of(r["info_path"]) / f"{r['id']}.whisper.fail").exists()
            and os.path.exists(r["path"])]


def cmd_transcribe(args):
    todo = transcribe_pending(connect())
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


def cmd_diarize(args):
    """Speaker turns for every video lacking <id>.turns.json, one container per series (models load once)."""
    import subprocess
    by_series = {}
    for r in step_rows(connect(), "diarize"):
        tdir = tdir_of(r["info_path"])
        if not (tdir / f"{r['id']}.turns.json").exists() and os.path.exists(r["path"]):
            by_series.setdefault(tdir, []).append({"id": r["id"], "path": r["path"]})
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
               "--user", f"{os.getuid()}:{os.getgid()}", *step_mounts("diarize"), "-v", f"{MODELS}:/models:ro",
               "-v", f"{tdir}:/out", "-v", f"{HERE / 'asr'}:/asr:ro",
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
    slide. Tesseract returned noise on busy photo backgrounds; RapidOCR ~1.7 s/slide on 4 cores."""
    import subprocess
    by_series = {}
    for r in step_rows(connect(), "ocr"):
        series = series_of(r["info_path"])
        if r["is_photo"] and not (series / ".transcripts" / f"{r['id']}.ocr.txt").exists():
            by_series.setdefault(series, []).append({"id": r["id"], "dir": str(series / ".photos" / r["id"])})
    for series, jobs in by_series.items():
        jobs = jobs[: args.max] if args.max else jobs
        tdir = series / ".transcripts"
        tdir.mkdir(parents=True, exist_ok=True)
        cmd = ["docker", "run", "--rm", "-i", f"--cpus={WHISPER_CPUS}", "--memory=3g", "--cpu-shares=128",
               "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", *step_mounts("ocr"),
               "-v", f"{tdir}:/out", "-v", f"{HERE / 'asr'}:/asr:ro", ASR_IMAGE, "python", "/asr/ocr.py"]
        p = subprocess.run(cmd, input=json.dumps(jobs), capture_output=True, text=True, timeout=3 * 3600)
        if not args.quiet or p.returncode:
            print(p.stdout.strip(), flush=True)
        if p.returncode:
            print(f"ocr: container exit {p.returncode}: {p.stderr.strip()[-300:]}", file=sys.stderr)
            return 3
    return 0


# ---------------------------------------------------------------- embeddings

_E = CFG.get("embed", {})
EMBED_IMAGE = _E.get("image", "vidrag-embed:2")
# EmbeddingGemma-300m won a bake-off (eval/eval_retrieval.py) against nomic-embed and Qwen3-Embedding-0.6B; every
# reranker tried was worse AND took most of a minute per query on 4 cores.
EMBED_MODEL = _E.get("model", "google/embeddinggemma-300m")
EMBED_MODEL_PATH = _E.get("model_path", "/models/fastembed/flat/embeddinggemma-300m")
_S = CFG.get("search", {})
# Keyword ranking's weight in the fusion. Pure embeddings scored a little higher on a judged set, but judged queries
# under-represent exact names (a creator's handle, a dish, a venue) where the keyword leg is the safety net.
KW_WEIGHT = float(_S.get("kw_weight", 0.25))
LOOKUP_MAX = int(_S.get("lookup_max", 5))  # see the lookup rule in search()
SEARCHER = None  # the warm in-process model, set by `serve`


def embed_run(mode, payload, timeout):
    import subprocess
    (MODELS / "fastembed").mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "-i", "--cpus=4", "--memory=3g", "--cpu-shares=256",
           "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-e", "THREADS=4",
           "-e", f"EMBED_MODEL={EMBED_MODEL}", *(["-e", f"MODEL_PATH={EMBED_MODEL_PATH}"] if EMBED_MODEL_PATH else []),
           "-v", f"{MODELS / 'fastembed'}:/models/fastembed", "-v", f"{DATA}:/data",
           "-v", f"{HERE / 'asr'}:/asr:ro", EMBED_IMAGE, "python", "/asr/embed.py", mode]
    p = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        err = [l for l in p.stderr.strip().splitlines() if "it/s" not in l and "Fetching" not in l]
        raise RuntimeError(f"exit {p.returncode}{' (OOM-killed?)' if p.returncode == 137 else ''}: "
                           + (err[-1][:300] if err else ""))
    return json.loads(p.stdout.strip().splitlines()[-1])


def embed_query(text, k=200, ids=None):
    """-> [[id, cosine], …] best first among `ids` (all when None), or None when the semantic leg is unavailable.
    Uses the warm model when running inside `serve`, else a running server's (api.url), else a one-shot container."""
    if SEARCHER is not None:
        try:
            return SEARCHER.query(text, k, ids)
        except Exception as e:  # noqa: BLE001 — degrade to keyword search, never fail the query
            print(f"semantic leg failed: {e}", file=sys.stderr)
            return None
    if not (DATA / "vectors.npz").exists():
        return None
    if API_URL:
        try:
            return api_call("/v1/embed", {"q": text, "k": k, "ids": ids})
        except Exception as e:  # noqa: BLE001 — fall back to the container
            print(f"api embed failed ({e}); starting the model locally", file=sys.stderr)
    try:
        return embed_run("query", {"q": text, "k": k, "ids": ids}, timeout=120)
    except Exception as e:  # noqa: BLE001 — degrade to keyword search, never fail the query
        print(f"semantic leg failed: {e}", file=sys.stderr)
        return None


def doc_text(r):
    parts = [r["title"], r["caption"], "#" + r["hashtags"].replace(" ", " #") if r["hashtags"] else "", r["location"],
             r["keywords"], r["topics"], r["visual"], r["ocr"], r["transcript_whisper"] or r["transcript_subs"]]
    return "\n".join(p for p in parts if p)


def cmd_embed(args):
    """(Re)embed every video whose text changed. Single instance: a long first run (a big new collection) and the
    pipeline's routine call never both embed the same backlog."""
    lock = single_instance("embed")
    if lock is None:
        if not args.quiet:
            print("embed: another run holds the lock, skipping")
        return 0
    with lock:
        with closing(connect()) as db:
            docs = [{"id": r["id"], "text": doc_text(r)} for r in db.execute("select * from videos")]
        r = embed_run("index", {"docs": docs}, timeout=6 * 3600)
    if not args.quiet:
        print(f"embed: {r['embedded']} (re)embedded, {r['total']} vectors")
    return 0


# ---------------------------------------------------------------- Gemini watches each video (the one cloud step)
# Only collections that list "watch" in their steps, and only with a key. Nothing else is sent: one 1-fps 480p
# re-encode + audio + the caption per video, over the stateless generateContent endpoint (the Interactions API stores
# requests 55 days by default). Use a paid-tier key so Google does not train on it. Off: touch <home>/GEMINI_OFF.

_G = CFG.get("gemini", {})
GEMINI_MODEL = _G.get("model", "gemini-3.5-flash-lite")
GEMINI_KEY_FILE = expand(_G.get("key_file", HOME / ".config/vidrag/gemini.env"))
GEMINI_PRICE = tuple(_G.get("price", (0.30, 2.50)))  # $/M input, output (thinking included)
INLINE_MAX = 14 * 2**20  # generateContent caps a request at 20 MB and base64 adds a third
GEMINI_PROMPT = """You are indexing a {kind}video for a private search engine. Watch the video and listen to the audio.
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
        h = json.loads((DATA / ".gemini-hold").read_text())
    except (OSError, ValueError):
        return None
    return h.get("reason") if time.time() < h.get("until", 0) else None


def gemini_ready():
    return bool(step_colls("watch")) and bool(gemini_key()) and not GEMINI_OFF.exists() and not gemini_hold()


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


def watch_one(mp4, tdir, vid, caption, key, kind=""):
    """Gemini watches one video -> <id>.gemini.json {description, raw, usage} or {error} (retried weekly)."""
    import base64
    tmp = DATA / ".watch-tmp" / f"{vid}.mp4"
    try:
        clip = watch_clip(mp4, tmp)
    finally:
        tmp.unlink(missing_ok=True)
    body = {"contents": [{"parts": [
        {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(clip).decode()}},
        {"text": GEMINI_PROMPT.format(kind=f"{kind} " if kind else "",
                                      caption=" ".join((caption or "(none)").split())[:1000])}]}],
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
    """-> [(tdir, id, mp4, caption, kind)] newest first: no sidecar yet, or an error sidecar older than a week."""
    now, out = time.time(), []
    for r in step_rows(db, "watch"):
        tdir = tdir_of(r["info_path"])
        p = tdir / f"{r['id']}.gemini.json"
        if p.exists():
            try:
                old = json.loads(p.read_text())
            except ValueError:
                old = {"error": "corrupt", "fetched": 0}
            if "error" not in old or now - old.get("fetched", 0) < RETRY_ERROR_AFTER:
                continue
        if os.path.exists(r["path"]):
            out.append((tdir, r["id"], r["path"], r["caption"], COLLS[r["collection"]].kind))
    return out


def cmd_watch(args):
    key = gemini_key()
    if not key or GEMINI_OFF.exists() or gemini_hold() or not step_colls("watch"):
        if not args.quiet:
            why = ("no collection has the watch step" if not step_colls("watch") else "no key" if not key
                   else "GEMINI_OFF" if GEMINI_OFF.exists() else gemini_hold())
            print(f"watch: skipped ({why})")
        return 0
    todo = watch_pending(connect())
    todo = todo[: args.max] if args.max else todo
    if not args.quiet:
        print(f"watch: {len(todo)} pending, model {GEMINI_MODEL}", flush=True)
    ok = err = streak = 0
    for tdir, vid, mp4, caption, kind in todo:
        if OFF.exists() or GEMINI_OFF.exists():
            break
        t0 = time.time()
        try:
            good, note = watch_one(mp4, tdir, vid, caption, key, kind)
        except GeminiHold as e:
            # 402 no credits / 401-403 key / 429 quota: pause the step (6 h; 30 min for quota) instead of retrying.
            wait = 1800 if "HTTP 429" in str(e) else 6 * 3600
            atomic_write(DATA / ".gemini-hold", json.dumps({"until": int(time.time()) + wait, "reason": str(e)[:300]}))
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


# ---------------------------------------------------------------- search

def parse_since(s):
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            pass
    raise ValueError(f"bad date {s!r}: use YYYY, YYYY-MM or YYYY-MM-DD")


# Concept groups: a query word in a group also matches its siblings; keep them small and literal (multi-word entries
# are phrases). The weight scales a group's IDF: "place/spot/things to do" says what KIND of answer is wanted, not
# what the video is about, so it only breaks ties. [[search.aliases]] in the config replaces these.
ALIASES = [(float(a.get("weight", 1.0)), set(a["words"])) for a in _S.get("aliases", [])] or [
    (1.0, {"nyc", "new york", "newyork", "newyorkcity", "manhattan", "brooklyn", "queens", "bronx", "harlem",
     "soho", "williamsburg", "nycfood", "nyceats", "thingstodoinnyc"}),
    (1.0, {"eat", "eats", "food", "foodie", "restaurant", "restaurants", "dinner", "lunch", "brunch", "breakfast",
     "cafe", "bakery", "pizza", "bar", "dessert", "nycfood", "nyceats", "foodtok", "eating"}),
    (0.25, {"place", "places", "spot", "spots", "visit", "things to do", "thingstodo", "hidden gem", "hiddengem"}),
    (1.0, {"recipe", "recipes", "cook", "cooking", "easyrecipe", "dinnerideas"}),
]
# Expansion is one-way: a generic word ("food", "nyc") matches its specific siblings, but a specific word the user
# typed is matched literally. Otherwise "pizza nyc" means "any food + nyc" and every NYC food video ties with the
# actual pizza one, and "brooklyn" returns Manhattan.
SPECIFIC = set(_S.get("specific", [])) or {
    "manhattan", "brooklyn", "queens", "bronx", "harlem", "soho", "williamsburg", "cafe", "bakery", "pizza",
    "bar", "dessert", "brunch", "breakfast", "lunch", "dinner"}
FTS_FIELDS = "{caption hashtags author location keywords transcript ocr visual title}"  # music left out on purpose
BM25 = "bm25(fts, 2.0, 1.5, 0.5, 2.0, 1.2, 0.8, 0.8, 0.6, 1.0, 2.0)"


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
        groups.append((weight, FTS_FIELDS + " : (" + " OR ".join(f'"{t}"' for t in sorted(g)) + ")"))
    return groups


def contact_of(scope):
    for n in scope:
        c = getattr(COLLS[n].plugin, "contact", None)
        if c:
            return c
    return "them"


def filters(o, alias="v"):
    """Every filter except the collection scope, which callers apply separately."""
    where, params = [], []
    if o.since:
        where.append(f"{alias}.date_ts >= ?"); params.append(parse_since(o.since))
    if o.until:
        where.append(f"{alias}.date_ts < ?"); params.append(parse_since(o.until))
    if o.sender and o.sender != "any":
        where.append(f"{alias}.sender = ?"); params.append(contact_of(o.collections) if o.sender == "them" else o.sender)
    if o.source and o.source != "all":
        where.append(f"{alias}.source = ?"); params.append(o.source)
    if getattr(o, "series", None):
        where.append(f"lower({alias}.series) = lower(?)"); params.append(o.series)
    if getattr(o, "author", None):
        where.append(f"(lower({alias}.author) = lower(?) or lower({alias}.author_name) = lower(?))")
        params += [o.author.lstrip("@"), o.author.lstrip("@")]
    return where, params


def scope_sql(o, alias="v"):
    return f"{alias}.collection in ({qmarks(o.collections)})", list(o.collections)


def search(db, o):
    """-> (rows, notes). Keyword leg: rank by how many query concepts a video matches (each weighted by its IDF within
    the searched collections), then bm25. Semantic leg: local embeddings. Hybrid fuses them by weighted RRF."""
    notes = []
    where, params = filters(o)
    sc, sp = scope_sql(o)
    groups = fts_terms(o.text) or [(1.0, o.text)]
    whole = " OR ".join(f"({g})" for _, g in groups)
    sql = f"""select v.*, snippet(fts, -1, '[', ']', '…', 24) as snip, {BM25} as score
              from fts join videos v on v.rowid = fts.rowid
              where fts match ? and {sc}{''.join(' and ' + w for w in where)}"""
    try:
        rows = db.execute(sql, [whole, *sp, *params]).fetchall()
        # Coverage score: each matched concept adds its IDF, so a rare concept ("nyc") outweighs common
        # ones ("eat", "place") — a video about NYC food beats a non-NYC video that mentions both.
        n_docs = max(1, db.execute(f"select count(*) from videos v where {sc}", sp).fetchone()[0])
        hits, n_hit = {}, {}
        for weight, g in groups:
            ids = [vid for (vid,) in db.execute(
                f"select v.id from fts join videos v on v.rowid = fts.rowid where fts match ? and {sc}", [g, *sp])]
            idf = weight * math.log(1 + n_docs / max(1, len(ids)))
            for vid in ids:
                hits[vid] = hits.get(vid, 0) + idf
                n_hit[vid] = n_hit.get(vid, 0) + 1
    except sqlite3.OperationalError as e:
        raise ValueError(f"query error: {e} (fts query was {whole!r})")
    if o.newest:
        rows.sort(key=lambda r: (-round(hits.get(r["id"], 0), 1), -(r["date_ts"] or 0)))
    else:
        rows.sort(key=lambda r: (-round(hits.get(r["id"], 0), 1), r["score"]))
    if o.all_terms:
        rows = [r for r in rows if n_hit.get(r["id"], 0) == len(groups)]
    if o.mode == "keyword" or o.all_terms:
        return rows[: o.limit], notes
    # Semantic leg: cosine over local embeddings of the searched collections, fused with the keyword ranking by
    # weighted reciprocal rank (k=60, keyword x KW_WEIGHT), so "Katz's on the Lower East Side" answers "nyc places to
    # eat" with no shared word.
    scope_ids = [r[0] for r in db.execute(f"select v.id from videos v where {sc}", sp)]
    sem = embed_query(o.text, k=200, ids=scope_ids)
    if sem is None:
        notes.append("semantic search unavailable — keyword results only")
        return rows[: o.limit], notes
    allowed = {r[0] for r in db.execute(f"select v.id from videos v where {' and '.join([sc, *where])}", [*sp, *params])}
    kw_rank = [r["id"] for r in rows]
    sem_rank = [i for i, _ in sem if i in allowed]
    if o.mode == "semantic":
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
        # every word by chance; firing there cost nDCG on a judged set, so they are left to fusion.
        exact = [r["id"] for r in rows if n_hit.get(r["id"], 0) == len(groups)]
        if len(groups) == 1 and 0 < len(exact) <= LOOKUP_MAX:
            fused = exact + [v for v in fused if v not in exact]
    by_id = {r["id"]: r for r in rows}
    out = [by_id.get(vid) or db.execute("select v.*, '' as snip from videos v where id=?", (vid,)).fetchone()
           for vid in fused[: o.limit]]
    if o.newest:
        out.sort(key=lambda r: -(r["date_ts"] or 0))
    return out, notes


def recent(db, o):
    where, params = filters(o)
    sc, sp = scope_sql(o)
    # Chat position is the true order for DMs; everything else goes by date.
    chat = all(getattr(COLLS[n].plugin, "chat_ordered", False) for n in o.collections) and o.source not in ("saved", "fav")
    order = "coalesce(v.chat_pos, -1) desc, v.date_ts desc" if chat else "v.date_ts desc"
    return db.execute(f"select v.* from videos v where {' and '.join([sc, *where])} order by {order} limit ?",
                      [*sp, *params, o.n]).fetchall()


def fmt_date(ts, prec=None):
    if not ts:
        return "?"
    d = datetime.fromtimestamp(ts, timezone.utc)
    if prec == "est":
        return ">=" + d.strftime("%Y-%m-%d")
    return d.strftime("%Y-%m-%d %H:%MZ") if prec == "poll" else d.strftime("%Y-%m-%d")


def row_url(r):
    p = COLLS[r["collection"]].plugin if r["collection"] in COLLS else None
    return p.url(r) if p and hasattr(p, "url") else r["url"] or ""


def record(r, snippet_col=None, legacy=False):
    """One result as a dict. legacy=True keeps the field names and order of the old tiktok-rag output."""
    snippet = " ".join((r[snippet_col] if snippet_col and r[snippet_col] else (r["caption"] or "")[:200]).split())
    common = {"seen": (r["visual"] or "").split("\n")[0],  # the watch step's one-line summary of what it shows
              "has_transcript": bool(r["transcript_whisper"] or r["transcript_subs"]),
              # "meaning": only the semantic leg found it (no query word in the video) — a guess, so say so
              "match": "meaning" if snippet_col and not r[snippet_col] else "words"}
    if legacy:
        rec = {k: r[k] for k in ("id", "source", "sender", "author", "caption", "hashtags", "music", "location", "topics")}
        rec.update(nas_path=r["path"], share_precision=r["date_precision"], is_photo=r["is_photo"],
                   shared=fmt_date(r["date_ts"], r["date_precision"]), uploaded=fmt_date(r["upload_ts"]),
                   url=row_url(r), snippet=snippet)
        return {**rec, **common}
    rec = {k: r[k] for k in ("id", "collection", "series", "source", "sender", "title", "author", "caption", "hashtags",
                             "music", "location", "topics")}
    rec.update(date=fmt_date(r["date_ts"], r["date_precision"]), date_precision=r["date_precision"],
               uploaded=fmt_date(r["upload_ts"]), duration=r["duration"], is_photo=r["is_photo"], path=r["path"],
               url=row_url(r), snippet=snippet)
    return {**rec, **common}


def print_records(recs, snippet_used, legacy=False):
    if not recs:
        print("(no results)")
    for i, rec in enumerate(recs, 1):
        who = rec["sender"] or {"fav": "my favorites"}.get(rec["source"], rec["source"])
        if legacy:
            label, date, extra = f"from {who}", rec["shared"], ""
        else:
            label = " · ".join(x for x in (rec["collection"], f"from {who}" if who else rec["series"]) if x)
            date = rec["date"]
            dur = int(rec["duration"] or 0)
            extra = f" · {dur // 60}:{dur % 60:02d}" if dur else ""
        print(f"{i:>2}. [{date}] {label} · @{rec['author']} · uploaded {rec['uploaded']}{extra}"
              f"{' · photo post' if rec['is_photo'] else ''}{' · ≈ related by meaning only' if rec['match'] == 'meaning' else ''}")
        if rec["location"]:
            print(f"    📍 {rec['location']}")
        if not legacy and rec["title"]:
            print(f"    {rec['title']}")
        cap = " ".join((rec["caption"] or "").split())
        print(f"    {cap[:220]}{'…' if len(cap) > 220 else ''}")
        plain = re.sub(r"[\[\]…]", "", rec["snippet"]).strip()
        if rec["seen"]:
            print(f"    seen: {rec['seen'][:220]}{'…' if len(rec['seen']) > 220 else ''}")
        if snippet_used and plain and plain[:40] not in cap and plain[:40] not in rec["seen"]:
            print(f"    match: {rec['snippet']}")
        print(f"    {rec.get('nas_path') or rec.get('path')}")
        print(f"    {rec['url']}")


def show_row(db, vid, legacy=False):
    r = db.execute("select * from videos where id=?", (vid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["url"] = row_url(r)
    d["date"] = fmt_date(d["date_ts"], d["date_precision"])
    d["uploaded"] = fmt_date(d["upload_ts"])
    if legacy:
        for new, old in (("path", "nas_path"), ("date_ts", "share_ts"), ("date_precision", "share_precision"),
                         ("transcript_subs", "transcript_tiktok"), ("date", "shared")):
            d[old] = d.pop(new)
    return d


def collections_info(db):
    out = []
    for c in COLLS.values():
        q = lambda s, *a: db.execute(s, (c.name, *a)).fetchall()
        upd = meta_get(db, f"updated:{c.name}")
        out.append({
            "name": c.name, "kind": c.kind, "videos": q("select count(*) from videos where collection=?")[0][0],
            "steps": c.steps, "gemini": "watch" in c.steps, "default": c.name in DEFAULT_SCOPE,
            "updated": fmt_date(int(upd), "poll") if upd else None,
            "sources": dict(q("select source, count(*) from videos where collection=? and source is not null group by 1")),
            "series": dict(q("select series, count(*) from videos where collection=? group by 1 order by 2 desc")),
        })
    return out


def gemini_stats(db, scope):
    """Tokens and estimated spend so far, summed from the sidecars' usageMetadata."""
    n = tin = tout = 0
    names = [s for s in scope if "watch" in COLLS[s].steps]
    rows = db.execute(f"select id, info_path from videos where collection in ({qmarks(names)})", names) if names else []
    for r in rows:
        try:
            u = json.loads((tdir_of(r["info_path"]) / f"{r['id']}.gemini.json").read_text())
        except (OSError, ValueError):
            continue
        u = u.get("usage") or {}
        n += 1
        tin += u.get("promptTokenCount", 0)
        tout += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return {"sidecars": n, "input_tokens": tin, "output_tokens": tout,
            "est_usd": round(tin / 1e6 * GEMINI_PRICE[0] + tout / 1e6 * GEMINI_PRICE[1], 2),
            "ready": gemini_ready(), "hold": gemini_hold(), "off": GEMINI_OFF.exists()}


def stats(db, scope, legacy=False):
    """-> (ok, dict). legacy: the old single-archive keys for one plugin collection."""
    ok = True
    upd = meta_get(db, "updated")
    if legacy:
        c = COLLS[scope[0]]
        good, msg = floor_check(db, c)
        return good, {"db": str(DB), "updated": fmt_date(int(upd), "poll") if upd else None,
                      **c.plugin.legacy_stats(db), "gemini": gemini_stats(db, scope),
                      **c.plugin.legacy_stats_tail(db), "floor": msg}
    out = {"db": str(DB), "updated": fmt_date(int(upd), "poll") if upd else None, "collections": {}}
    for n in scope:
        c = COLLS[n]
        q = lambda s: db.execute(s.replace("WHERE", "where collection=? and"), (n,)).fetchone()[0]
        good, msg = floor_check(db, c)
        ok = ok and good
        out["collections"][n] = {
            "videos": db.execute("select count(*) from videos where collection=?", (n,)).fetchone()[0],
            "with_subtitles": q("select count(*) from videos WHERE coalesce(transcript_subs,'')<>''"),
            "with_whisper_transcript": q("select count(*) from videos WHERE coalesce(transcript_whisper,'')<>''"),
            "with_speaker_turns": q("select count(*) from videos WHERE n_speakers is not null"),
            "with_ocr_or_stickers": q("select count(*) from videos WHERE coalesce(ocr,'')<>''"),
            "watched_by_gemini": q("select count(*) from videos WHERE coalesce(visual,'')<>''"),
            "with_location": q("select count(*) from videos WHERE coalesce(location,'')<>''"),
            "steps": c.steps, "floor": msg}
    out["gemini"] = gemini_stats(db, scope)
    return ok, out


# ---------------------------------------------------------------- pipeline

def pending_counts(db):
    """{step: videos still to do}, from sidecar existence (a few stats per video). watch counts only while Gemini can
    run (key, no GEMINI_OFF, no billing pause), so a pause cannot spin the loop."""
    out = dict.fromkeys(STEPS, 0)
    for c in COLLS.values():
        if "enrich" in c.steps and c.plugin and hasattr(c.plugin, "enrich_pending"):
            out["enrich"] += c.plugin.enrich_pending(db)
    for r in step_rows(db, "ocr"):
        out["ocr"] += bool(r["is_photo"]) and not (tdir_of(r["info_path"]) / f"{r['id']}.ocr.txt").exists()
    for r in step_rows(db, "diarize"):
        out["diarize"] += os.path.exists(r["path"]) and not (tdir_of(r["info_path"]) / f"{r['id']}.turns.json").exists()
    out["transcribe"] = len(transcribe_pending(db))
    out["watch"] = len(watch_pending(db)) if gemini_ready() else 0
    return out


def due_collections(db):
    """Collections whose update_every has elapsed since their last scan."""
    now = time.time()
    return [c.name for c in COLLS.values()
            if c.update_every and now - int(meta_get(db, f"updated:{c.name}", 0)) >= c.update_every]


def cmd_pipeline(args):
    """Drain every enrichment step, newest videos first, so a fresh video is fully processed within one Whisper
    video's time: each pass = enrich <=20, ocr <=10, diarize <=20, transcribe 1, watch <=10, re-index. Collections with
    update_every are re-scanned first when due. Start it detached after each poll (single instance); stops when nothing
    is pending or when <home>/OFF exists."""
    ns = argparse.Namespace
    passes = 0
    with closing(connect()) as db:
        due = due_collections(db)
    if due and not OFF.exists():
        update(ns(quiet=True, collection=due))
        embed_safely()
    stepped = [c.name for c in COLLS.values() if c.steps] or None
    last = None
    while not OFF.exists():
        with closing(connect()) as db:
            pend = pending_counts(db)
        if not any(pend.values()):
            break
        if pend == last:  # a whole pass changed nothing (a step keeps failing): stop instead of spinning
            print(f"{datetime.now(timezone.utc):%FT%TZ} pipeline: no progress in a pass, stopping; pending "
                  + " ".join(f"{k}={v}" for k, v in pend.items() if v), file=sys.stderr, flush=True)
            break
        last = pend
        local = any(v for k, v in pend.items() if k != "watch")
        if pend["ocr"]:
            cmd_ocr(ns(max=10, quiet=True))
        if pend["enrich"]:
            cmd_enrich(ns(max=20, delay=args.delay, quiet=True))
        if pend["diarize"]:
            cmd_diarize(ns(max=20, quiet=True))
        if pend["transcribe"]:
            cmd_transcribe(ns(max=1, budget=0, quiet=True))
        if pend["watch"]:
            lock = single_instance("watch")  # a hand-started `watch` backlog run holds it: leave that run be
            if lock:
                cmd_watch(ns(max=10, quiet=True))
                lock.close()
            elif not local:
                # Only Gemini work is left and that run owns it: stop instead of re-indexing in a loop until it ends
                # (that once spun for thousands of idle passes). The next poll restarts the pipeline for anything left.
                update(ns(quiet=True, collection=stepped))
                break
        update(ns(quiet=True, collection=stepped))
        passes += 1
        if passes % 10 == 0:
            embed_safely()
        if not args.quiet:
            print(f"{datetime.now(timezone.utc):%FT%TZ} pass {passes}: pending "
                  + " ".join(f"{k}={v}" for k, v in pend.items()), flush=True)
    if passes:
        embed_safely()
        print(f"{datetime.now(timezone.utc):%FT%TZ} pipeline done after {passes} passes"
              f"{' (OFF switch)' if OFF.exists() else ''}", flush=True)
    return 0


def embed_safely():
    try:
        cmd_embed(argparse.Namespace(quiet=True))
    except Exception as ex:  # noqa: BLE001
        print(f"embed failed: {ex}", file=sys.stderr)


def wait_lock(name, wait):
    """Like single_instance, but waits up to `wait` seconds: two updates of one collection run one after the other
    (the second then finds nothing changed) instead of scanning the same tree twice at once."""
    deadline = time.time() + wait
    while True:
        f = single_instance(name)
        if f or time.time() >= deadline:
            return f
        time.sleep(1)


def single_instance(name):
    """Non-blocking lock so a per-poll run never overlaps the background backlog run of the same step."""
    import fcntl
    DATA.mkdir(parents=True, exist_ok=True)
    f = open(DATA / f".{name}.lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    return f


# ---------------------------------------------------------------- HTTP API (`vidrag serve`)

_API = CFG.get("api", {})
API_LISTEN = _API.get("listen", "127.0.0.1:8790")
API_TOKEN_FILE = expand(_API.get("token_file", HOME / ".config/vidrag/api-token"))
API_URL = _API.get("url", "").rstrip("/")  # the CLI borrows a running server's warm model for the semantic leg
MAX_LIMIT = 100
MAX_BODY = 64 * 1024


def api_token():
    if os.environ.get("VIDRAG_API_TOKEN"):
        return os.environ["VIDRAG_API_TOKEN"].strip()
    try:
        return API_TOKEN_FILE.read_text().strip() or None
    except OSError:
        return None


def api_call(path, body, timeout=30):
    import urllib.request
    req = urllib.request.Request(API_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_token()}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def listify(v):
    if v is None or v == "":
        return []
    return [x.strip() for x in (v if isinstance(v, list) else str(v).split(",")) if str(x).strip()]


def truthy(v):
    return v is True or str(v).lower() in ("1", "true", "yes", "on")


def opts_from(p, need_text=True):
    """Query params / JSON body -> the same options the CLI builds. Raises ValueError on bad input."""
    colls = listify(p.get("collection") or p.get("collections")) or DEFAULT_SCOPE
    bad = [c for c in colls if c not in COLLS]
    if bad:
        raise ValueError(f"unknown collection {bad[0]!r}; have {', '.join(COLLS)}")
    text = (p.get("q") or p.get("text") or "").strip()
    if need_text and not text:
        raise ValueError("q is required")
    mode = p.get("mode") or "hybrid"
    if mode not in ("hybrid", "keyword", "semantic"):
        raise ValueError("mode is hybrid, keyword or semantic")
    o = argparse.Namespace(
        text=text, collections=colls, limit=max(1, min(MAX_LIMIT, int(p.get("limit") or p.get("n") or 15))),
        n=max(1, min(MAX_LIMIT, int(p.get("n") or p.get("limit") or 10))), newest=truthy(p.get("newest")),
        all_terms=truthy(p.get("all")), mode=mode, since=p.get("since"), until=p.get("until"),
        sender=p.get("from") or p.get("sender") or "any", source=p.get("source") or "all",
        series=p.get("series"), author=p.get("author"))
    for d in (o.since, o.until):
        if d:
            parse_since(d)
    return o


OPENAPI_PARAMS = [
    ("q", "string", "what to find, in plain words (\"dog catching a frisbee\", \"pizza in brooklyn\")"),
    ("collection", "string", "comma-separated collection names (GET /v1/collections); default: the configured default"),
    ("limit", "integer", f"results, 1-{MAX_LIMIT} (default 15)"),
    ("mode", "string", "hybrid (default) | keyword | semantic"),
    ("since", "string", "date >= YYYY[-MM[-DD]]"), ("until", "string", "date < YYYY[-MM[-DD]]"),
    ("series", "string", "one series folder, e.g. a channel name"), ("author", "string", "uploader handle or channel"),
    ("from", "string", "chat collections: who sent it (me | them | a contact name)"),
    ("source", "string", "chat collections: dm | saved | fav"),
    ("newest", "boolean", "newest first among the top matches"), ("all", "boolean", "only videos matching every word"),
]


def openapi():
    params = [{"name": n, "in": "query", "schema": {"type": t}, "description": d} for n, t, d in OPENAPI_PARAMS]
    ok = {"200": {"description": "JSON"}}
    return {
        "openapi": "3.1.0",
        "info": {"title": "vidrag", "version": "1", "description": GUIDE},
        "components": {"securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}}},
        "security": [{"bearer": []}],
        "paths": {
            "/v1/search": {"get": {"summary": "Search videos by what they say and show", "parameters": params,
                                   "responses": ok},
                           "post": {"summary": "Same as GET, parameters as a JSON body", "responses": ok}},
            "/v1/recent": {"get": {"summary": "Newest videos first",
                                   "parameters": [{"name": "n", "in": "query", "schema": {"type": "integer"}}]
                                   + params[1:2] + params[4:10], "responses": ok}},
            "/v1/videos/{id}": {"get": {"summary": "Every indexed field of one video, full transcript included",
                                        "parameters": [{"name": "id", "in": "path", "required": True,
                                                        "schema": {"type": "string"}}], "responses": ok}},
            "/v1/collections": {"get": {"summary": "What can be searched: collections, their series and counts",
                                        "responses": ok}},
            "/v1/stats": {"get": {"summary": "Coverage per collection and the floor verdicts", "responses": ok}},
            "/v1/health": {"get": {"summary": "Liveness and floor verdicts (no auth)", "security": [], "responses": ok}},
        }}


def make_handler(token):
    import hmac
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    class H(BaseHTTPRequestHandler):
        server_version = "vidrag"
        timeout = 30  # a client that stops sending mid-request is dropped

        def log_message(self, fmt, *a):  # one line per request, no query strings (they carry what was searched)
            sys.stderr.write(f"{datetime.now(timezone.utc):%FT%TZ} {self.command} {urlparse(self.path).path} "
                             f"{a[1] if len(a) > 1 else ''}\n")

        def send(self, code, obj, ctype="application/json"):
            body = (json.dumps(obj, ensure_ascii=False) if ctype == "application/json" else obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authed(self):
            got = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            return bool(token) and hmac.compare_digest(got.encode(), token.encode())

        def params(self):
            u = urlparse(self.path)
            p = {k: v[-1] for k, v in parse_qs(u.query).items()}
            if self.command == "POST" and self.authed():  # never read an unauthenticated body
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    raise ValueError(f"body over {MAX_BODY} bytes")
                if n:
                    body = json.loads(self.rfile.read(n))
                    if not isinstance(body, dict):
                        raise ValueError("JSON body must be an object")
                    p.update(body)
            return u.path.rstrip("/") or "/", p

        def do_GET(self):
            self.route()

        def do_POST(self):
            self.route()

        def route(self):
            try:
                path, p = self.params()
                if path == "/":
                    return self.send(200, GUIDE, "text/plain")
                if path == "/openapi.json":
                    return self.send(200, openapi())
                if path == "/v1/health":
                    with closing(connect(readonly=True)) as db:
                        floors = {n: safe_floor(db, c) for n, c in COLLS.items()}
                    return self.send(200 if all(ok for ok, _ in floors.values()) else 503,
                                     {"ok": all(ok for ok, _ in floors.values()),
                                      "floors": {n: m for n, (_, m) in floors.items()}})
                if not self.authed():
                    return self.send(401, {"error": "Authorization: Bearer <token> required"})
                with closing(connect(readonly=True)) as db:
                    if path == "/v1/search":
                        o = opts_from(p)
                        rows, notes = search(db, o)
                        return self.send(200, {"query": o.text, "collections": o.collections, "notes": notes,
                                               "results": [record(r, "snip") for r in rows]})
                    if path == "/v1/recent":
                        o = opts_from(p, need_text=False)
                        return self.send(200, {"collections": o.collections,
                                               "results": [record(r) for r in recent(db, o)]})
                    if path.startswith("/v1/videos/"):
                        d = show_row(db, path.rsplit("/", 1)[1])
                        return self.send(200, d) if d else self.send(404, {"error": "no such id"})
                    if path == "/v1/collections":
                        return self.send(200, {"collections": collections_info(db)})
                    if path == "/v1/stats":
                        return self.send(200, stats(db, list(COLLS))[1])
                    if path == "/v1/embed" and self.command == "POST":
                        return self.send(200, SEARCHER.query(p["q"], int(p.get("k", 200)), p.get("ids")))
                return self.send(404, {"error": f"no route {path}; see GET / or /openapi.json"})
            except ValueError as e:
                return self.send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 — one bad request must not take the server down
                print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
                return self.send(500, {"error": f"{type(e).__name__}: {e}"})
    return H


def safe_floor(db, c):
    try:
        return floor_check(db, c)
    except Exception as e:  # noqa: BLE001 — e.g. a plugin file the server cannot see
        return False, f"floor check failed: {type(e).__name__}: {e}"


def cmd_serve(args):
    """The HTTP API, with the embedding model loaded once. Run it inside the embed image (it has fastembed)."""
    global SEARCHER
    from http.server import ThreadingHTTPServer
    if not COLLS:
        raise SystemExit("serve: no collections configured (is $VIDRAG_CONFIG / ~/.config/vidrag/config.toml visible?)")
    if not DB.exists():
        raise SystemExit(f"serve: no index at {DB}; run `vidrag update` first")
    host, _, port = (args.listen or API_LISTEN).rpartition(":")
    token = api_token()
    if not token and host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(f"serve: refusing to listen on {host} without a token in {API_TOKEN_FILE}")
    os.environ.update(EMBED_MODEL=EMBED_MODEL, STORE=str(DATA / "vectors.npz"), THREADS=str(args.threads),
                      **({"MODEL_PATH": EMBED_MODEL_PATH} if EMBED_MODEL_PATH else {}))
    sys.path.insert(0, str(HERE / "asr"))
    import embed
    SEARCHER = embed.Searcher()
    srv = ThreadingHTTPServer((host, int(port)), make_handler(token))
    print(f"{datetime.now(timezone.utc):%FT%TZ} vidrag serving on {host}:{port} "
          f"({', '.join(f'{n}={c.kind or n}' for n, c in COLLS.items())})", file=sys.stderr, flush=True)
    srv.serve_forever()


GUIDE = """vidrag: search downloaded videos by what they say and show (speech, captions, on-screen text, and what a
vision model saw), across collections such as a TikTok chat archive or YouTube channels.

CLI (on the server, or through the ssh wrapper):
  vidrag query "dog catching a frisbee"      hybrid keyword + semantic search, best first
      -n 15  --collection a,b  --series NAME  --author NAME  --since YYYY[-MM[-DD]]  --until …  --newest
      --mode keyword|semantic  --all (every word must match)  --from me|them|NAME  --source dm|saved|fav  --json
  vidrag recent 20 [filters]                 newest first
  vidrag show ID                             every field of one video, full transcript (JSON)
  vidrag collections                         what can be searched, with series and counts (JSON)
  vidrag stats | check                       coverage | floor check (exit 2: the index shrank or is empty)
Results: id, collection, series, title, author, caption, date, path (the file on disk), url (its original page),
  seen (what the video shows), snippet (the matching words), match ("meaning" = found by the semantic leg only).

HTTP API (Authorization: Bearer <token>; the same filters as query parameters, or a JSON body on POST):
  GET  /v1/search?q=…&collection=…&limit=…   POST /v1/search {"q": "…", "collection": "…", …}
  GET  /v1/recent?n=…   GET /v1/videos/ID   GET /v1/collections   GET /v1/stats
  GET  /v1/health and /openapi.json need no token.
"""


# ---------------------------------------------------------------- CLI

def main(argv=None, compat=None):
    """compat="tiktok": the old tiktok-rag command (one chat collection, --source dm by default, old field names)."""
    compat = compat or os.environ.get("VIDRAG_COMPAT")
    legacy_coll = next((n for n, c in COLLS.items() if c.plugin and compat and type(c.plugin).__module__ == f"vidrag_{compat}"), None)
    if compat and not legacy_coll:
        raise SystemExit(f"{compat}-rag: no collection with plugin = \"{compat}\" in the config")
    legacy = bool(legacy_coll)
    prog = f"{compat}-rag" if legacy else "vidrag"
    ap = argparse.ArgumentParser(prog=prog, description="Search downloaded videos by what they say and show.",
                                 epilog="Agents: `vidrag guide` prints a one-screen usage guide.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--since", help="date >= YYYY[-MM[-DD]]")
        p.add_argument("--until", help="date < YYYY[-MM[-DD]]")
        if legacy:
            contact = COLLS[legacy_coll].plugin.contact
            p.add_argument("--from", dest="sender", choices=list(dict.fromkeys([contact, "them", "me", "any"])), default="any")
            p.add_argument("--source", choices=["dm", "saved", "fav", "all"], default="dm")
        else:
            p.add_argument("--collection", "-c", help=f"comma-separated: {', '.join(COLLS) or '(none configured)'}")
            p.add_argument("--series", help="one series folder (a channel, a show)")
            p.add_argument("--author", help="uploader handle or channel name")
            p.add_argument("--from", dest="sender", default="any", help="chat collections: me, them or a contact name")
            p.add_argument("--source", default="all", help="chat collections: dm, saved or fav")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("query", help="search titles, captions, tags, transcripts, on-screen text and video descriptions")
    p.add_argument("text")
    p.add_argument("-n", "--limit", type=int, default=15)
    p.add_argument("--newest", action="store_true", help="newest first among the top matches")
    p.add_argument("--all", dest="all_terms", action="store_true", help="only videos matching every query concept (keyword)")
    p.add_argument("--mode", choices=["hybrid", "keyword", "semantic"], default="hybrid",
                   help="hybrid (default) fuses keyword and local-embedding rankings")
    common(p)
    p = sub.add_parser("recent", help="the N newest videos (chat order for a chat)")
    p.add_argument("n", type=int, nargs="?", default=10)
    common(p)
    p = sub.add_parser("show", help="every indexed field of one video (JSON)")
    p.add_argument("id")
    if not legacy:
        sub.add_parser("collections", help="what can be searched: collections, series, counts (JSON)")
        sub.add_parser("guide", help="a one-screen usage guide for agents and scripts")
        p = sub.add_parser("serve", help="the HTTP API (run it in the embed image; see README)")
        p.add_argument("--listen", help=f"host:port (default {API_LISTEN})")
        p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("update", help="incremental re-index; exits 2 on a floor failure")
    if not legacy:
        p.add_argument("--collection", "-c", help="comma-separated (default: all)")
        p.add_argument("--prune", action="store_true", help="also drop collections that are no longer configured")
    p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("enrich", help="fetch public pages: TikTok place tag, keywords, stickers, subtitles")
    p.add_argument("--max", type=int, default=0, help="at most N videos this run (0 = all pending)")
    p.add_argument("--delay", type=float, default=5.0, help="seconds between videos")
    p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("transcribe", help="Whisper (whisper.cpp, local, CPU-capped) transcripts, newest first")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--budget", type=float, default=0, help="stop starting new videos after S seconds")
    p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("diarize", help="speaker turns (pyannote + TitaNet via sherpa-onnx, local, CPU-capped)")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("pipeline", help="drain every step, newest first (start it detached after each download run)")
    p.add_argument("--delay", type=float, default=6.0, help="seconds between page fetches")
    p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("ocr", help="RapidOCR over photo-post slides (local, CPU-capped)")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("watch", help="Gemini watches each video once (the one cloud step; only collections listing it)")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    p = sub.add_parser("embed", help="(re)embed changed videos with the local embedding model")
    p.add_argument("-q", "--quiet", action="store_true")
    sub.add_parser("stats", help="counts, coverage and the floor verdicts (JSON)")
    sub.add_parser("check", help="floor assertion only; exit 2 if an index is empty or shrank")
    args = ap.parse_args(argv)

    if args.cmd == "embed":
        return cmd_embed(args)
    if args.cmd in ("transcribe", "diarize", "enrich", "pipeline", "ocr", "watch"):
        lock = single_instance(args.cmd)
        if lock is None:
            if not getattr(args, "quiet", False):
                print(f"{args.cmd}: another run holds the lock, skipping")
            return 0
        return {"transcribe": cmd_transcribe, "diarize": cmd_diarize, "enrich": cmd_enrich, "pipeline": cmd_pipeline,
                "ocr": cmd_ocr, "watch": cmd_watch}[args.cmd](args)
    if args.cmd == "guide":
        print(GUIDE)
        return 0
    if args.cmd == "serve":
        return cmd_serve(args)
    if not COLLS:
        raise SystemExit(f"{prog}: no collections configured; see config.example.toml")
    scope = [legacy_coll] if legacy else None
    if args.cmd == "update":
        return update(argparse.Namespace(quiet=args.quiet, collection=scope or listify(args.collection) or None,
                                         prune=getattr(args, "prune", False)))
    db = connect()
    if args.cmd == "show":
        d = show_row(db, args.id, legacy)
        if not d:
            print(f"no such id {args.id}", file=sys.stderr)
            return 1
        print(json.dumps(d, ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "collections":
        print(json.dumps(collections_info(db), ensure_ascii=False, indent=1))
        return 0
    if args.cmd in ("stats", "check"):
        names = scope or list(COLLS)
        if args.cmd == "stats":
            ok, d = stats(db, names, legacy)
            print(json.dumps(d, indent=1))
            return 0 if ok else 2
        res = [(n, *floor_check(db, COLLS[n])) for n in names]
        for n, ok, msg in res:
            line = msg if legacy else f"{n}: {msg}"
            print(line, file=sys.stdout if ok else sys.stderr)
        return 0 if all(ok for _, ok, _ in res) else 2
    try:
        o = opts_from({"q": getattr(args, "text", ""), "collection": scope or args.collection,
                       "limit": getattr(args, "limit", None), "n": getattr(args, "n", None),
                       "newest": getattr(args, "newest", False), "all": getattr(args, "all_terms", False),
                       "mode": getattr(args, "mode", None), "since": args.since, "until": args.until,
                       "from": args.sender, "source": args.source, "series": getattr(args, "series", None),
                       "author": getattr(args, "author", None)}, need_text=args.cmd == "query")
        o.limit = getattr(args, "limit", o.limit)  # the CLI is not capped
        o.n = getattr(args, "n", o.n)
        if args.cmd == "query":
            rows, notes = search(db, o)
        else:
            rows, notes = recent(db, o), []
    except ValueError as e:
        raise SystemExit(f"{prog}: {e}")
    for n in notes:
        print(f"({n})", file=sys.stderr)
    snip = "snip" if args.cmd == "query" else None
    recs = [record(r, snip, legacy) for r in rows]
    if args.json:
        print(json.dumps(recs, ensure_ascii=False, indent=1))
    else:
        print_records(recs, bool(snip), legacy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
