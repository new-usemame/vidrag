"""vidrag plugin for a TikTok DM archive: chat order and sender, favorites, share dates, the public post page.

Config (under [collections.<name>] with plugin = "tiktok"):
  [collections.<name>.tiktok]
  contact = "sam"                     # the other person: `--from sam` (or `--from them`)
  chat_index = "~/dm-poller/state/chat_index.json"   # {cards: [[id, isMyself], ...]}, oldest first
  favs_index = "<data>/favs_index.json"              # favs/favs.py: {"items": {id: {fav_ts, ...}}}
  sources = {"TikTok DMs" = "dm", "TikTok Saved" = "saved"}   # series folder -> source label
  backfill_index_size = 0             # chat size when the backfill began (see share dates)
  max_missing = 15                    # chat videos allowed to lack a file before the floor fails

Share dates: the chat index has order but no timestamps. A video that entered the chat after the backfill (chat
position >= backfill_index_size) was downloaded within one poll of being shared, so its MeTube timestamp is the share
time ("poll" precision). Older videos get a lower bound: the latest upload date among it and every video before it in
the chat ("est" precision, shown as ">="). Favorites are dated by favorite time (also a lower bound).
"""
import json
import re
import sys
import time


class Plugin:
    chat_ordered = True

    def __init__(self, core, coll, cfg):
        self.core, self.coll = core, coll
        self.contact = str(cfg.get("contact", "them")).strip().lower() or "them"
        self.chat_index = core.expand(cfg.get("chat_index", "~/dm-poller/state/chat_index.json"))
        self.favs_index = core.expand(cfg.get("favs_index", core.DATA / "favs_index.json"))
        self.sources = dict(cfg.get("sources", {"TikTok DMs": "dm", "TikTok Saved": "saved"}))
        self.backfill = int(cfg.get("backfill_index_size", 0))
        self.max_missing = int(cfg.get("max_missing", 15))
        coll.titles = False  # a TikTok "title" is just the caption cut short
        coll.kind = coll.kind or "TikTok"

    # ------------------------------------------------------------ index hooks

    def parse(self, info, d):
        info["chapters"] = info["categories"] = ""

    def source_for(self, series_dir):
        return self.sources.get(series_dir.name, series_dir.name)

    def url(self, r):
        return f"https://www.tiktok.com/@{r['author'] or '_'}/video/{r['id']}"

    def load_chat(self):
        """-> (complete, [(id, mine)]) oldest first."""
        d = json.loads(self.chat_index.read_text())
        return bool(d.get("complete")), [(str(c[0]), c[1]) for c in d.get("cards", [])]

    def load_favs(self):
        try:
            return json.loads(self.favs_index.read_text()).get("items", {})
        except (OSError, ValueError):
            return {}

    def metube_times(self):
        """id -> download unix time, from each series' MeTube history (ns timestamps)."""
        out = {}
        for sdir in self.coll.series_dirs():
            try:
                items = json.loads((sdir / ".metube" / "completed.json").read_text()).get("items", [])
            except (OSError, ValueError):
                continue
            for it in items:
                info = it.get("info") or {}
                if info.get("status") == "finished" and info.get("id") and info.get("timestamp"):
                    out[str(info["id"])] = int(info["timestamp"]) // 1_000_000_000
        return out

    def after_scan(self, db):
        """Source, chat membership, order and sender are re-applied to every row each run (cheap; the chat moves)."""
        name = self.coll.name
        complete, cards = self.load_chat()
        self.cards = cards
        db.execute("update videos set chat_pos=null, sender=null where collection=?", (name,))
        for folder, label in self.sources.items():
            db.execute("update videos set source=? where collection=? and series=?", (label, name, folder))
        for pos, (vid, is_me) in enumerate(cards):
            db.execute("update videos set chat_pos=?, source='dm', sender=? where id=? and collection=?",
                       (pos, None if is_me is None else ("me" if is_me else self.contact), vid, name))
        favs = self.load_favs()
        for vid in favs:
            db.execute("update videos set source='fav' where id=? and chat_pos is null and collection=?", (vid, name))
        self.assign_share_dates(db, self.metube_times(), favs)
        self.core.meta_set(db, "chat_complete", int(complete))

    def assign_share_dates(self, db, dl, favs):
        rows = db.execute("select id, chat_pos, upload_ts, first_seen from videos where collection=? order by chat_pos",
                          (self.coll.name,)).fetchall()
        running = 0
        for r in rows:
            if r["chat_pos"] is None and (favs.get(r["id"]) or {}).get("fav_ts"):
                ts, prec = favs[r["id"]]["fav_ts"], "est"
            elif r["chat_pos"] is None:
                ts, prec = dl.get(r["id"]) or r["first_seen"], "download"
            else:
                running = max(running, r["upload_ts"] or 0)
                if r["chat_pos"] >= self.backfill and r["id"] in dl:
                    ts, prec = min(dl[r["id"]], r["first_seen"] or dl[r["id"]]), "poll"
                else:
                    ts, prec = running or None, "est"
            db.execute("update videos set date_ts=?, date_precision=? where id=?", (ts, prec, r["id"]))

    def floor(self, db):
        """Fail when DM rows drop below floor_min, below the last high-water mark minus slack, or when more chat
        videos lack a row than max_missing."""
        cards = getattr(self, "cards", None)
        if cards is None:
            _, cards = self.load_chat()
        name = self.coll.name
        n_dm = db.execute("select count(*) from videos where source='dm' and collection=?", (name,)).fetchone()[0]
        hw = int(self.core.meta_get(db, "high_water_dm", 0))
        have = {r[0] for r in db.execute("select id from videos where source='dm' and collection=?", (name,))}
        missing = [vid for vid, _ in cards if vid not in have]
        problems = []
        if n_dm < self.coll.floor_min:
            problems.append(f"DM rows {n_dm} < floor {self.coll.floor_min}")
        if hw and n_dm < hw - self.coll.floor_slack:
            problems.append(f"DM rows {n_dm} shrank below high-water {hw} - {self.coll.floor_slack}")
        if len(missing) > self.max_missing:
            problems.append(f"{len(missing)} chat videos have no indexed file (allowance {self.max_missing})")
        msg = f"dm={n_dm} high_water={hw} chat={len(cards)} chat_missing={len(missing)}"
        if problems:
            return False, "FLOOR FAIL: " + "; ".join(problems) + f" ({msg})"
        return True, "floor ok (" + msg + ")"

    def bump_high_water(self, db):
        n_dm = db.execute("select count(*) from videos where source='dm' and collection=?",
                          (self.coll.name,)).fetchone()[0]
        self.core.meta_set(db, "high_water_dm", max(n_dm, int(self.core.meta_get(db, "high_water_dm", 0))))

    # ------------------------------------------------------------ the old tiktok-rag `stats` shape

    def legacy_stats(self, db):
        q = lambda s, *a: db.execute(s.replace("WHERE", "where collection=? and"), (self.coll.name, *a)).fetchone()[0]
        return {
            "total": db.execute("select count(*) from videos where collection=?", (self.coll.name,)).fetchone()[0],
            "dm": q("select count(*) from videos WHERE source='dm'"),
            "saved": q("select count(*) from videos WHERE source='saved'"),
            "favorites": q("select count(*) from videos WHERE source='fav'"),
            f"from_{self.contact}": q("select count(*) from videos WHERE sender=?", self.contact),
            "from_me": q("select count(*) from videos WHERE sender='me'"),
            "photo_posts": q("select count(*) from videos WHERE is_photo=1"),
            "with_tiktok_transcript": q("select count(*) from videos WHERE transcript_subs<>''"),
            "with_whisper_transcript": q("select count(*) from videos WHERE transcript_whisper<>''"),
            "with_speaker_turns": q("select count(*) from videos WHERE n_speakers is not null"),
            "multi_speaker": q("select count(*) from videos WHERE n_speakers >= 2"),
            "with_ocr_or_stickers": q("select count(*) from videos WHERE coalesce(ocr,'')<>''"),
            "watched_by_gemini": q("select count(*) from videos WHERE coalesce(visual,'')<>''"),
        }

    def legacy_stats_tail(self, db):
        q = lambda s: db.execute(s.replace("WHERE", "where collection=? and"), (self.coll.name,)).fetchone()[0]
        return {
            "with_location": q("select count(*) from videos WHERE coalesce(location,'')<>''"),
            "with_keywords": q("select count(*) from videos WHERE coalesce(keywords,'')<>''"),
            "share_precision": dict(db.execute("select date_precision, count(*) from videos where collection=? "
                                               "group by 1", (self.coll.name,)).fetchall()),
        }

    # ------------------------------------------------------------ enrich step: the public post page

    def page_item(self, vid):
        """Public post page (no login) -> itemStruct dict. Raises on an unavailable post."""
        html = self.core.http_get(f"https://www.tiktok.com/@_/video/{vid}")
        m = re.search(rb'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', html, re.S)
        if not m:
            raise RuntimeError("no rehydration data (challenge page?)")
        detail = json.loads(m.group(1))["__DEFAULT_SCOPE__"].get("webapp.video-detail") or {}
        if detail.get("statusCode") not in (0, None):
            raise RuntimeError(f"post unavailable: statusCode {detail.get('statusCode')} {detail.get('statusMsg', '')}")
        return detail["itemInfo"]["itemStruct"]

    @staticmethod
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
            "subtitles": [{"lang": x.get("LanguageCodeName"), "format": x.get("Format"), "source": x.get("Source")}
                          for x in subs],
        }, subs

    @staticmethod
    def pick_subtitle(subs):
        """Prefer English; else the video's own-language ASR; else anything webvtt."""
        vtt = [x for x in subs if (x.get("Format") or "").lower() == "webvtt" and x.get("Url")]
        for pref in (lambda x: (x.get("LanguageCodeName") or "").startswith("eng"),
                     lambda x: x.get("Source") == "ASR", lambda x: True):
            for x in vtt:
                if pref(x):
                    return x
        return None

    def enrich_one(self, tdir, vid):
        """Fetch one post page; write <id>.page.json (always) and <id>.tiktok.vtt (when TikTok has captions)."""
        aw = self.core.atomic_write
        try:
            it = self.page_item(vid)
        except Exception as e:  # noqa: BLE001 — recorded, retried weekly
            aw(tdir / f"{vid}.page.json", json.dumps({"id": vid, "fetched": int(time.time()), "error": str(e)[:300]}))
            return "error", str(e)[:120]
        rec, subs = self.page_record(it)
        sub = self.pick_subtitle(subs)
        if sub:
            try:
                vtt = self.core.http_get(sub["Url"], referer="https://www.tiktok.com/")
                if vtt.lstrip().startswith(b"WEBVTT"):
                    aw(tdir / f"{vid}.tiktok.vtt", vtt)
                    rec["subtitle_saved"] = sub.get("LanguageCodeName")
                else:
                    rec["subtitle_error"] = "not webvtt"
            except Exception as e:  # noqa: BLE001
                rec["subtitle_error"] = str(e)[:200]
        aw(tdir / f"{vid}.page.json", json.dumps(rec, ensure_ascii=False))
        return "ok", (rec.get("poi") or {}).get("name") or ""

    def enrich_todo(self, db):
        """Never-fetched pages first (they are what the pipeline counts as pending), then the weekly error retries."""
        now, todo, retry = time.time(), [], []
        for r in db.execute("select id, info_path from videos where collection=? "
                            "order by chat_pos is null, chat_pos desc, date_ts desc", (self.coll.name,)):
            tdir = self.core.tdir_of(r["info_path"])
            pj = tdir / f"{r['id']}.page.json"
            if pj.exists():
                try:
                    old = json.loads(pj.read_text())
                except ValueError:
                    old = {"error": "corrupt", "fetched": 0}
                if "error" not in old or now - old.get("fetched", 0) < self.core.RETRY_ERROR_AFTER:
                    continue
                retry.append((tdir, r["id"]))
                continue
            todo.append((tdir, r["id"]))
        return todo + retry

    def enrich_pending(self, db):
        """Only never-fetched pages count toward the pipeline's pending work (weekly error retries do not)."""
        return sum(not (self.core.tdir_of(r["info_path"]) / f"{r['id']}.page.json").exists()
                   for r in db.execute("select id, info_path from videos where collection=?", (self.coll.name,)))

    def enrich(self, db, args):
        todo = self.enrich_todo(db)
        todo = todo[: args.max] if args.max else todo
        if not args.quiet:
            print(f"enrich: {len(todo)} to fetch (delay {args.delay}s)")
        ok = err = streak = 0
        for i, (tdir, vid) in enumerate(todo):
            if i:
                time.sleep(args.delay)
            status, note = self.enrich_one(tdir, vid)
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
