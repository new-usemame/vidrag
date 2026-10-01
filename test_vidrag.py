"""Fixture tests for vidrag (stdlib unittest; run: python3 -m unittest -v test_vidrag)."""
import argparse
import importlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

ns = argparse.Namespace


def info(vid, desc, uploader="someone", date="20260801", ts=None, **extra):
    return {"id": vid, "description": desc, "title": desc[:40], "uploader": uploader, "channel": uploader.title(),
            "upload_date": date, "timestamp": ts, "track": "original sound", "artist": "x", "duration": 12, **extra}


class Base(unittest.TestCase):
    """A TikTok chat archive (TikTok DMs + TikTok Saved) and, when asked, a YouTube tree of channel folders."""
    youtube = False

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.nas = self.tmp / "nas"
        (self.nas / "TikTok DMs/Season 2026").mkdir(parents=True)
        (self.nas / "TikTok Saved/Season 2026").mkdir(parents=True)
        self.chat = self.tmp / "chat_index.json"
        os.environ.pop("GEMINI_API_KEY", None)  # never the real key: gemini_post is always faked here
        os.environ.pop("VIDRAG_API_TOKEN", None)
        vids = [("1", "Best dumplings in Chinatown #nyc #food", "20260101", False),
                ("2", "pigeons eat seeds in places", "20260301", False),
                ("3", "cozy soup recipe #cooking", "20260201", True),
                ("4", "hidden bakery in Brooklyn", "20260920", False)]
        for vid, desc, date, _ in vids:
            self.write("TikTok DMs", vid, desc, date)
        (self.nas / "TikTok DMs/.photos/3").mkdir(parents=True)
        self.write("TikTok Saved", "9", "saved one-off about nyc", "20260915")
        self.set_chat([(v, m) for v, _, _, m in vids])
        mt = self.nas / "TikTok DMs/.metube"
        mt.mkdir()
        mt.joinpath("completed.json").write_text(json.dumps({"items": [
            {"info": {"id": "4", "status": "finished", "timestamp": 1790500000 * 10**9}},
            {"info": {"id": "1", "status": "finished", "timestamp": 1790000000 * 10**9}}]}))
        if self.youtube:
            yt = self.tmp / "youtube"
            self.write_yt(yt, "Veritasium", "aaaaaaaaaa1", "Why the sky is blue", "Rayleigh scattering explained.",
                          "20110202", chapters=[{"title": "Sunsets"}], categories=["Education"],
                          subs="WEBVTT\n\n00:00.000 --> 00:02.000\nlight from the sun scatters\n")
            self.write_yt(yt, "Veritasium", "aaaaaaaaaa2", "The dumplings paradox", "Folding math.", "20200505")
            self.write_yt(yt, "Doug DeMuro", "bbbbbbbbbb1", "Quirks and features of a minivan", "A family car.",
                          "20190101")
            (yt / "TikTok DMs").mkdir()  # excluded: another collection's folder inside the same tree
        self.configure()

    def configure(self, contact=None, nas=None, extra=""):
        nas = nas or self.nas
        cfg = f'''
home = "{self.tmp}/state"
data = "{self.tmp}"
[gemini]
key_file = "{self.tmp}/no-such-gemini.env"
[api]
token_file = "{self.tmp}/no-such-token"
[collections.tiktok]
paths = ["{nas}/TikTok DMs", "{nas}/TikTok Saved"]
plugin = "tiktok"
steps = ["enrich", "ocr", "diarize", "transcribe", "watch"]
floor_min = 3
floor_slack = 0
[collections.tiktok.tiktok]
chat_index = "{self.chat}"
backfill_index_size = 3
max_missing = 1
{f'contact = "{contact}"' if contact else ""}
'''
        if self.youtube:
            cfg += f'''
[collections.youtube]
paths = ["{self.tmp}/youtube"]
exclude = ["TikTok DMs"]
kind = "YouTube"
floor_min = 2
'''
        (self.tmp / "config.toml").write_text(cfg + extra)
        os.environ["VIDRAG_CONFIG"] = str(self.tmp / "config.toml")
        for m in ("vidrag", "vidrag_tiktok"):
            sys.modules.pop(m, None)
        self.t = importlib.import_module("vidrag")
        self.t.OFF, self.t.GEMINI_OFF = self.tmp / "OFF", self.tmp / "GEMINI_OFF"  # never the host's real switches

    def write(self, series, vid, desc, date):
        p = self.nas / series / "Season 2026" / f"s2026.e{vid} - {desc[:10]}"
        Path(str(p) + ".info.json").write_text(json.dumps(info(vid, desc, date=date)))
        Path(str(p) + ".mp4").write_bytes(b"")

    def write_yt(self, root, channel, vid, title, desc, date, subs=None, **extra):
        d = root / channel / f"Season {date[:4]}"
        d.mkdir(parents=True, exist_ok=True)
        base = d / f"s{date[:4]}.e{date[4:]}01 - {title}"
        meta = {"id": vid, "title": title, "description": desc, "uploader": channel, "channel": channel,
                "upload_date": date, "duration": 193, "webpage_url": f"https://www.youtube.com/watch?v={vid}", **extra}
        Path(str(base) + ".info.json").write_text(json.dumps(meta))
        Path(str(base) + ".mkv").write_bytes(b"")
        if subs:
            Path(str(base) + ".en.vtt").write_text(subs)

    def set_chat(self, cards):
        self.chat.write_text(json.dumps({"complete": True, "updated": "x", "videos": len(cards),
                                         "cards": [[v, m] for v, m in cards]}))

    def run_cli(self, *argv, compat="tiktok"):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self.t.main(list(argv), compat=compat)
        return rc, buf.getvalue()

    def vidrag(self, *argv):
        return self.run_cli(*argv, compat=None)


class TikTokCompatTest(Base):
    """The old tiktok-rag command, unchanged: these are its original tests, run through the compat mode."""

    def test_update_fields_and_share_dates(self):
        rc, _ = self.run_cli("update", "-q")
        self.assertEqual(rc, 0)
        rc, out = self.run_cli("recent", "10", "--json")
        rows = json.loads(out)
        self.assertEqual([r["id"] for r in rows], ["4", "3", "2", "1"])  # chat order, newest first
        by = {r["id"]: r for r in rows}
        self.assertEqual(by["3"]["sender"], "me")
        self.assertEqual(by["1"]["sender"], "them")
        self.assertEqual(by["3"]["is_photo"], 1)
        self.assertEqual(by["4"]["share_precision"], "poll")  # chat pos 3 >= backfill size 3
        self.assertTrue(by["4"]["shared"].startswith("2026-09-27"))
        # lower bound is the running max of upload dates along the chat: video 3 (Feb) after 2 (Mar) -> >= Mar
        self.assertEqual(by["3"]["shared"], ">=2026-03-01")
        self.assertIn("#nyc", by["1"]["caption"])
        self.assertEqual(list(by["1"])[:3], ["id", "source", "sender"])  # the old field order
        self.assertTrue(by["1"]["nas_path"].endswith(".mp4"))
        self.assertEqual(by["1"]["url"], "https://www.tiktok.com/@someone/video/1")
        rc, out = self.run_cli("recent", "5", "--source", "saved", "--json")
        self.assertEqual([r["id"] for r in json.loads(out)], ["9"])

    def test_contact_name_comes_from_config(self):
        self.configure(contact="Sam")
        self.run_cli("update", "-q")
        rows = json.loads(self.run_cli("query", "dumplings", "--from", "sam", "--mode", "keyword", "--json")[1])
        self.assertEqual([(r["id"], r["sender"]) for r in rows], [("1", "sam")])
        self.assertEqual(json.loads(self.run_cli("query", "dumplings", "--from", "them", "--mode", "keyword", "--json")[1])[0]["id"], "1")
        self.assertEqual(json.loads(self.run_cli("stats")[1])["from_sam"], 3)  # videos 1, 2 and 4

    def test_favorites_labelled_and_dated_by_favorite_time(self):
        self.write("TikTok Saved", "8", "favorited noodle spot nyc", "20250101")
        fav_ts = 1780000000  # 2026-05-28
        (self.tmp / "favs_index.json").write_text(json.dumps({"complete": True, "items": {
            "8": {"fav_ts": fav_ts}, "1": {"fav_ts": fav_ts}}}))  # "1" is also a DM: it must stay a DM
        self.assertEqual(self.run_cli("update", "-q")[0], 0)
        rows = {r["id"]: r for r in json.loads(self.run_cli("recent", "10", "--source", "all", "--json")[1])}
        self.assertEqual(rows["8"]["source"], "fav")
        self.assertEqual(rows["8"]["shared"], ">=2026-05-28")
        self.assertEqual(rows["9"]["source"], "saved")  # a one-off link is not a favorite
        self.assertEqual((rows["1"]["source"], rows["1"]["sender"]), ("dm", "them"))
        fav = json.loads(self.run_cli("recent", "10", "--source", "fav", "--json")[1])
        self.assertEqual([r["id"] for r in fav], ["8"])
        self.assertIn("my favorites", self.run_cli("query", "noodle", "--source", "fav")[1])
        # un-favorited: back to saved on the next run (the label is recomputed, not sticky)
        (self.tmp / "favs_index.json").write_text(json.dumps({"complete": True, "items": {}}))
        self.run_cli("update", "-q")
        rows = {r["id"]: r for r in json.loads(self.run_cli("recent", "10", "--source", "all", "--json")[1])}
        self.assertEqual(rows["8"]["source"], "saved")

    def test_query_ranks_rare_concept_and_filters(self):
        self.run_cli("update", "-q")
        rc, out = self.run_cli("query", "nyc places to eat", "--json")
        ids = [r["id"] for r in json.loads(out)]
        # nyc + food (1; 4 via the brooklyn/bakery aliases) both beat the pigeon video with places + eat
        self.assertEqual(set(ids[:2]), {"1", "4"})
        self.assertEqual(ids[2], "2")
        rc, out = self.run_cli("query", "soup", "--from", "them", "--json")
        self.assertEqual(json.loads(out), [])
        rc, out = self.run_cli("query", "nyc", "--all", "--since", "2026-09", "--json")
        self.assertEqual([r["id"] for r in json.loads(out)], ["4"])

    def test_specific_word_is_not_widened_to_its_alias_group(self):
        self.write("TikTok DMs", "5", "Pizza margherita #pizza #nyc", "20260310")
        self.write("TikTok DMs", "6", "late night pizza in Ohio", "20260311")
        self.set_chat([("1", True), ("2", True), ("3", False), ("4", True), ("5", True), ("6", True)])
        self.run_cli("update", "-q")
        ids = [r["id"] for r in json.loads(self.run_cli("query", "pizza nyc", "--mode", "keyword", "--json")[1])]
        # pizza AND nyc first; then pizza alone; NYC food that is not pizza no longer ties ("pizza" != "any food")
        self.assertEqual(ids[:2], ["5", "6"])
        # "new york" is one concept (the nyc group), not "new" + "york" outvoting "pizza"
        groups = [g for _, g in self.t.fts_terms("best pizza in new york")]
        self.assertEqual(len(groups), 3)  # best · pizza · new york
        self.assertTrue(any('"new york"' in g and '"nyc"' in g for g in groups))
        self.assertFalse(any('"york"' in g for g in groups))
        rows = json.loads(self.run_cli("query", "pizza", "--mode", "keyword", "--json")[1])
        self.assertEqual({r["match"] for r in rows}, {"words"})
        ids = [r["id"] for r in json.loads(self.run_cli("query", "brooklyn", "--mode", "keyword", "--json")[1])]
        self.assertEqual(ids, ["4"])  # brooklyn is not all of NYC
        ids = [r["id"] for r in json.loads(self.run_cli("query", "nyc food", "--mode", "keyword", "--json")[1])]
        self.assertTrue({"1", "4", "5"} <= set(ids))  # generic words still expand to their siblings

    def test_floor_fails_on_empty_and_shrink(self):
        self.assertEqual(self.run_cli("update", "-q")[0], 0)
        # shrink: two DM files vanish -> below high-water AND 2 chat videos missing
        for f in (self.nas / "TikTok DMs/Season 2026").glob("s2026.e2 *"):
            f.unlink()
        for f in (self.nas / "TikTok DMs/Season 2026").glob("s2026.e3 *"):
            f.unlink()
        self.assertEqual(self.run_cli("update", "-q")[0], 2)
        # rolled back: the last good index still answers, and the floor keeps failing until the files return
        self.assertEqual(len(json.loads(self.run_cli("recent", "10", "--json")[1])), 4)
        self.assertEqual(self.run_cli("update", "-q")[0], 2)

    def test_floor_fails_on_unmounted_nas(self):
        self.assertEqual(self.run_cli("update", "-q")[0], 0)
        self.configure(nas=self.tmp / "empty-mountpoint")
        self.assertEqual(self.run_cli("update", "-q")[0], 2)
        self.assertEqual(len(json.loads(self.run_cli("recent", "10", "--json")[1])), 4)

    def test_empty_index_fails_check(self):
        self.assertEqual(self.run_cli("check")[0], 2)

    def test_transcript_sidecar_is_indexed_incrementally(self):
        self.run_cli("update", "-q")
        tdir = self.nas / "TikTok DMs/.transcripts"
        tdir.mkdir()
        (tdir / "2.whisper.txt").write_text("we went to Katz's Deli on the Lower East Side")
        (tdir / "2.tiktok.vtt").write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nkatz deli\n")
        self.assertEqual(self.run_cli("update", "-q")[0], 0)
        rc, out = self.run_cli("query", "katz", "--json")
        self.assertEqual([r["id"] for r in json.loads(out)], ["2"])
        self.assertTrue(json.loads(out)[0]["has_transcript"])
        # the old row's words are gone from the keyword index after the sidecar changes again
        (tdir / "2.whisper.txt").write_text("a different day")
        (tdir / "2.tiktok.vtt").unlink()
        os.utime(tdir / "2.whisper.txt", (2e9, 2e9))
        self.run_cli("update", "-q")
        self.assertEqual(json.loads(self.run_cli("query", "katz", "--mode", "keyword", "--json")[1]), [])

    def test_semantic_only_hits_are_marked(self):
        self.run_cli("update", "-q")
        self.t.embed_query = lambda text, k=200, ids=None: [["2", 0.8], ["1", 0.5]]  # 2 = pigeons: no "dumplings"
        rows = {r["id"]: r["match"] for r in json.loads(self.run_cli("query", "dumplings", "--json")[1])}
        self.assertEqual(rows, {"1": "words", "2": "meaning"})
        self.assertIn("related by meaning only", self.run_cli("query", "dumplings")[1])

    def test_rare_exact_match_beats_semantic_list(self):
        self.run_cli("update", "-q")
        # semantic leg ranks 20 other videos first and never returns "4"; keyword finds "brooklyn" only in "4"
        self.t.embed_query = lambda text, k=200, ids=None: [[i, 0.9] for i in ("1", "2", "3", "9")]
        ids = [r["id"] for r in json.loads(self.run_cli("query", "brooklyn", "--source", "all", "--json")[1])]
        self.assertEqual(ids[0], "4")
        self.assertEqual(json.loads(self.run_cli("query", "brooklyn", "--source", "all", "--json")[1])[0]["match"], "words")
        # a multi-concept description is not a lookup: the semantic order stands
        self.t.embed_query = lambda text, k=200, ids=None: [[i, 0.9] for i in ("2", "1", "3", "9")]
        ids = [r["id"] for r in json.loads(self.run_cli("query", "hidden bakery", "--source", "all", "--json")[1])]
        self.assertEqual(ids[0], "2")

    def test_gemini_description_is_indexed_and_searchable(self):
        self.run_cli("update", "-q")
        tdir = self.nas / "TikTok DMs/.transcripts"
        tdir.mkdir()
        (tdir / "2.gemini.json").write_text(json.dumps({"description": "A golden retriever catches a frisbee in a park.\nTags: dog"}))
        self.assertEqual(self.run_cli("update", "-q")[0], 0)
        rows = json.loads(self.run_cli("query", "frisbee", "--mode", "keyword", "--json")[1])
        self.assertEqual([r["id"] for r in rows], ["2"])
        self.assertEqual(rows[0]["seen"], "A golden retriever catches a frisbee in a park.")  # summary line only
        self.assertIn("seen: A golden retriever catches a frisbee", self.run_cli("query", "frisbee", "--mode", "keyword")[1])
        db = self.t.connect()
        self.assertIn("frisbee", self.t.doc_text(db.execute("select * from videos where id='2'").fetchone()))
        (tdir / "3.gemini.json").write_text(json.dumps({"error": "no JSON: SAFETY", "fetched": 1}))  # errors index nothing
        self.run_cli("update", "-q")
        self.assertEqual(db.execute("select visual from videos where id='3'").fetchone()[0], "")

    def test_watch_writes_sidecars_and_pauses_on_billing_error(self):
        self.run_cli("update", "-q")
        t = self.t
        self.assertEqual(t.pending_counts(t.connect())["watch"], 0)  # no key: the cloud step does not exist
        os.environ["GEMINI_API_KEY"] = "fake"
        self.addCleanup(os.environ.pop, "GEMINI_API_KEY", None)
        self.assertEqual(t.pending_counts(t.connect())["watch"], 5)
        sent = []
        t.watch_clip = lambda mp4, out: b"clip"
        def post(body, key):
            sent.append(body)
            return {"candidates": [{"content": {"parts": [{"text": json.dumps(
                {"summary": "A cat knocks a cup off a table.", "things": ["cat", "cup"], "tags": ["cat", "funny"]})}]}}],
                "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 60}}
        t.gemini_post = post
        self.assertEqual(t.cmd_watch(ns(max=2, quiet=True)), 0)
        side = json.loads((self.nas / "TikTok DMs/.transcripts/4.gemini.json").read_text())  # newest chat video first
        self.assertEqual(side["description"], "A cat knocks a cup off a table.\nThings: cat, cup\nTags: cat, funny")
        self.assertEqual(sent[0]["generationConfig"]["mediaResolution"], "MEDIA_RESOLUTION_LOW")
        prompt = sent[0]["contents"][0]["parts"][1]["text"]
        self.assertIn("hidden bakery in Brooklyn", prompt)  # caption as context
        self.assertTrue(prompt.startswith("You are indexing a TikTok video for a private search engine."))
        self.assertEqual(t.pending_counts(t.connect())["watch"], 3)
        # out of credits: pause the whole step (no sidecar, no retry loop), and the pipeline stops counting it
        def broke(body, key):
            raise t.GeminiHold("HTTP 402: Your prepayment credits are depleted.")
        t.gemini_post = broke
        self.assertEqual(t.cmd_watch(ns(max=0, quiet=True)), 3)
        self.assertIn("402", t.gemini_hold())
        self.assertFalse(t.gemini_ready())
        self.assertEqual(t.pending_counts(t.connect())["watch"], 0)
        self.assertEqual(len(list((self.nas / "TikTok DMs/.transcripts").glob("*.gemini.json"))), 2)
        stats = json.loads(self.run_cli("stats")[1])["gemini"]
        self.assertEqual((stats["sidecars"], stats["input_tokens"], stats["output_tokens"]), (2, 1800, 120))

    def test_pipeline_stops_when_only_a_held_watch_run_remains(self):
        self.run_cli("update", "-q")
        t = self.t
        for vid in ("1", "2", "3", "4", "9"):  # every local step done: only Gemini is pending
            td = self.nas / ("TikTok Saved" if vid == "9" else "TikTok DMs") / ".transcripts"
            td.mkdir(exist_ok=True)
            for suf in (".page.json", ".turns.json", ".whisper.txt", ".ocr.txt"):
                (td / f"{vid}{suf}").write_text("{}" if suf.endswith("json") else "")
        os.environ["GEMINI_API_KEY"] = "fake"
        self.addCleanup(os.environ.pop, "GEMINI_API_KEY", None)
        self.assertEqual(t.pending_counts(t.connect())["watch"], 5)
        held = t.single_instance("watch")  # a backlog `watch` run owns the Gemini work
        self.addCleanup(held.close)
        calls = []
        t.update = lambda args: calls.append(1) or 0
        t.cmd_embed = lambda args: 0
        self.assertEqual(t.cmd_pipeline(ns(delay=0, quiet=True)), 0)
        self.assertEqual(len(calls), 1)  # one re-index, then out: no busy loop

    def test_speaker_transcript_join(self):
        w = {"transcription": [{"offsets": {"from": 0, "to": 3000}, "text": " Welcome back."},
                               {"offsets": {"from": 3000, "to": 6000}, "text": " Thanks for having me."},
                               {"offsets": {"from": 6500, "to": 7000}, "text": " Sure."},  # gap: keeps S2
                               {"offsets": {"from": 7000, "to": 9000}, "text": " So, bananas."}]}
        turns = {"n_speakers": 2, "turns": [[0, 3.1, 0], [3.1, 6.2, 1], [7.2, 9, 0]]}
        text, n = self.t.speaker_transcript(w, turns)
        self.assertEqual(n, 2)
        self.assertEqual(text, "S1: Welcome back.\nS2: Thanks for having me. Sure.\nS1: So, bananas.")
        text, n = self.t.speaker_transcript(w, {"n_speakers": 1, "turns": [[0, 9, 0]]})
        self.assertEqual(text, "Welcome back. Thanks for having me. Sure. So, bananas.")


class MultiCollectionTest(Base):
    """A YouTube tree next to the chat: generic metadata, scoping, and the old command seeing only its chat."""
    youtube = True

    def test_youtube_metadata_is_indexed(self):
        self.assertEqual(self.vidrag("update", "-q")[0], 0)
        rows = json.loads(self.vidrag("query", "rayleigh", "--collection", "youtube", "--mode", "keyword", "--json")[1])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["collection"], r["series"], r["title"]), ("youtube", "Veritasium", "Why the sky is blue"))
        self.assertEqual(r["url"], "https://www.youtube.com/watch?v=aaaaaaaaaa1")
        self.assertTrue(r["path"].endswith(".mkv"))  # whatever container sits beside the .info.json
        self.assertEqual((r["date"], r["date_precision"]), ("2011-02-02", "upload"))
        self.assertTrue(r["has_transcript"])  # the .en.vtt beside the video
        for word in ("sunsets", "scatters", "education", "sky"):  # chapters, subtitles, categories, title
            got = json.loads(self.vidrag("query", word, "-c", "youtube", "--mode", "keyword", "--json")[1])
            self.assertEqual([x["id"] for x in got], ["aaaaaaaaaa1"], word)
        cols = {c["name"]: c for c in json.loads(self.vidrag("collections")[1])}
        self.assertEqual(cols["youtube"]["series"], {"Veritasium": 2, "Doug DeMuro": 1})  # TikTok DMs excluded
        self.assertFalse(cols["youtube"]["gemini"])
        self.assertTrue(cols["tiktok"]["gemini"])

    def test_scope_and_filters(self):
        self.vidrag("update", "-q")
        both = json.loads(self.vidrag("query", "dumplings", "--mode", "keyword", "--json")[1])
        self.assertEqual({r["collection"] for r in both}, {"tiktok", "youtube"})  # default: every collection
        yt = json.loads(self.vidrag("query", "dumplings", "-c", "youtube", "--mode", "keyword", "--json")[1])
        self.assertEqual([r["id"] for r in yt], ["aaaaaaaaaa2"])
        got = json.loads(self.vidrag("recent", "5", "-c", "youtube", "--series", "doug demuro", "--json")[1])
        self.assertEqual([r["id"] for r in got], ["bbbbbbbbbb1"])
        got = json.loads(self.vidrag("recent", "5", "-c", "youtube", "--since", "2015", "--json")[1])
        self.assertEqual([r["id"] for r in got], ["aaaaaaaaaa2", "bbbbbbbbbb1"])  # by date, newest first
        got = json.loads(self.vidrag("recent", "5", "-c", "youtube", "--author", "Veritasium", "--json")[1])
        self.assertEqual({r["id"] for r in got}, {"aaaaaaaaaa1", "aaaaaaaaaa2"})
        # the old command still sees only its chat
        old = json.loads(self.run_cli("query", "dumplings", "--source", "all", "--mode", "keyword", "--json")[1])
        self.assertEqual([r["id"] for r in old], ["1"])
        self.assertEqual(json.loads(self.run_cli("stats")[1])["total"], 5)
        with self.assertRaises(SystemExit):
            self.vidrag("query", "x", "-c", "nope")

    def test_semantic_leg_is_asked_only_about_the_searched_collections(self):
        self.vidrag("update", "-q")
        asked = []
        def fake(text, k=200, ids=None):
            asked.append(set(ids))
            return [[i, 0.5] for i in ids]
        self.t.embed_query = fake
        self.vidrag("query", "anything", "-c", "tiktok", "--json")
        self.assertEqual(asked[-1], {"1", "2", "3", "4", "9"})  # YouTube vectors cannot crowd the chat out
        self.run_cli("query", "anything", "--json")
        self.assertEqual(asked[-1], {"1", "2", "3", "4", "9"})

    def test_only_listed_collections_get_steps(self):
        self.vidrag("update", "-q")
        os.environ["GEMINI_API_KEY"] = "fake"
        self.addCleanup(os.environ.pop, "GEMINI_API_KEY", None)
        t = self.t
        pend = t.pending_counts(t.connect())
        self.assertEqual((pend["watch"], pend["diarize"], pend["transcribe"]), (5, 5, 5))  # the chat only
        self.assertEqual({p[4] for p in t.watch_pending(t.connect())}, {"TikTok"})

    def test_generic_floor_and_collection_isolation(self):
        self.assertEqual(self.vidrag("update", "-q")[0], 0)
        self.assertEqual(self.vidrag("check")[0], 0)
        for f in (self.tmp / "youtube/Veritasium").rglob("*.info.json"):
            f.unlink()
        rc = self.vidrag("update", "-q")[0]
        self.assertEqual(rc, 2)  # 1 row < floor 2, and below the high-water mark
        db = self.t.connect()
        self.assertEqual(db.execute("select count(*) from videos where collection='youtube'").fetchone()[0], 3)
        self.assertEqual(self.vidrag("update", "-q", "-c", "tiktok")[0], 0)  # the chat is unaffected

    def test_due_collections_rescan_in_pipeline(self):
        self.configure(extra='[collections.news]\npaths = ["' + str(self.tmp / "youtube") + '"]\nupdate_every = 60\n'
                             'exclude = ["TikTok DMs"]\nfloor_min = 0\n')
        # "news" reads the same tree as "youtube": the first collection to index an id keeps it
        self.vidrag("update", "-q", "-c", "tiktok,youtube")
        t = self.t
        self.assertEqual(t.due_collections(t.connect()), ["news"])
        ran = []
        for step in ("cmd_embed", "cmd_ocr", "cmd_enrich", "cmd_diarize", "cmd_transcribe", "cmd_watch"):
            setattr(t, step, lambda args, step=step: ran.append(step) or 0)  # steps that do nothing
        t.cmd_pipeline(ns(delay=0, quiet=True))
        self.assertEqual(t.due_collections(t.connect()), [])
        db = t.connect()
        self.assertEqual(db.execute("select count(*) from videos where collection='news'").fetchone()[0], 0)
        # the chat's steps made no progress: one pass, then the pipeline stops instead of spinning
        self.assertEqual(ran.count("cmd_diarize"), 1)


class RobustnessTest(Base):
    """Review findings 2026-09-30: each of these once broke or could break a live index."""
    youtube = True

    def test_first_seen_survives_a_schema_rebuild(self):
        self.vidrag("update", "-q")
        db = self.t.connect()
        db.execute("update videos set first_seen=1000 where id='9'")
        db.execute("update meta set v='4' where k='schema'")  # an older schema: the next connect rebuilds
        db.commit()
        db.close()
        self.vidrag("update", "-q")
        db = self.t.connect()
        self.assertEqual(db.execute("select first_seen from videos where id='9'").fetchone()[0], 1000)
        self.assertEqual(db.execute("select date_ts from videos where id='9'").fetchone()[0], 1000)  # no MeTube time

    def test_subtitles_added_later_are_picked_up(self):
        self.vidrag("update", "-q")
        d = self.tmp / "youtube/Doug DeMuro/Season 2019"
        (d / "s2019.e010101 - Quirks and features of a minivan.en.srt").write_text(
            "1\n00:00:01,000 --> 00:00:02,000\nthe cupholders fold away\n")
        self.vidrag("update", "-q")
        got = json.loads(self.vidrag("query", "cupholders", "-c", "youtube", "--mode", "keyword", "--json")[1])
        self.assertEqual([r["id"] for r in got], ["bbbbbbbbbb1"])

    def test_playlist_info_files_are_not_videos(self):
        (self.tmp / "youtube/Veritasium/Season 2011/Veritasium - Videos.info.json").write_text(
            json.dumps({"id": "UCplaylist", "_type": "playlist", "title": "Videos"}))
        self.vidrag("update", "-q")
        db = self.t.connect()
        self.assertIsNone(db.execute("select id from videos where id='UCplaylist'").fetchone())

    def test_update_prunes_unconfigured_collections_only_when_asked(self):
        self.vidrag("update", "-q")
        self.youtube = False
        self.configure()  # youtube dropped from the config
        self.vidrag("update", "-q")
        db = self.t.connect()
        self.assertEqual(db.execute("select count(*) from videos where collection='youtube'").fetchone()[0], 3)
        self.vidrag("update", "-q", "--prune")
        self.assertEqual(db.execute("select count(*) from videos where collection='youtube'").fetchone()[0], 0)
        self.assertEqual(db.execute("select count(*) from fts").fetchone()[0], 5)

    def test_updated_stamp_stays_put_after_a_failed_update(self):
        self.vidrag("update", "-q")
        db = self.t.connect()
        db.execute("update meta set v='1' where k='updated'")
        db.commit()
        for f in (self.tmp / "youtube").rglob("*.info.json"):
            f.unlink()
        self.assertEqual(self.vidrag("update", "-q")[0], 2)
        self.assertEqual(self.t.meta_get(self.t.connect(), "updated"), "1")

    def test_enrich_fetches_never_fetched_pages_before_weekly_retries(self):
        self.vidrag("update", "-q")
        tdir = self.nas / "TikTok DMs/.transcripts"
        tdir.mkdir()
        for vid in ("1", "2", "3", "4"):  # the chat's newest videos: old errors, due for their weekly retry
            (tdir / f"{vid}.page.json").write_text(json.dumps({"error": "gone", "fetched": 0}))
        plugin = self.t.COLLS["tiktok"].plugin
        todo = [v for _, v in plugin.enrich_todo(self.t.connect())]
        self.assertEqual(todo[0], "9")  # the never-fetched Saved video, then the retries
        self.assertEqual(plugin.enrich_pending(self.t.connect()), 1)

    def test_concurrent_update_of_one_collection_waits_then_skips(self):
        held = self.t.single_instance("update-youtube")
        self.addCleanup(held.close)
        rc = self.t.update(ns(quiet=True, collection=["youtube", "tiktok"], wait=0))
        self.assertEqual(rc, 0)
        db = self.t.connect()
        self.assertEqual(db.execute("select count(*) from videos where collection='youtube'").fetchone()[0], 0)
        self.assertEqual(db.execute("select count(*) from videos where collection='tiktok'").fetchone()[0], 5)

    def test_text_output_with_float_durations(self):
        self.write_yt(self.tmp / "youtube", "Primer", "cccccccccc1", "Simulating an epidemic", "Agents.", "20200101",
                      duration=1203.5)
        self.vidrag("update", "-q")
        out = self.vidrag("query", "epidemic", "-c", "youtube", "--mode", "keyword")[1]
        self.assertIn("20:03", out)


class ApiTest(Base):
    youtube = True

    def setUp(self):
        super().setUp()
        (self.tmp / "no-such-token").write_text("s3cret\n")
        self.configure()
        self.vidrag("update", "-q")
        t = self.t

        class Fake:
            def query(self, q, k=50, ids=None):
                return [[i, 0.5] for i in (ids or [])][:k]
        t.SEARCHER = Fake()
        from http.server import ThreadingHTTPServer
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), t.make_handler(t.api_token()))
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.addCleanup(self.srv.shutdown)

    def call(self, path, body=None, token="s3cret"):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_search_recent_show_collections(self):
        code, d = self.call("/v1/search?q=dumplings&mode=keyword")
        self.assertEqual(code, 200)
        self.assertEqual({r["id"] for r in d["results"]}, {"1", "aaaaaaaaaa2"})
        code, d = self.call("/v1/search", {"q": "dumplings", "collection": ["youtube"], "limit": 5})
        self.assertEqual([r["id"] for r in d["results"]][0], "aaaaaaaaaa2")
        self.assertTrue(all(r["collection"] == "youtube" for r in d["results"]))
        code, d = self.call("/v1/recent?n=2&collection=tiktok&source=dm")
        self.assertEqual([r["id"] for r in d["results"]], ["4", "3"])  # chat order
        code, d = self.call("/v1/videos/aaaaaaaaaa1")
        self.assertIn("light from the sun scatters", d["transcript_subs"])
        self.assertEqual(self.call("/v1/videos/nope")[0], 404)
        code, d = self.call("/v1/collections")
        self.assertEqual({c["name"] for c in d["collections"]}, {"tiktok", "youtube"})
        self.assertEqual(self.call("/v1/stats")[0], 200)

    def test_auth_errors_and_open_routes(self):
        self.assertEqual(self.call("/v1/search?q=x", token="wrong")[0], 401)
        self.assertEqual(self.call("/v1/search?q=x", token="")[0], 401)
        self.assertEqual(self.call("/v1/search?q=x&collection=nope")[0], 400)
        self.assertEqual(self.call("/v1/search")[0], 400)  # no q
        self.assertEqual(self.call("/v1/search?q=x&since=yesterday")[0], 400)
        code, d = self.call("/v1/health", token="")
        self.assertEqual((code, d["ok"]), (200, True))
        code, d = self.call("/openapi.json", token="")
        self.assertIn("/v1/search", d["paths"])
        with urllib.request.urlopen(self.base + "/", timeout=5) as r:
            self.assertIn(b"vidrag", r.read())

    def test_body_is_not_read_without_a_token_and_is_capped(self):
        import http.client
        from urllib.parse import urlsplit

        def declared(token):  # announce a 5 MB body, send none: the server must answer from the headers alone
            u = urlsplit(self.base)
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
            c.putrequest("POST", "/v1/search")
            c.putheader("Authorization", f"Bearer {token}")
            c.putheader("Content-Length", str(5 << 20))
            c.endheaders()
            code = c.getresponse().status
            c.close()
            return code
        self.assertEqual(declared("wrong"), 401)
        self.assertEqual(declared("s3cret"), 400)

    def test_embed_takes_a_large_collections_id_allowlist(self):
        seen = {}

        class Stub:
            def query(self, q, k, ids):
                seen["n"] = len(ids)
                return [[ids[0], 0.9]]
        self.t.SEARCHER = Stub()
        ids = [f"{i:019d}" for i in range(20000)]  # bigger than a 6k-video YouTube tree
        code, d = self.call("/v1/embed", {"q": "dog", "k": 5, "ids": ids})
        self.assertEqual(code, 200)
        self.assertEqual(seen["n"], 20000)

    def test_semantic_failure_degrades_to_keyword(self):
        class Broken:
            def query(self, *a, **k):
                raise ValueError("corrupt vectors")
        self.t.SEARCHER = Broken()
        code, d = self.call("/v1/search?q=dumplings")
        self.assertEqual(code, 200)
        self.assertIn("semantic search unavailable — keyword results only", d["notes"])
        self.assertTrue(d["results"])

    def test_refuses_lan_without_token(self):
        (self.tmp / "no-such-token").unlink()
        self.configure()
        with self.assertRaises(SystemExit):
            self.t.cmd_serve(ns(listen="10.9.9.9:8790", threads=1))


if __name__ == "__main__":
    unittest.main()
