"""Fixture tests for tiktok_rag (stdlib unittest; run: python3 -m unittest -v test_tiktok_rag)."""
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def info(vid, desc, uploader="someone", date="20260801", ts=None):
    return {"id": vid, "description": desc, "title": desc[:40], "uploader": uploader, "channel": uploader.title(),
            "upload_date": date, "timestamp": ts, "track": "original sound", "artist": "x", "duration": 12}


class RagTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.nas = self.tmp / "nas"
        (self.nas / "TikTok DMs/Season 2026").mkdir(parents=True)
        (self.nas / "TikTok Saved/Season 2026").mkdir(parents=True)
        self.chat = self.tmp / "chat_index.json"
        os.environ.update({"TIKTOK_RAG_NAS": str(self.nas), "TIKTOK_RAG_CHAT_INDEX": str(self.chat),
                           "TIKTOK_RAG_DB": str(self.tmp / "index.db"), "TIKTOK_RAG_FLOOR_MIN_DM": "3",
                           "TIKTOK_RAG_FLOOR_SLACK": "0", "TIKTOK_RAG_MAX_MISSING": "1",
                           "TIKTOK_RAG_BACKFILL_INDEX_SIZE": "3", "TIKTOK_RAG_CONFIG": str(self.tmp / "no-config.env"),
                           "TIKTOK_RAG_GEMINI_KEY_FILE": str(self.tmp / "no-such-gemini.env")})
        os.environ.pop("GEMINI_API_KEY", None)  # never the real key: gemini_post is always faked here
        os.environ.pop("TIKTOK_RAG_CONTACT", None)  # the host's real config must not leak in
        sys.modules.pop("tiktok_rag", None)
        self.t = importlib.import_module("tiktok_rag")
        self.t.OFF, self.t.GEMINI_OFF = self.tmp / "OFF", self.tmp / "GEMINI_OFF"  # never the host's real switches
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

    def write(self, series, vid, desc, date):
        p = self.nas / series / "Season 2026" / f"s2026.e{vid} - {desc[:10]}"
        Path(str(p) + ".info.json").write_text(json.dumps(info(vid, desc, date=date)))
        Path(str(p) + ".mp4").write_bytes(b"")

    def set_chat(self, cards):
        self.chat.write_text(json.dumps({"complete": True, "updated": "x", "videos": len(cards),
                                         "cards": [[v, m] for v, m in cards]}))

    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self.t.main(list(argv))
        return rc, buf.getvalue()

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
        rc, out = self.run_cli("recent", "5", "--source", "saved", "--json")
        self.assertEqual([r["id"] for r in json.loads(out)], ["9"])

    def test_contact_name_comes_from_config(self):
        cfg = self.tmp / "config.env"
        cfg.write_text("# site settings\nTIKTOK_RAG_CONTACT=Sam\n")
        os.environ["TIKTOK_RAG_CONFIG"] = str(cfg)
        self.addCleanup(os.environ.pop, "TIKTOK_RAG_CONTACT", None)
        sys.modules.pop("tiktok_rag", None)
        self.t = importlib.import_module("tiktok_rag")
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
        os.environ["TIKTOK_RAG_FLOOR_MIN_DM"] = "3"
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
        os.environ["TIKTOK_RAG_NAS"] = str(self.tmp / "empty-mountpoint")
        sys.modules.pop("tiktok_rag", None)
        self.t = importlib.import_module("tiktok_rag")
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

    def test_semantic_only_hits_are_marked(self):
        self.run_cli("update", "-q")
        self.t.embed_query = lambda text, k=200: [["2", 0.8], ["1", 0.5]]  # 2 = pigeons: no "dumplings" in it
        rows = {r["id"]: r["match"] for r in json.loads(self.run_cli("query", "dumplings", "--json")[1])}
        self.assertEqual(rows, {"1": "words", "2": "meaning"})
        self.assertIn("related by meaning only", self.run_cli("query", "dumplings")[1])

    def test_rare_exact_match_beats_semantic_list(self):
        self.run_cli("update", "-q")
        # semantic leg ranks 20 other videos first and never returns "4"; keyword finds "brooklyn" only in "4"
        self.t.embed_query = lambda text, k=200: [[i, 0.9] for i in ("1", "2", "3", "9")]
        ids = [r["id"] for r in json.loads(self.run_cli("query", "brooklyn", "--source", "all", "--json")[1])]
        self.assertEqual(ids[0], "4")
        self.assertEqual(json.loads(self.run_cli("query", "brooklyn", "--source", "all", "--json")[1])[0]["match"], "words")
        # a multi-concept description is not a lookup: the semantic order stands
        self.t.embed_query = lambda text, k=200: [[i, 0.9] for i in ("2", "1", "3", "9")]
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
        ns = __import__("argparse").Namespace
        self.assertEqual(t.pending_counts(t.connect())[4], 0)  # no key: the cloud step does not exist
        os.environ["GEMINI_API_KEY"] = "fake"
        self.addCleanup(os.environ.pop, "GEMINI_API_KEY", None)
        self.assertEqual(t.pending_counts(t.connect())[4], 5)
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
        self.assertIn("hidden bakery in Brooklyn", sent[0]["contents"][0]["parts"][1]["text"])  # caption as context
        self.assertEqual(t.pending_counts(t.connect())[4], 3)
        # out of credits: pause the whole step (no sidecar, no retry loop), and the pipeline stops counting it
        def broke(body, key):
            raise t.GeminiHold("HTTP 402: Your prepayment credits are depleted.")
        t.gemini_post = broke
        self.assertEqual(t.cmd_watch(ns(max=0, quiet=True)), 3)
        self.assertIn("402", t.gemini_hold())
        self.assertFalse(t.gemini_ready())
        self.assertEqual(t.pending_counts(t.connect())[4], 0)
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
        self.assertEqual(t.pending_counts(t.connect())[4], 5)
        held = t.single_instance("watch")  # a backlog `watch` run owns the Gemini work
        self.addCleanup(held.close)
        calls = []
        t.update = lambda args: calls.append(1) or 0
        t.cmd_embed = lambda args: 0
        self.assertEqual(t.cmd_pipeline(__import__("argparse").Namespace(delay=0, quiet=True)), 0)
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


if __name__ == "__main__":
    unittest.main()
