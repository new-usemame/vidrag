import json, random, sqlite3, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import vidrag as t  # noqa: E402  (the DB location and the chat collection's folders come from its config)
chat = next(c for c in t.COLLS.values() if c.plugin)
root = str(chat.paths[0].parent)  # the folder that holds the series folders; mounted as /media in the bench
db = sqlite3.connect(t.DB); db.row_factory = sqlite3.Row
rows = [dict(r) for r in db.execute("""select id, path, duration, transcript_subs, transcript_whisper from videos
        where collection=? and length(transcript_subs) > 200 and length(transcript_whisper) > 200
        and duration between 15 and 120 and not is_photo""", (chat.name,))]
random.Random(7).shuffle(rows)
print(json.dumps([{"id": r["id"], "path": r["path"].replace(root, "/media"), "dur": r["duration"],
                   "tiktok": r["transcript_subs"], "whisper": r["transcript_whisper"]} for r in rows[:20]]))
