import json, random, sqlite3, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tiktok_rag as t  # noqa: E402  (NAS path and DB location come from its config)
db = sqlite3.connect(t.DB); db.row_factory = sqlite3.Row
rows = [dict(r) for r in db.execute("""select id, nas_path, duration, transcript_tiktok, transcript_whisper from videos
        where length(transcript_tiktok) > 200 and length(transcript_whisper) > 200 and duration between 15 and 120
        and not is_photo""")]
random.Random(7).shuffle(rows)
print(json.dumps([{"id": r["id"], "path": r["nas_path"].replace(str(t.NAS), "/media"), "dur": r["duration"],
                   "tiktok": r["transcript_tiktok"], "whisper": r["transcript_whisper"]} for r in rows[:20]]))
