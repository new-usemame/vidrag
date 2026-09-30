"""List production top-10 results missing from qrels.json, with the fields a grader needs (eval/unjudged.json)."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.argv = sys.argv[:1]
from eval_retrieval import QUERIES, t  # noqa: E402
Q = json.loads((HERE / "qrels.json").read_text())
db = t.connect()
out = {}
for q in [q for q in QUERIES if any(v > 0 for v in Q[q].values())]:
    res = subprocess.run([str(HERE.parent / "tiktok-rag"), "query", q, "--source", "all", "--json", "-n", "10"],
                         capture_output=True, text=True)
    if not res.stdout.strip(): print(q, "FAILED", res.stderr[-400:], file=sys.stderr); continue
    res = res.stdout
    for i in [r["id"] for r in json.loads(res)][:10]:
        if i in Q[q]:
            continue
        r = db.execute("select * from videos where id=?", (i,)).fetchone()
        tr = r["transcript_whisper"] or r["transcript_tiktok"] or ""
        out.setdefault(q, []).append({"id": i, "caption": (r["caption"] or "")[:200], "place": r["location"] or "",
                                      "visual": (r["visual"] or "")[:400], "ocr": (r["ocr"] or "")[:120],
                                      "transcript": " ".join(tr.split())[:300]})
(HERE / "unjudged.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(sum(map(len, out.values())), "unjudged across", len(out), "queries")
