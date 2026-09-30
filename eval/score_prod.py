"""Score the production `tiktok-rag query` (the chat collection) (all its rules) on the judged queries; unjudged results count as 0."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.argv = sys.argv[:1]
from eval_retrieval import QUERIES, ndcg
Q = json.loads((HERE / "qrels.json").read_text())
qs = [q for q in QUERIES if any(v > 0 for v in Q[q].values())]
tot, p5, unj = 0, 0, 0
for q in qs:
    out = subprocess.run([str(HERE.parent / "tiktok-rag"), "query", q, "--source", "all", "--json", "-n", "20"],
                         capture_output=True, text=True).stdout
    ids = [r["id"] for r in json.loads(out)]
    tot += ndcg(ids, Q[q]); p5 += sum(Q[q].get(v, 0) > 0 for v in ids[:5]) / 5; unj += sum(v not in Q[q] for v in ids[:10])
print(f"production: nDCG@10 {tot/len(qs):.3f}  P@5 {p5/len(qs):.3f}  unjudged-in-top10 {unj}  (n={len(qs)})")
