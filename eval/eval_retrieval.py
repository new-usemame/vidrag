#!/usr/bin/env python3
"""Retrieval bake-off for tiktok-rag on the real index (runs on the server, drives the embed container).

  eval_retrieval.py embed <tag>       embed every video with EMBEDDERS[tag] into eval/vec-<tag>.npz (incremental)
  eval_retrieval.py run               keyword + each embedder + RRF fusions + rerankers -> eval/runs.json, eval/pool.json
  eval_retrieval.py score             eval/qrels.json + eval/runs.json -> nDCG@10 / P@5 / MRR / recall@20 per system

queries.json = ["query", ...] (your own; falls back to queries.example.json). Like qrels.json, pool.json and
runs.json it describes your archive, so it stays out of git.
qrels.json = {query: {video_id: 0|1|2}} (2 = exactly what was asked, 1 = relevant, 0 = not), judged on the pooled
top-10 of every system, so every system is scored on the same judged pool (unjudged = 0).
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import tiktok_rag as t  # noqa: E402

QUERIES = json.loads(next((HERE / f).read_text() for f in ("queries.json", "queries.example.json")
                           if (HERE / f).exists()))
FLAT = {"gemma": "/models/fastembed/flat/embeddinggemma-300m"}  # see embed.py MODEL_PATH
EMBEDDERS = {
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "gemma": "google/embeddinggemma-300m",
    "qwen3": "Qwen/Qwen3-Embedding-0.6B-Q",
}
RERANKERS = {
    "bge-m3": "BAAI/bge-reranker-v2-m3",
    "qwen3r": "n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo",
    "jina2": "jinaai/jina-reranker-v2-base-multilingual",
}
RERANK_DEPTH = 30
IMAGE = "tiktok-rag-embed:2"


def container(mode, payload, env, timeout=6 * 3600):
    cmd = ["docker", "run", "--rm", "-i", "--cpus=3", "--memory=4g", "--cpu-shares=128", "--user", "1000:1000",
           "-e", "HOME=/tmp", "-e", "THREADS=3", *sum((["-e", f"{k}={v}"] for k, v in env.items()), []),
           "-v", f"{t.MODELS / 'fastembed'}:/models/fastembed", "-v", f"{HERE}:/data",
           "-v", f"{HERE / 'embed.py'}:/asr/embed.py:ro", IMAGE, "python", "/asr/embed.py", mode]
    p = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise SystemExit(f"{mode} {env}: exit {p.returncode}: {p.stderr.strip()[-600:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def docs():
    db = t.connect()
    return {r["id"]: t.doc_text(r) for r in db.execute("select * from videos")}


def keyword(q):
    out = subprocess.run([str(HERE.parent / "tiktok-rag"), "query", q, "--mode", "keyword", "--source", "all",
                          "--json", "-n", "50"], capture_output=True, text=True).stdout
    return [r["id"] for r in json.loads(out or "[]")]


def rrf(*rankings, k=60):
    s = {}
    for rk in rankings:
        for pos, vid in enumerate(rk):
            s[vid] = s.get(vid, 0) + 1 / (k + pos)
    return sorted(s, key=lambda v: -s[v])


def cmd_embed(tag):
    d = docs()
    t0 = time.time()
    r = container("index", {"docs": [{"id": i, "text": x} for i, x in d.items()]},
                  {"EMBED_MODEL": EMBEDDERS[tag], "STORE": f"/data/vec-{tag}.npz", "BATCH": "2",
                   **({"MODEL_PATH": FLAT[tag]} if tag in FLAT else {})})
    print(json.dumps({"tag": tag, **r, "sec": round(time.time() - t0)}))


def cmd_run():
    d = docs()
    runs, timing = {}, {}
    runs["keyword"] = {q: keyword(q) for q in QUERIES}
    for tag, model in EMBEDDERS.items():
        if not (HERE / f"vec-{tag}.npz").exists():
            continue
        t0 = time.time()
        res = container("rank", {"qs": QUERIES, "k": 50}, {"EMBED_MODEL": model, "STORE": f"/data/vec-{tag}.npz",
                                                           **({"MODEL_PATH": FLAT[tag]} if tag in FLAT else {})})
        timing[f"sem-{tag}"] = round((time.time() - t0) / len(QUERIES), 2)
        runs[f"sem-{tag}"] = {q: [i for i, _ in r] for q, r in zip(QUERIES, res)}
        runs[f"fused-{tag}"] = {q: rrf(runs["keyword"][q], runs[f"sem-{tag}"][q]) for q in QUERIES}
    base = max((k for k in runs if k.startswith("fused-")), key=lambda k: k != "fused-nomic")  # rerank the newest fusion
    for tag, model in RERANKERS.items():
        name, t0 = f"{base}+{tag}", time.time()
        try:
            out = {}
            for q in QUERIES:
                cand = runs[base][q][:RERANK_DEPTH]
                res = container("rerank", {"q": q, "docs": [{"id": i, "text": d[i]} for i in cand]}, {"RERANKER": model})
                out[q] = [i for i, _ in res] + runs[base][q][RERANK_DEPTH:]
        except SystemExit as e:  # one broken reranker must not lose the others' results
            print(f"{name} FAILED: {str(e)[-300:]}", file=sys.stderr)
            continue
        runs[name] = out
        timing[name] = round((time.time() - t0) / len(QUERIES), 1)
    (HERE / "runs.json").write_text(json.dumps({"runs": runs, "sec_per_query": timing}))
    db = t.connect()
    pool = {}
    for q in QUERIES:
        ids = sorted({i for sysrun in runs.values() for i in sysrun[q][:10]})
        pool[q] = []
        for i in ids:
            r = db.execute("select * from videos where id=?", (i,)).fetchone()
            tr = r["transcript_whisper"] or r["transcript_tiktok"] or ""
            pool[q].append({"id": i, "caption": (r["caption"] or "")[:200], "place": r["location"] or "",
                            "keywords": (r["keywords"] or "")[:160], "ocr": (r["ocr"] or "")[:120],
                            "transcript": " ".join(tr.split())[:300]})
    (HERE / "pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=0))
    print(json.dumps({"systems": list(runs), "pooled_docs": sum(len(v) for v in pool.values()), "sec_per_query": timing}))


def ndcg(rank, rel, k=10):
    dcg = sum((2 ** rel.get(v, 0) - 1) / math.log2(i + 2) for i, v in enumerate(rank[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def cmd_score():
    runs = json.loads((HERE / "runs.json").read_text())
    qrels = json.loads((HERE / "qrels.json").read_text())
    rows = []
    for name, run in runs["runs"].items():
        qs = [q for q in QUERIES if any(v > 0 for v in qrels.get(q, {}).values())]
        n = len(qs)
        m = {"ndcg@10": sum(ndcg(run[q], qrels[q]) for q in qs) / n,
             "p@5": sum(sum(qrels[q].get(v, 0) > 0 for v in run[q][:5]) / 5 for q in qs) / n,
             "mrr": sum(next((1 / (i + 1) for i, v in enumerate(run[q]) if qrels[q].get(v, 0) > 0), 0) for q in qs) / n,
             "recall@20": sum(sum(qrels[q].get(v, 0) > 0 for v in run[q][:20]) / max(1, sum(g > 0 for g in qrels[q].values()))
                              for q in qs) / n}
        rows.append((name, m))
    rows.sort(key=lambda x: -x[1]["ndcg@10"])
    print(f"{'system':28} {'nDCG@10':>8} {'P@5':>6} {'MRR':>6} {'R@20':>6}  s/query  (n={n} queries)")
    for name, m in rows:
        print(f"{name:28} {m['ndcg@10']:8.3f} {m['p@5']:6.3f} {m['mrr']:6.3f} {m['recall@20']:6.3f}  "
              f"{runs['sec_per_query'].get(name, '')}")


if __name__ == "__main__":
    {"embed": lambda: cmd_embed(sys.argv[2]), "run": cmd_run, "score": cmd_score}[sys.argv[1]]()
