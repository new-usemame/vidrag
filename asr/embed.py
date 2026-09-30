#!/usr/bin/env python3
"""Local text embeddings and reranking for tiktok-rag (runs in the tiktok-rag-embed container; ONNX on CPU, no
network after the one-time model download into /models/fastembed).

  embed.py index   stdin {"docs": [{"id", "text"}]}           -> updates $STORE (only docs whose text changed)
  embed.py query   stdin {"q": "...", "k": 50}                -> stdout [[id, cosine], ...]
  embed.py rank    stdin {"qs": ["...", ...], "k": 50}        -> stdout [[[id, cosine], ...], ...]  (one load, many queries)
  embed.py rerank  stdin {"q": "...", "docs": [{"id", "text"}]} -> stdout [[id, score], ...] best first ($RERANKER)

EMBED_MODEL picks the embedder; each model family wants its own query/document prefixes (PREFIXES).
"""
import hashlib
import json
import os
import sys

import numpy as np

MODEL = os.environ.get("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
STORE = os.environ.get("STORE", "/data/vectors.npz")
RERANKER = os.environ.get("RERANKER", "")
THREADS = int(os.environ.get("THREADS", "4"))
MAX_CHARS = 6000  # ~1.5k tokens: caption + metadata + most of a transcript
RERANK_CHARS = int(os.environ.get("RERANK_CHARS", "2000"))  # rerankers are O(len): caption, place, first ~500 words
QWEN_TASK = "Given a search query about short social videos, retrieve the videos whose content matches the query"
# (document prefix, query prefix) — from each model card
PREFIXES = {
    "nomic": ("search_document: ", "search_query: "),
    "embeddinggemma": ("title: none | text: ", "task: search result | query: "),
    "qwen3": ("", f"Instruct: {QWEN_TASK}\nQuery: "),
}


def prefixes():
    for key, p in PREFIXES.items():
        if key in MODEL.lower():
            return p
    return "", ""


def model():
    from fastembed import TextEmbedding
    # MODEL_PATH: a flat copy of the model files. onnxruntime >=1.30 refuses external weights (model.onnx_data) that
    # the HF cache symlinks into another blobs/ directory ("External data path escapes model directory").
    path = os.environ.get("MODEL_PATH")
    return TextEmbedding(MODEL, cache_dir="/models/fastembed", threads=THREADS,
                         **({"specific_model_path": path} if path else {}))


def load():
    if not os.path.exists(STORE):
        return {}
    z = np.load(STORE, allow_pickle=False)
    if str(z["model"]) != MODEL:
        return {}
    return {i: (h, v) for i, h, v in zip(z["ids"], z["hashes"], z["vecs"])}


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def reranker():
    """-> fn(query, texts) -> scores. bge-reranker-v2-m3 and Qwen3-Reranker are not in this fastembed's registry."""
    if "qwen3-reranker" in RERANKER.lower():  # qwen3-embed names it n24q02m/Qwen3-Reranker-0.6B-ONNX[-YesNo]
        from qwen3_embed import TextCrossEncoder as Q3
        m = Q3(RERANKER, cache_dir="/models/fastembed", threads=THREADS)
        return lambda q, ts: list(m.rerank(q, ts))
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    if RERANKER == "BAAI/bge-reranker-v2-m3":
        from fastembed.common.model_description import ModelSource
        try:
            TextCrossEncoder.add_custom_model(model=RERANKER, model_file="onnx/model_quantized.onnx",
                                              sources=ModelSource(hf="onnx-community/bge-reranker-v2-m3-ONNX"))
        except ValueError:
            pass  # already registered
    m = TextCrossEncoder(RERANKER, cache_dir="/models/fastembed", threads=THREADS)
    return lambda q, ts: list(m.rerank(q, ts, batch_size=4))


def main():
    mode = sys.argv[1]
    req = json.load(sys.stdin)
    if mode == "rerank":
        docs = req["docs"]
        scores = reranker()(req["q"], [d["text"][:RERANK_CHARS] for d in docs])
        out = sorted(((d["id"], round(float(s), 4)) for d, s in zip(docs, scores)), key=lambda x: -x[1])
        print(json.dumps(out))
        return
    dp, qp = prefixes()
    m = model()
    if mode == "index":
        have = load()
        docs = [(d["id"], d["text"][:MAX_CHARS]) for d in req["docs"]]
        todo = [(i, t) for i, t in docs if have.get(i, (None,))[0] != hashlib.sha1(t.encode()).hexdigest()]
        if todo:
            vecs = list(m.embed([dp + t for _, t in todo], batch_size=int(os.environ.get("BATCH", "4"))))
            for (i, t), v in zip(todo, vecs):
                have[i] = (hashlib.sha1(t.encode()).hexdigest(), unit(v))
        keep = {i for i, _ in docs}
        ids = [i for i in have if i in keep]
        tmp = STORE + ".tmp.npz"
        dim = len(next(iter(have.values()))[1]) if have else 768
        np.savez(tmp, model=np.array(MODEL), ids=np.array(ids), hashes=np.array([have[i][0] for i in ids]),
                 vecs=np.stack([have[i][1] for i in ids]).astype(np.float32) if ids else np.zeros((0, dim), np.float32))
        os.replace(tmp, STORE)
        print(json.dumps({"embedded": len(todo), "total": len(ids)}))
        return
    have = load()
    qs = req["qs"] if mode == "rank" else [req["q"]]
    if not have:
        print(json.dumps([[] for _ in qs] if mode == "rank" else []))
        return
    ids = list(have)
    mat = np.stack([have[i][1] for i in ids])
    res = []
    for qv in m.embed([qp + q for q in qs]):
        sims = mat @ unit(qv)
        top = np.argsort(-sims)[: req.get("k", 50)]
        res.append([[ids[j], round(float(sims[j]), 4)] for j in top])
    print(json.dumps(res if mode == "rank" else res[0]))


if __name__ == "__main__":
    main()
