#!/usr/bin/env python3
"""Qwen3-ASR-0.6B (ONNX, CPU) vs the Whisper large-v3-turbo transcripts already on disk, scored against TikTok's own
captions on the same videos (word error rate after normalisation). stdin: [{"id","path","dur","tiktok","whisper"}]."""
import json, re, subprocess, sys, time
sys.path.insert(0, "/qwen")
from onnx_inference import OnnxAsrPipeline  # noqa: E402
import numpy as np  # noqa: E402

UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
NUM = {**{w: i for i, w in enumerate(UNITS)}, **TENS}


def words_to_digits(ws):
    """'forty five percent' -> '45' so 45% and forty-five percent score the same (formatting is not an error)."""
    out, cur, big = [], None, 0
    for w in ws + ["<end>"]:
        if w in NUM:
            cur = (cur or 0) + NUM[w]
        elif w in ("hundred", "thousand", "million") and cur is not None:
            cur *= {"hundred": 100, "thousand": 1000, "million": 10 ** 6}[w]
            if w != "hundred":
                big, cur = big + cur, 0
        elif w == "and" and cur is not None:
            continue
        else:
            if cur is not None:
                out.append(str(big + cur)); cur, big = None, 0
            if w != "<end>":
                out.append(w)
    return out


def norm(t):
    t = re.sub(r"<[^>]+>|\d\d:\d\d[:.\d]* --> [\d:.]+|WEBVTT", " ", t or "")
    t = t.lower().replace("’", "'").replace("%", " ").replace("-", " ")
    ws = [w for w in re.sub(r"[^a-z0-9' ]+", " ", t).split() if w not in ("percent", "um", "uh")]
    return words_to_digits(ws)

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r)), len(r)

def load(path):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000",
                    "/tmp/a.wav"], check=True)
    return "/tmp/a.wav"

pipe = OnnxAsrPipeline("/qwen/onnx_models", num_threads=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
out = []
for v in json.load(sys.stdin):
    wav = load(v["path"])
    t0 = time.time()
    r = pipe.transcribe(wav, chunk_sec=int(__import__("os").environ.get("CHUNK", "20")))
    dt = time.time() - t0
    text = r["text"]
    ww, n = wer(v["tiktok"], v["whisper"])
    qw, _ = wer(v["tiktok"], text)
    out.append({"id": v["id"], "dur": round(r["timing"]["audio_duration_s"], 1), "lang": r.get("language"), "sec": round(dt, 1), "ref_words": n,
                "wer_whisper": round(ww, 3), "wer_qwen": round(qw, 3), "qwen": text, "whisper": v["whisper"], "tiktok": v["tiktok"]})
    print(json.dumps(out[-1]), file=sys.stderr, flush=True)
print(json.dumps(out))
