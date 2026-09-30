#!/usr/bin/env python3
"""Batch speech-to-text + speaker turns for vidrag, run inside the vidrag-asr container.

One process loads the models once and handles many videos:
  silero VAD  -> speech segments (music/silence dropped, so nothing is hallucinated over a soundtrack)
  whisper-tiny spoken-language ID on the first ~30 s of speech
  Parakeet TDT 0.6B v3 (25 European languages) per speech segment, with token timestamps
  pyannote-3.0 segmentation + TitaNet embeddings -> speaker turns, joined to words by time overlap
A language Parakeet does not cover is reported as needs_whisper (the caller runs whisper.cpp for it).

stdin: JSON list of {"id", "path"}; writes /out/<id>.asr.json (atomic); one status line per video on stdout.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import sherpa_onnx

M = os.environ.get("MODELS", "/models")
THREADS = int(os.environ.get("THREADS", "4"))
DIARIZE = os.environ.get("DIARIZE", "1") == "1"
MODE = os.environ.get("MODE", "parakeet")  # parakeet | diarize (speaker turns only, for joining to Whisper text)
EMBED = os.environ.get("EMBED", "nemo_en_titanet_small.onnx")
MIN_SHARE = float(os.environ.get("MIN_SHARE", "0.08"))
SR = 16000
PARAKEET_LANGS = set("bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split())


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def models():
    pk = f"{M}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{pk}/encoder.int8.onnx", decoder=f"{pk}/decoder.int8.onnx", joiner=f"{pk}/joiner.int8.onnx",
        tokens=f"{pk}/tokens.txt", num_threads=THREADS, model_type="nemo_transducer")
    wt = f"{M}/sherpa-onnx-whisper-tiny"
    lid = sherpa_onnx.SpokenLanguageIdentification(sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
            encoder=f"{wt}/tiny-encoder.int8.onnx", decoder=f"{wt}/tiny-decoder.int8.onnx"),
        num_threads=THREADS))
    sd = None
    if DIARIZE:
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=f"{M}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"), num_threads=THREADS),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=f"{M}/{EMBED}", num_threads=THREADS),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=-1, threshold=float(os.environ.get("DIAR_THRESHOLD", "0.9"))),
            min_duration_on=0.3, min_duration_off=0.5)
        sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return rec, lid, sd


def vad_segments(samples):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = f"{M}/silero_vad.onnx"
    cfg.silero_vad.min_silence_duration = 0.4
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.max_speech_duration = 25
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max(60, len(samples) / SR + 5))
    w = cfg.silero_vad.window_size
    out = []
    for i in range(0, len(samples), w):
        vad.accept_waveform(samples[i:i + w])
        while not vad.empty():
            out.append((vad.front.start, np.array(vad.front.samples, dtype=np.float32)))
            vad.pop()
    vad.flush()
    while not vad.empty():
        out.append((vad.front.start, np.array(vad.front.samples, dtype=np.float32)))
        vad.pop()
    return out


def words_from(result, offset):
    """Group Parakeet sentencepiece tokens into words with start times (seconds, absolute)."""
    words = []
    for tok, ts in zip(result.tokens, result.timestamps):
        t = offset + ts
        if not words or tok.startswith((" ", "▁")):
            words.append([tok.replace("▁", " ").strip(), t, t])
        else:
            words[-1][0] += tok
            words[-1][2] = t
    return [w for w in words if w[0]]


def speaker_at(turns, t0, t1):
    best, spk = 0.0, None
    for s, e, k in turns:
        ov = min(e, t1 + 0.2) - max(s, t0)
        if ov > best:
            best, spk = ov, k
    return spk


def process(rec, lid, sd, path):
    samples = load_audio(path)
    dur = len(samples) / SR
    segs = vad_segments(samples)
    speech = sum(len(s) for _, s in segs) / SR
    out = {"engine": "parakeet-tdt-0.6b-v3-int8", "duration": round(dur, 1), "speech_s": round(speech, 1),
           "segments": [], "text": "", "n_speakers": 0}
    if speech < 0.5:
        out["lang"] = None
        return out
    probe = np.concatenate([s for _, s in segs])[: 30 * SR]
    st = lid.create_stream()
    st.accept_waveform(SR, probe)
    out["lang"] = lid.compute(st)
    if out["lang"] not in PARAKEET_LANGS:
        out["needs_whisper"] = True
        return out
    words, texts = [], []
    for start, s in segs:
        stream = rec.create_stream()
        stream.accept_waveform(SR, s)
        rec.decode_stream(stream)
        r = stream.result
        if r.text.strip():
            texts.append(r.text.strip())
            words += words_from(r, start / SR)
    out["text"] = " ".join(texts)
    turns = []
    if sd is not None and words:
        res = sd.process(samples).sort_by_start_time()
        turns = [(x.start, x.end, x.speaker) for x in res]
    # speaker turns: consecutive words with the same speaker become one segment
    cur = None
    for w, t0, t1 in words:
        spk = speaker_at(turns, t0, t1) if turns else 0
        if cur and (spk == cur["speaker"] or spk is None) and t0 - cur["end"] < 1.5:
            cur["text"] += " " + w
            cur["end"] = round(t1, 2)
        else:
            cur = {"start": round(t0, 2), "end": round(t1, 2), "speaker": spk if spk is not None else 0, "text": w}
            out["segments"].append(cur)
    # renumber speakers by first appearance
    order = {}
    for sg in out["segments"]:
        sg["speaker"] = order.setdefault(sg["speaker"], len(order))
    out["n_speakers"] = len(order)
    return out


def diarize_only(sd, path):
    samples = load_audio(path)
    res = sd.process(samples).sort_by_start_time()
    raw = [[round(x.start, 2), round(x.end, 2), x.speaker] for x in res]
    # On TikTok audio (music, jump cuts) a single narrator also yields 0-5 % "speakers"; drop any speaker
    # below MIN_SHARE of the talk time (benchmarked 2026-09-28 on 6 clips: keeps a 51/49 podcast and a
    # 4-voice promo, collapses solo narrators to 1). Their stretches fall to the neighbouring speaker at join.
    talk = {}
    for s, e, k in raw:
        talk[k] = talk.get(k, 0) + e - s
    total = sum(talk.values()) or 1
    keep = {k for k, v in talk.items() if v / total >= MIN_SHARE}
    order = {}
    turns = []
    for s, e, k in raw:
        if k in keep:
            turns.append([s, e, order.setdefault(k, len(order))])
    return {"engine": f"pyannote-3.0+{EMBED}", "threshold": float(os.environ.get("DIAR_THRESHOLD", "0.9")),
            "min_share": MIN_SHARE, "duration": round(len(samples) / SR, 1), "turns": turns,
            "n_speakers": len(order), "raw_speakers": len(talk)}


def main():
    jobs = json.load(sys.stdin)
    t = time.time()
    if MODE == "diarize":
        global DIARIZE
        DIARIZE = True
        _, _, sd = models()
        for j in jobs:
            t = time.time()
            try:
                r = diarize_only(sd, j["path"])
                r["elapsed_s"] = round(time.time() - t, 1)
                with open(f"/out/{j['id']}.turns.json.tmp", "w") as f:
                    json.dump(r, f)
                os.replace(f"/out/{j['id']}.turns.json.tmp", f"/out/{j['id']}.turns.json")
                print(f"{j['id']} spk={r['n_speakers']} turns={len(r['turns'])} {r['elapsed_s']}s", flush=True)
            except Exception as e:  # noqa: BLE001 — marker so the pipeline does not retry it every pass
                with open(f"/out/{j['id']}.turns.json", "w") as f:
                    json.dump({"error": f"{type(e).__name__}: {str(e)[:200]}", "turns": [], "n_speakers": None}, f)
                print(f"{j['id']} FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)
        return
    rec, lid, sd = models()
    print(f"models loaded {time.time() - t:.1f}s", flush=True)
    for j in jobs:
        t = time.time()
        try:
            r = process(rec, lid, sd, j["path"])
            r["elapsed_s"] = round(time.time() - t, 1)
            tmp = f"/out/{j['id']}.asr.json.tmp"
            with open(tmp, "w") as f:
                json.dump(r, f, ensure_ascii=False)
            os.replace(tmp, f"/out/{j['id']}.asr.json")
            print(f"{j['id']} ok lang={r.get('lang')} dur={r['duration']} speech={r['speech_s']} "
                  f"spk={r['n_speakers']} {r['elapsed_s']}s{' NEEDS_WHISPER' if r.get('needs_whisper') else ''}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{j['id']} FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
