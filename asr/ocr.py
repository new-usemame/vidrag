#!/usr/bin/env python3
"""OCR for photo-post slides (tiktok-rag-asr container). stdin: [{"id", "dir"}]; writes /out/<id>.ocr.txt.
ENGINE=rapidocr (PaddleOCR ONNX; default) or tesseract. One line of text per slide, in slide order.
rapidocr 3.x (PP-OCRv6, image tiktok-rag-asr:3) when installed, else rapidocr_onnxruntime (PP-OCRv4): v6 stops gluing
words together ("helloworld", "TheCafe") at the same speed (2026-09-29 side-by-side, eval/ocr_bench.py)."""
import glob
import json
import os
import subprocess
import sys
import time

ENGINE = os.environ.get("ENGINE", "rapidocr")


def slides(d):
    return sorted(glob.glob(os.path.join(d, "*.jp*g")) + glob.glob(os.path.join(d, "*.png")) +
                  glob.glob(os.path.join(d, "*.webp")))


def main():
    jobs = json.load(sys.stdin)
    if ENGINE == "rapidocr":
        try:
            from rapidocr import RapidOCR

            v6 = RapidOCR()

            def eng(img):
                r = v6(img)
                return list(zip(r.boxes if r.boxes is not None else [], r.txts or (), r.scores or ())), None
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR
            eng = RapidOCR()
    for j in jobs:
        t = time.time()
        lines = []
        try:
            for img in slides(j["dir"]):
                if ENGINE == "rapidocr":
                    res, _ = eng(img)
                    txt = " ".join(r[1] for r in (res or []) if r[2] >= 0.6)
                else:
                    txt = subprocess.run(["tesseract", img, "-", "--psm", "11"], capture_output=True,
                                         text=True).stdout
                    txt = " ".join(w for w in txt.split() if len(w) > 1 or w.isalnum())
                lines.append(txt.strip())
            out = f"/out/{j['id']}.ocr.txt"
            with open(out + ".tmp", "w") as f:
                f.write("\n".join(lines) + "\n")
            os.replace(out + ".tmp", out)
            print(f"{j['id']} ok {len(lines)} slides {sum(map(len, lines))} chars {time.time() - t:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{j['id']} FAIL {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
