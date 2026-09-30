"""PP-OCRv4 (rapidocr_onnxruntime, current) vs PP-OCRv5 (rapidocr 3.x) on the same slides: text + time."""
import glob, json, sys, time
from rapidocr_onnxruntime import RapidOCR as V4
from rapidocr import RapidOCR as V5, ModelType
from rapidocr.utils.typings import OCRVersion
v4 = V4()
v5 = V5(params={"Det.ocr_version": OCRVersion.PPOCRV5, "Rec.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE, "Rec.model_type": ModelType.MOBILE})
v6 = V5()  # rapidocr 3.9 default = PP-OCRv6 small
imgs = sorted(glob.glob("/photos/*/*.jpg"))[:: max(1, len(glob.glob("/photos/*/*.jpg")) // 12)][:12]
for p in imgs:
    t0 = time.time(); r4, _ = v4(p); t4 = time.time() - t0
    t0 = time.time(); r5 = v5(p); t5 = time.time() - t0
    t0 = time.time(); r6 = v6(p); t6 = time.time() - t0
    a = " | ".join(x[1] for x in (r4 or []))
    print(json.dumps({"img": p[-30:], "t": [round(t4, 1), round(t5, 1), round(t6, 1)], "v4": a[:220],
                      "v5": " | ".join(r5.txts or ())[:220], "v6": " | ".join(r6.txts or ())[:220]}, ensure_ascii=False))
