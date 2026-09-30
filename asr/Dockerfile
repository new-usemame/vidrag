FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg bzip2 tesseract-ocr libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir sherpa-onnx numpy soundfile rapidocr_onnxruntime rapidocr onnxruntime \
 && python -c "from rapidocr import RapidOCR; RapidOCR()"  # bake the PP-OCRv6 models in: the container runs as non-root
