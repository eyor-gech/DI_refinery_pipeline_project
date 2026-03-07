FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y tesseract-ocr libgl1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./ 
COPY src ./src
COPY run_pipeline.py ./run_pipeline.py
COPY data ./data

# Install runtime dependencies (matching pyproject)
RUN pip install --no-cache-dir langdetect pdfplumber pillow pydantic pytesseract pdf2image tqdm python-dotenv

CMD ["python", "run_pipeline.py", "--input", "data/test_documents"]
