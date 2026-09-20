FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-por tesseract-ocr-eng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Histórico, hashes e fila de revisão ficam fora da imagem (volume), com retenção configurável.
ENV IMAGEGUARD_DB=/data/imageguard.sqlite \
    IMAGEGUARD_RETENTION_DAYS=90 \
    IMAGEGUARD_STORE_OCR_TEXT=0
VOLUME ["/data"]

# API protegida por padrão. Monte o registro de credenciais somente para leitura
# e informe IMAGEGUARD_AUTH_FILE; sem ele, operações de negócio retornam 503.
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "fraud_detector.cli", "serve", "--port", "8000"]
