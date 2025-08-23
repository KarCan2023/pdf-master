# OCR Route A: Python + Tesseract + Spanish/English languages
FROM python:3.11-slim

# System deps: Tesseract with spa & eng, plus libgl for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
