# 📄 PDF Toolkit — Streamlit (Full)

Herramienta web en **una sola interfaz** para manipular PDFs:

- **Comprimir** (sin pérdida o agresiva por rasterización)
- **Dividir** por rangos (`1-3,5,7-`)
- **Combinar** varios PDFs
- **Extraer** texto e **imágenes**
- **Convertir** PDF→Imágenes (PNG/JPG) e Imágenes→PDF
- **Desbloquear** (requiere contraseña)
- **Ordenar/Rotar** páginas
- **OCR** con **Tesseract** (Ruta A: local)

## 🚀 Uso local (Ubuntu/Debian)

1. Instala Tesseract y paquetes de idioma (opcional `eng`/`spa`):
   ```bash
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng
   ```
2. Instala dependencias Python:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la app:
   ```bash
   streamlit run app.py
   ```

> **macOS** (Homebrew):
> ```bash
> brew install tesseract
> pip install -r requirements.txt
> ```
> Para idiomas extra: `brew install tesseract-lang` (o descarga los traineddata y colócalos en `TESSDATA_PREFIX`).
>
> **Windows**: instala Tesseract (UB Mannheim build), agrega `tesseract.exe` al PATH.

## 🐳 Ruta A con Docker (recomendada para despliegue)

1. Construye la imagen:
   ```bash
   docker build -t pdf-toolkit .
   ```
2. Ejecuta el contenedor:
   ```bash
   docker run --rm -p 8501:8501 pdf-toolkit
   ```
3. Abre `http://localhost:8501`.

> **Nota Streamlit Cloud**: no permite `apt-get`. Para OCR usa Docker (Render, Fly.io, Railway, ECS, etc.).

## 📦 Estructura
- `app.py` — código de la app (todas las pestañas).
- `requirements.txt` — dependencias Python.
- `Dockerfile` — incluye Tesseract + idiomas `spa` y `eng`.
- `.gitignore`, `LICENSE`, `README.md`.

## 🧠 OCR (consejos)
- Usa **DPI 200–300** para mejor reconocimiento.
- Idiomas: `"spa"`, `"eng"` o `"spa+eng"`.
- Marca **“Omitir OCR en páginas con texto”** para PDFs mixtos (texto+escaneo).
- Si la calidad es baja, sube DPI o prueba el modo agresivo de compresión antes.

## 🛡️ Legal
- El módulo **Desbloquear** solo funciona si **proporcionas la contraseña**. No evadimos protecciones sin autorización.

## 🛠️ Roadmap opcional
- Detección automática de rotación (deskew) con OpenCV.
- Firmas, sellos y marcas de agua.
- Encriptar con contraseña (owner/user passwords).
