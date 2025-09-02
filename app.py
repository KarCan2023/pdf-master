#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit PDF Toolkit — Compresión, Extracción, Conversión, División, Combinación,
Desbloquear, Ordenar/Rotar y OCR (Tesseract)

Autor: Tú
Licencia: MIT

Requisitos: streamlit, pymupdf (fitz), pillow, pytesseract (y Tesseract instalado en el sistema).
Opcional: opencv-python-headless para preprocesado de OCR.
"""
import io
import re
import zipfile
import tempfile, os
from typing import List, Tuple, Optional
import time
import json
import requests


import streamlit as st
import fitz  # PyMuPDF
from PIL import Image

# OCR (opcional; el motor real es Tesseract en el sistema)
try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

# OpenCV opcional para mejores resultados de OCR
try:
    import cv2
    HAVE_CV = True
except Exception:
    HAVE_CV = False

# =========================
# Utilidades
# =========================

def parse_page_ranges(ranges_str: str, total_pages: int) -> List[int]:
    """
    Convierte una cadena de rangos estilo "1-3,5,7-" en una lista de índices de página (0-based).
    - "7-" significa desde la 7 hasta el final.
    - Valida límites y evita duplicados.
    """
    if not ranges_str or ranges_str.strip() == "":
        return list(range(total_pages))

    pages = set()
    parts = [p.strip() for p in ranges_str.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start, end = part.split("-", 1)
            start = start.strip()
            end = end.strip()
            if start and not end:
                # "7-" => desde start al final
                s = max(1, int(start))
                e = total_pages
            elif start and end:
                s = max(1, int(start))
                e = min(total_pages, int(end))
            else:
                continue
            for i in range(s-1, e):
                if 0 <= i < total_pages:
                    pages.add(i)
        else:
            # página individual
            idx = max(1, int(part)) - 1
            if 0 <= idx < total_pages:
                pages.add(idx)
    return sorted(pages)


def parse_order(order_str: str, total_pages: int) -> List[int]:
    """
    Analiza cadenas del tipo "3,1,2,5-7,4" preservando el orden e incluyendo rangos.
    Permite repetidos.
    """
    if not order_str or order_str.strip() == "":
        return list(range(total_pages))  # por defecto, mismo orden
    order = []
    parts = [p.strip() for p in order_str.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip()
            b = b.strip()
            if not a:
                continue
            start = max(1, int(a))
            end = total_pages if b == "" else min(total_pages, int(b))
            order.extend(list(range(start-1, end)))
        else:
            idx = max(1, int(part)) - 1
            if 0 <= idx < total_pages:
                order.append(idx)
    return order


def pdf_bytes_to_doc(pdf_bytes: bytes) -> fitz.Document:
    return fitz.open(stream=pdf_bytes, filetype="pdf")


def doc_to_bytes(doc):
    # opciones seguras y efectivas para compresión
    opts = dict(garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, clean=True)
    return doc.write(**opts)  # escribe a bytes en memoria

def doc_to_bytes_linear(doc):
    opts = dict(garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, clean=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = tmp.name
    doc.save(path, **opts, linear=True)  # linearización OK en archivo
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data

def doc_to_bytes_with_options(doc, **opts):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        path = tmp.name
    doc.save(path, **opts)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    return data


def _summarize_extract_text_from_pdf(pdf_bytes: bytes, page_from: int | None = None, page_to: int | None = None) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        n = doc.page_count
        if page_from is None:
            page_from = 1
        if page_to is None or page_to <= 0:
            page_to = n
        page_from = max(1, min(page_from, n))
        page_to = max(1, min(page_to, n))
        if page_from > page_to:
            page_from, page_to = page_to, page_from
        out = []
        for i in range(page_from - 1, page_to):
            out.append(doc.load_page(i).get_text("text"))
        return "\n".join(out).strip()

def _split_into_chunks(text: str, chunk_size: int = 4000, overlap: int = 200):
    if not text:
        return []
    chunk_size = max(500, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def _call_openai_chat(api_key: str, base_url: str, model: str, messages, temperature: float = 0.2, timeout: int = 90) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": max(0.0, min(1.0, float(temperature)))}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)

def _map_reduce_summarize(api_key: str, base_url: str, model: str, chunks, language: str = "es", temperature: float = 0.2):
    if not chunks:
        return "", []
    partials = []
    for ch in chunks:
        messages = [
            {"role": "system", "content": f"Eres un analista experto que resume documentos en {language}. Sé fiel al texto, sin inventar."},
            {"role": "user", "content": "Resume el siguiente fragmento en viñetas claras (5-10 bullets), con cifras, definiciones y nombres propios relevantes. No añadas opiniones ni información que no esté en el texto.\n\n" + ch},
        ]
        partials.append(_call_openai_chat(api_key, base_url, model, messages, temperature=temperature))
    corpus = "\n\n".join(partials)
    reduce_messages = [
        {"role": "system", "content": f"Eres un analista que consolida resúmenes parciales en un resumen único, en {language}, conciso y accionable."},
        {"role": "user", "content": (
            "A partir de los resúmenes parciales siguientes, crea un único resumen final con estas secciones:\n"
            "1) Resumen ejecutivo (150-250 palabras).\n"
            "2) Puntos clave (bullets concretos).\n"
            "3) Riesgos/limitaciones del documento.\n"
            "4) Acciones recomendadas (bullets accionables).\n"
            "5) Citas textuales relevantes (si las hay).\n\n"
            "Resúmenes parciales:\n" + corpus
        )},
    ]
    final_summary = _call_openai_chat(api_key, base_url, model, reduce_messages, temperature=temperature)
    return final_summary, partials


# =========================
# Operaciones base
# =========================

def compress_pdf(pdf_bytes: bytes, mode: str = "lossless", dpi: int = 120, jpg_quality: int = 70) -> bytes:
    """
    Comprime un PDF.
    - mode = "lossless": limpia y deflacta sin tocar imágenes (pierde muy poco o nada de calidad).
    - mode = "aggressive": rasteriza cada página a imagen (control por dpi y calidad JPEG).
      Nota: en este modo se pierde la capacidad de seleccionar texto (convierte a imagen).
    """
    doc = pdf_bytes_to_doc(pdf_bytes)
    if mode == "lossless":
        return doc_to_bytes(doc)

    # Modo agresivo: render a imágenes y reconstruir
    new_doc = fitz.open()
    for page in doc:
        # Render según DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # sin canal alpha
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # JPEG para mejor compresión
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=jpg_quality, optimize=True)
        img_bytes = img_bytes.getvalue()

        # Crear página del tamaño original y pegar imagen
        p = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=img_bytes, keep_proportion=False)
    return doc_to_bytes(new_doc)


def split_pdf(pdf_bytes: bytes, ranges_str: str) -> List[Tuple[str, bytes]]:
    """
    Divide un PDF en múltiples PDFs según rangos.
    Devuelve lista de tuplas (nombre_sugerido, bytes_pdf).
    """
    doc = pdf_bytes_to_doc(pdf_bytes)
    pages = parse_page_ranges(ranges_str, doc.page_count)
    # Agrupar páginas contiguas para producir un archivo por bloque contiguo
    if not pages:
        return []
    blocks = []
    start = pages[0]
    prev = start
    for idx in pages[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            blocks.append((start, prev))
            start = idx
            prev = idx
    blocks.append((start, prev))

    outputs = []
    for i, (s, e) in enumerate(blocks, start=1):
        new_doc = fitz.open()
        for pg in range(s, e + 1):
            new_doc.insert_pdf(doc, from_page=pg, to_page=pg)
        name = f"split_{i}_{s+1}-{e+1}.pdf"
        outputs.append((name, doc_to_bytes(new_doc)))
    return outputs


def merge_pdfs(files: List[bytes]) -> bytes:
    """
    Combina varios PDFs en uno solo, respetando el orden de la lista.
    """
    out_doc = fitz.open()
    for i, fb in enumerate(files, start=1):
        d = pdf_bytes_to_doc(fb)
        out_doc.insert_pdf(d)
    return doc_to_bytes(out_doc)


def extract_text(pdf_bytes: bytes, ranges_str: Optional[str] = None) -> bytes:
    """
    Extrae texto plano de páginas seleccionadas.
    Devuelve bytes de un archivo .txt en UTF-8.
    """
    doc = pdf_bytes_to_doc(pdf_bytes)
    pages = parse_page_ranges(ranges_str or "", doc.page_count)
    buf = io.StringIO()
    for i, pg in enumerate(pages, start=1):
        page = doc.load_page(pg)
        buf.write(page.get_text("text"))
        buf.write("\n")
        if i < len(pages):
            buf.write("\n" + "-" * 24 + f" [Página {pg+1}] " + "-" * 24 + "\n\n")
    return buf.getvalue().encode("utf-8")


def extract_images(pdf_bytes: bytes, ranges_str: Optional[str] = None) -> List[Tuple[str, bytes]]:
    """
    Extrae imágenes embebidas en las páginas seleccionadas.
    Devuelve lista de (nombre, bytes_img) con la extensión nativa (jpg/png/etc.).
    """
    doc = pdf_bytes_to_doc(pdf_bytes)
    pages = parse_page_ranges(ranges_str or "", doc.page_count)
    outputs = []
    counter = 1
    for pg in pages:
        page = doc.load_page(pg)
        img_list = page.get_images(full=True)  # lista de tuplas con (xref, ...)
        for img_info in img_list:
            xref = img_info[0]
            base = doc.extract_image(xref)
            ext = base.get("ext", "png")
            content = base.get("image")
            if content:
                name = f"page{pg+1}_img{counter}.{ext}"
                outputs.append((name, content))
                counter += 1
    return outputs


def pdf_to_images(pdf_bytes: bytes, ranges_str: Optional[str] = None, dpi: int = 144, fmt: str = "PNG", jpg_quality: int = 85) -> List[Tuple[str, bytes]]:
    """
    Convierte páginas del PDF a imágenes.
    fmt en {"PNG", "JPEG"}.
    """
    fmt = fmt.upper()
    if fmt not in {"PNG", "JPEG"}:
        fmt = "PNG"

    doc = pdf_bytes_to_doc(pdf_bytes)
    pages = parse_page_ranges(ranges_str or "", doc.page_count)
    outputs = []
    for pg in pages:
        page = doc.load_page(pg)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        bio = io.BytesIO()
        if fmt == "PNG":
            img.save(bio, format="PNG", optimize=True)
            ext = "png"
        else:
            img.save(bio, format="JPEG", quality=jpg_quality, optimize=True)
            ext = "jpg"
        outputs.append((f"page_{pg+1}.{ext}", bio.getvalue()))
    return outputs


def images_to_pdf(images: List[bytes]) -> bytes:
    """
    Ensambla múltiples imágenes en un solo PDF (una imagen por página).
    Mantiene el tamaño de cada imagen en puntos (72 dpi base) para ocupar la página.
    """
    if not images:
        return b""
    doc = fitz.open()
    for img_bytes in images:
        # Garantizar RGB sin alfa
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=90, optimize=True)
        b = b.getvalue()
        page = doc.new_page(width=w, height=h)  # puntos ~ pixeles a 72 dpi
        page.insert_image(page.rect, stream=b, keep_proportion=False)
    return doc_to_bytes(doc)


def make_zip(filetuples: List[Tuple[str, bytes]]) -> bytes:
    """
    Recibe lista de (nombre, contenido_bytes) y devuelve bytes de un .zip
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in filetuples:
            zf.writestr(name, content)
    return buf.getvalue()


# =========================
# Nuevas funciones: Desbloquear, Ordenar/Rotar, OCR
# =========================

def unlock_pdf(pdf_bytes: bytes, password: str) -> bytes:
    doc = pdf_bytes_to_doc(pdf_bytes)
    if doc.needs_pass:
        if not doc.authenticate(password):
            raise ValueError("Contraseña incorrecta.")
    return doc_to_bytes_with_options(doc, encryption=fitz.PDF_ENCRYPT_NONE)



def reorder_pdf(pdf_bytes: bytes, order_str: str) -> bytes:
    src = pdf_bytes_to_doc(pdf_bytes)
    idxs = parse_order(order_str, src.page_count)
    if not idxs:
        raise ValueError("Orden vacío o inválido.")
    dst = fitz.open()
    for i in idxs:
        dst.insert_pdf(src, from_page=i, to_page=i)
    return doc_to_bytes(dst)


def rotate_pages(pdf_bytes: bytes, ranges_str: str, degrees: int = 90) -> bytes:
    doc = pdf_bytes_to_doc(pdf_bytes)
    pages = parse_page_ranges(ranges_str, doc.page_count)
    degrees = (degrees // 90) * 90
    for p in pages:
        page = doc.load_page(p)
        page.set_rotation((page.rotation + degrees) % 360)
    return doc_to_bytes(doc)


def annotate_pdf(
    pdf_bytes: bytes,
    sig_bytes: Optional[bytes] = None,
    sig_page: int = 1,
    sig_x: float = 50,
    sig_y: float = 50,
    sig_width: float = 150,
    text: str = "",
    text_page: int = 1,
    text_x: float = 50,
    text_y: float = 50,
    font_size: int = 12,
) -> bytes:
    """Añade una firma en PNG y/o texto a un PDF."""
    doc = pdf_bytes_to_doc(pdf_bytes)

    if sig_bytes:
        p = doc.load_page(max(0, min(sig_page - 1, doc.page_count - 1)))
        img = Image.open(io.BytesIO(sig_bytes))
        ratio = img.height / img.width
        rect = fitz.Rect(sig_x, sig_y, sig_x + sig_width, sig_y + sig_width * ratio)
        p.insert_image(rect, stream=sig_bytes)

    if text:
        p = doc.load_page(max(0, min(text_page - 1, doc.page_count - 1)))
        p.insert_text((text_x, text_y), text, fontsize=font_size)

    return doc_to_bytes(doc)


def _preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Preprocesado opcional para mejorar OCR: gris, binarización/adaptativo, nitidez.
    Usa OpenCV si está disponible; si no, cae a PIL básico.
    """
    try:
        if HAVE_CV:
            import numpy as np
            img = np.array(pil_img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Umbral adaptativo (reduce ruido de iluminación)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 15)
            # Mediana para quitar sal/pimienta
            th = cv2.medianBlur(th, 3)
            return Image.fromarray(th)
        else:
            # PIL fallback
            return pil_img.convert("L")
    except Exception:
        return pil_img


def ocr_pdf(pdf_bytes: bytes, lang: str = "spa+eng", dpi: int = 200, skip_text_pages: bool = True, to_txt: bool = False) -> bytes:
    """
    Aplica OCR con Tesseract. Si skip_text_pages=True, copia tal cual páginas que ya tienen texto.
    - to_txt=False => devuelve PDF con capa de texto (buscable).
    - to_txt=True  => devuelve TXT concatenado.
    """
    if not HAVE_TESS:
        raise RuntimeError("pytesseract no está disponible. Instala Tesseract en el sistema y la librería pytesseract.")

    src = pdf_bytes_to_doc(pdf_bytes)

    if to_txt:
        # Extraer texto como TXT
        out_buf = io.StringIO()
        for page in src:
            # Si ya tiene texto y se quiere omitir OCR, se intenta extraer texto nativo
            native_text = page.get_text("text").strip()
            if skip_text_pages and native_text:
                out_buf.write(native_text + "\n")
                out_buf.write("\n" + "-" * 16 + f" [Página {page.number+1}] " + "-" * 16 + "\n\n")
                continue

            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil_img = _preprocess_for_ocr(pil_img)
            txt = pytesseract.image_to_string(pil_img, lang=lang)
            out_buf.write(txt + "\n")
            out_buf.write("\n" + "-" * 16 + f" [Página {page.number+1}] " + "-" * 16 + "\n\n")
        return out_buf.getvalue().encode("utf-8")

    # PDF con capa de texto OCR
    out_doc = fitz.open()
    for page in src:
        native_text = page.get_text("text").strip()
        if skip_text_pages and native_text:
            # Copiar página original
            out_doc.insert_pdf(src, from_page=page.number, to_page=page.number)
            continue
        # Renderizar a imagen
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pil_img = _preprocess_for_ocr(pil_img)
        # Generar PDF de una página con capa de texto
        pdf_page_bytes = pytesseract.image_to_pdf_or_hocr(pil_img, lang=lang, extension="pdf")
        tmp = fitz.open(stream=pdf_page_bytes, filetype="pdf")
        out_doc.insert_pdf(tmp)
    return doc_to_bytes(out_doc)


# =========================
# Interfaz Streamlit
# =========================

st.set_page_config(page_title="PDF Toolkit — Streamlit", page_icon="📄", layout="wide")

st.title("📄 PDF Toolkit — Streamlit")
st.caption("Comprimir • Dividir • Combinar • Extraer • Convertir • Desbloquear • Ordenar/Rotar • OCR")

with st.sidebar:
    st.header("⚙️ Parámetros globales")
    st.write("Ajusta DPI y calidad (cuando aplique).")
    dpi = st.slider("DPI (renderización)", min_value=72, max_value=300, value=144, step=12,
                    help="Para convertir PDF→Imágenes, compresión agresiva y OCR (renderizado).")
    jpg_quality = st.slider("Calidad JPEG", min_value=30, max_value=95, value=85, step=5,
                            help="Afecta compresión agresiva e imágenes JPG.")
    st.markdown("---")
    st.info("Consejo: para resultados más livianos, baja DPI y/o Calidad JPEG. Para OCR, DPI 200–300 suele mejorar resultados.")

tabs = st.tabs(["Comprimir", "Dividir", "Combinar", "Extraer", "Convertir", "Desbloquear", "Ordenar/Rotar", "OCR", "Resumir con IA", "Anotar"])

# --------- Comprimir ---------
with tabs[0]:
    st.subheader("🗜️ Comprimir PDF")
    comp_file = st.file_uploader("Sube un PDF", type=["pdf"], key="comp")
    mode = st.radio("Modo de compresión", options=["Sin pérdida (recomendado)", "Agresiva (rasterizar)"],
                    help="La compresión agresiva convierte cada página a imagen (pierdes texto seleccionable).")
    if comp_file:
        pdf_bytes = comp_file.read()
        if st.button("Comprimir ahora", type="primary"):
            with st.spinner("Comprimiendo..."):
                out = compress_pdf(pdf_bytes, mode="lossless" if mode.startswith("Sin") else "aggressive", dpi=dpi, jpg_quality=jpg_quality)
            st.success("Listo ✅")
            st.download_button("⬇️ Descargar PDF comprimido", data=out, file_name=f"compressed_{comp_file.name}", mime="application/pdf")

# --------- Dividir ---------
with tabs[1]:
    st.subheader("✂️ Dividir PDF por rangos")
    split_file = st.file_uploader("Sube un PDF", type=["pdf"], key="split")
    ranges_help = 'Ejemplos: "1-3,5" (páginas 1 a 3 y 5), "4-" (de la 4 al final). Vacío = todas.'
    ranges_str = st.text_input("Rangos de páginas", value="", help=ranges_help)
    if split_file:
        pdf_bytes = split_file.read()
        if st.button("Dividir ahora"):
            with st.spinner("Dividiendo..."):
                outputs = split_pdf(pdf_bytes, ranges_str=ranges_str)
            if not outputs:
                st.warning("No se generó ninguna salida. Verifica tus rangos.")
            elif len(outputs) == 1:
                name, content = outputs[0]
                st.success("Listo ✅")
                st.download_button("⬇️ Descargar PDF dividido", data=content, file_name=name, mime="application/pdf")
            else:
                zip_bytes = make_zip(outputs)
                st.success(f"Listo ✅ — {len(outputs)} archivos generados")
                st.download_button("⬇️ Descargar ZIP con partes", data=zip_bytes, file_name="splits.zip", mime="application/zip")

# --------- Combinar ---------
with tabs[2]:
    st.subheader("➕ Combinar múltiples PDFs")
    merge_files = st.file_uploader("Sube varios PDFs (en orden)", type=["pdf"], accept_multiple_files=True, key="merge")
    if merge_files and st.button("Combinar ahora", type="primary"):
        with st.spinner("Combinando..."):
            pdfs = [f.read() for f in merge_files]
            merged = merge_pdfs(pdfs)
        st.success("Listo ✅")
        st.download_button("⬇️ Descargar PDF combinado", data=merged, file_name="merged.pdf", mime="application/pdf")

# --------- Extraer ---------
with tabs[3]:
    st.subheader("🔍 Extraer texto o imágenes")
    ext_file = st.file_uploader("Sube un PDF", type=["pdf"], key="extract")
    ext_mode = st.radio("¿Qué quieres extraer?", options=["Texto", "Imágenes"], horizontal=True)
    ranges_help2 = 'Rangos de páginas (opcional). Ej: "1-2,4". Vacío = todas.'
    ranges_ext = st.text_input("Rangos de páginas", value="", help=ranges_help2, key="ranges_ext")
    if ext_file:
        pdf_bytes = ext_file.read()
        if ext_mode == "Texto":
            if st.button("Extraer texto"):
                with st.spinner("Extrayendo texto..."):
                    txt = extract_text(pdf_bytes, ranges_ext)
                st.success("Listo ✅")
                st.download_button("⬇️ Descargar .txt", data=txt, file_name="extracted.txt", mime="text/plain")
        else:
            st.caption("Las imágenes se extraen en su formato original (jpg/png/etc.).")
            if st.button("Extraer imágenes"):
                with st.spinner("Extrayendo imágenes..."):
                    imgs = extract_images(pdf_bytes, ranges_ext)
                if not imgs:
                    st.warning("No se encontraron imágenes embebidas en las páginas seleccionadas.")
                elif len(imgs) == 1:
                    name, content = imgs[0]
                    st.success("Listo ✅")
                    st.download_button("⬇️ Descargar imagen", data=content, file_name=name)
                else:
                    zip_bytes = make_zip(imgs)
                    st.success(f"Listo ✅ — {len(imgs)} imágenes extraídas")
                    st.download_button("⬇️ Descargar ZIP", data=zip_bytes, file_name="images.zip", mime="application/zip")

# --------- Convertir ---------
with tabs[4]:
    st.subheader("🔄 Convertir PDF ↔ Imágenes")
    conv_mode = st.radio("Selecciona conversión", options=["PDF → Imágenes", "Imágenes → PDF"], horizontal=True)
    if conv_mode == "PDF → Imágenes":
        pdf_img_file = st.file_uploader("Sube un PDF", type=["pdf"], key="pdf2img")
        fmt = st.selectbox("Formato de salida", options=["PNG", "JPEG"], index=0)
        ranges_help3 = 'Rangos de páginas (opcional). Ej: "1-5". Vacío = todas.'
        ranges_conv = st.text_input("Rangos de páginas", value="", help=ranges_help3, key="ranges_conv")
        if pdf_img_file and st.button("Convertir a imágenes"):
            with st.spinner("Convirtiendo..."):
                outputs = pdf_to_images(pdf_img_file.read(), ranges_conv, dpi=dpi, fmt=fmt, jpg_quality=jpg_quality)
            if not outputs:
                st.warning("No se generó ninguna imagen. Verifica tus rangos.")
            elif len(outputs) == 1:
                name, content = outputs[0]
                st.success("Listo ✅")
                st.image(content, caption=name)
                st.download_button("⬇️ Descargar imagen", data=content, file_name=name)
            else:
                zip_bytes = make_zip(outputs)
                st.success(f"Listo ✅ — {len(outputs)} imágenes")
                st.download_button("⬇️ Descargar ZIP", data=zip_bytes, file_name="pages_as_images.zip", mime="application/zip")

    else:
        imgs = st.file_uploader("Sube imágenes (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="img2pdf")
        if imgs and st.button("Crear PDF"):
            with st.spinner("Creando PDF..."):
                img_bytes_list = [f.read() for f in imgs]
                pdf_bytes = images_to_pdf(img_bytes_list)
            st.success("Listo ✅")
            st.download_button("⬇️ Descargar PDF", data=pdf_bytes, file_name="from_images.pdf", mime="application/pdf")

# --------- Desbloquear ---------
with tabs[5]:
    st.subheader("🔓 Desbloquear (quitar contraseña con clave)")
    u_file = st.file_uploader("Sube un PDF protegido", type=["pdf"], key="unlock")
    password = st.text_input("Contraseña", type="password")
    if u_file and password and st.button("Quitar contraseña"):
        try:
            with st.spinner("Desbloqueando..."):
                unlocked = unlock_pdf(u_file.read(), password)
            st.success("Listo ✅")
            st.download_button("⬇️ Descargar PDF desbloqueado", data=unlocked, file_name=f"unlocked_{u_file.name}", mime="application/pdf")
        except Exception as e:
            st.error(f"Error: {e}")

# --------- Ordenar/Rotar ---------
with tabs[6]:
    st.subheader("🧩 Ordenar y Rotar páginas")
    or_file = st.file_uploader("Sube un PDF", type=["pdf"], key="ordrotate")
    col1, col2 = st.columns(2)
    with col1:
        order_str = st.text_input("Orden de páginas (ej: 3,1,2,5-7,4)", value="")
        if or_file and st.button("Reordenar"):
            try:
                with st.spinner("Reordenando..."):
                    out = reorder_pdf(or_file.read(), order_str)
                st.success("Listo ✅")
                st.download_button("⬇️ Descargar PDF reordenado", data=out, file_name=f"reordered_{or_file.name}", mime="application/pdf")
            except Exception as e:
                st.error(f"Error: {e}")
    with col2:
        ranges_rot = st.text_input('Rangos a rotar (ej: "1-3,5")', value="")
        degrees = st.selectbox("Grados de rotación", options=[0, 90, 180, 270], index=1)
        if or_file and st.button("Rotar"):
            try:
                with st.spinner("Rotando..."):
                    out = rotate_pages(or_file.read(), ranges_rot, degrees)
                st.success("Listo ✅")
                st.download_button("⬇️ Descargar PDF rotado", data=out, file_name=f"rotated_{or_file.name}", mime="application/pdf")
            except Exception as e:
                st.error(f"Error: {e}")

# --------- OCR ---------
with tabs[7]:
    st.subheader("🧠 OCR (Tesseract) — convierte escaneos a PDF buscable o TXT")
    if not HAVE_TESS:
        st.warning("pytesseract no está disponible en este entorno. Para usar OCR necesitas instalar Tesseract en el sistema y la librería pytesseract. Revisa el README para la ruta A (Docker).")
    ocr_file = st.file_uploader("Sube un PDF escaneado (o mixto)", type=["pdf"], key="ocr")
    lang_options = ["spa", "eng"]
    langs = st.multiselect("Idiomas del OCR (Tesseract)", options=lang_options, default=["spa", "eng"])
    ocr_dpi = st.slider("DPI para OCR", min_value=150, max_value=300, value=200, step=10)
    skip_text_pages = st.checkbox("Omitir OCR en páginas que ya tienen texto", value=True)
    out_mode = st.radio("Salida", options=["PDF buscable", "Texto (.txt)"], horizontal=True)
    if ocr_file and st.button("Ejecutar OCR", type="primary"):
        try:
            with st.spinner("Ejecutando OCR..."):
                lang = "+".join(langs) if langs else "spa+eng"
                to_txt = out_mode.startswith("Texto")
                out = ocr_pdf(ocr_file.read(), lang=lang, dpi=ocr_dpi, skip_text_pages=skip_text_pages, to_txt=to_txt)
            st.success("Listo ✅")
            if to_txt:
                st.download_button("⬇️ Descargar .txt", data=out, file_name=f"ocr_{ocr_file.name.rsplit('.',1)[0]}.txt", mime="text/plain")
            else:
                st.download_button("⬇️ Descargar PDF buscable", data=out, file_name=f"ocr_{ocr_file.name}", mime="application/pdf")
        except Exception as e:
            st.error(f"Error: {e}")

with tabs[8]:
    st.subheader("🤖 Resumir con IA")
    sum_file = st.file_uploader("Sube un PDF", type=["pdf"], key="summarize")
    colp = st.columns(2)
    with colp[0]:
        page_from = st.number_input("Desde página (1-indexed)", min_value=1, value=1, step=1)
    with colp[1]:
        page_to = st.number_input("Hasta página (0 = hasta el final)", min_value=0, value=0, step=1)

    colc = st.columns(2)
    with colc[0]:
        chunk_size = st.slider("Tamaño de fragmento", 1000, 8000, 4000, 500)
        overlap = st.slider("Solape", 0, 1000, 200, 50)
    with colc[1]:
        language = st.selectbox("Idioma", ["es", "en"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("**Proveedor LLM**")
    provider = st.selectbox("Tipo", ["OpenAI (api.openai.com)", "OpenAI-compatible (custom base_url)"])
    default_base = "https://api.openai.com/v1" if provider.startswith("OpenAI (") else ""

    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    api_key = st.text_input("API Key", type="password", value=st.session_state.llm_api_key)
    if api_key and api_key != st.session_state.llm_api_key:
        st.session_state.llm_api_key = api_key
    base_url = st.text_input("Base URL", value=default_base)
    model = st.text_input("Modelo", value="gpt-4o-mini")

    if sum_file and st.button("Generar resumen", type="primary", use_container_width=True):
        if not api_key:
            st.error("Ingresa tu API Key.")
        elif not base_url or not model:
            st.error("Base URL y modelo son obligatorios.")
        else:
            with st.spinner("Extrayendo texto..."):
                n_from = int(page_from)
                n_to = int(page_to) if int(page_to) > 0 else None
                text = _summarize_extract_text_from_pdf(sum_file.read(), n_from, n_to)
            if not text:
                st.warning("No se encontró texto. Si el PDF es escaneado, usa la pestaña OCR primero.")
            else:
                chunks = _split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                st.info(f"Fragmentos a resumir: {len(chunks)}")
                try:
                    with st.spinner("Resumiendo..."):
                        final_summary, partials = _map_reduce_summarize(
                            api_key=api_key,
                            base_url=base_url,
                            model=model,
                            chunks=chunks,
                            language=language,
                            temperature=temperature,
                        )
                    st.subheader("Resumen final")
                    st.markdown(final_summary)
                    st.download_button("⬇️ Descargar resumen (.md)", data=final_summary.encode("utf-8"),
                                       file_name="resumen.md", mime="text/markdown")
                    with st.expander("Resúmenes parciales"):
                        for i, p in enumerate(partials, 1):
                            st.markdown(f"**Parte {i}**")
                            st.markdown(p)
                            st.divider()
                except requests.HTTPError as e:
                    details = getattr(e.response, "text", str(e))
                    st.error("Error del proveedor LLM. Revisa modelo/base_url o saldo.")
                    st.code(details[:1000])
                except Exception as e:
                    st.error("Fallo al generar el resumen.")
                    st.code(str(e))

with tabs[9]:
    st.subheader("✍️ Anotar PDF")
    ann_file = st.file_uploader("Sube un PDF", type=["pdf"], key="annotate_pdf")
    sig_file = st.file_uploader("Firma (PNG)", type=["png"], key="sig_img")
    text_to_add = st.text_input("Texto a añadir", value="", key="annot_text")
    col_sig, col_txt = st.columns(2)
    with col_sig:
        sig_page = st.number_input("Página firma", min_value=1, value=1, step=1)
        sig_x = st.number_input("Posición X firma", min_value=0.0, value=50.0, step=10.0)
        sig_y = st.number_input("Posición Y firma", min_value=0.0, value=50.0, step=10.0)
        sig_w = st.number_input("Ancho firma", min_value=10.0, value=150.0, step=10.0)
    with col_txt:
        text_page = st.number_input("Página texto", min_value=1, value=1, step=1)
        text_x = st.number_input("Posición X texto", min_value=0.0, value=50.0, step=10.0)
        text_y = st.number_input("Posición Y texto", min_value=0.0, value=50.0, step=10.0)
        font_size = st.number_input("Tamaño de fuente", min_value=4, max_value=72, value=12, step=1)
    if ann_file and (sig_file or text_to_add) and st.button("Aplicar anotaciones", type="primary"):
        with st.spinner("Anotando..."):
            out = annotate_pdf(
                ann_file.read(),
                sig_bytes=sig_file.read() if sig_file else None,
                sig_page=int(sig_page),
                sig_x=float(sig_x),
                sig_y=float(sig_y),
                sig_width=float(sig_w),
                text=text_to_add,
                text_page=int(text_page),
                text_x=float(text_x),
                text_y=float(text_y),
                font_size=int(font_size),
            )
        st.success("Listo ✅")
        st.download_button(
            "⬇️ Descargar PDF anotado",
            data=out,
            file_name=f"annotated_{ann_file.name}",
            mime="application/pdf",
        )
    elif ann_file:
        st.info("Sube una firma o escribe un texto para añadir.")

st.markdown("---")
with st.expander("ℹ️ Notas y límites prácticos"):
    st.write(
        "- La **compresión sin pérdida** limpia objetos y deflacta streams. Ganancias moderadas.\n"
        "- La **compresión agresiva** rasteriza las páginas según tu DPI y calidad JPEG. Ahorra más, pero **convierte el texto en imagen**.\n"
        "- Para dividir, usa rangos tipo `1-3,5,7-`. Vacío = todas las páginas.\n"
        "- Para PDF→Imágenes, 144 DPI suele ser suficiente para lectura nítida.\n"
        "- **Desbloquear** requiere que proporciones la contraseña del documento.\n"
        "- **Ordenar** acepta patrones como `3,1,2,5-7,4`. **Rotar** admite 0/90/180/270 grados.\n"
        "- **OCR** necesita Tesseract instalado en el sistema. Usa DPI 200–300 para mejores resultados.\n"
    )
