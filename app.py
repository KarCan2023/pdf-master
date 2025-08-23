
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit PDF Toolkit — Compresión, Extracción, Conversión, División y Combinación
Autor: Tú
Licencia: MIT

Funciones clave:
- Comprimir PDF (modo sin pérdida y modo agresivo por rasterización).
- Dividir por rangos de páginas (ej: "1-3,5,7-").
- Combinar varios PDFs en uno.
- Extraer texto o imágenes.
- Convertir PDF→Imágenes (PNG/JPG) y Imágenes→PDF.
- Interfaz única y sencilla.

Requisitos: streamlit, pymupdf (fitz), pillow
"""
import io
import zipfile
from typing import List, Tuple, Optional

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image


def parse_page_ranges(ranges_str: str, total_pages: int) -> List[int]:
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
                s = max(1, int(start)); e = total_pages
            elif start and end:
                s = max(1, int(start)); e = min(total_pages, int(end))
            else:
                continue
            for i in range(s-1, e):
                if 0 <= i < total_pages:
                    pages.add(i)
        else:
            idx = max(1, int(part)) - 1
            if 0 <= idx < total_pages:
                pages.add(idx)
    return sorted(pages)


def pdf_bytes_to_doc(pdf_bytes: bytes) -> fitz.Document:
    return fitz.open(stream=pdf_bytes, filetype="pdf")

def doc_to_bytes(doc: fitz.Document, **save_kwargs) -> bytes:
    out = io.BytesIO()
    default_args = dict(garbage=4, deflate=True, clean=True, linear=True)
    default_args.update(save_kwargs or {})
    doc.save(out, **default_args)
    return out.getvalue()


def compress_pdf(pdf_bytes: bytes, mode: str = "lossless", dpi: int = 120, jpg_quality: int = 70) -> bytes:
    doc = pdf_bytes_to_doc(pdf_bytes)
    if mode == "lossless":
        return doc_to_bytes(doc)
    new_doc = fitz.open()
    for page in doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=jpg_quality, optimize=True)
        img_bytes = img_bytes.getvalue()
        p = new_doc.new_page(width=page.rect.width, height=page.rect.height)
        p.insert_image(p.rect, stream=img_bytes, keep_proportion=False)
    return doc_to_bytes(new_doc)

def split_pdf(pdf_bytes: bytes, ranges_str: str) -> List[Tuple[str, bytes]]:
    doc = pdf_bytes_to_doc(pdf_bytes)
    pages = parse_page_ranges(ranges_str, doc.page_count)
    if not pages:
        return []
    blocks = []
    start = pages[0]; prev = start
    for idx in pages[1:]:
        if idx == prev + 1: prev = idx
        else: blocks.append((start, prev)); start = idx; prev = idx
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
    out_doc = fitz.open()
    for fb in files:
        d = pdf_bytes_to_doc(fb)
        out_doc.insert_pdf(d)
    return doc_to_bytes(out_doc)


def extract_text(pdf_bytes: bytes, ranges_str: Optional[str] = None) -> bytes:
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
    doc = pdf_bytes_to_doc(pdf_bytes)
    pages = parse_page_ranges(ranges_str or "", doc.page_count)
    outputs = []; counter = 1
    for pg in pages:
        page = doc.load_page(pg)
        img_list = page.get_images(full=True)
        for img_info in img_list:
            xref = img_info[0]
            base = doc.extract_image(xref)
            ext = base.get("ext", "png")
            content = base.get("image")
            if content:
                name = f"page{pg+1}_img{counter}.{ext}"
                outputs.append((name, content)); counter += 1
    return outputs

def pdf_to_images(pdf_bytes: bytes, ranges_str: Optional[str] = None, dpi: int = 144, fmt: str = "PNG", jpg_quality: int = 85) -> List[Tuple[str, bytes]]:
    fmt = fmt.upper()
    if fmt not in {"PNG", "JPEG"}: fmt = "PNG"
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
            img.save(bio, format="PNG", optimize=True); ext = "png"
        else:
            img.save(bio, format="JPEG", quality=jpg_quality, optimize=True); ext = "jpg"
        outputs.append((f"page_{pg+1}.{ext}", bio.getvalue()))
    return outputs

def images_to_pdf(images: List[bytes]) -> bytes:
    if not images: return b""
    doc = fitz.open()
    for img_bytes in images:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        b = io.BytesIO(); img.save(b, format="JPEG", quality=90, optimize=True); b = b.getvalue()
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=b, keep_proportion=False)
    return doc_to_bytes(doc)

def make_zip(filetuples: List[Tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in filetuples: zf.writestr(name, content)
    return buf.getvalue()


st.set_page_config(page_title="PDF Toolkit — Streamlit", page_icon="📄", layout="wide")
st.title("📄 PDF Toolkit — Streamlit")
st.caption("Comprimir • Dividir • Combinar • Extraer • Convertir — en una sola interfaz")

with st.sidebar:
    st.header("⚙️ Parámetros globales")
    st.write("Ajusta DPI y calidad (cuando aplique).")
    dpi = st.slider("DPI (renderización)", min_value=72, max_value=300, value=144, step=12,
                    help="Para convertir PDF→Imágenes o compresión agresiva (rasterización).")
    jpg_quality = st.slider("Calidad JPEG", min_value=30, max_value=95, value=85, step=5,
                            help="Afecta compresión agresiva e imágenes JPG.")
    st.markdown("---")
    st.info("Consejo: para resultados más livianos, baja DPI y/o Calidad JPEG.")

tabs = st.tabs(["Comprimir", "Dividir", "Combinar", "Extraer", "Convertir"])


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


with tabs[2]:
    st.subheader("➕ Combinar múltiples PDFs")
    merge_files = st.file_uploader("Sube varios PDFs (en orden)", type=["pdf"], accept_multiple_files=True, key="merge")
    if merge_files and st.button("Combinar ahora", type="primary"):
        with st.spinner("Combinando..."):
            pdfs = [f.read() for f in merge_files]
            merged = merge_pdfs(pdfs)
        st.success("Listo ✅")
        st.download_button("⬇️ Descargar PDF combinado", data=merged, file_name="merged.pdf", mime="application/pdf")


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

st.markdown("---")
with st.expander("ℹ️ Notas y límites prácticos"):
    st.write(
        "- La **compresión sin pérdida** limpia objetos y deflacta streams. Ganancias moderadas.\n"
        "- La **compresión agresiva** rasteriza las páginas según tu DPI y calidad JPEG. Ahorra más, pero **convierte el texto en imagen**.\n"
        "- Para dividir, usa rangos tipo `1-3,5,7-`. Vacío = todas las páginas.\n"
        "- Para PDF→Imágenes, sube DPI para mayor nitidez (y tamaño). 144 DPI suele ser suficiente.\n"
        "- Para Imágenes→PDF, cada imagen ocupa una página completa.\n"
    )
