"""
utils/rich_pdf_extractor.py

Section-aware PDF extractor with multimodal figure captioning.

Produces structured markdown that is fully compatible with the `paper_text`
string expected by the existing pipeline — drop-in replacement for
`utils.pdf_extractor.extract_text`.

Figure captioning uses llama4:scout (multimodal) on the same remote Ollama
server as the rest of the pipeline.

Quick test (no pipeline needed):
    python -m utils.rich_pdf_extractor data/papers/raw/yourpaper.pdf
    python -m utils.rich_pdf_extractor data/papers/raw/yourpaper.pdf --no-figures
"""

import base64
import logging
import sys
from pathlib import Path

import requests

from utils.model_config import get_model

# Must match the Ollama server used by the rest of the pipeline (llm_client.py)
_OLLAMA_BASE = "https://ollama.draco.uni-jena.de/api/generate"


def _vision_model() -> str:
    return get_model("rag_pipeline", "vision_model", "llama4:scout")

# Skip images smaller than this — likely icons, borders, or decorative elements
_MIN_IMAGE_BYTES = 5_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _count_tables_from_text(text: str) -> int:
    """
    Count unique table labels in the text. Handles:
      - Arabic numerals : Table 1, TABLE 2, Tab. 3
      - Roman numerals  : TABLE I, TABLE IV, Table VIII
    """
    import re
    _ROMAN = r'(?:X{0,3})(?:IX|IV|V?I{0,3})'
    pattern = rf'\b(?:TABLE|Table|Tab\.?)\s+({_ROMAN}|\d+)\b'
    return len(set(re.findall(pattern, text)))


def extract_paper_stats(file_path: Path) -> dict:
    """
    Return lightweight structural statistics about a PDF.

    Uses a single fitz pass (no LLM calls) — safe to call even when
    extraction is already cached.  For non-PDF files only char/section
    counts are available; page/figure/table fields will be None.

    Fields
    ------
    n_pages         : int   — total pages
    n_figures       : int   — images >= _MIN_IMAGE_BYTES (proxy for data figures)
    n_tables        : int   — unique 'Table N' captions found in text (reliable
                              for borderless journal tables; fitz find_tables()
                              only detects tables with explicit border lines)
    file_size_kb    : float — raw PDF size on disk
    """
    stats: dict = {
        "n_pages":      None,
        "n_figures":    None,
        "n_tables":     None,
        "file_size_kb": round(file_path.stat().st_size / 1024, 1),
    }

    if file_path.suffix.lower() != ".pdf":
        return stats

    try:
        import fitz  # noqa: PLC0415
    except ImportError:
        return stats

    try:
        doc = fitz.open(str(file_path))
        stats["n_pages"]   = len(doc)
        n_figures  = 0
        full_text  = []

        for page in doc:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    if len(doc.extract_image(xref)["image"]) >= _MIN_IMAGE_BYTES:
                        n_figures += 1
                except Exception:
                    pass
            full_text.append(page.get_text())

        doc.close()
        stats["n_figures"] = n_figures
        stats["n_tables"]  = _count_tables_from_text("\n".join(full_text))
    except Exception as e:
        logging.warning("[RichPDF] extract_paper_stats failed: %s", e)

    return stats


def extract_rich_text(file_path: Path, caption_figures: bool = True) -> str:
    """
    Extract structured markdown from a PDF.

    Sections are detected via font-size heuristics and promoted to ## headings.
    Each figure is extracted and described by llama4:scout, then inserted inline
    as a bracketed annotation so the LLM downstream can reason about it.

    Non-PDF files (txt, md) fall back to plain UTF-8 read — identical to the
    behaviour of the original `extract_text` function.

    Parameters
    ----------
    file_path:       Path to the input file.
    caption_figures: Set False to skip image captioning (faster, text-only).

    Returns
    -------
    str — structured markdown, compatible with `paper_text` in the pipeline.
    """
    if file_path.suffix.lower() != ".pdf":
        return file_path.read_text(encoding="utf-8")

    try:
        import fitz  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "Rich PDF extraction requires pymupdf — run: pip install 'pymupdf>=1.24'"
        )

    doc  = fitz.open(str(file_path))
    parts: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        text = _page_to_markdown(page)
        if text:
            parts.append(text)

        if caption_figures:
            for caption in _caption_page_figures(doc, page, page_num):
                parts.append(caption)

    doc.close()
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _page_to_markdown(page) -> str:
    """Try PyMuPDF's native markdown output first; fall back to heuristic."""
    try:
        # Available in pymupdf >= 1.24.1 — preserves headings automatically
        md = page.get_text("markdown").strip()
        if md:
            return md
    except Exception:
        pass
    return _page_to_markdown_heuristic(page)


def _page_to_markdown_heuristic(page) -> str:
    """
    Promote bold / oversized spans to ## headings; emit everything else as
    plain text lines. Relies on relative font-size comparison so it works
    across different paper layouts without hard-coded point sizes.
    """
    blocks    = page.get_text("dict")["blocks"]
    body_size = _body_font_size(blocks)
    lines: list[str] = []

    for block in blocks:
        if block.get("type") != 0:   # skip embedded-image blocks
            continue
        for line in block["lines"]:
            if not line["spans"]:
                continue
            text     = " ".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            max_size = max(s["size"] for s in line["spans"])
            is_bold  = any(bool(s["flags"] & 16) for s in line["spans"])

            # Heading heuristic: noticeably larger OR slightly larger + bold
            if max_size >= body_size * 1.25 or (max_size >= body_size * 1.1 and is_bold):
                lines.append(f"\n## {text}\n")
            else:
                lines.append(text)

    return "\n".join(lines)


def _body_font_size(blocks) -> float:
    """Return the modal (most common) font size — a reliable proxy for body text."""
    sizes: list[float] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    sizes.append(round(span["size"], 1))
    return max(set(sizes), key=sizes.count) if sizes else 10.0


# ---------------------------------------------------------------------------
# Figure captioning
# ---------------------------------------------------------------------------

def _caption_page_figures(doc, page, page_num: int) -> list[str]:
    """Extract all images on a page and caption each with llama4:scout."""
    captions: list[str] = []
    for fig_idx, img_info in enumerate(page.get_images(full=True), start=1):
        xref = img_info[0]
        try:
            base_image  = doc.extract_image(xref)
            image_bytes = base_image["image"]
            if len(image_bytes) < _MIN_IMAGE_BYTES:
                continue   # skip tiny decorative images
            b64     = base64.b64encode(image_bytes).decode("utf-8")
            caption = _ollama_caption(b64, page_num, fig_idx)
            captions.append(caption)
        except Exception as e:
            logging.warning(
                "[RichPDF] Skipping image (page %d, fig %d): %s",
                page_num, fig_idx, e,
            )
    return captions


_CAPTION_PROMPT = (
    "You are analyzing a figure extracted from a polymer science research paper. "
    "In 2-4 sentences describe: "
    "(1) what type of data this is — e.g. GPC/SEC trace, DSC/TGA curve, NMR spectrum, "
    "reaction scheme, SEM/TEM image, conversion-vs-time plot, or table; "
    "(2) the key numerical values or trends visible; and "
    "(3) what this reveals about the polymer system studied. "
    "Be concise and factual. Do not say 'the figure shows' — start directly with the content."
)


def _ollama_caption(image_b64: str, page_num: int, fig_idx: int) -> str:
    label = f"Figure — Page {page_num}, Fig {fig_idx}"
    try:
        resp = requests.post(
            _OLLAMA_BASE,
            json={
                "model":   _vision_model(),
                "prompt":  _CAPTION_PROMPT,
                "images":  [image_b64],
                "stream":  False,
                "options": {"temperature": 0.1},
            },
            timeout=180,
        )
        resp.raise_for_status()
        description = resp.json().get("response", "").strip()
        return f"\n[{label}: {description}]\n"
    except Exception as e:
        logging.warning("[RichPDF] Caption failed (%s): %s", label, e)
        return f"\n[{label}: (caption unavailable — {e})]\n"


# ---------------------------------------------------------------------------
# CLI — quick smoke-test without running the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m utils.rich_pdf_extractor path/to/paper.pdf [--no-figures]")
        sys.exit(1)

    _path     = Path(sys.argv[1])
    _do_figs  = "--no-figures" not in sys.argv

    print(f"Extracting: {_path}  (figures={'yes' if _do_figs else 'no'})\n")
    _result = extract_rich_text(_path, caption_figures=_do_figs)

    # Print first 3 000 chars as a preview
    print(_result[:3_000])
    print(f"\n... [{len(_result):,} total chars extracted]")
