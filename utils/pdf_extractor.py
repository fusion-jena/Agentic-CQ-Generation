from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}


def extract_text(file_path: Path) -> str:
    """Return plain text from a PDF, .txt, or .md file."""
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED))}"
        )

    if suffix == ".pdf":
        try:
            import pypdf
        except ImportError:
            raise ImportError("PDF support requires pypdf — run: pip install pypdf")

        reader = pypdf.PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    return file_path.read_text(encoding="utf-8")
