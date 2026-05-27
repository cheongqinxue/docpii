"""Document loading helpers for multimodal inference."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


def encode_image_path(image_path: str | Path) -> str:
    """Base64-encode an image file for OpenAI-compatible image_url payloads.

    Args:
        image_path: Path to a PNG, JPEG, or other supported image file.

    Returns:
        Base64-encoded image bytes (without data URL prefix).
    """
    path = Path(image_path)
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def image_path_to_data_url(image_path: str | Path) -> str:
    """Convert an image file to a data URL.

    Args:
        image_path: Path to an image file.

    Returns:
        Data URL string suitable for vLLM ``image_url`` content.
    """
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    encoded = encode_image_path(path)
    return f"data:{mime_type};base64,{encoded}"


def load_document_images(document_path: str | Path, *, dpi: int = 170) -> list[str]:
    """Load a document as one or more image data URLs.

    PDFs are rendered page-by-page with PyMuPDF when available; otherwise a
    single image path is returned.

    Args:
        document_path: Path to a PDF or image file.
        dpi: Render resolution for PDF pages.

    Returns:
        Ordered list of image data URLs (one per page for PDFs).

    Raises:
        FileNotFoundError: If the document does not exist.
        ValueError: If the file type is unsupported.
    """
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PDF support requires pymupdf. Install with: pip install 'docpii[pdf]'"
            ) from exc

        data_urls: list[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                png_bytes = pix.tobytes("png")
                encoded = base64.b64encode(png_bytes).decode("utf-8")
                data_urls.append(f"data:image/png;base64,{encoded}")
        return data_urls

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
        return [image_path_to_data_url(path)]

    raise ValueError(f"Unsupported document type: {suffix}")
