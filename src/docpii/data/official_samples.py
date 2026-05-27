"""Download official US identity document reference samples."""

from __future__ import annotations

import json
import logging
import mimetypes
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CATALOG = Path("data/sources/us_official_id_catalog.yaml")
DEFAULT_OUTPUT_DIR = Path("data/raw/official_samples")
DEFAULT_INDEX_PATH = Path("data/samples/official_download_index.json")

USER_AGENT = "docpii/0.1 (research; official .gov reference samples)"


@dataclass(frozen=True)
class OfficialSampleSpec:
    """One catalog entry for an official reference document."""

    id: str
    url: str
    document_type: str
    jurisdiction: str
    source_page: str
    description: str


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of downloading one sample."""

    sample_id: str
    success: bool
    local_path: str | None = None
    error: str | None = None
    content_type: str | None = None
    bytes_written: int = 0


@dataclass(frozen=True)
class DownloadReport:
    """Summary of a batch download run."""

    downloaded_at: str
    catalog_path: str
    output_dir: str
    index_path: str
    results: list[DownloadResult]

    @property
    def success_count(self) -> int:
        """Number of successful downloads."""
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        """Number of failed downloads."""
        return sum(1 for r in self.results if not r.success)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-friendly dictionary."""
        return {
            "downloaded_at": self.downloaded_at,
            "catalog_path": self.catalog_path,
            "output_dir": self.output_dir,
            "index_path": self.index_path,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "results": [asdict(r) for r in self.results],
        }


def load_catalog(catalog_path: str | Path) -> list[OfficialSampleSpec]:
    """Load sample specifications from a YAML catalog.

    Args:
        catalog_path: Path to the catalog YAML file.

    Returns:
        Parsed sample specifications.

    Raises:
        FileNotFoundError: If the catalog file does not exist.
        ValueError: If the catalog format is invalid.
    """
    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError(f"Invalid catalog format: {path}")

    specs: list[OfficialSampleSpec] = []
    for record in payload["samples"]:
        specs.append(
            OfficialSampleSpec(
                id=str(record["id"]),
                url=str(record["url"]),
                document_type=str(record["document_type"]),
                jurisdiction=str(record["jurisdiction"]),
                source_page=str(record["source_page"]),
                description=str(record["description"]),
            )
        )
    return specs


def _extension_from_url(url: str, content_type: str | None) -> str:
    """Infer a file extension from URL path or Content-Type."""
    path_name = unquote(urlparse(url).path)
    suffix = Path(path_name).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".tif", ".tiff"}:
        return suffix

    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext

    return ".bin"


def _local_path_for_sample(output_dir: Path, spec: OfficialSampleSpec, extension: str) -> Path:
    """Build a stable local path for a sample file."""
    jurisdiction = spec.jurisdiction.lower().replace(" ", "-")
    doc_type = spec.document_type.replace(" ", "_")
    return output_dir / jurisdiction / doc_type / f"{spec.id}{extension}"


def download_official_samples(
    catalog_path: str | Path = DEFAULT_CATALOG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    *,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> DownloadReport:
    """Download all samples listed in the catalog.

    Args:
        catalog_path: YAML catalog of official sample URLs.
        output_dir: Directory for downloaded binaries (gitignored).
        index_path: JSON index with provenance and local paths (committed).
        overwrite: Re-download files even if they already exist.
        timeout: HTTP timeout in seconds per request.

    Returns:
        Download report with per-sample results.
    """
    catalog_file = Path(catalog_path)
    out_dir = Path(output_dir)
    index_file = Path(index_path)
    specs = load_catalog(catalog_file)

    results: list[DownloadResult] = []
    index_entries: list[dict[str, Any]] = []

    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for spec in specs:
            try:
                result, entry = _download_one(client, out_dir, spec, overwrite=overwrite)
            except httpx.HTTPError as exc:
                logger.exception("HTTP error downloading %s", spec.id)
                result = DownloadResult(sample_id=spec.id, success=False, error=str(exc))
                entry = _index_entry(spec, None, error=str(exc))
            results.append(result)
            index_entries.append(entry)

    index_payload = {
        "downloaded_at": datetime.now(tz=UTC).isoformat(),
        "catalog_path": str(catalog_file),
        "output_dir": str(out_dir),
        "samples": index_entries,
    }
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index_file.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")

    report = DownloadReport(
        downloaded_at=index_payload["downloaded_at"],
        catalog_path=str(catalog_file),
        output_dir=str(out_dir),
        index_path=str(index_file),
        results=results,
    )
    logger.info(
        "Download complete: %d succeeded, %d failed",
        report.success_count,
        report.failure_count,
    )
    return report


def _index_entry(
    spec: OfficialSampleSpec,
    local_path: Path | None,
    *,
    error: str | None = None,
    content_type: str | None = None,
    bytes_written: int = 0,
) -> dict[str, Any]:
    """Build one provenance record for the download index."""
    return {
        "id": spec.id,
        "document_type": spec.document_type,
        "jurisdiction": spec.jurisdiction,
        "description": spec.description,
        "source_page": spec.source_page,
        "source_url": spec.url,
        "local_path": str(local_path) if local_path else None,
        "content_type": content_type,
        "bytes_written": bytes_written,
        "error": error,
    }


def _download_one(
    client: httpx.Client,
    output_dir: Path,
    spec: OfficialSampleSpec,
    *,
    overwrite: bool,
) -> tuple[DownloadResult, dict[str, Any]]:
    """Download a single catalog entry."""
    # Probe extension with HEAD when supported; many .gov hosts reject HEAD.
    content_type: str | None = None
    try:
        head = client.head(spec.url)
        if head.status_code < 400:
            content_type = head.headers.get("content-type")
    except httpx.HTTPError:
        logger.debug("HEAD unsupported for %s; will GET directly", spec.id)

    extension = _extension_from_url(spec.url, content_type)
    local_path = _local_path_for_sample(output_dir, spec, extension)

    if local_path.exists() and not overwrite:
        size = local_path.stat().st_size
        result = DownloadResult(
            sample_id=spec.id,
            success=True,
            local_path=str(local_path),
            content_type=content_type,
            bytes_written=size,
        )
        entry = _index_entry(spec, local_path, content_type=content_type, bytes_written=size)
        return result, entry

    response = client.get(spec.url)
    response.raise_for_status()
    content_type = response.headers.get("content-type", content_type)
    extension = _extension_from_url(spec.url, content_type)
    local_path = _local_path_for_sample(output_dir, spec, extension)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(response.content)

    result = DownloadResult(
        sample_id=spec.id,
        success=True,
        local_path=str(local_path),
        content_type=content_type,
        bytes_written=len(response.content),
    )
    entry = _index_entry(
        spec,
        local_path,
        content_type=content_type,
        bytes_written=len(response.content),
    )
    return result, entry
