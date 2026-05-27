"""Tests for official sample downloads."""

from pathlib import Path

import httpx

from docpii.data.official_samples import (
    OfficialSampleSpec,
    _download_one,
    load_catalog,
)


class TestLoadCatalog:
    """Catalog loading tests."""

    def test_load_default_catalog(self, project_root: Path) -> None:
        """Default catalog should parse with expected sample count."""
        catalog_path = project_root / "data/sources/us_official_id_catalog.yaml"
        specs = load_catalog(catalog_path)
        assert len(specs) >= 20
        assert specs[0].id
        assert specs[0].url.startswith("http")


class TestDownloadOne:
    """Single-sample download tests."""

    def test_download_writes_file(self, tmp_path: Path) -> None:
        """A successful HTTP response should be written to disk."""
        spec = OfficialSampleSpec(
            id="test-sample",
            url="https://example.com/sample.png",
            document_type="drivers_license",
            jurisdiction="US-TEST",
            source_page="https://example.com",
            description="Test sample",
        )
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "image/png"},
                content=b"\x89PNG\r\n",
            )
        )
        client = httpx.Client(transport=transport)

        result, entry = _download_one(client, tmp_path, spec, overwrite=False)

        assert result.success
        assert result.local_path is not None
        assert Path(result.local_path).exists()
        assert entry["source_url"] == spec.url
