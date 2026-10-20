# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for retrieving the UK-specific Zenodo data archive."""

from io import BytesIO
import zipfile

import pytest

from scripts import retrieve_uk_data


def make_zip(entries: dict[str, bytes]) -> bytes:
    """Create an in-memory ZIP archive from named byte contents."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class FakeResponse:
    """Minimal streamed-response stub for the downloader tests."""

    def __init__(self, content: bytes):
        """Initialize the response with the archive bytes."""
        self.content = content
        self.closed = False

    def raise_for_status(self):
        """Match the requests response method."""
        pass

    def iter_content(self, chunk_size):
        """Yield the archive bytes as one streamed chunk."""
        yield self.content

    def close(self):
        """Record that the downloader closed the response."""
        self.closed = True


def test_download_extracts_archive_and_writes_marker(tmp_path, monkeypatch):
    """Download and extract archive contents, then write the completion marker."""
    response = FakeResponse(make_zip({"folder/UK_dataset.csv": b"value\n1\n"}))
    monkeypatch.setattr(
        retrieve_uk_data.requests, "get", lambda *args, **kwargs: response
    )

    retrieve_uk_data.download_uk_data(tmp_path)

    assert (tmp_path / "folder/UK_dataset.csv").read_bytes() == b"value\n1\n"
    assert (tmp_path / ".zenodo_20522643_complete").is_file()
    assert response.closed


def test_extract_rejects_paths_outside_destination(tmp_path):
    """Reject ZIP members that attempt path traversal."""
    archive_path = tmp_path / "unsafe.zip"
    archive_path.write_bytes(make_zip({"../outside.txt": b"unsafe"}))

    with pytest.raises(ValueError, match="Unsafe path"):
        retrieve_uk_data.extract_zip_safely(archive_path, tmp_path / "data_UK")

    assert not (tmp_path / "outside.txt").exists()


def test_extract_rejects_symbolic_links(tmp_path):
    """Reject ZIP members that encode symbolic links."""
    archive_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (0o120777 << 16)
        archive.writestr(member, "target")

    with pytest.raises(ValueError, match="Symbolic links"):
        retrieve_uk_data.extract_zip_safely(archive_path, tmp_path / "data_UK")
