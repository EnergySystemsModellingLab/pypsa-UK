# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Download and safely extract the UK-specific data archive from Zenodo."""

import logging
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import requests

from scripts._helpers import configure_logging

logger = logging.getLogger(__name__)

ZENODO_ARCHIVE_URL = "https://zenodo.org/api/records/20522643/files-archive"


def extract_zip_safely(archive_path: Path, destination: Path) -> None:
    """Extract archive members only when their paths remain inside destination."""
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise zipfile.BadZipFile(f"Corrupt member in Zenodo archive: {corrupt_member}")

        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or "\\" in member.filename
            ):
                raise ValueError(f"Unsafe path in Zenodo archive: {member.filename}")

            target = (destination / Path(*member_path.parts)).resolve()
            if os.path.commonpath((root, target)) != str(root):
                raise ValueError(f"Unsafe path in Zenodo archive: {member.filename}")

            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed: {member.filename}")

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


def download_uk_data(destination: Path) -> None:
    """Download and extract the archived UK datasets into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".zenodo_20522643_complete"
    marker.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "uk_data.zip"
        response = requests.get(
            ZENODO_ARCHIVE_URL, stream=True, timeout=(10, 120)
        )
        try:
            response.raise_for_status()
            with archive_path.open("wb") as archive_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive_file.write(chunk)
        finally:
            response.close()

        extract_zip_safely(archive_path, destination)

    marker.write_text(f"Downloaded from {ZENODO_ARCHIVE_URL}\n", encoding="utf-8")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_uk_data")

    configure_logging(snakemake)
    try:
        download_uk_data(Path(snakemake.output.marker).parent)
    except Exception:
        logger.exception("Failed to retrieve UK-specific Zenodo data")
        raise
