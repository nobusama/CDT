#!/usr/bin/env python3
"""
Download hg38 reference genome from UCSC.

Output: data/raw/reference/hg38.fa (~3GB uncompressed)

Usage:
    python scripts/data_prep/download_hg38.py
"""

import os
import sys
import urllib.request
import gzip
import shutil
from pathlib import Path

# URLs for hg38 reference genome
UCSC_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
OUTPUT_DIR = Path("data/raw/reference")
OUTPUT_FILE = OUTPUT_DIR / "hg38.fa"
COMPRESSED_FILE = OUTPUT_DIR / "hg38.fa.gz"


def download_with_progress(url: str, output_path: Path) -> None:
    """Download file with progress indicator."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print("\nDownload complete!")


def decompress_gzip(input_path: Path, output_path: Path) -> None:
    """Decompress gzip file."""
    print(f"Decompressing {input_path} -> {output_path}")
    with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Decompression complete!")


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    if OUTPUT_FILE.exists():
        size_gb = OUTPUT_FILE.stat().st_size / (1024**3)
        print(f"hg38.fa already exists ({size_gb:.2f} GB)")
        print("Delete it to re-download.")
        return

    # Download compressed file
    if not COMPRESSED_FILE.exists():
        download_with_progress(UCSC_URL, COMPRESSED_FILE)
    else:
        print(f"Compressed file already exists: {COMPRESSED_FILE}")

    # Decompress
    decompress_gzip(COMPRESSED_FILE, OUTPUT_FILE)

    # Verify
    if OUTPUT_FILE.exists():
        size_gb = OUTPUT_FILE.stat().st_size / (1024**3)
        print(f"\nSuccess! hg38.fa: {size_gb:.2f} GB")

        # Show first few lines
        print("\nFirst 5 lines:")
        with open(OUTPUT_FILE, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"  {line.rstrip()[:80]}...")

        # Clean up compressed file
        response = input("\nDelete compressed file to save space? [y/N]: ")
        if response.lower() == 'y':
            COMPRESSED_FILE.unlink()
            print("Compressed file deleted.")
    else:
        print("Error: Output file not created!")
        sys.exit(1)


if __name__ == "__main__":
    main()
