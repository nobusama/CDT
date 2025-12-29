#!/usr/bin/env python3
"""
CDT v2 Data Download Script

Downloads embeddings and training data for CDT v2:
1. Enformer embeddings from seq2cells (Zenodo)
2. ESM-C embeddings from HuggingFace (Bitbol-Lab)
3. Gasperini CRISPRi data (training)
4. Morris CRISPR-Flow-FISH data (test)

Usage:
    python scripts/download_data.py --all           # Download all data
    python scripts/download_data.py --enformer      # Enformer only
    python scripts/download_data.py --proteome      # ESM-C embeddings only
    python scripts/download_data.py --gasperini     # Gasperini data only
    python scripts/download_data.py --morris        # Morris data only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = DATA_DIR / "config" / "data_paths.yaml"


def load_config():
    """Load data configuration from YAML file."""
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found at {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def check_disk_space(required_gb=15):
    """Check if there's enough disk space."""
    import shutil
    total, used, free = shutil.disk_usage(DATA_DIR)
    free_gb = free // (2**30)

    print(f"Disk space: {free_gb} GB available")

    if free_gb < required_gb:
        print(f"Warning: Less than {required_gb} GB free. Recommended: 15+ GB")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(0)

    return free_gb


def download_enformer(config):
    """
    Download Enformer embeddings from seq2cells Zenodo.

    Source: https://zenodo.org/records/13754626
    Size: ~2-3 GB
    """
    print("\n" + "="*60)
    print("Downloading Enformer embeddings from seq2cells...")
    print("="*60)

    output_dir = DATA_DIR / "raw" / "enformer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zenodo direct download URL
    zenodo_url = "https://zenodo.org/records/13754626/files"

    print(f"\nSource: {zenodo_url}")
    print(f"Output: {output_dir}")
    print("\nNote: Check Zenodo page for exact file names.")
    print("URL: https://zenodo.org/records/13754626")
    print("\nManual download steps:")
    print("1. Visit the Zenodo URL above")
    print("2. Download the HDF5 files containing Enformer embeddings")
    print(f"3. Save to: {output_dir}")

    # TODO: Implement automatic download once file names are confirmed
    # Files to download may include:
    # - seq2cells_embeddings.h5 or similar

    print("\n[INFO] Enformer download requires manual verification of file names.")


def download_proteome(config):
    """
    Download ESM-C embeddings from HuggingFace (Bitbol-Lab/ProteomeLM-dataset).

    Size: ~6.89 GB (Parquet files)
    """
    print("\n" + "="*60)
    print("Downloading ESM-C embeddings from HuggingFace...")
    print("="*60)

    output_dir = DATA_DIR / "raw" / "proteome"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = "Bitbol-Lab/ProteomeLM-dataset"

    print(f"\nSource: https://huggingface.co/datasets/{dataset_id}")
    print(f"Output: {output_dir}")

    # Use huggingface_hub to download
    try:
        from huggingface_hub import snapshot_download

        print("\nDownloading with huggingface_hub...")
        print("This may take a while (~6.89 GB)...")

        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=output_dir / "esm_c_embeddings",
            local_dir_use_symlinks=False,
        )

        print(f"\n[SUCCESS] ESM-C embeddings downloaded to {output_dir}")

    except ImportError:
        print("\n[INFO] huggingface_hub not installed.")
        print("Install with: pip install huggingface_hub")
        print("\nAlternative manual download:")
        print(f"  1. Visit: https://huggingface.co/datasets/{dataset_id}")
        print("  2. Download the Parquet files")
        print(f"  3. Save to: {output_dir}/esm_c_embeddings/")

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Please download manually from HuggingFace.")


def download_gasperini(config):
    """
    Download Gasperini et al. (2019) CRISPRi data.

    Source: GEO/supplementary from Cell paper
    Use: Training data
    """
    print("\n" + "="*60)
    print("Gasperini et al. (2019) CRISPRi Data")
    print("="*60)

    output_dir = DATA_DIR / "raw" / "gasperini"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput: {output_dir}")
    print("\nPaper: Gasperini et al. (2019) Cell")
    print("Title: A Genome-wide Framework for Mapping Gene Regulation via Cellular Genetic Screens")
    print("\nData sources:")
    print("  - GEO: Check paper supplementary for GEO accession")
    print("  - Supplementary tables from Cell website")
    print("\nRequired files:")
    print("  - Enhancer-gene pairs (validated)")
    print("  - CRISPRi effect sizes")
    print("  - K562 cell line data")

    print("\n[INFO] Gasperini data requires manual download from paper/GEO.")
    print("This is the TRAINING data for CDT v2.")


def download_morris(config):
    """
    Download Morris et al. (2023) CRISPR-Flow-FISH data.

    Source: GEO/supplementary from Science paper
    Use: Independent test data (DO NOT use for training)
    """
    print("\n" + "="*60)
    print("Morris et al. (2023) CRISPR-Flow-FISH Data")
    print("="*60)

    output_dir = DATA_DIR / "raw" / "morris"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput: {output_dir}")
    print("\nPaper: Morris et al. (2023) Science")
    print("Title: Discovery of target genes and pathways at GWAS loci by pooled single-cell CRISPR screens")
    print("\nData sources:")
    print("  - GEO: Check paper supplementary for GEO accession")
    print("  - Likely in H5AD format (AnnData)")
    print("\nExpected content:")
    print("  - 507 mutation targets")
    print("  - 38,916 cells")
    print("  - 188 ADT markers (surface proteins)")
    print("  - K562 cell line")

    print("\n[INFO] Morris data requires manual download from paper/GEO.")
    print("\n*** IMPORTANT: This is the TEST data. ***")
    print("*** DO NOT use for training or validation. ***")
    print("*** Reserve for final evaluation only. ***")


def verify_downloads():
    """Verify downloaded data exists and report status."""
    print("\n" + "="*60)
    print("Download Status Verification")
    print("="*60)

    checks = [
        ("Enformer", DATA_DIR / "raw" / "enformer"),
        ("ESM-C/ProteomeLM", DATA_DIR / "raw" / "proteome"),
        ("Gasperini (train)", DATA_DIR / "raw" / "gasperini"),
        ("Morris (test)", DATA_DIR / "raw" / "morris"),
    ]

    for name, path in checks:
        files = list(path.glob("*")) if path.exists() else []
        # Filter out .gitkeep
        files = [f for f in files if f.name != ".gitkeep"]

        if files:
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"  [{name:20s}] {len(files):3d} files, {size_mb:.1f} MB")
        else:
            print(f"  [{name:20s}] Not downloaded yet")


def main():
    parser = argparse.ArgumentParser(
        description="Download data for CDT v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--all", action="store_true", help="Download all data")
    parser.add_argument("--enformer", action="store_true", help="Download Enformer embeddings")
    parser.add_argument("--proteome", action="store_true", help="Download ESM-C embeddings")
    parser.add_argument("--gasperini", action="store_true", help="Download Gasperini training data")
    parser.add_argument("--morris", action="store_true", help="Download Morris test data")
    parser.add_argument("--verify", action="store_true", help="Verify download status")

    args = parser.parse_args()

    # If no args specified, show help
    if not any([args.all, args.enformer, args.proteome, args.gasperini, args.morris, args.verify]):
        parser.print_help()
        print("\n" + "="*60)
        print("Current download status:")
        verify_downloads()
        return

    # Load config
    config = load_config()

    print("="*60)
    print("CDT v2 Data Download Script")
    print("="*60)

    # Check disk space
    check_disk_space()

    # Download requested data
    if args.all or args.enformer:
        download_enformer(config)

    if args.all or args.proteome:
        download_proteome(config)

    if args.all or args.gasperini:
        download_gasperini(config)

    if args.all or args.morris:
        download_morris(config)

    # Always verify at end
    verify_downloads()

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Complete any manual downloads listed above")
    print("2. Run: python scripts/download_data.py --verify")
    print("3. Proceed to data preprocessing scripts")


if __name__ == "__main__":
    main()
