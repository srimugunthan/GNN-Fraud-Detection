"""
================================================================================
Dataset Downloader — IEEE-CIS Fraud Detection (Kaggle)
================================================================================

Downloads the raw competition archive (ieee-fraud-detection.zip) from Kaggle
into DATA_DIR, using the Kaggle CLI under the hood.

export KAGGLE_API_TOKEN=
One-time setup:
    1. pip install kaggle
    2. Get an API token: https://www.kaggle.com/settings -> "Create New Token"
       This downloads a kaggle.json file.
    3. Place it at ~/.kaggle/kaggle.json (chmod 600 ~/.kaggle/kaggle.json), or
       set the KAGGLE_USERNAME / KAGGLE_KEY environment variables instead.
    4. Accept the competition rules (Kaggle refuses the download otherwise):
       https://www.kaggle.com/c/ieee-fraud-detection/rules

Usage:
    python download_dataset.py                  # download the zip only
    python download_dataset.py --unzip           # download and extract the CSVs
    python download_dataset.py --force           # re-download even if present
    python download_dataset.py --data-dir other  # use a different destination
================================================================================
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile

COMPETITION = "ieee-fraud-detection"
DATA_DIR = "ieee_cis_dataset"
ZIP_NAME = f"{COMPETITION}.zip"


def check_kaggle_cli():
    """Verify the `kaggle` CLI is installed and on PATH."""
    if shutil.which("kaggle") is None:
        sys.exit(
            "The 'kaggle' CLI is not installed or not on PATH.\n"
            "Install it with:  pip install kaggle\n"
            "Then set up credentials: https://www.kaggle.com/docs/api#authentication"
        )


def download_dataset(data_dir: str = DATA_DIR, competition: str = COMPETITION, force: bool = False) -> str:
    """Download the competition zip into data_dir. Returns the path to the zip."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, ZIP_NAME)

    if os.path.exists(zip_path) and not force:
        print(f"Already downloaded: {zip_path} (use --force to re-download)")
        return zip_path

    check_kaggle_cli()

    print(f"Downloading '{competition}' into {data_dir}/ ...")
    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", data_dir]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(
            "Download failed. Common causes:\n"
            "  - Kaggle credentials not configured (missing ~/.kaggle/kaggle.json)\n"
            f"  - Competition rules not accepted: https://www.kaggle.com/c/{competition}/rules"
        )

    if not os.path.exists(zip_path):
        sys.exit(
            f"Expected zip not found at {zip_path} after download — "
            "check the Kaggle CLI output above for the actual filename."
        )

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Downloaded: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def extract_zip(zip_path: str, data_dir: str = DATA_DIR):
    """Extract the downloaded zip's CSVs into data_dir."""
    print(f"Extracting {zip_path} -> {data_dir}/ ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    print("Extraction complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Download the IEEE-CIS Fraud Detection dataset from Kaggle."
    )
    parser.add_argument("--data-dir", default=DATA_DIR,
                         help=f"Destination directory (default: {DATA_DIR})")
    parser.add_argument("--competition", default=COMPETITION,
                         help=f"Kaggle competition slug (default: {COMPETITION})")
    parser.add_argument("--unzip", action="store_true",
                         help="Extract the CSVs after downloading")
    parser.add_argument("--force", action="store_true",
                         help="Re-download even if the zip already exists")
    args = parser.parse_args()

    zip_path = download_dataset(args.data_dir, args.competition, args.force)

    if args.unzip:
        extract_zip(zip_path, args.data_dir)


if __name__ == "__main__":
    main()
