"""Download datasets and model checkpoints used by the experiments."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import hashlib
import zipfile
import requests



# ZENODO RECORDS

REAL_GOBO_RECORD_ID = "8085511"
SYNTHETIC_GOBO_RECORD_ID = "20311815"


# FILE NAMES

REAL_GOBO_ZIP_FILENAME = "GoBo_v1-0.zip"

SYNTHETIC_GOBO_ONE_SHOT_ZIP_FILENAME = "01_GoBo_synthetic_one_shot.zip"
SYNTHETIC_GOBO_FEW_SHOT_ZIP_FILENAME = "02_GoBo_synthetic_few_shot.zip"
BIG_STYLE_TRANSFER_ZIP_FILENAME = "03_GoBo_synthetic_few_shot_10k_common_words.zip"

GENERIC_IAM_MODEL_FILENAME = "AttentionHTR_IAM_baseline_model.pth"


# OUTPUT PATHS

REAL_GOBO_OUTPUT_DIR = Path("data") / "real"

SYNTHETIC_ROOT_DIR = Path("data") / "synthetic"

SYNTHETIC_GOBO_ONE_SHOT_OUTPUT_DIR = (
    SYNTHETIC_ROOT_DIR / "synthetic_gobo_one_shot"
)

SYNTHETIC_GOBO_FEW_SHOT_OUTPUT_DIR = (
    SYNTHETIC_ROOT_DIR / "synthetic_gobo_few_shot"
)

BIG_STYLE_TRANSFER_OUTPUT_DIR = (
    SYNTHETIC_ROOT_DIR / "big_style_transfer"
)

GENERIC_IAM_MODEL_OUTPUT_DIR = Path("generic_model")


# EXTRACTION MARKERS

REAL_GOBO_EXTRACT_MARKER = ".real_gobo_extracted"
SYNTHETIC_GOBO_ONE_SHOT_EXTRACT_MARKER = ".synthetic_gobo_one_shot_extracted"
SYNTHETIC_GOBO_FEW_SHOT_EXTRACT_MARKER = ".synthetic_gobo_few_shot_extracted"
BIG_STYLE_TRANSFER_EXTRACT_MARKER = ".big_style_transfer_extracted"


# BASIC HELPERS

def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_zenodo_record(record_id: str) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"

    print("=" * 80)
    print(f"Reading Zenodo metadata: {record_id}")
    print("=" * 80)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    return response.json()


def get_file_info_from_record(record: dict, filename: str) -> dict:
    files = record.get("files", [])

    for file_info in files:
        if file_info.get("key") == filename:
            return file_info

    available_files = [file_info.get("key") for file_info in files]

    raise FileNotFoundError(
        f"{filename} was not found in Zenodo record.\n"
        f"Available files:\n{available_files}"
    )


def get_file_download_url(file_info: dict) -> str:
    links = file_info.get("links", {})

    if "self" in links:
        return links["self"]

    if "content" in links:
        return links["content"]

    raise KeyError("No download link found in Zenodo file metadata.")


def download_file(
    url: str,
    output_path: Path,
    expected_size: int | None = None,
    expected_checksum: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if expected_size is not None:
            current_size = output_path.stat().st_size

            if current_size == expected_size:
                print(f"File already exists with correct size: {output_path}")
            else:
                print(f"Existing file has wrong size. Re-downloading: {output_path}")
                output_path.unlink()
        else:
            print(f"File already exists: {output_path}")

    if not output_path.exists():
        print("=" * 80)
        print(f"Downloading: {output_path.name}")
        print(f"Target: {output_path}")
        print("=" * 80)

        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            downloaded = 0

            with output_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        percent = downloaded / total * 100
                        print(f"\r{percent:.1f}% downloaded", end="")

        print(f"\nSaved to: {output_path}")

    if expected_checksum and expected_checksum.startswith("md5:"):
        expected_md5 = expected_checksum.replace("md5:", "")
        actual_md5 = md5_file(output_path)

        if actual_md5 != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {output_path.name}: "
                f"expected {expected_md5}, got {actual_md5}"
            )

        print(f"MD5 OK: {output_path.name}")

    return output_path


def safe_extract_zip(zip_path: Path, output_dir: Path, marker_name: str) -> None:
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)

    marker_path = output_dir / marker_name

    if marker_path.exists():
        print(f"Already extracted, skipping: {output_dir}")
        return

    print("=" * 80)
    print(f"Extracting: {zip_path.name}")
    print(f"Into: {output_dir}")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        resolved_output = output_dir.resolve()

        for member in zip_ref.infolist():
            target_path = output_dir / member.filename
            resolved_target = target_path.resolve()

            if not str(resolved_target).startswith(str(resolved_output)):
                raise RuntimeError(f"Unsafe path in ZIP file: {member.filename}")

        zip_ref.extractall(output_dir)

    marker_path.write_text("Extracted successfully.\n", encoding="utf-8")

    print(f"Extracted successfully: {output_dir}")


def download_zenodo_file(
    record_id: str,
    filename: str,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir)
    output_path = output_dir / filename

    record = get_zenodo_record(record_id)
    file_info = get_file_info_from_record(record, filename)

    file_url = get_file_download_url(file_info)
    file_size = file_info.get("size")
    checksum = file_info.get("checksum", "")

    downloaded_path = download_file(
        url=file_url,
        output_path=output_path,
        expected_size=file_size,
        expected_checksum=checksum,
    )

    return downloaded_path


# REAL GOBO DATA

def download_real_gobo_data(
    output_dir: Path = REAL_GOBO_OUTPUT_DIR,
    extract: bool = True,
) -> Path:
    output_dir = Path(output_dir)

    zip_path = download_zenodo_file(
        record_id=REAL_GOBO_RECORD_ID,
        filename=REAL_GOBO_ZIP_FILENAME,
        output_dir=output_dir,
    )

    if extract:
        safe_extract_zip(
            zip_path=zip_path,
            output_dir=output_dir,
            marker_name=REAL_GOBO_EXTRACT_MARKER,
        )

    return output_dir


# SYNTHETIC GOBO ONE-SHOT DATA

def download_synthetic_gobo_one_shot_data(
    output_dir: Path = SYNTHETIC_GOBO_ONE_SHOT_OUTPUT_DIR,
    extract: bool = True,
) -> Path:
    output_dir = Path(output_dir)

    zip_path = download_zenodo_file(
        record_id=SYNTHETIC_GOBO_RECORD_ID,
        filename=SYNTHETIC_GOBO_ONE_SHOT_ZIP_FILENAME,
        output_dir=output_dir,
    )

    if extract:
        safe_extract_zip(
            zip_path=zip_path,
            output_dir=output_dir,
            marker_name=SYNTHETIC_GOBO_ONE_SHOT_EXTRACT_MARKER,
        )

    return output_dir


# SYNTHETIC GOBO FEW-SHOT DATA

def download_synthetic_gobo_few_shot_data(
    output_dir: Path = SYNTHETIC_GOBO_FEW_SHOT_OUTPUT_DIR,
    extract: bool = True,
) -> Path:
    output_dir = Path(output_dir)

    zip_path = download_zenodo_file(
        record_id=SYNTHETIC_GOBO_RECORD_ID,
        filename=SYNTHETIC_GOBO_FEW_SHOT_ZIP_FILENAME,
        output_dir=output_dir,
    )

    if extract:
        safe_extract_zip(
            zip_path=zip_path,
            output_dir=output_dir,
            marker_name=SYNTHETIC_GOBO_FEW_SHOT_EXTRACT_MARKER,
        )

    return output_dir


# BIG STYLE TRANSFER DATA (10K WORDS)

def download_big_style_transfer_data(
    output_dir: Path = BIG_STYLE_TRANSFER_OUTPUT_DIR,
    extract: bool = True,
) -> Path:
    output_dir = Path(output_dir)

    zip_path = download_zenodo_file(
        record_id=SYNTHETIC_GOBO_RECORD_ID,
        filename=BIG_STYLE_TRANSFER_ZIP_FILENAME,
        output_dir=output_dir,
    )

    if extract:
        safe_extract_zip(
            zip_path=zip_path,
            output_dir=output_dir,
            marker_name=BIG_STYLE_TRANSFER_EXTRACT_MARKER,
        )

    return output_dir


# GENERIC IAM MODEL

def download_generic_IAM_model(
    output_dir: Path = GENERIC_IAM_MODEL_OUTPUT_DIR,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = download_zenodo_file(
        record_id=SYNTHETIC_GOBO_RECORD_ID,
        filename=GENERIC_IAM_MODEL_FILENAME,
        output_dir=output_dir,
    )

    return model_path


# OPTIONAL: DOWNLOAD EVERYTHING

def download_all_data(extract: bool = True) -> None:
    download_real_gobo_data(extract=extract)
    download_synthetic_gobo_one_shot_data(extract=extract)
    download_synthetic_gobo_few_shot_data(extract=extract)
    download_big_style_transfer_data(extract=extract)
    download_generic_IAM_model()

    print("=" * 80)
    print("All requested data is ready.")
    print("=" * 80)


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real GoBo data, synthetic GoBo data, and generic IAM model."
    )

    parser.add_argument(
        "--real-gobo",
        action="store_true",
        help="Download real GoBo dataset.",
    )

    parser.add_argument(
        "--synthetic-gobo-one-shot",
        action="store_true",
        help="Download synthetic GoBo one-shot dataset.",
    )

    parser.add_argument(
        "--synthetic-gobo-few-shot",
        action="store_true",
        help="Download synthetic GoBo few-shot dataset.",
    )

    parser.add_argument(
        "--big-style-transfer",
        action="store_true",
        help="Download the 10,000-sample style-transfer dataset for Big Style Transfer.",
    )

    parser.add_argument(
        "--generic-iam-model",
        action="store_true",
        help="Download generic IAM baseline model.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all data and model.",
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download ZIP files, do not extract them.",
    )

    args = parser.parse_args()
    extract = not args.no_extract

    if args.all:
        download_all_data(extract=extract)
        return

    if args.real_gobo:
        download_real_gobo_data(extract=extract)

    if args.synthetic_gobo_one_shot:
        download_synthetic_gobo_one_shot_data(extract=extract)

    if args.synthetic_gobo_few_shot:
        download_synthetic_gobo_few_shot_data(extract=extract)

    if args.big_style_transfer:
        download_big_style_transfer_data(extract=extract)

    if args.generic_iam_model:
        download_generic_IAM_model()

    if (
        not args.real_gobo
        and not args.synthetic_gobo_one_shot
        and not args.synthetic_gobo_few_shot
        and not args.big_style_transfer
        and not args.generic_iam_model
        and not args.all
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
