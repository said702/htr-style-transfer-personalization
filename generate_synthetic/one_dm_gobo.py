"""Generate the one-shot GoBo-style synthetic handwriting dataset with One-DM."""

import inspect
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np


# Settings

WRITER_IDS = [i for i in range(41) if i != 33]

NUM_STYLE_SAMPLES = 1
SEED = 1

TRAIN_TXTS = [
    "brown",
    "cedar",
    "nonwords",
    "domain_A_train",
    "domain_B_train",
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

ONE_DM_DIR_NAME = "One-DM"

MODEL_DIR_NAME = "onedm_model"
MODEL_FILENAME = "One-DM-ckpt.pt"

# Official One-DM Google Drive model_zoo folder from the GitHub README.
ONE_DM_GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/10KOQ05HeN2kaR2_OCZNl9D_Kh1p8BDaa"

# Official / prepared One-DM data folder.
# It contains English_Data.zip, which includes data/unifont.pickle.
ONE_DM_DATA_GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/108TB-z2ytAZSIEzND94dyufybjpqVyn6"
REQUIRED_ONE_DM_PICKLE = "unifont.pickle"

STYLE_OUTPUT_DIR = "one_shot_styles"
GENERATED_OUTPUT_DIR = "one_shot_samples"

SKIP_EXISTING = True

DEVICE = "cuda"
SAMPLING_TIMESTEPS = 50
SAMPLE_METHOD = "ddim"
ETA = 0.0

STABLE_DIFFUSION_PATH = "runwayml/stable-diffusion-v1-5"


# Basic helpers

def format_seconds(seconds):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def run_cmd(cmd, cwd=None):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))

    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
    )


def check_gdown_available():
    try:
        import gdown  # noqa: F401
    except ImportError:
        raise ImportError(
            "gdown is not installed. Install it first:\n"
            "pip install gdown"
        )


def filename_from_gt_path(rel_img):
    """
    Keep only the image filename from the left side of the GT line.

    Example:
    brown/0-brown-0-0-0.png
    -> 0-brown-0-0-0.png
    """
    return Path(rel_img.replace("\\", "/")).name


def get_project_root():
    return PROJECT_ROOT


def get_one_dm_dir(project_root):
    one_dm_dir = project_root / ONE_DM_DIR_NAME

    if not one_dm_dir.exists():
        raise FileNotFoundError(
            f"One-DM folder not found: {one_dm_dir}\n"
            "This script assumes that the One-DM repository already exists."
        )

    return one_dm_dir


# One-DM checkpoint

def find_existing_one_dm_checkpoint(project_root):
    candidates = [
        project_root / MODEL_DIR_NAME / MODEL_FILENAME,
        project_root / MODEL_DIR_NAME / "epoch-ckpt.pt",
        project_root / MODEL_DIR_NAME / "one_dm.pt",
        project_root / MODEL_DIR_NAME / "one_dm.pth",
        project_root / MODEL_DIR_NAME / "One-DM-ckpt.pth",
        project_root / "model_zoo" / MODEL_FILENAME,
        project_root / "model_zoo" / "epoch-ckpt.pt",
        project_root / "One-DM-ckpt.pt",
        project_root / "ONE_DM_Model" / MODEL_FILENAME,
    ]

    for path in candidates:
        if path.exists():
            return path

    model_dir = project_root / MODEL_DIR_NAME

    if model_dir.exists():
        pt_files = list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.pth"))

        if pt_files:
            pt_files = sorted(pt_files, key=lambda p: p.stat().st_size, reverse=True)
            return pt_files[0]

    return None


def find_downloaded_one_dm_checkpoint(download_dir):
    pt_files = list(download_dir.rglob("*.pt")) + list(download_dir.rglob("*.pth"))

    if not pt_files:
        return None

    priority_keywords = [
        "one",
        "dm",
        "epoch",
        "ckpt",
    ]

    def score(path):
        name = path.name.lower()
        s = 0

        for keyword in priority_keywords:
            if keyword in name:
                s += 10

        s += path.stat().st_size / 1_000_000_000
        return s

    pt_files = sorted(pt_files, key=score, reverse=True)
    return pt_files[0]


def download_google_drive_folder(url, output_dir):
    """
    Download a Google Drive folder using the installed gdown version.

    This avoids unsupported arguments such as remaining_ok.
    """
    check_gdown_available()
    import gdown

    output_dir.mkdir(parents=True, exist_ok=True)

    signature = inspect.signature(gdown.download_folder)
    supported_args = set(signature.parameters.keys())

    kwargs = {
        "url": url,
        "output": str(output_dir),
        "quiet": False,
        "use_cookies": False,
    }

    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_args
    }

    try:
        return gdown.download_folder(**kwargs)

    except TypeError:
        print("gdown.download_folder failed with current API. Trying CLI fallback...")

        cmd = [
            sys.executable,
            "-m",
            "gdown",
            "--folder",
            url,
            "-O",
            str(output_dir),
        ]

        subprocess.run(cmd, check=True)
        return None


def ensure_one_dm_checkpoint(project_root):
    """
    Find the One-DM checkpoint locally.

    If it is missing, download the official model_zoo folder from Google Drive,
    select the most likely One-DM checkpoint, and save it as:

    onedm_model/One-DM-ckpt.pt
    """
    model_dir = project_root / MODEL_DIR_NAME
    model_dir.mkdir(parents=True, exist_ok=True)

    existing_checkpoint = find_existing_one_dm_checkpoint(project_root)

    if existing_checkpoint is not None:
        print(f"One-DM checkpoint found: {existing_checkpoint}")
        return existing_checkpoint

    print("One-DM checkpoint not found.")
    print(f"Downloading official One-DM model_zoo to: {model_dir}")

    temp_download_dir = model_dir / "_downloaded_model_zoo"

    if temp_download_dir.exists():
        shutil.rmtree(temp_download_dir)

    temp_download_dir.mkdir(parents=True, exist_ok=True)

    download_google_drive_folder(
        url=ONE_DM_GOOGLE_DRIVE_FOLDER_URL,
        output_dir=temp_download_dir,
    )

    downloaded_checkpoint = find_downloaded_one_dm_checkpoint(temp_download_dir)

    if downloaded_checkpoint is None:
        raise FileNotFoundError(
            "No .pt or .pth file was found after downloading the One-DM model_zoo.\n"
            f"Download folder: {temp_download_dir}"
        )

    target_checkpoint = model_dir / MODEL_FILENAME

    if target_checkpoint.exists():
        target_checkpoint.unlink()

    shutil.copy2(downloaded_checkpoint, target_checkpoint)

    print(f"Downloaded checkpoint selected: {downloaded_checkpoint}")
    print(f"Checkpoint saved as: {target_checkpoint}")

    return target_checkpoint


# One-DM data files, especially data/unifont.pickle

def find_file_by_name(root_dir, filename):
    """
    Search recursively for a file with an exact filename.
    """
    matches = list(root_dir.rglob(filename))
    if not matches:
        return None

    # Prefer shorter paths, because they are usually the intended extracted file.
    matches = sorted(matches, key=lambda p: (len(p.parts), str(p).lower()))
    return matches[0]


def extract_zip_files(download_dir):
    """
    Extract all ZIP files found inside download_dir.

    The Google Drive folder may contain English_Data.zip. This function is
    robust to different capitalization or nested folder structures.
    """
    zip_files = sorted(download_dir.rglob("*.zip"))

    if not zip_files:
        print(f"No ZIP files found in: {download_dir}")
        return []

    extracted_roots = []
    extract_base_dir = download_dir / "_extracted_zip_files"
    extract_base_dir.mkdir(parents=True, exist_ok=True)

    for zip_path in zip_files:
        target_dir = extract_base_dir / zip_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting ZIP: {zip_path}")
        print(f"Extract target: {target_dir}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        extracted_roots.append(target_dir)

    return extracted_roots


def ensure_one_dm_data_files(one_dm_dir):
    """
    Ensure that One-DM/data/unifont.pickle exists.

    If it is missing, download the One-DM data Google Drive folder, extract
    English_Data.zip, find unifont.pickle, and copy it to:

    One-DM/data/unifont.pickle
    """
    data_dir = one_dm_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    required_pickle_path = data_dir / REQUIRED_ONE_DM_PICKLE

    if required_pickle_path.exists():
        print(f"One-DM data file found: {required_pickle_path}")
        return required_pickle_path

    print(f"One-DM data file not found: {required_pickle_path}")
    print("Downloading One-DM English data folder...")

    temp_download_dir = data_dir / "_downloaded_english_data"

    if temp_download_dir.exists():
        shutil.rmtree(temp_download_dir)

    temp_download_dir.mkdir(parents=True, exist_ok=True)

    download_google_drive_folder(
        url=ONE_DM_DATA_GOOGLE_DRIVE_FOLDER_URL,
        output_dir=temp_download_dir,
    )

    # First search directly in the downloaded folder.
    found_pickle = find_file_by_name(temp_download_dir, REQUIRED_ONE_DM_PICKLE)

    # If not found directly, extract ZIP files such as English_Data.zip.
    extracted_roots = []

    if found_pickle is None:
        extracted_roots = extract_zip_files(temp_download_dir)

        for root in extracted_roots:
            found_pickle = find_file_by_name(root, REQUIRED_ONE_DM_PICKLE)
            if found_pickle is not None:
                break

    if found_pickle is None:
        searched_locations = [str(temp_download_dir)] + [str(root) for root in extracted_roots]
        raise FileNotFoundError(
            f"Could not find {REQUIRED_ONE_DM_PICKLE} after downloading and extracting One-DM data.\n"
            "Searched locations:\n"
            + "\n".join(searched_locations)
        )

    shutil.copy2(found_pickle, required_pickle_path)

    print(f"Found pickle: {found_pickle}")
    print(f"Copied pickle to: {required_pickle_path}")

    return required_pickle_path


# GoBo reading

def get_gobo_root(project_root, writer_id):
    return project_root / "data" / "real" / "GoBo_v1-0" / "words" / str(writer_id)


def deduplicate_entries_by_output_filename(entries, writer_id):
    """
    Keep only one entry per output filename.

    If the same filename appears multiple times, the later entry is kept.
    """
    filename_to_entry = {}
    duplicate_count = 0

    for entry in entries:
        filename = entry["output_filename"]

        if filename in filename_to_entry:
            duplicate_count += 1

        filename_to_entry[filename] = entry

    deduplicated_entries = list(filename_to_entry.values())

    if duplicate_count > 0:
        print(
            f"writer_{writer_id}: {duplicate_count} duplicate output filenames found. "
            f"Later entries were kept."
        )

    return deduplicated_entries


def collect_gobo_entries(project_root, writer_id):
    """
    Read all valid GoBo train entries for one writer.

    No intersection is used:
    each writer gets all valid samples from its own train TXT files.

    The output filename is taken from the left side of the GT line.
    """
    gobo_root = get_gobo_root(project_root, writer_id)

    if not gobo_root.exists():
        raise FileNotFoundError(f"GoBo writer folder not found: {gobo_root}")

    entries = []

    for txt_name in TRAIN_TXTS:
        txt_path = gobo_root / f"{txt_name}.txt"
        if not txt_path.exists():
            print(f"Warning: missing txt file: {txt_path}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                parts = line.strip().split(maxsplit=2)

                if len(parts) != 3:
                    continue

                rel_img, status, word = parts

                if status not in  {"ok","rw"}:
                    continue
                if not rel_img.lower().endswith(IMAGE_EXTS):
                    continue

                img_path = gobo_root / rel_img
                if not img_path.exists():
                    continue

                output_filename = filename_from_gt_path(rel_img)

                entries.append({
                    "img_path": img_path,
                    "rel_img": rel_img,
                    "word": word,
                    "dataset": txt_name,
                    "source_filename": img_path.name,
                    "output_filename": output_filename,
                    "line_idx": line_idx,
                })

    if not entries:
        raise ValueError(f"No valid GoBo train entries found for writer {writer_id}")

    entries = deduplicate_entries_by_output_filename(entries, writer_id)
    return entries


def collect_entries_per_writer(project_root):
    entries_per_writer = {}

    for writer_id in WRITER_IDS:
        entries = collect_gobo_entries(project_root, writer_id)
        entries_per_writer[writer_id] = entries

        unique_words = len(set(entry["word"] for entry in entries))

        print(
            f"writer_{writer_id}: "
            f"{len(entries)} train samples after duplicate handling, "
            f"{unique_words} unique words"
        )

    return entries_per_writer


# Style preparation

def laplace_otsu(image):
    """
    Create a Laplace-Otsu image for One-DM style conditioning.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplace = cv2.Laplacian(gray, cv2.CV_64F)
    laplace = np.absolute(laplace)
    laplace = np.clip(laplace, 0, 255).astype(np.uint8)

    _, threshold = cv2.threshold(
        laplace,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    threshold = cv2.resize(threshold, (224, 64))
    return threshold


def prepare_one_shot_style(project_root, writer_id, entries, seed):
    """
    Select one fixed style sample and save:
    - resized grayscale style image
    - Laplace-Otsu image

    The same style filename is used in both folders.
    """
    rng = random.Random(seed)
    selected_entry = rng.sample(entries, 1)[0]

    style_base_dir = project_root / STYLE_OUTPUT_DIR / f"writer_{writer_id}"
    style_img_dir = style_base_dir / "style"
    laplace_img_dir = style_base_dir / "laplace"

    if style_base_dir.exists():
        shutil.rmtree(style_base_dir)

    style_img_dir.mkdir(parents=True, exist_ok=True)
    laplace_img_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(selected_entry["img_path"]))

    if image is None:
        raise ValueError(f"Could not read style image: {selected_entry['img_path']}")

    resized = cv2.resize(image, (224, 64))
    gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    laplace_image = laplace_otsu(resized)

    style_filename = filename_from_gt_path(selected_entry["rel_img"])

    style_img_path = style_img_dir / style_filename
    laplace_img_path = laplace_img_dir / style_filename

    cv2.imwrite(str(style_img_path), gray_resized)
    cv2.imwrite(str(laplace_img_path), laplace_image)

    print(f"writer_{writer_id}: one-shot style sample selected: {style_filename}")
    print(f"writer_{writer_id}: style saved to {style_img_path}")
    print(f"writer_{writer_id}: laplace saved to {laplace_img_path}")

    return style_img_path, laplace_img_path


# One-DM worker script

def write_one_dm_worker_script(one_dm_dir):
    """
    Write a small One-DM worker script into the existing One-DM repository.

    The worker loads the model once per writer and then generates all target
    samples from a manifest file.
    """
    worker_path = one_dm_dir / "generate_gobo_one_shot_worker.py"

    worker_code = r'''
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL

from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
from data_loader.loader import ContentData, style_len
from models.unet import UNetModel
from models.diffusion import Diffusion
from utils.util import fix_seed


def load_gray_image(path):
    image = cv2.imread(str(path), flags=0)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    image = cv2.resize(image, (224, 64))

    if image.shape[1] > style_len:
        image = image[:, :style_len]

    image = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


def read_manifest(manifest_path):
    rows = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t", maxsplit=1)

            if len(parts) != 2:
                continue

            output_path, text = parts
            rows.append((Path(output_path), text))

    return rows


def append_gt_line(gt_path, output_path, text):
    with open(gt_path, "a", encoding="utf-8") as f:
        f.write(f"{output_path.name}\t{text}\n")


def append_failed_line(failed_path, output_path, text, error):
    with open(failed_path, "a", encoding="utf-8") as f:
        f.write(f"{output_path.name}\t{text}\t{error}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", required=True)
    parser.add_argument("--one_dm", required=True)
    parser.add_argument("--style_img", required=True)
    parser.add_argument("--laplace_img", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--failed", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stable_dif_path", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--sampling_timesteps", type=int, default=50)
    parser.add_argument("--sample_method", default="ddim")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--skip_existing", type=int, default=1)

    args = parser.parse_args()

    cfg_from_file(args.cfg)
    assert_and_infer_cfg()
    fix_seed(cfg.TRAIN.SEED)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA is not available. Falling back to CPU.")
    else:
        device = torch.device(args.device)

    rows = read_manifest(args.manifest)

    if not rows:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    gt_path = Path(args.gt)
    failed_path = Path(args.failed)

    gt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(gt_path, "w", encoding="utf-8"):
        pass

    with open(failed_path, "w", encoding="utf-8"):
        pass

    style_input = load_gray_image(args.style_img).to(device)
    laplace_input = load_gray_image(args.laplace_img).to(device)

    load_content = ContentData()

    diffusion = Diffusion(device=str(device))

    unet = UNetModel(
        in_channels=cfg.MODEL.IN_CHANNELS,
        model_channels=cfg.MODEL.EMB_DIM,
        out_channels=cfg.MODEL.OUT_CHANNELS,
        num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
        attention_resolutions=(1, 1),
        channel_mult=(1, 1),
        num_heads=cfg.MODEL.NUM_HEADS,
        context_dim=cfg.MODEL.EMB_DIM,
    ).to(device)

    state_dict = torch.load(args.one_dm, map_location=torch.device("cpu"))
    unet.load_state_dict(state_dict)
    unet.eval()

    vae = AutoencoderKL.from_pretrained(
        args.stable_dif_path,
        subfolder="vae",
    )
    vae = vae.to(device)
    vae.requires_grad_(False)

    total = len(rows)
    generated_count = 0
    skipped_count = 0
    failed_count = 0

    with torch.no_grad():
        for idx, (output_path, text) in enumerate(rows, start=1):
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists() and args.skip_existing == 1:
                append_gt_line(gt_path, output_path, text)
                skipped_count += 1

                if idx % 25 == 0 or idx == total:
                    print(
                        f"{idx}/{total} done | "
                        f"generated={generated_count}, skipped={skipped_count}, failed={failed_count}"
                    )

                continue

            try:
                text_ref = load_content.get_content(text)
                text_ref = text_ref.to(device)

                noise = torch.randn(
                    (
                        1,
                        4,
                        style_input.shape[2] // 8,
                        (text_ref.shape[1] * 32) // 8,
                    )
                ).to(device)

                if args.sample_method == "ddim":
                    sampled_images = diffusion.ddim_sample(
                        unet,
                        vae,
                        style_input.shape[0],
                        noise,
                        style_input,
                        laplace_input,
                        text_ref,
                        args.sampling_timesteps,
                        args.eta,
                    )
                elif args.sample_method == "ddpm":
                    sampled_images = diffusion.ddpm_sample(
                        unet,
                        vae,
                        style_input.shape[0],
                        noise,
                        style_input,
                        laplace_input,
                        text_ref,
                    )
                else:
                    raise ValueError(f"Unsupported sample method: {args.sample_method}")

                image = torchvision.transforms.ToPILImage()(sampled_images[0])
                image = image.convert("L")
                image.save(output_path)

                append_gt_line(gt_path, output_path, text)
                generated_count += 1

            except Exception as e:
                failed_count += 1
                append_failed_line(failed_path, output_path, text, str(e))
                print(f"Failed: {output_path.name} | text='{text}' | error={e}")

            if idx % 25 == 0 or idx == total:
                print(
                    f"{idx}/{total} done | "
                    f"generated={generated_count}, skipped={skipped_count}, failed={failed_count}"
                )


if __name__ == "__main__":
    main()
'''

    with open(worker_path, "w", encoding="utf-8") as f:
        f.write(worker_code)

    return worker_path


# Manifest creation

def write_manifest(manifest_path, writer_output_dir, entries):
    """
    Create a manifest for the One-DM worker.

    Each row:
    absolute_output_path<TAB>text
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            output_path = writer_output_dir / entry["output_filename"]
            word = entry["word"]
            f.write(f"{output_path}\t{word}\n")

    return manifest_path


# Main

def main():
    from htr_personalization.attention_htr_adapter import ensure_git_repo

    project_root = get_project_root()
    ensure_git_repo(
        project_root,
        "One-DM",
        "https://github.com/dailenson/One-DM.git",
        "configs/IAM64.yml",
    )
    one_dm_dir = get_one_dm_dir(project_root)

    # Needed by One-DM/data_loader/loader.py when ContentData() is created.
    ensure_one_dm_data_files(one_dm_dir)

    one_dm_checkpoint = ensure_one_dm_checkpoint(project_root)

    one_dm_cfg = one_dm_dir / "configs" / "IAM64.yml"

    if not one_dm_cfg.exists():
        raise FileNotFoundError(f"One-DM config not found: {one_dm_cfg}")

    print("=" * 80)
    print("One-DM setup")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"One-DM dir: {one_dm_dir}")
    print(f"One-DM checkpoint: {one_dm_checkpoint}")
    print(f"One-DM config: {one_dm_cfg}")

    worker_script = write_one_dm_worker_script(one_dm_dir)

    print(f"Worker script written to: {worker_script}")

    print("\n" + "=" * 80)
    print("Collecting all GoBo train samples per writer")
    print("=" * 80)

    entries_per_writer = collect_entries_per_writer(project_root)

    print("\nSamples to generate per writer:")

    for writer_id in WRITER_IDS:
        entries = entries_per_writer[writer_id]
        writer_output_dir = project_root / GENERATED_OUTPUT_DIR / f"writer_{writer_id}"

        existing_files = set()

        if writer_output_dir.exists():
            existing_files = {
                p.name
                for p in writer_output_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            }

        target_files = {entry["output_filename"] for entry in entries}
        existing_target = len(existing_files.intersection(target_files))

        if SKIP_EXISTING:
            to_generate = len(entries) - existing_target
        else:
            to_generate = len(entries)

        print(
            f"writer_{writer_id}: "
            f"{to_generate} samples will be generated "
            f"({existing_target} existing, target {len(entries)})"
        )

    print("\n" + "=" * 80)
    print("Starting One-DM one-shot generation")
    print("=" * 80)

    total_writers = len(WRITER_IDS)
    global_start = time.time()

    for writer_index, writer_id in enumerate(WRITER_IDS, start=1):
        writer_start = time.time()

        print("\n" + "=" * 80)
        print(f"[Writer {writer_index}/{total_writers}] writer_{writer_id}")
        print("=" * 80)

        entries = entries_per_writer[writer_id]

        style_img_path, laplace_img_path = prepare_one_shot_style(
            project_root=project_root,
            writer_id=writer_id,
            entries=entries,
            seed=SEED,
        )

        writer_output_dir = project_root / GENERATED_OUTPUT_DIR / f"writer_{writer_id}"
        writer_output_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = (
            project_root
            / STYLE_OUTPUT_DIR
            / "manifests"
            / f"writer_{writer_id}.tsv"
        )

        write_manifest(
            manifest_path=manifest_path,
            writer_output_dir=writer_output_dir,
            entries=entries,
        )

        gt_path = writer_output_dir / "gt.txt"
        failed_path = writer_output_dir / "failed.txt"

        cmd = [
            sys.executable,
            str(worker_script),
            "--cfg",
            str(one_dm_cfg),
            "--one_dm",
            str(one_dm_checkpoint),
            "--style_img",
            str(style_img_path),
            "--laplace_img",
            str(laplace_img_path),
            "--manifest",
            str(manifest_path),
            "--gt",
            str(gt_path),
            "--failed",
            str(failed_path),
            "--device",
            DEVICE,
            "--stable_dif_path",
            STABLE_DIFFUSION_PATH,
            "--sampling_timesteps",
            str(SAMPLING_TIMESTEPS),
            "--sample_method",
            SAMPLE_METHOD,
            "--eta",
            str(ETA),
            "--skip_existing",
            "1" if SKIP_EXISTING else "0",
        ]

        run_cmd(cmd, cwd=one_dm_dir)

        writer_elapsed = time.time() - writer_start
        global_elapsed = time.time() - global_start

        print(f"writer_{writer_id}: gt.txt written incrementally to {gt_path}")
        print(f"writer_{writer_id}: failed samples written to {failed_path}")
        print(f"writer_{writer_id}: total target samples = {len(entries)}")
        print(f"writer_{writer_id}: finished in {format_seconds(writer_elapsed)}")
        print(f"Total elapsed so far: {format_seconds(global_elapsed)}")

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Styles saved in: {project_root / STYLE_OUTPUT_DIR}")
    print(f"Generated samples saved in: {project_root / GENERATED_OUTPUT_DIR}")
    print(f"Checkpoint stored in: {project_root / MODEL_DIR_NAME}")


if __name__ == "__main__":
    main()
