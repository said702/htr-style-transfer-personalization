"""Generate the few-shot GoBo-style synthetic handwriting dataset with VATr."""

import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Settings

WRITER_IDS = [i for i in range(41) if i != 33]

NUM_STYLE_SAMPLES = 15
SEED = 1

# Direct Google Drive file ID for vatr.pth
GDOWN_MODEL_ID = "1cn0t_I3mUDjKSx9Na0dNHB0nPa0fDf1I"

TRAIN_TXTS = [
    "brown",
    "cedar",
    "nonwords",
    "domain_A_train",
    "domain_B_train",
]

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

STYLE_OUTPUT_DIR = "few_shot_styles"
GENERATED_OUTPUT_DIR = "generated_vatr_all"

# If True, existing final image files are skipped.
# If False, existing final image files are overwritten.
SKIP_EXISTING = True


# Basic helpers

def run_cmd(cmd, cwd=None):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))

    subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True
    )


def check_gdown_available():
    try:
        import gdown  # noqa: F401
    except ImportError:
        raise ImportError(
            "gdown is not installed. Please install the project requirements first:\n"
            "pip install -r requirements.txt"
        )


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


def filename_from_gt_path(rel_img):
    """
    Keep only the image filename from the left side of the GT line.

    Example:
    domain_A_train/0-brown-0-0-0.png
    -> 0-brown-0-0-0.png
    """
    return Path(rel_img.replace("\\", "/")).name


def vatr_auto_suffix_path(final_path):
    """
    VATr may automatically create an output file with _000 added.

    Example:
    final_path = 0-brown-0-0-0.png
    VATr output = 0-brown-0-0-0_000.png
    """
    return final_path.with_name(f"{final_path.stem}_000{final_path.suffix}")


def normalize_vatr_output(final_path):
    """
    Rename VATr's automatic *_000.png output to the desired final filename.

    Example:
    0-brown-0-0-0_000.png -> 0-brown-0-0-0.png
    """
    suffix_path = vatr_auto_suffix_path(final_path)

    if final_path.exists():
        if suffix_path.exists() and suffix_path != final_path:
            suffix_path.unlink()
        return final_path

    if suffix_path.exists():
        suffix_path.replace(final_path)
        return final_path

    # Fallback: find any file that starts with the final stem and was produced by VATr.
    candidates = sorted(
        final_path.parent.glob(f"{final_path.stem}_*{final_path.suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if candidates:
        candidates[0].replace(final_path)
        return final_path

    raise FileNotFoundError(
        "VATr output was not found.\n"
        f"Expected final file: {final_path}\n"
        f"Expected VATr auto-suffix file: {suffix_path}"
    )


# VATr model

def find_vatr_model(project_root):
    model_dir = project_root / "vatr_model"

    candidates = [
        model_dir / "vatr.pth",
        project_root / "VATr" / "files" / "vatr.pth",
    ]

    for path in candidates:
        if path.exists():
            return path

    if model_dir.exists():
        found = list(model_dir.rglob("vatr.pth"))
        if found:
            return found[0]

    return None


def ensure_vatr_model(project_root):
    model_dir = project_root / "vatr_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = find_vatr_model(project_root)

    if model_path is not None:
        print(f"VATr model found: {model_path}")
        return model_path

    print("VATr model not found. Downloading only vatr.pth to vatr_model...")

    check_gdown_available()

    target_model_path = model_dir / "vatr.pth"

    run_cmd(
        [
            sys.executable,
            "-m",
            "gdown",
            GDOWN_MODEL_ID,
            "-O",
            str(target_model_path),
        ],
        cwd=project_root,
    )

    if not target_model_path.exists():
        raise FileNotFoundError(
            "vatr.pth was not found after download.\n"
            f"Expected path: {target_model_path}"
        )

    print(f"VATr model downloaded: {target_model_path}")
    return target_model_path


# GoBo reading

def get_gobo_root(project_root, writer_id):
    return project_root / "data" / "real" / "GoBo_v1-0" / "words" / str(writer_id)


def deduplicate_entries_by_output_filename(entries, writer_id):
    """
    Keep only one entry per output filename.

    If the same output filename appears multiple times, the later entry
    overwrites the earlier one.
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

    Example:
    brown/0-brown-0-0-0.png ok action
    -> output_filename = 0-brown-0-0-0.png
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

                if status not in {"ok","rw"}:
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


# Style folder

def make_style_folder(project_root, writer_id, entries, num_style_samples, seed):
    if len(entries) < num_style_samples:
        print(
            f"Warning: writer_{writer_id} has only {len(entries)} style candidates. "
            f"Using all."
        )

    rng = random.Random(seed)
    selected = rng.sample(entries, min(num_style_samples, len(entries)))

    style_dir = (
        project_root
        / STYLE_OUTPUT_DIR
        / f"writer_{writer_id}"
        / f"samples_{len(selected)}"
    )

    if style_dir.exists():
        shutil.rmtree(style_dir)

    style_dir.mkdir(parents=True, exist_ok=True)

    used_names = set()

    for entry in selected:
        src = entry["img_path"]
        dst_name = src.name

        if dst_name in used_names:
            stem = src.stem
            suffix = src.suffix
            counter = 1

            while dst_name in used_names:
                dst_name = f"{stem}_{counter}{suffix}"
                counter += 1

        used_names.add(dst_name)
        shutil.copy2(src, style_dir / dst_name)

    print(f"writer_{writer_id}: style folder created with {len(selected)} samples")
    return style_dir


# GT writing

def reset_gt_file(writer_output_dir):
    """
    Create an empty gt.txt at the beginning of each writer.
    This avoids duplicate GT lines when the script is restarted.
    """
    gt_path = writer_output_dir / "gt.txt"

    with open(gt_path, "w", encoding="utf-8") as f:
        pass

    return gt_path


def append_gt_line(gt_path, filename, word):
    """
    Append one GT line immediately after each generated or skipped sample.
    """
    with open(gt_path, "a", encoding="utf-8") as f:
        f.write(f"{filename}\t{word}\n")


# VATr generation

def generate_entry(project_root, writer_id, entry, style_dir, model_path):
    vatr_dir = project_root / "VATr"
    generator = vatr_dir / "generator.py"

    if not generator.exists():
        raise FileNotFoundError(f"generator.py not found: {generator}")

    out_dir = project_root / GENERATED_OUTPUT_DIR / f"writer_{writer_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    word = entry["word"]
    output_filename = entry["output_filename"]

    final_path = out_dir / output_filename
    suffix_path = vatr_auto_suffix_path(final_path)

    # If a previous VATr run created *_000.png, normalize it before checking skip.
    if not final_path.exists() and suffix_path.exists():
        suffix_path.replace(final_path)

    if final_path.exists() and SKIP_EXISTING:
        print(f"writer_{writer_id}: skip existing {final_path.name}")
        return final_path

    # Remove old files if overwriting is enabled.
    if not SKIP_EXISTING:
        if final_path.exists():
            final_path.unlink()
        if suffix_path.exists():
            suffix_path.unlink()

    cmd = [
        sys.executable,
        "generator.py",
        "--style-folder",
        str(style_dir),
        "--checkpoint",
        str(model_path),
        "--output",
        str(final_path),
        "--text",
        word,
    ]

    subprocess.run(cmd, cwd=vatr_dir, check=True)

    # Rename VATr's automatic *_000.png output to the final GT filename.
    final_path = normalize_vatr_output(final_path)

    print(f"writer_{writer_id}: generated {final_path.name}")

    return final_path


# Main

def main():
    from htr_personalization.attention_htr_adapter import ensure_git_repo

    project_root = PROJECT_ROOT
    ensure_git_repo(
        project_root,
        "VATr",
        "https://github.com/aimagelab/VATr.git",
        "generator.py",
    )

    print("=" * 80)
    print("Checking VATr model")
    print("=" * 80)
    model_path = ensure_vatr_model(project_root)

    print("\n" + "=" * 80)
    print("Collecting all GoBo train samples per writer")
    print("=" * 80)

    entries_per_writer = collect_entries_per_writer(project_root)

    print("\nSamples to generate per writer:")

    for writer_id in WRITER_IDS:
        entries = entries_per_writer[writer_id]
        out_dir = project_root / GENERATED_OUTPUT_DIR / f"writer_{writer_id}"

        existing_files = set()

        if out_dir.exists():
            # Normalize old VATr *_000.png files before counting.
            for entry in entries:
                final_path = out_dir / entry["output_filename"]
                suffix_path = vatr_auto_suffix_path(final_path)

                if not final_path.exists() and suffix_path.exists():
                    suffix_path.replace(final_path)

            existing_files = {
                p.name
                for p in out_dir.iterdir()
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
    print("Starting generation")
    print("=" * 80)

    total_writers = len(WRITER_IDS)
    global_start = time.time()

    for writer_index, writer_id in enumerate(WRITER_IDS, start=1):
        writer_start = time.time()

        print("\n" + "=" * 80)
        print(f"[Writer {writer_index}/{total_writers}] writer_{writer_id}")
        print("=" * 80)

        entries = entries_per_writer[writer_id]

        style_dir = make_style_folder(
            project_root=project_root,
            writer_id=writer_id,
            entries=entries,
            num_style_samples=NUM_STYLE_SAMPLES,
            seed=SEED,
        )

        writer_output_dir = project_root / GENERATED_OUTPUT_DIR / f"writer_{writer_id}"
        writer_output_dir.mkdir(parents=True, exist_ok=True)

        gt_path = reset_gt_file(writer_output_dir)

        total = len(entries)

        for idx, entry in enumerate(entries, start=1):
            final_path = generate_entry(
                project_root=project_root,
                writer_id=writer_id,
                entry=entry,
                style_dir=style_dir,
                model_path=model_path,
            )

            append_gt_line(
                gt_path=gt_path,
                filename=final_path.name,
                word=entry["word"],
            )

            if idx % 25 == 0 or idx == total:
                elapsed = time.time() - writer_start
                avg_time = elapsed / idx
                remaining = total - idx
                eta = avg_time * remaining

                print(
                    f"writer_{writer_id}: {idx}/{total} done | "
                    f"elapsed={format_seconds(elapsed)}, eta={format_seconds(eta)}"
                )

        writer_elapsed = time.time() - writer_start
        global_elapsed = time.time() - global_start

        print(f"writer_{writer_id}: gt.txt written incrementally to {gt_path}")
        print(f"writer_{writer_id}: total samples in gt.txt = {len(entries)}")
        print(f"writer_{writer_id}: finished in {format_seconds(writer_elapsed)}")
        print(f"Total elapsed so far: {format_seconds(global_elapsed)}")

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Styles saved in: {project_root / STYLE_OUTPUT_DIR}")
    print(f"Generated samples saved in: {project_root / GENERATED_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
