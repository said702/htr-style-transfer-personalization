"""Dataset preparation for style-transfer personalization experiments."""

import os
import random
import shutil
import subprocess
from pathlib import Path

from config import (
    REAL_GOBO_ROOT,
    STYLE_TRANSFER_FEW_SHOT_ROOT,
    MIXED_REAL_SYNTHETIC_DATA_ROOT,
    ORDERED_REAL_THEN_SYNTHETIC_DATA_ROOT,
    IMAGE_EXTENSIONS,
    ORDERED_BEGIN_SAMPLES,
    ORDERED_REAL_LIMIT,
    ORDERED_END_SAMPLES,
    ORDERED_STEP_SAMPLES,
)


REAL_TRAIN_TXTS = [
    "brown",
    "cedar",
    "nonwords",
    "domain_A_train",
    "domain_B_train",
]

ALLOWED_REAL_STATUS = {"ok", "rw"}



def count_images(folder):
    folder = Path(folder)

    if not folder.exists():
        return 0

    return len([
        file for file in os.listdir(folder)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ])


def count_gt_lines(gt_path):
    gt_path = Path(gt_path)

    if not gt_path.exists():
        return 0

    with open(gt_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_command(command, cwd):
    print("\n" + "-" * 80)
    print(" ".join(str(part) for part in command))
    print("-" * 80)

    subprocess.run(
        command,
        cwd=cwd,
        check=True,
    )


def safe_copy_name(prefix, original_name, used_names):
    original_name = Path(original_name).name
    stem = Path(original_name).stem
    ext = Path(original_name).suffix

    output_name = f"{prefix}_{original_name}"

    counter = 1
    while output_name in used_names:
        output_name = f"{prefix}_{stem}_{counter}{ext}"
        counter += 1

    used_names.add(output_name)

    return output_name


# Real GoBo collection

def collect_real_gobo_train_entries(writer_id):
    writer_folder = (
        REAL_GOBO_ROOT
        / "GoBo_v1-0"
        / "words"
        / str(writer_id)
    )

    if not writer_folder.exists():
        raise FileNotFoundError(f"Real GoBo writer folder not found: {writer_folder}")

    entries = []

    for dataset_name in REAL_TRAIN_TXTS:
        txt_path = writer_folder / f"{dataset_name}.txt"

        if not txt_path.exists():
            print(f"Warning: missing real TXT file: {txt_path}")
            continue

        print(f"Reading real: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                parts = line.split(maxsplit=2)

                if len(parts) != 3:
                    print(f"Warning: invalid real line {line_number} in {txt_path}: {line}")
                    continue

                relative_img_path, status, word = parts
                status = status.lower().strip()

                if status not in ALLOWED_REAL_STATUS:
                    continue

                if not relative_img_path.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                src_img_path = writer_folder / relative_img_path

                if not src_img_path.exists():
                    print(f"Warning: missing real image: {src_img_path}")
                    continue

                entries.append({
                    "source": "real",
                    "src_img_path": src_img_path,
                    "original_filename": Path(relative_img_path).name,
                    "word": word,
                    "dataset_name": dataset_name,
                    "line_number": line_number,
                })

    if not entries:
        raise ValueError(f"No real train entries found for writer {writer_id}")

    return entries


# Synthetic Few-Shot GoBo collection

def collect_synthetic_few_shot_entries(writer_id):
    writer_folder = STYLE_TRANSFER_FEW_SHOT_ROOT / f"writer_{writer_id}"
    gt_path = writer_folder / "gt.txt"

    if not writer_folder.exists():
        raise FileNotFoundError(f"Synthetic few-shot writer folder not found: {writer_folder}")

    if not gt_path.exists():
        raise FileNotFoundError(f"Synthetic few-shot gt.txt not found: {gt_path}")

    entries = []

    print(f"Reading synthetic few-shot: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            if "\t" in line:
                image_name, word = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)

                if len(parts) != 2:
                    print(f"Warning: invalid synthetic line {line_number} in {gt_path}: {line}")
                    continue

                image_name, word = parts

            image_name = image_name.strip()
            word = word.strip()

            if not image_name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            src_img_path = writer_folder / image_name

            if not src_img_path.exists():
                print(f"Warning: missing synthetic image: {src_img_path}")
                continue

            entries.append({
                "source": "synthetic",
                "src_img_path": src_img_path,
                "original_filename": Path(image_name).name,
                "word": word,
                "dataset_name": "synthetic_few_shot",
                "line_number": line_number,
            })

    if not entries:
        raise ValueError(f"No synthetic few-shot entries found for writer {writer_id}")

    return entries


# Mixed real/synthetic counts

def normalize_mixed_ratio(mixed_ratio):
    """
    Returns (real_ratio, synthetic_ratio).

    Preferred format:
        (0.80, 0.20) -> 80% real samples + 20% synthetic samples
        (0.40, 0.60) -> 40% real samples + 60% synthetic samples

    Ratios are interpreted relative to the number of available real GoBo
    training samples for the writer. They do not have to sum to 1.0.

    Backward compatibility:
        a single float x is treated as the old synthetic ratio, i.e.
        (1 - x, x).
    """

    if isinstance(mixed_ratio, dict):
        real_ratio = mixed_ratio.get("real_ratio")
        synthetic_ratio = mixed_ratio.get("synthetic_ratio")
    elif isinstance(mixed_ratio, (tuple, list)) and len(mixed_ratio) == 2:
        real_ratio, synthetic_ratio = mixed_ratio
    else:
        synthetic_ratio = float(mixed_ratio)
        real_ratio = 1.0 - synthetic_ratio

    real_ratio = float(real_ratio)
    synthetic_ratio = float(synthetic_ratio)

    if real_ratio < 0 or synthetic_ratio < 0:
        raise ValueError("Real and synthetic ratios must be non-negative.")

    return real_ratio, synthetic_ratio


def ratio_to_name(mixed_ratio):
    real_ratio, synthetic_ratio = normalize_mixed_ratio(mixed_ratio)
    real_percent = int(round(real_ratio * 100))
    synthetic_percent = int(round(synthetic_ratio * 100))
    return f"real_{real_percent}_synthetic_{synthetic_percent}"


def calculate_mixed_counts(real_available, synthetic_available, mixed_ratio):
    """
    Calculate how many real and synthetic samples are used.

    Example:
        mixed_ratio = (0.80, 0.20)
        -> use 80% of the real GoBo training set as real samples
        -> add 20% of the real GoBo training set as synthetic samples

    This is not replacement logic. The mixed dataset contains both selected
    real samples and selected synthetic samples. No separate real-control
    condition is created here.
    """

    real_ratio, synthetic_ratio = normalize_mixed_ratio(mixed_ratio)

    requested_real_count = round(real_available * real_ratio)
    requested_synthetic_count = round(real_available * synthetic_ratio)

    real_count = min(requested_real_count, real_available)
    synthetic_count = min(requested_synthetic_count, synthetic_available)
    mixed_total_count = real_count + synthetic_count

    return {
        "real_ratio_requested": real_ratio,
        "synthetic_ratio_requested": synthetic_ratio,
        "real_count": real_count,
        "synthetic_count": synthetic_count,
        "mixed_total_count": mixed_total_count,
    }


# Dataset writing

def write_entries_to_folder(entries, output_folder, recreate=True):
    output_folder = Path(output_folder)
    gt_output_path = output_folder / "gt.txt"

    if output_folder.exists() and recreate:
        shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    used_names = set()
    gt_lines = []

    real_written = 0
    synthetic_written = 0

    for entry in entries:
        prefix = "real" if entry["source"] == "real" else "synthetic"

        output_filename = safe_copy_name(
            prefix=prefix,
            original_name=entry["original_filename"],
            used_names=used_names,
        )

        dst_img_path = output_folder / output_filename
        shutil.copy2(entry["src_img_path"], dst_img_path)

        gt_lines.append(f"{output_filename}\t{entry['word']}")

        if entry["source"] == "real":
            real_written += 1
        elif entry["source"] == "synthetic":
            synthetic_written += 1

    with open(gt_output_path, "w", encoding="utf-8") as f:
        for gt_line in gt_lines:
            f.write(gt_line + "\n")

    image_count = count_images(output_folder)
    gt_count = count_gt_lines(gt_output_path)

    if image_count != gt_count:
        raise ValueError(
            f"Dataset mismatch: images={image_count}, gt_lines={gt_count}, folder={output_folder}"
        )

    return {
        "train_path": output_folder,
        "gt_path": gt_output_path,
        "real_used": real_written,
        "synthetic_used": synthetic_written,
        "total_used": image_count,
    }


def prepare_mixed_train_folder(
    writer_id,
    mixed_ratio,
    seed=1,
    recreate=True,
):
    """
    Create one mixed training folder for the requested real/synthetic ratio.

    Example:
        mixed_ratio = (0.80, 0.20)
        -> 80% real samples + 20% synthetic samples

    The function no longer creates a separate real-only control folder.
    """

    ratio_name = ratio_to_name(mixed_ratio)

    real_entries = collect_real_gobo_train_entries(writer_id)
    synthetic_entries = collect_synthetic_few_shot_entries(writer_id)

    real_available = len(real_entries)
    synthetic_available = len(synthetic_entries)

    count_info = calculate_mixed_counts(
        real_available=real_available,
        synthetic_available=synthetic_available,
        mixed_ratio=mixed_ratio,
    )

    real_ratio_requested = count_info["real_ratio_requested"]
    synthetic_ratio_requested = count_info["synthetic_ratio_requested"]
    real_count = count_info["real_count"]
    synthetic_count = count_info["synthetic_count"]
    mixed_total_count = count_info["mixed_total_count"]

    rng = random.Random(seed)

    shuffled_real_entries = real_entries.copy()
    shuffled_synthetic_entries = synthetic_entries.copy()

    rng.shuffle(shuffled_real_entries)
    rng.shuffle(shuffled_synthetic_entries)

    selected_real_entries = shuffled_real_entries[:real_count]
    selected_synthetic_entries = shuffled_synthetic_entries[:synthetic_count]

    mixed_entries = selected_real_entries + selected_synthetic_entries
    rng.shuffle(mixed_entries)

    mixed_folder = (
        MIXED_REAL_SYNTHETIC_DATA_ROOT
        / ratio_name
        / f"writer_{writer_id}"
        / f"mixed_{real_count}_real_{synthetic_count}_synthetic"
    )

    mixed_info = write_entries_to_folder(
        entries=mixed_entries,
        output_folder=mixed_folder,
        recreate=recreate,
    )

    actual_synthetic_ratio = mixed_info["synthetic_used"] / max(1, mixed_info["total_used"])
    actual_real_ratio = mixed_info["real_used"] / max(1, mixed_info["total_used"])

    print("\n" + "=" * 80)
    print("Mixed real/synthetic dataset created")
    print("=" * 80)
    print(f"writer_id:                    {writer_id}")
    print(f"ratio_name:                   {ratio_name}")
    print(f"requested_real_ratio:         {real_ratio_requested:.4f}")
    print(f"requested_synthetic_ratio:    {synthetic_ratio_requested:.4f}")
    print(f"actual_real_ratio_in_dataset: {actual_real_ratio:.4f}")
    print(f"actual_synthetic_ratio_in_dataset: {actual_synthetic_ratio:.4f}")
    print("-" * 80)
    print(f"real_available:               {real_available}")
    print(f"synthetic_available:          {synthetic_available}")
    print(f"selected_real_count:          {real_count}")
    print(f"selected_synthetic_count:     {synthetic_count}")
    print("-" * 80)
    print("Mixed condition:")
    print(f"real_used:                    {mixed_info['real_used']}")
    print(f"synthetic_used:               {mixed_info['synthetic_used']}")
    print(f"total_used:                   {mixed_info['total_used']}")
    print(f"folder:                       {mixed_info['train_path']}")
    print("=" * 80)

    if mixed_info["real_used"] != real_count:
        raise ValueError("Mixed real count does not match expected real_count.")

    if mixed_info["synthetic_used"] != synthetic_count:
        raise ValueError("Mixed synthetic count does not match expected synthetic_count.")

    if mixed_info["total_used"] != mixed_total_count:
        raise ValueError("Mixed total count does not match expected mixed_total_count.")

    return {
        "ratio_name": ratio_name,
        "mixed_ratio": mixed_ratio,
        "real_ratio_requested": real_ratio_requested,
        "synthetic_ratio_requested": synthetic_ratio_requested,
        "real_available": real_available,
        "synthetic_available": synthetic_available,
        "real_count": real_count,
        "synthetic_count": synthetic_count,
        "mixed_total_count": mixed_total_count,
        "actual_synthetic_ratio": actual_synthetic_ratio,
        "actual_real_ratio": actual_real_ratio,
        "mixed": mixed_info,
    }


# Training



def build_ordered_sample_sizes(
    begin_samples=ORDERED_BEGIN_SAMPLES,
    end_samples=ORDERED_END_SAMPLES,
    step_samples=ORDERED_STEP_SAMPLES,
):
    sample_sizes = list(range(begin_samples, end_samples + 1, step_samples))

    if end_samples not in sample_sizes:
        sample_sizes.append(end_samples)

    return sample_sizes



# Ordered mixed dataset creation

def calculate_ordered_counts(sample_size, real_limit=ORDERED_REAL_LIMIT):
    """
    Ordered logic:

    sample_size <= real_limit:
        only real data

    sample_size > real_limit:
        real data is fixed at real_limit
        additional samples are synthetic
    """

    if sample_size <= real_limit:
        real_count = sample_size
        synthetic_count = 0
    else:
        real_count = real_limit
        synthetic_count = sample_size - real_limit

    return real_count, synthetic_count


def prepare_ordered_mixed_train_folder(
    writer_id,
    sample_size,
    seed=1,
    recreate=True,
):
    real_entries = collect_real_gobo_train_entries(writer_id)
    synthetic_entries = collect_synthetic_few_shot_entries(writer_id)

    real_available = len(real_entries)
    synthetic_available = len(synthetic_entries)

    real_count, synthetic_count = calculate_ordered_counts(
        sample_size=sample_size,
        real_limit=ORDERED_REAL_LIMIT,
    )

    if real_count > real_available:
        raise ValueError(
            f"Not enough real samples for writer {writer_id}: "
            f"needed={real_count}, available={real_available}"
        )

    if synthetic_count > synthetic_available:
        raise ValueError(
            f"Not enough synthetic samples for writer {writer_id}: "
            f"needed={synthetic_count}, available={synthetic_available}"
        )

    rng = random.Random(seed)

    shuffled_real_entries = real_entries.copy()
    shuffled_synthetic_entries = synthetic_entries.copy()

    rng.shuffle(shuffled_real_entries)
    rng.shuffle(shuffled_synthetic_entries)

    selected_real_entries = shuffled_real_entries[:real_count]
    selected_synthetic_entries = shuffled_synthetic_entries[:synthetic_count]

    ordered_entries = selected_real_entries + selected_synthetic_entries

    output_folder = (
        ORDERED_REAL_THEN_SYNTHETIC_DATA_ROOT
        / f"writer_{writer_id}"
        / f"samples_{sample_size}"
    )

    gt_output_path = output_folder / "gt.txt"

    if output_folder.exists() and recreate:
        shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    used_names = set()
    gt_lines = []

    real_written = 0
    synthetic_written = 0

    for entry in ordered_entries:
        prefix = "real" if entry["source"] == "real" else "synthetic"

        output_filename = safe_copy_name(
            prefix=prefix,
            original_name=entry["original_filename"],
            used_names=used_names,
        )

        dst_img_path = output_folder / output_filename
        shutil.copy2(entry["src_img_path"], dst_img_path)

        gt_lines.append(f"{output_filename}\t{entry['word']}")

        if entry["source"] == "real":
            real_written += 1
        else:
            synthetic_written += 1

    with open(gt_output_path, "w", encoding="utf-8") as f:
        for gt_line in gt_lines:
            f.write(gt_line + "\n")

    image_count = count_images(output_folder)
    gt_count = count_gt_lines(gt_output_path)

    print("\n" + "=" * 80)
    print("Ordered mixed dataset created")
    print("=" * 80)
    print(f"writer_id:            {writer_id}")
    print(f"sample_size:          {sample_size}")
    print(f"real_limit:           {ORDERED_REAL_LIMIT}")
    print("-" * 80)
    print(f"real_available:       {real_available}")
    print(f"synthetic_available:  {synthetic_available}")
    print(f"real_used:            {real_written}")
    print(f"synthetic_used:       {synthetic_written}")
    print(f"total_used:           {image_count}")
    print("-" * 80)
    print(f"output_folder:        {output_folder}")
    print(f"gt_output_path:       {gt_output_path}")
    print(f"image_count:          {image_count}")
    print(f"gt_count:             {gt_count}")
    print("=" * 80)

    if image_count != gt_count:
        raise ValueError(
            f"Ordered mixed dataset mismatch: images={image_count}, gt_lines={gt_count}"
        )

    if image_count != sample_size:
        raise ValueError(
            f"Ordered mixed dataset size mismatch: "
            f"images={image_count}, expected={sample_size}"
        )

    return {
        "train_path": output_folder,
        "sample_size": sample_size,
        "real_available": real_available,
        "synthetic_available": synthetic_available,
        "real_used": real_written,
        "synthetic_used": synthetic_written,
        "total_used": image_count,
    }


# Training

