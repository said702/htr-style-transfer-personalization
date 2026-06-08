import os
import random
import shutil
from pathlib import Path

from config import (
    BEGIN_SAMPLES,
    END_SAMPLES,
    STEP_SAMPLES,
    PROGRESSIVE_TRAIN_ROOT,
    RECREATE_PROGRESSIVE_TRAIN_FOLDERS,
)


TRAIN_TXTS = [
    "brown",
    "cedar",
    "nonwords",
    "domain_A_train",
    "domain_B_train",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# GoBo train data collection

def get_gobo_words_root(project_root, writer_id):
    return (
        Path(project_root)
        / "data"
        / "real"
        / "GoBo_v1-0"
        / "words"
        / str(writer_id)
    )


def collect_real_gobo_train_entries(project_root, writer_id):
    """
    Collects valid GoBo training entries for one writer.

    Duplicate output filenames are handled by keeping the last entry.
    This avoids crashes and avoids overwriting during output creation.
    """
    gobo_words_root = get_gobo_words_root(project_root, writer_id)

    if not gobo_words_root.exists():
        raise FileNotFoundError(f"GoBo writer folder not found: {gobo_words_root}")

    entries_by_filename = {}

    for txt_name in TRAIN_TXTS:
        txt_path = gobo_words_root / f"{txt_name}.txt"

        if not txt_path.exists():
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(maxsplit=2)

                if len(parts) != 3:
                    continue

                relative_img_path, status, word = parts

                if status.lower() not in {"ok", "rw"}:
                    continue

                if not relative_img_path.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                src_img_path = gobo_words_root / relative_img_path

                if not src_img_path.exists():
                    continue

                output_filename = Path(relative_img_path).name

                entries_by_filename[output_filename] = {
                    "src_img_path": src_img_path,
                    "output_filename": output_filename,
                    "word": word,
                }

    entries = list(entries_by_filename.values())

    if len(entries) == 0:
        raise ValueError(f"No valid GoBo training entries found for writer_{writer_id}")

    entries = sorted(entries, key=lambda item: item["output_filename"])

    return entries


# Progressive nested sample schedule

def create_progressive_sample_sizes(
    available_samples,
    begin_samples=BEGIN_SAMPLES,
    end_samples=END_SAMPLES,
    step_samples=STEP_SAMPLES,
):
    """
    Creates nested sample sizes.

    Example with available_samples=528:
    10, 20, 30, ..., 520, 528

    The final available sample size is added, even if it is not exactly
    divisible by the step size.
    """
    if available_samples <= 0:
        raise ValueError("available_samples must be greater than 0")

    if begin_samples <= 0:
        raise ValueError("begin_samples must be greater than 0")

    if step_samples <= 0:
        raise ValueError("step_samples must be greater than 0")

    max_samples = min(end_samples, available_samples)

    if available_samples < begin_samples:
        return [available_samples]

    sample_sizes = list(range(begin_samples, max_samples + 1, step_samples))

    if max_samples not in sample_sizes:
        sample_sizes.append(max_samples)

    sample_sizes = sorted(set(sample_sizes))

    return sample_sizes


def create_nested_entry_order(entries, seed, writer_id):
    """
    Creates one fixed shuffled order per writer.

    This guarantees:
    samples_20 contains all samples from samples_10 plus 10 new samples.
    samples_30 contains all samples from samples_20 plus 10 new samples.
    """
    ordered_entries = entries.copy()
    rng = random.Random(seed + writer_id)
    rng.shuffle(ordered_entries)
    return ordered_entries


# Progressive train folder creation

def count_images(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        return 0

    return len([
        file for file in os.listdir(folder_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ])


def count_gt_lines(gt_path):
    gt_path = Path(gt_path)

    if not gt_path.exists():
        return 0

    with open(gt_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def progressive_train_folder_is_complete(output_folder, expected_count):
    output_folder = Path(output_folder)
    gt_path = output_folder / "gt.txt"

    if not output_folder.exists():
        return False

    image_count = count_images(output_folder)
    gt_count = count_gt_lines(gt_path)

    return image_count == expected_count and gt_count == expected_count


def prepare_progressive_train_folder(
    project_root,
    writer_id,
    ordered_entries,
    sample_size,
    recreate_output=RECREATE_PROGRESSIVE_TRAIN_FOLDERS,
):
    """
    Creates a train folder for one writer and one progressive sample size.

    The first N samples are selected from the fixed nested order.
    """
    if sample_size > len(ordered_entries):
        sample_size = len(ordered_entries)

    output_folder = (
        Path(project_root)
        / PROGRESSIVE_TRAIN_ROOT.relative_to(Path(project_root))
        / f"writer_{writer_id}"
        / f"samples_{sample_size}"
    )

    if not recreate_output and progressive_train_folder_is_complete(output_folder, sample_size):
        return output_folder

    if output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    selected_entries = ordered_entries[:sample_size]

    gt_lines = []

    for entry in selected_entries:
        src_img_path = entry["src_img_path"]
        output_filename = entry["output_filename"]
        word = entry["word"]

        dst_img_path = output_folder / output_filename
        shutil.copy2(src_img_path, dst_img_path)

        gt_lines.append(f"{output_filename}\t{word}")

    gt_path = output_folder / "gt.txt"

    with open(gt_path, "w", encoding="utf-8") as f:
        for gt_line in gt_lines:
            f.write(gt_line + "\n")

    image_count = count_images(output_folder)
    gt_count = count_gt_lines(gt_path)

    if image_count != sample_size or gt_count != sample_size:
        raise ValueError(
            f"Progressive folder is inconsistent for writer_{writer_id}, samples_{sample_size}: "
            f"images={image_count}, gt_lines={gt_count}"
        )

    return output_folder


def get_writer_progressive_data(
    project_root,
    writer_id,
    seed,
    begin_samples=BEGIN_SAMPLES,
    end_samples=END_SAMPLES,
    step_samples=STEP_SAMPLES,
):
    """
    Returns all information needed for progressive training of one writer.
    """
    entries = collect_real_gobo_train_entries(project_root, writer_id)
    ordered_entries = create_nested_entry_order(entries, seed=seed, writer_id=writer_id)

    sample_sizes = create_progressive_sample_sizes(
        available_samples=len(ordered_entries),
        begin_samples=begin_samples,
        end_samples=end_samples,
        step_samples=step_samples,
    )

    return {
        "writer_id": writer_id,
        "available_samples": len(ordered_entries),
        "ordered_entries": ordered_entries,
        "sample_sizes": sample_sizes,
    }


# Train/validation splits for large synthetic datasets



def read_gt_entries(gt_path):
    gt_path = Path(gt_path)

    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    entries = []

    with open(gt_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            if "\t" in line:
                filename, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)

                if len(parts) != 2:
                    print(f"Warning: invalid gt line {line_number}: {line}")
                    continue

                filename, text = parts

            entries.append({
                "filename": filename,
                "text": text,
                "line_number": line_number,
            })

    if not entries:
        raise ValueError(f"No valid gt entries found in {gt_path}")

    return entries


def count_images(folder):
    folder = Path(folder)

    if not folder.exists():
        return 0

    return len([
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ])


def count_gt_lines(gt_path):
    gt_path = Path(gt_path)

    if not gt_path.exists():
        return 0

    with open(gt_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_split(entries, source_folder, output_folder):
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    gt_path = output_folder / "gt.txt"
    gt_lines = []

    copied_count = 0

    for entry in entries:
        filename = entry["filename"]
        text = entry["text"]

        src_img = source_folder / filename
        dst_img = output_folder / filename

        if not src_img.exists():
            print(f"Warning: missing image: {src_img}")
            continue

        shutil.copy2(src_img, dst_img)
        gt_lines.append(f"{filename}\t{text}")
        copied_count += 1

    with open(gt_path, "w", encoding="utf-8") as f:
        for gt_line in gt_lines:
            f.write(gt_line + "\n")

    image_count = count_images(output_folder)
    gt_count = count_gt_lines(gt_path)

    if image_count != gt_count:
        raise ValueError(
            f"Split folder mismatch: {output_folder} | "
            f"images={image_count}, gt_lines={gt_count}"
        )

    if copied_count != image_count:
        raise ValueError(
            f"Copied count mismatch: {output_folder} | "
            f"copied={copied_count}, images={image_count}"
        )

    return {
        "folder": output_folder,
        "gt_path": gt_path,
        "count": image_count,
    }


def create_train_val_split_for_big_style_transfer(
    source_folder,
    output_root,
    writer_id,
    experiment_name,
    val_ratio=0.30,
    seed=1,
    recreate=True,
):
    source_folder = Path(source_folder)
    output_root = Path(output_root)

    gt_path = source_folder / "gt.txt"

    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    split_root = (
        output_root
        / experiment_name
        / f"writer_{writer_id}"
    )

    train_folder = split_root / "train"
    val_folder = split_root / "val"

    if split_root.exists() and recreate:
        shutil.rmtree(split_root)

    split_root.mkdir(parents=True, exist_ok=True)

    entries = read_gt_entries(gt_path)

    rng = random.Random(seed + int(writer_id))
    shuffled_entries = entries[:]
    rng.shuffle(shuffled_entries)

    total_count = len(shuffled_entries)
    val_count = max(1, round(total_count * val_ratio))
    train_count = total_count - val_count

    if train_count <= 0:
        raise ValueError(
            f"Not enough samples for train/val split: total={total_count}"
        )

    val_entries = shuffled_entries[:val_count]
    train_entries = shuffled_entries[val_count:]

    train_info = write_split(
        entries=train_entries,
        source_folder=source_folder,
        output_folder=train_folder,
    )

    val_info = write_split(
        entries=val_entries,
        source_folder=source_folder,
        output_folder=val_folder,
    )

    print("\n" + "=" * 80)
    print("Big Style Transfer train/validation split created")
    print("=" * 80)
    print(f"writer_id:      {writer_id}")
    print(f"experiment:     {experiment_name}")
    print(f"source_folder:  {source_folder}")
    print(f"split_root:     {split_root}")
    print(f"total_samples:  {total_count}")
    print(f"train_samples:  {train_info['count']}")
    print(f"val_samples:    {val_info['count']}")
    print(f"val_ratio:      {val_ratio}")
    print("=" * 80)

    return {
        "train_path": train_info["folder"],
        "train_gt": train_info["gt_path"],
        "train_samples": train_info["count"],
        "val_path": val_info["folder"],
        "val_gt": val_info["gt_path"],
        "val_samples": val_info["count"],
        "total_samples": total_count,
    }