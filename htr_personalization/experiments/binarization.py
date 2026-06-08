import os
import sys
import random
import shutil
import subprocess
from pathlib import Path

import cv2

from config import (
    PROJECT_ROOT,
    STYLE_TRANSFER_FEW_SHOT_ROOT,
    BIG_STYLE_TRANSFER_ROOT,
    BINARIZED_STYLE_TRANSFER_FEW_SHOT_ROOT,
    BINARIZED_BIG_STYLE_TRANSFER_ROOT,
    STYLE_TRANSFER_BINARIZATION_RESULTS_CSV,
    BIG_STYLE_TRANSFER_BINARIZATION_RESULTS_CSV,
    RECREATE_BINARIZED_SYNTHETIC_DATA,
    LMDB_ROOT,
    SAVED_MODELS_ROOT,
    TEST_SAMPLES,
    BATCH_SIZE,
    EPOCHS,
    KEEP_MODELS,
    KEEP_LMDB,
    IMAGE_EXTENSIONS,
    CREATE_LMDB_SCRIPT,
    TRAIN_SCRIPT,
    ATTENTION_HTR_MODEL_ROOT,
    GENERIC_MODEL_PATH,
)

from htr_personalization.attention_htr_adapter import (
    evaluate_model_on_writer,
    get_best_accuracy_model,
    delete_model_checkpoints,
    delete_lmdb_cache,
    recreate_lmdb_folder,
)

from htr_personalization.result_tables import append_result_and_save


# Binarization setup

BINARIZATION_DATASETS = {
    "Style_Transfer_Binarization": {
        "source_root": STYLE_TRANSFER_FEW_SHOT_ROOT,
        "output_root": BINARIZED_STYLE_TRANSFER_FEW_SHOT_ROOT,
        "results_csv": STYLE_TRANSFER_BINARIZATION_RESULTS_CSV,
    },
    "Big_Style_Transfer_Binarization": {
        "source_root": BIG_STYLE_TRANSFER_ROOT,
        "output_root": BINARIZED_BIG_STYLE_TRANSFER_ROOT,
        "results_csv": BIG_STYLE_TRANSFER_BINARIZATION_RESULTS_CSV,
    },
}


# Big Style Transfer validation split settings

BIG_STYLE_TRANSFER_VAL_RATIO = 0.30

BINARIZED_BIG_STYLE_TRANSFER_SPLIT_ROOT = (
    PROJECT_ROOT / "binarized_big_style_transfer_train_val_split"
)


# Helpers

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


def read_gt_lines(gt_path):
    gt_path = Path(gt_path)

    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    lines = []

    with open(gt_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            if "\t" in line:
                image_name, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)

                if len(parts) != 2:
                    print(f"Warning: invalid gt line {line_number} in {gt_path}: {line}")
                    continue

                image_name, text = parts

            lines.append({
                "image_name": image_name,
                "text": text,
                "line_number": line_number,
            })

    if not lines:
        raise ValueError(f"No valid gt lines found in {gt_path}")

    return lines


def write_gt_lines(output_gt_path, entries):
    output_gt_path = Path(output_gt_path)
    output_gt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_gt_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(f"{entry['image_name']}\t{entry['text']}\n")


# Otsu binarization

def binarize_image_otsu(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    _, binary_image = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        cv2.imwrite(
            str(output_path),
            binary_image,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
    else:
        cv2.imwrite(str(output_path), binary_image)


# Binarize synthetic data

def binarize_synthetic_writer_folder(
    source_writer_folder,
    output_writer_folder,
    recreate=True,
):
    source_writer_folder = Path(source_writer_folder)
    output_writer_folder = Path(output_writer_folder)

    source_gt_path = source_writer_folder / "gt.txt"
    output_gt_path = output_writer_folder / "gt.txt"

    if not source_writer_folder.exists():
        raise FileNotFoundError(f"Source writer folder not found: {source_writer_folder}")

    if not source_gt_path.exists():
        raise FileNotFoundError(f"Source gt.txt not found: {source_gt_path}")

    if output_writer_folder.exists() and recreate:
        shutil.rmtree(output_writer_folder)

    output_writer_folder.mkdir(parents=True, exist_ok=True)

    gt_entries = read_gt_lines(source_gt_path)

    output_gt_entries = []
    binarized_count = 0

    for entry in gt_entries:
        image_name = entry["image_name"]
        text = entry["text"]

        source_image_path = source_writer_folder / image_name
        output_image_path = output_writer_folder / image_name

        if not source_image_path.exists():
            print(f"Warning: missing synthetic image: {source_image_path}")
            continue

        binarize_image_otsu(
            input_path=source_image_path,
            output_path=output_image_path,
        )

        output_gt_entries.append({
            "image_name": image_name,
            "text": text,
        })

        binarized_count += 1

    write_gt_lines(
        output_gt_path=output_gt_path,
        entries=output_gt_entries,
    )

    image_count = count_images(output_writer_folder)
    gt_count = count_gt_lines(output_gt_path)

    if image_count != gt_count:
        raise ValueError(
            f"Binarized folder mismatch: {output_writer_folder} | "
            f"images={image_count}, gt_lines={gt_count}"
        )

    print("\n" + "=" * 80)
    print("Binarized writer folder created")
    print("=" * 80)
    print(f"source:          {source_writer_folder}")
    print(f"output:          {output_writer_folder}")
    print(f"binarized_count: {binarized_count}")
    print(f"image_count:     {image_count}")
    print(f"gt_count:        {gt_count}")
    print("=" * 80)

    return {
        "folder": output_writer_folder,
        "gt_path": output_gt_path,
        "count": image_count,
    }


def binarize_synthetic_dataset(
    source_root,
    output_root,
    writer_ids,
    recreate=True,
):
    source_root = Path(source_root)
    output_root = Path(output_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Synthetic source root not found: {source_root}")

    print("\n" + "=" * 80)
    print("Binarizing synthetic dataset with Otsu")
    print("=" * 80)
    print(f"source_root: {source_root}")
    print(f"output_root: {output_root}")
    print(f"recreate:    {recreate}")
    print("=" * 80)

    writer_infos = {}

    for writer_id in writer_ids:
        source_writer_folder = source_root / f"writer_{writer_id}"
        output_writer_folder = output_root / f"writer_{writer_id}"

        if not source_writer_folder.exists():
            print(f"Warning: source writer folder missing, skipping: {source_writer_folder}")
            continue

        writer_info = binarize_synthetic_writer_folder(
            source_writer_folder=source_writer_folder,
            output_writer_folder=output_writer_folder,
            recreate=recreate,
        )

        writer_infos[writer_id] = writer_info

    return writer_infos


# Big Style Transfer 70/30 split after binarization

def copy_entries_to_split_folder(entries, source_folder, output_folder):
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    if output_folder.exists():
        shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    output_gt_path = output_folder / "gt.txt"
    output_gt_entries = []

    copied_count = 0

    for entry in entries:
        image_name = entry["image_name"]
        text = entry["text"]

        source_image_path = source_folder / image_name
        output_image_path = output_folder / image_name

        if not source_image_path.exists():
            print(f"Warning: missing image for split: {source_image_path}")
            continue

        shutil.copy2(source_image_path, output_image_path)

        output_gt_entries.append({
            "image_name": image_name,
            "text": text,
        })

        copied_count += 1

    write_gt_lines(
        output_gt_path=output_gt_path,
        entries=output_gt_entries,
    )

    image_count = count_images(output_folder)
    gt_count = count_gt_lines(output_gt_path)

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
        "gt_path": output_gt_path,
        "count": image_count,
    }


def create_binarized_big_style_transfer_train_val_split(
    writer_id,
    source_folder,
    val_ratio=BIG_STYLE_TRANSFER_VAL_RATIO,
    seed=1,
    recreate=True,
):
    source_folder = Path(source_folder)
    source_gt_path = source_folder / "gt.txt"

    if not source_folder.exists():
        raise FileNotFoundError(f"Binarized Big Style Transfer source folder not found: {source_folder}")

    if not source_gt_path.exists():
        raise FileNotFoundError(f"Binarized Big Style Transfer gt.txt not found: {source_gt_path}")

    split_root = (
        BINARIZED_BIG_STYLE_TRANSFER_SPLIT_ROOT
        / f"writer_{writer_id}"
    )

    train_folder = split_root / "train"
    val_folder = split_root / "val"

    if split_root.exists() and recreate:
        shutil.rmtree(split_root)

    split_root.mkdir(parents=True, exist_ok=True)

    entries = read_gt_lines(source_gt_path)

    rng = random.Random(seed + int(writer_id))
    shuffled_entries = entries[:]
    rng.shuffle(shuffled_entries)

    total_samples = len(shuffled_entries)
    val_samples = max(1, round(total_samples * val_ratio))
    train_samples = total_samples - val_samples

    if train_samples <= 0:
        raise ValueError(
            f"Not enough samples for 70/30 split: total={total_samples}"
        )

    val_entries = shuffled_entries[:val_samples]
    train_entries = shuffled_entries[val_samples:]

    train_info = copy_entries_to_split_folder(
        entries=train_entries,
        source_folder=source_folder,
        output_folder=train_folder,
    )

    val_info = copy_entries_to_split_folder(
        entries=val_entries,
        source_folder=source_folder,
        output_folder=val_folder,
    )

    print("\n" + "=" * 80)
    print("Binarized Big Style Transfer train/validation split created")
    print("=" * 80)
    print(f"writer_id:      {writer_id}")
    print(f"source_folder:  {source_folder}")
    print(f"split_root:     {split_root}")
    print(f"total_samples:  {total_samples}")
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
        "total_samples": total_samples,
    }


# Training

def train_attentionhtr_on_binarized_synthetic(
    writer_id,
    dataset_name,
    train_path,
    pretrained_model_path,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    val_path=None,
    keep_lmdb=KEEP_LMDB,
):
    train_path = Path(train_path)
    pretrained_model_path = Path(pretrained_model_path)

    if val_path is None:
        val_path = train_path
        validation_mode = "same_as_training"
    else:
        val_path = Path(val_path)
        validation_mode = "70_train_30_val"

    experiment_root_name = "Binarization"

    train_gt_path = train_path / "gt.txt"
    val_gt_path = val_path / "gt.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Train folder not found: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation folder not found: {val_path}")

    if not train_gt_path.exists():
        raise FileNotFoundError(f"Train gt.txt not found: {train_gt_path}")

    if not val_gt_path.exists():
        raise FileNotFoundError(f"Validation gt.txt not found: {val_gt_path}")

    if not CREATE_LMDB_SCRIPT.exists():
        raise FileNotFoundError(f"create_lmdb_dataset.py not found: {CREATE_LMDB_SCRIPT}")

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"train.py not found: {TRAIN_SCRIPT}")

    if not pretrained_model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")

    train_image_count = count_images(train_path)
    train_gt_count = count_gt_lines(train_gt_path)

    val_image_count = count_images(val_path)
    val_gt_count = count_gt_lines(val_gt_path)

    if train_image_count == 0:
        raise ValueError(f"No images found in train folder: {train_path}")

    if val_image_count == 0:
        raise ValueError(f"No images found in validation folder: {val_path}")

    if train_image_count != train_gt_count:
        raise ValueError(
            f"Train folder inconsistent: images={train_image_count}, gt_lines={train_gt_count}"
        )

    if val_image_count != val_gt_count:
        raise ValueError(
            f"Validation folder inconsistent: images={val_image_count}, gt_lines={val_gt_count}"
        )

    sample_name = f"train_{train_image_count}_val_{val_image_count}"

    exp_name = (
        f"{experiment_root_name}/"
        f"{dataset_name}/"
        f"writer_{writer_id}/"
        f"{sample_name}"
    )

    train_lmdb = (
        LMDB_ROOT
        / experiment_root_name
        / dataset_name
        / f"writer_{writer_id}"
        / sample_name
        / "train_lmdb"
    )

    val_lmdb = (
        LMDB_ROOT
        / experiment_root_name
        / dataset_name
        / f"writer_{writer_id}"
        / sample_name
        / "val_lmdb"
    )

    expected_model_folder = (
        SAVED_MODELS_ROOT
        / experiment_root_name
        / dataset_name
        / f"writer_{writer_id}"
        / sample_name
    )

    iters_per_epoch = max(1, train_image_count // batch_size)
    num_iter = epochs * iters_per_epoch

    print("\n" + "=" * 80)
    print("Training on binarized synthetic data")
    print("=" * 80)
    print(f"writer_id:        {writer_id}")
    print(f"dataset_name:     {dataset_name}")
    print(f"validation_mode:  {validation_mode}")
    print(f"train_path:       {train_path}")
    print(f"val_path:         {val_path}")
    print(f"train_images:     {train_image_count}")
    print(f"train_gt_lines:   {train_gt_count}")
    print(f"val_images:       {val_image_count}")
    print(f"val_gt_lines:     {val_gt_count}")
    print(f"pretrained:       {pretrained_model_path}")
    print(f"batch_size:       {batch_size}")
    print(f"epochs:           {epochs}")
    print(f"num_iter:         {num_iter}")
    print(f"exp_name:         {exp_name}")
    print(f"train_lmdb:       {train_lmdb}")
    print(f"val_lmdb:         {val_lmdb}")
    print(f"model_folder:     {expected_model_folder}")
    print("=" * 80)

    recreate_lmdb_folder(train_lmdb)
    recreate_lmdb_folder(val_lmdb)
    expected_model_folder.mkdir(parents=True, exist_ok=True)

    create_train_lmdb_command = [
        sys.executable,
        str(CREATE_LMDB_SCRIPT),
        "--inputPath", str(train_path),
        "--gtFile", str(train_gt_path),
        "--outputPath", str(train_lmdb),
    ]

    create_val_lmdb_command = [
        sys.executable,
        str(CREATE_LMDB_SCRIPT),
        "--inputPath", str(val_path),
        "--gtFile", str(val_gt_path),
        "--outputPath", str(val_lmdb),
    ]

    train_command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--exp_name", exp_name,
        "--train_data", str(train_lmdb),
        "--valid_data", str(val_lmdb),
        "--select_data", "/",
        "--batch_ratio", "1",
        "--manualSeed", "1",
        "--Transformation", "TPS",
        "--FeatureExtraction", "ResNet",
        "--SequenceModeling", "BiLSTM",
        "--Prediction", "Attn",
        "--sensitive",
        "--num_iter", str(num_iter),
        "--batch_size", str(batch_size),
        "--saved_model", str(pretrained_model_path),
    ]

    try:
        print("Creating train LMDB...")
        run_command(create_train_lmdb_command, cwd=ATTENTION_HTR_MODEL_ROOT)

        print("Creating validation LMDB...")
        run_command(create_val_lmdb_command, cwd=ATTENTION_HTR_MODEL_ROOT)

        print("Starting training...")
        run_command(train_command, cwd=PROJECT_ROOT)

    finally:
        delete_lmdb_cache(train_lmdb, val_lmdb, keep_lmdb=keep_lmdb)

    best_model_path = get_best_accuracy_model(expected_model_folder)

    print("\n" + "=" * 80)
    print("Binarization training finished")
    print("=" * 80)
    print(f"best_model_path: {best_model_path}")
    print("=" * 80)

    return best_model_path


# Main experiment

def run_binarization_experiment(
    writer_ids,
    generic_model_path=GENERIC_MODEL_PATH,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
):
    rows_by_dataset = {
        dataset_name: []
        for dataset_name in BINARIZATION_DATASETS
    }

    print("\n" + "=" * 80)
    print("Running binarization experiment")
    print("=" * 80)
    print(f"writer_ids:             {writer_ids}")
    print(f"generic_model_path:     {generic_model_path}")
    print(f"batch_size:             {batch_size}")
    print(f"epochs:                 {epochs}")
    print(f"keep_models:            {keep_models}")
    print("results_csv_files:")
    for dataset_name, dataset_info in BINARIZATION_DATASETS.items():
        print(f"  {dataset_name}: {dataset_info['results_csv']}")
    print(f"recreate_binarization:  {RECREATE_BINARIZED_SYNTHETIC_DATA}")
    print("=" * 80)

    # --------------------------------------------------
    # 1) Binarize both synthetic datasets
    # --------------------------------------------------

    binarized_dataset_infos = {}

    for dataset_name, dataset_info in BINARIZATION_DATASETS.items():
        source_root = dataset_info["source_root"]
        output_root = dataset_info["output_root"]

        writer_infos = binarize_synthetic_dataset(
            source_root=source_root,
            output_root=output_root,
            writer_ids=writer_ids,
            recreate=RECREATE_BINARIZED_SYNTHETIC_DATA,
        )

        binarized_dataset_infos[dataset_name] = {
            "output_root": output_root,
            "writer_infos": writer_infos,
            "results_csv": dataset_info["results_csv"],
        }

    # --------------------------------------------------
    # 2) Train and evaluate
    # --------------------------------------------------

    for writer_id in writer_ids:
        print("\n" + "#" * 80)
        print(f"Binarization experiment | writer_{writer_id}")
        print("#" * 80)

        for dataset_name, dataset_info in binarized_dataset_infos.items():
            writer_infos = dataset_info["writer_infos"]

            if writer_id not in writer_infos:
                print(
                    f"Skipping writer_{writer_id}, dataset={dataset_name}: "
                    f"no binarized data available."
                )
                continue

            original_train_path = writer_infos[writer_id]["folder"]
            original_train_samples = writer_infos[writer_id]["count"]

            if dataset_name == "Big_Style_Transfer_Binarization":
                split_info = create_binarized_big_style_transfer_train_val_split(
                    writer_id=writer_id,
                    source_folder=original_train_path,
                    val_ratio=BIG_STYLE_TRANSFER_VAL_RATIO,
                    seed=1,
                    recreate=True,
                )

                train_path = split_info["train_path"]
                val_path = split_info["val_path"]
                train_samples = split_info["train_samples"]
                val_samples = split_info["val_samples"]
                total_synthetic_samples = split_info["total_samples"]
                validation_mode = "70_train_30_val"

            else:
                train_path = original_train_path
                val_path = None
                train_samples = original_train_samples
                val_samples = original_train_samples
                total_synthetic_samples = original_train_samples
                validation_mode = "same_as_training"

            model_path = train_attentionhtr_on_binarized_synthetic(
                writer_id=writer_id,
                dataset_name=dataset_name,
                train_path=train_path,
                val_path=val_path,
                pretrained_model_path=generic_model_path,
                batch_size=batch_size,
                epochs=epochs,
            )

            cer_value, _, test_path = evaluate_model_on_writer(
                writer_id=writer_id,
                model_path=model_path,
                test_samples=TEST_SAMPLES,
            )

            row = {
                "experiment": dataset_name,
                "writer_id": writer_id,
                "training_writer_id": writer_id,
                "dataset_name": dataset_name,
                "binarization_method": "otsu",
                "validation_mode": validation_mode,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "original_train_samples": original_train_samples,
                "total_synthetic_samples": total_synthetic_samples,
                "test_data": "normal_previous_test_data",
                "test_samples": TEST_SAMPLES,
                "cer": cer_value,
                "model_path": str(model_path),
                "original_train_path": str(original_train_path),
                "train_path": str(train_path),
                "val_path": str(val_path) if val_path is not None else str(train_path),
                "test_path": str(test_path),
            }

            append_result_and_save(
                path=dataset_info["results_csv"],
                rows=rows_by_dataset[dataset_name],
                row=row,
            )

            print("\n" + "=" * 80)
            print("Binarization result")
            print("=" * 80)
            print(f"writer_id:              {writer_id}")
            print(f"dataset_name:           {dataset_name}")
            print(f"validation_mode:        {validation_mode}")
            print(f"original_train_samples: {original_train_samples}")
            print(f"train_samples:          {train_samples}")
            print(f"val_samples:            {val_samples}")
            print(f"test_samples:           {TEST_SAMPLES}")
            print(f"CER:                    {cer_value:.4f}")
            print("=" * 80)

            delete_model_checkpoints(
                model_path=model_path,
                keep_models=keep_models,
            )

    print("\n" + "=" * 80)
    print("Finished binarization experiment")
    print("=" * 80)
    print("CSV files saved to:")

    outputs = {}

    for dataset_name, dataset_info in BINARIZATION_DATASETS.items():
        dataset_rows = rows_by_dataset[dataset_name]
        results_csv = dataset_info["results_csv"]

        print(f"{dataset_name}: {results_csv}")

        outputs[dataset_name] = {
            "experiment": dataset_name,
            "results": dataset_rows,
            "csv": results_csv,
        }

    print("\nMean CER by binarized dataset:")

    for dataset_name, dataset_rows in rows_by_dataset.items():
        if not dataset_rows:
            print(f"{dataset_name}: no results")
            continue

        values = [row["cer"] for row in dataset_rows]
        mean_cer = sum(values) / len(values)
        print(f"{dataset_name}: mean CER = {mean_cer:.4f} over {len(values)} writers")

    print("=" * 80)

    return {
        "experiment": "Binarization",
        "outputs": outputs,
    }
