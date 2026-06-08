import os
import sys
import shutil
import subprocess
from pathlib import Path

from config import (
    PROJECT_ROOT,
    REAL_GOBO_ROOT,
    GENERIC_MODEL_PATH,
    WORD_GROUP_DATA_ROOT,
    WORD_GROUP_RESULTS_CSV,
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
)

from htr_personalization.attention_htr_adapter import (
    evaluate_model_on_writer,
    get_best_accuracy_model,
    delete_model_checkpoints,
    delete_lmdb_cache,
    recreate_lmdb_folder,
)

from htr_personalization.result_tables import append_result_and_save


# Word-group setup
#
# Available groups:
#   brown
#   cedar
#   nonwords
#   domain_specific = domain_A_train + domain_B_train
#
# Default: all groups are used.
#
# To run only selected groups, pass for example:
#
# run_word_group_personalization_experiment(
#     writer_ids=WRITER_IDS,
#     generic_model_path=GENERIC_MODEL_PATH,
#     selected_groups=["brown", "nonwords"],
# )
#

WORD_GROUPS = {
    "brown": ["brown"],
    "cedar": ["cedar"],
    "nonwords": ["nonwords"],
    "domain_specific": ["domain_A_train", "domain_B_train"],
}

DEFAULT_SELECTED_WORD_GROUPS = list(WORD_GROUPS.keys())

ALLOWED_REAL_STATUS = {"ok", "rw"}


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


def get_writer_folder(writer_id):
    return (
        REAL_GOBO_ROOT
        / "GoBo_v1-0"
        / "words"
        / str(writer_id)
    )


def normalize_selected_groups(selected_groups):
    if selected_groups is None:
        return DEFAULT_SELECTED_WORD_GROUPS

    if isinstance(selected_groups, str):
        selected_groups = [selected_groups]

    normalized = []

    for group_name in selected_groups:
        group_name = str(group_name).strip()

        if not group_name:
            continue

        if group_name not in WORD_GROUPS:
            raise ValueError(
                f"Unknown word group: {group_name}. "
                f"Available groups: {list(WORD_GROUPS.keys())}"
            )

        if group_name not in normalized:
            normalized.append(group_name)

    if not normalized:
        raise ValueError("No valid word groups selected.")

    return normalized


# Collect word-group entries

def collect_word_group_train_entries_from_txt(writer_id, dataset_name):
    writer_folder = get_writer_folder(writer_id)
    txt_path = writer_folder / f"{dataset_name}.txt"

    if not writer_folder.exists():
        raise FileNotFoundError(f"Writer folder not found: {writer_folder}")

    if not txt_path.exists():
        raise FileNotFoundError(f"Word-group txt not found: {txt_path}")

    entries = []

    print(f"Reading word group txt: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=2)

            if len(parts) != 3:
                print(f"Warning: invalid line {line_number} in {txt_path}: {line}")
                continue

            relative_img_path, status, word = parts
            status = status.lower().strip()

            if status not in ALLOWED_REAL_STATUS:
                continue

            if not relative_img_path.lower().endswith(IMAGE_EXTENSIONS):
                continue

            src_img_path = writer_folder / relative_img_path

            if not src_img_path.exists():
                print(f"Warning: missing image: {src_img_path}")
                continue

            entries.append({
                "source": "real",
                "src_img_path": src_img_path,
                "original_filename": Path(relative_img_path).name,
                "word": word,
                "dataset_name": dataset_name,
                "line_number": line_number,
            })

    return entries


def collect_word_group_train_entries(writer_id, dataset_names):
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    all_entries = []

    for dataset_name in dataset_names:
        entries = collect_word_group_train_entries_from_txt(
            writer_id=writer_id,
            dataset_name=dataset_name,
        )

        all_entries.extend(entries)

    if not all_entries:
        raise ValueError(
            f"No valid entries found for writer {writer_id}, "
            f"datasets={dataset_names}"
        )

    return all_entries


# Create train folder

def prepare_word_group_train_folder(
    writer_id,
    group_name,
    dataset_names,
    recreate=True,
):
    entries = collect_word_group_train_entries(
        writer_id=writer_id,
        dataset_names=dataset_names,
    )

    output_folder = (
        WORD_GROUP_DATA_ROOT
        / f"writer_{writer_id}"
        / group_name
        / "train"
    )

    gt_output_path = output_folder / "gt.txt"

    if output_folder.exists() and recreate:
        shutil.rmtree(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    used_names = set()
    gt_lines = []

    for entry in entries:
        dataset_prefix = entry["dataset_name"]

        output_filename = safe_copy_name(
            prefix=f"{group_name}_{dataset_prefix}",
            original_name=entry["original_filename"],
            used_names=used_names,
        )

        dst_img_path = output_folder / output_filename
        shutil.copy2(entry["src_img_path"], dst_img_path)

        gt_lines.append(f"{output_filename}\t{entry['word']}")

    with open(gt_output_path, "w", encoding="utf-8") as f:
        for gt_line in gt_lines:
            f.write(gt_line + "\n")

    image_count = count_images(output_folder)
    gt_count = count_gt_lines(gt_output_path)

    print("\n" + "=" * 80)
    print("Word-group train folder created")
    print("=" * 80)
    print(f"writer_id:        {writer_id}")
    print(f"group_name:       {group_name}")
    print(f"dataset_names:    {dataset_names}")
    print(f"train_folder:     {output_folder}")
    print(f"gt_path:          {gt_output_path}")
    print(f"image_count:      {image_count}")
    print(f"gt_count:         {gt_count}")
    print("=" * 80)

    if image_count != gt_count:
        raise ValueError(
            f"Word-group folder mismatch: images={image_count}, gt_lines={gt_count}"
        )

    return {
        "train_path": output_folder,
        "gt_path": gt_output_path,
        "train_samples": image_count,
        "group_name": group_name,
        "dataset_names": dataset_names,
    }


# Training

def train_attentionhtr_on_word_group_folder(
    writer_id,
    group_name,
    train_path,
    pretrained_model_path,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_lmdb=KEEP_LMDB,
):
    train_path = Path(train_path)
    pretrained_model_path = Path(pretrained_model_path)

    experiment_root_name = "Word_group_personalization"

    exp_name = (
        f"{experiment_root_name}/"
        f"writer_{writer_id}/"
        f"{group_name}"
    )

    train_lmdb = (
        LMDB_ROOT
        / experiment_root_name
        / f"writer_{writer_id}"
        / group_name
        / "train_lmdb"
    )

    val_lmdb = (
        LMDB_ROOT
        / experiment_root_name
        / f"writer_{writer_id}"
        / group_name
        / "val_lmdb"
    )

    expected_model_folder = (
        SAVED_MODELS_ROOT
        / experiment_root_name
        / f"writer_{writer_id}"
        / group_name
    )

    gt_path = train_path / "gt.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Word-group train folder not found: {train_path}")

    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found: {gt_path}")

    if not CREATE_LMDB_SCRIPT.exists():
        raise FileNotFoundError(f"create_lmdb_dataset.py not found: {CREATE_LMDB_SCRIPT}")

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"train.py not found: {TRAIN_SCRIPT}")

    if not pretrained_model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")

    image_count = count_images(train_path)
    gt_count = count_gt_lines(gt_path)

    if image_count == 0:
        raise ValueError(f"No images found in word-group train folder: {train_path}")

    if image_count != gt_count:
        raise ValueError(
            f"Word-group train folder inconsistent: images={image_count}, gt_lines={gt_count}"
        )

    iters_per_epoch = max(1, image_count // batch_size)
    num_iter = epochs * iters_per_epoch

    print("\n" + "=" * 80)
    print("Training word-group personalization")
    print("=" * 80)
    print(f"writer_id:       {writer_id}")
    print(f"group_name:      {group_name}")
    print(f"train_path:      {train_path}")
    print(f"images:          {image_count}")
    print(f"gt_lines:        {gt_count}")
    print(f"pretrained:      {pretrained_model_path}")
    print(f"batch_size:      {batch_size}")
    print(f"epochs:          {epochs}")
    print(f"num_iter:        {num_iter}")
    print(f"exp_name:        {exp_name}")
    print(f"train_lmdb:      {train_lmdb}")
    print(f"val_lmdb:        {val_lmdb}")
    print(f"model_folder:    {expected_model_folder}")
    print("=" * 80)

    recreate_lmdb_folder(train_lmdb)
    recreate_lmdb_folder(val_lmdb)
    expected_model_folder.mkdir(parents=True, exist_ok=True)

    create_train_lmdb_command = [
        sys.executable,
        str(CREATE_LMDB_SCRIPT),
        "--inputPath", str(train_path),
        "--gtFile", str(gt_path),
        "--outputPath", str(train_lmdb),
    ]

    create_val_lmdb_command = [
        sys.executable,
        str(CREATE_LMDB_SCRIPT),
        "--inputPath", str(train_path),
        "--gtFile", str(gt_path),
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

        print("Starting word-group training...")
        run_command(train_command, cwd=PROJECT_ROOT)

    finally:
        delete_lmdb_cache(train_lmdb, val_lmdb, keep_lmdb=keep_lmdb)

    best_model_path = get_best_accuracy_model(expected_model_folder)

    print("\n" + "=" * 80)
    print("Word-group training finished")
    print("=" * 80)
    print(f"best_model_path: {best_model_path}")
    print("=" * 80)

    return best_model_path


# Experiment runner

def run_word_group_personalization_experiment(
    writer_ids,
    generic_model_path=GENERIC_MODEL_PATH,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
    selected_groups=None,
):
    rows = []

    selected_groups = normalize_selected_groups(selected_groups)

    print("\n" + "=" * 80)
    print("Running word-group personalization experiment")
    print("=" * 80)
    print(f"writer_ids:        {writer_ids}")
    print(f"available_groups:  {WORD_GROUPS}")
    print(f"selected_groups:   {selected_groups}")
    print(f"result_csv:        {WORD_GROUP_RESULTS_CSV}")
    print("Evaluation:        combined normal test data via evaluate_model_on_writer")
    print("=" * 80)

    for writer_id in writer_ids:
        print("\n" + "#" * 80)
        print(f"Word-group personalization | writer_{writer_id}")
        print("#" * 80)

        for group_name in selected_groups:
            dataset_names = WORD_GROUPS[group_name]

            print("\n" + "-" * 80)
            print(f"writer_{writer_id} | train_group={group_name}")
            print(f"train_txt_files={dataset_names}")
            print("test_data = combined normal previous test data")
            print("-" * 80)

            train_info = prepare_word_group_train_folder(
                writer_id=writer_id,
                group_name=group_name,
                dataset_names=dataset_names,
                recreate=True,
            )

            model_path = train_attentionhtr_on_word_group_folder(
                writer_id=writer_id,
                group_name=group_name,
                train_path=train_info["train_path"],
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
                "experiment": "word_group_personalization",
                "writer_id": writer_id,
                "training_writer_id": writer_id,
                "train_group": group_name,
                "train_txt_files": "+".join(dataset_names),
                "train_samples": train_info["train_samples"],
                "test_data": "combined_normal_previous_test_data",
                "test_samples": TEST_SAMPLES,
                "cer": cer_value,
                "model_path": str(model_path),
                "train_path": str(train_info["train_path"]),
                "test_path": str(test_path),
            }

            append_result_and_save(
                path=WORD_GROUP_RESULTS_CSV,
                rows=rows,
                row=row,
            )

            print(
                f"Word-group result | writer_id={writer_id} | "
                f"train_group={group_name} | "
                f"train_txt_files={'+'.join(dataset_names)} | "
                f"train_samples={train_info['train_samples']} | "
                f"cer={cer_value:.4f}"
            )

            delete_model_checkpoints(
                model_path=model_path,
                keep_models=keep_models,
            )

    print("\n" + "=" * 80)
    print("Finished word-group personalization")
    print("=" * 80)
    print(f"CSV saved to: {WORD_GROUP_RESULTS_CSV}")

    if rows:
        print("\nMean CER by training word group:")
        group_to_values = {}

        for row in rows:
            group_to_values.setdefault(row["train_group"], []).append(row["cer"])

        for group_name, values in group_to_values.items():
            mean_cer = sum(values) / len(values)
            print(f"{group_name}: mean CER = {mean_cer:.4f} over {len(values)} writers")

    print("=" * 80)

    return {
        "experiment": "word_group_personalization",
        "results": rows,
        "csv": WORD_GROUP_RESULTS_CSV,
    }