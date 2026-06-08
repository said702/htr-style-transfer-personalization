import os
import random
import shutil
from pathlib import Path

from config import (
    PROJECT_ROOT,
    TEST_SAMPLES,
    BATCH_SIZE,
    EPOCHS,
    KEEP_MODELS,
    KEEP_LMDB,
    RANDOM_SEED,
    BIG_STYLE_TRANSFER_BEGIN_SAMPLES,
    BIG_STYLE_TRANSFER_END_SAMPLES,
    BIG_STYLE_TRANSFER_STEP_SAMPLES,
    BIG_STYLE_TRANSFER_VAL_RATIO,
    BIG_STYLE_TRANSFER_SPLIT_ROOT,
    RECREATE_BIG_STYLE_TRANSFER_SPLITS,
)

from htr_personalization.attention_htr_adapter import (
    evaluate_model_on_writer,
    delete_lmdb_cache,
    recreate_lmdb_folder,
)
from htr_personalization.result_tables import append_result_and_save


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# Style-transfer dataset paths

SYNTHETIC_DATASETS = {
    "Style_Transfer_One_Shot": (
        PROJECT_ROOT
        / "data"
        / "synthetic"
        / "synthetic_gobo_one_shot"
        / "01_GoBo_synthetic_one_shot"
    ),
    "Style_Transfer_Few_Shot": (
        PROJECT_ROOT
        / "data"
        / "synthetic"
        / "synthetic_gobo_few_shot"
        / "02_GoBo_synthetic_few_shot"
    ),
    "Big_Style_Transfer": (
        PROJECT_ROOT
        / "data"
        / "synthetic"
        / "big_style_transfer"
        / "GoBo_synthetic_few_shot_10k_common_words"
    ),
}


# Result CSV paths

STYLE_TRANSFER_RESULTS_ROOT = PROJECT_ROOT / "experiment_results" / "style_transfer"

SYNTHETIC_RESULTS_CSVS = {
    "Style_Transfer_One_Shot": (
        STYLE_TRANSFER_RESULTS_ROOT
        / "Style_Transfer_One_Shot"
        / "Style_Transfer_One_Shot.csv"
    ),
    "Style_Transfer_Few_Shot": (
        STYLE_TRANSFER_RESULTS_ROOT
        / "Style_Transfer_Few_Shot"
        / "Style_Transfer_Few_Shot.csv"
    ),
    "Big_Style_Transfer": (
        STYLE_TRANSFER_RESULTS_ROOT
        / "Big_Style_Transfer"
        / "Big_Style_Transfer.csv"
    ),
}


# Basic helpers

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


def quote_path(path):
    return f'"{str(path)}"'


def run_command(command, cwd=None):
    print("\n" + "-" * 80)
    print(command)
    print("-" * 80)

    old_cwd = Path.cwd()

    try:
        if cwd is not None:
            os.chdir(cwd)

        exit_code = os.system(command)

        if exit_code != 0:
            raise RuntimeError(f"Command failed with exit code {exit_code}")

    finally:
        os.chdir(old_cwd)


# GT helpers

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


def write_gt_entries_to_folder(entries, source_folder, output_folder):
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    gt_output_path = output_folder / "gt.txt"
    gt_lines = []

    copied_count = 0

    for entry in entries:
        filename = entry["filename"]
        text = entry["text"]

        source_image_path = source_folder / filename
        output_image_path = output_folder / filename

        if not source_image_path.exists():
            print(f"Warning: missing image: {source_image_path}")
            continue

        shutil.copy2(source_image_path, output_image_path)
        gt_lines.append(f"{filename}\t{text}")
        copied_count += 1

    with open(gt_output_path, "w", encoding="utf-8") as f:
        for gt_line in gt_lines:
            f.write(gt_line + "\n")

    image_count = count_images(output_folder)
    gt_count = count_gt_lines(gt_output_path)

    if image_count != gt_count:
        raise ValueError(
            f"Split folder mismatch:\n"
            f"{output_folder}\n"
            f"images={image_count}, gt_lines={gt_count}"
        )

    if copied_count != image_count:
        raise ValueError(
            f"Copied count mismatch:\n"
            f"{output_folder}\n"
            f"copied={copied_count}, images={image_count}"
        )

    return {
        "folder": output_folder,
        "gt_path": gt_output_path,
        "count": image_count,
    }


# Big Style Transfer progressive 70/30 splits

def build_big_style_transfer_sample_sizes(
    available_samples,
    begin_samples=BIG_STYLE_TRANSFER_BEGIN_SAMPLES,
    end_samples=BIG_STYLE_TRANSFER_END_SAMPLES,
    step_samples=BIG_STYLE_TRANSFER_STEP_SAMPLES,
):
    """Return progressive total sample sizes for Big Style Transfer."""
    available_samples = int(available_samples)
    begin_samples = int(begin_samples)
    end_samples = int(end_samples)
    step_samples = int(step_samples)

    if available_samples <= 0:
        raise ValueError("available_samples must be larger than 0")

    if begin_samples <= 0:
        raise ValueError("BIG_STYLE_TRANSFER_BEGIN_SAMPLES must be larger than 0")

    if end_samples < begin_samples:
        raise ValueError(
            "BIG_STYLE_TRANSFER_END_SAMPLES must be larger than or equal to "
            "BIG_STYLE_TRANSFER_BEGIN_SAMPLES"
        )

    if step_samples <= 0:
        raise ValueError("BIG_STYLE_TRANSFER_STEP_SAMPLES must be larger than 0")

    upper_bound = min(end_samples, available_samples)

    if upper_bound < begin_samples:
        return [upper_bound]

    sample_sizes = list(range(begin_samples, upper_bound + 1, step_samples))

    if upper_bound not in sample_sizes:
        sample_sizes.append(upper_bound)

    return sorted(set(sample_sizes))


def create_big_style_transfer_train_val_split(
    writer_id,
    source_folder,
    sample_size,
    val_ratio=BIG_STYLE_TRANSFER_VAL_RATIO,
    seed=RANDOM_SEED,
    recreate=RECREATE_BIG_STYLE_TRANSFER_SPLITS,
):
    """Create one progressive Big Style Transfer train/validation split.

    The requested sample_size is the total number of selected synthetic samples.
    At each step, 30% are used for validation and 70% for training by default.
    For example, sample_size=10000 produces 7000 train and 3000 validation samples.
    """
    source_folder = Path(source_folder)
    source_gt_path = source_folder / "gt.txt"
    sample_size = int(sample_size)

    if not source_folder.exists():
        raise FileNotFoundError(f"Big Style Transfer source folder not found: {source_folder}")

    if not source_gt_path.exists():
        raise FileNotFoundError(f"Big Style Transfer gt.txt not found: {source_gt_path}")

    if sample_size <= 1:
        raise ValueError(f"sample_size must be larger than 1, got {sample_size}")

    split_root = (
        BIG_STYLE_TRANSFER_SPLIT_ROOT
        / f"writer_{writer_id}"
        / f"samples_{sample_size}"
    )

    train_folder = split_root / "train"
    val_folder = split_root / "val"

    if split_root.exists() and recreate:
        shutil.rmtree(split_root)

    entries = read_gt_entries(source_gt_path)

    if sample_size > len(entries):
        raise ValueError(
            f"Requested {sample_size} Big Style Transfer samples for writer_{writer_id}, "
            f"but only {len(entries)} are available."
        )

    rng = random.Random(seed + int(writer_id))
    shuffled_entries = entries[:]
    rng.shuffle(shuffled_entries)

    selected_entries = shuffled_entries[:sample_size]
    total_samples = len(selected_entries)
    val_samples = max(1, round(total_samples * val_ratio))
    train_samples = total_samples - val_samples

    if train_samples <= 0:
        raise ValueError(
            f"Not enough samples for Big Style Transfer split: total={total_samples}"
        )

    val_entries = selected_entries[:val_samples]
    train_entries = selected_entries[val_samples:]

    if split_root.exists() and not recreate:
        train_gt_path = train_folder / "gt.txt"
        val_gt_path = val_folder / "gt.txt"
        train_count = count_images(train_folder)
        val_count = count_images(val_folder)

        if (
            train_gt_path.exists()
            and val_gt_path.exists()
            and train_count == count_gt_lines(train_gt_path)
            and val_count == count_gt_lines(val_gt_path)
            and train_count == train_samples
            and val_count == val_samples
        ):
            return {
                "train_path": train_folder,
                "train_gt": train_gt_path,
                "train_samples": train_count,
                "val_path": val_folder,
                "val_gt": val_gt_path,
                "val_samples": val_count,
                "sample_size": total_samples,
                "total_samples": total_samples,
                "available_samples": len(entries),
            }

        shutil.rmtree(split_root)

    split_root.mkdir(parents=True, exist_ok=True)

    train_info = write_gt_entries_to_folder(
        entries=train_entries,
        source_folder=source_folder,
        output_folder=train_folder,
    )

    val_info = write_gt_entries_to_folder(
        entries=val_entries,
        source_folder=source_folder,
        output_folder=val_folder,
    )

    print("\n" + "=" * 80)
    print("Big Style Transfer progressive train/validation split created")
    print("=" * 80)
    print(f"writer_id:         {writer_id}")
    print(f"source_folder:     {source_folder}")
    print(f"split_root:        {split_root}")
    print(f"available_samples: {len(entries)}")
    print(f"sample_size:       {total_samples}")
    print(f"train_samples:     {train_info['count']}")
    print(f"val_samples:       {val_info['count']}")
    print(f"val_ratio:         {val_ratio}")
    print("=" * 80)

    return {
        "train_path": train_info["folder"],
        "train_gt": train_info["gt_path"],
        "train_samples": train_info["count"],
        "val_path": val_info["folder"],
        "val_gt": val_info["gt_path"],
        "val_samples": val_info["count"],
        "sample_size": total_samples,
        "total_samples": total_samples,
        "available_samples": len(entries),
    }


# Find style-transfer writer folder

def get_synthetic_writer_folder(experiment_name, writer_id):
    if experiment_name not in SYNTHETIC_DATASETS:
        raise ValueError(f"Unknown style-transfer experiment: {experiment_name}")

    dataset_root = SYNTHETIC_DATASETS[experiment_name]

    writer_folder = dataset_root / f"writer_{writer_id}"

    if not writer_folder.exists():
        raise FileNotFoundError(
            f"Style-transfer writer folder not found:\n{writer_folder}"
        )

    gt_path = writer_folder / "gt.txt"

    if not gt_path.exists():
        raise FileNotFoundError(f"gt.txt not found:\n{gt_path}")

    image_count = count_images(writer_folder)
    gt_count = count_gt_lines(gt_path)

    if image_count == 0:
        raise ValueError(f"No images found in:\n{writer_folder}")

    if image_count != gt_count:
        raise ValueError(
            f"Style-transfer folder inconsistent:\n"
            f"{writer_folder}\n"
            f"images={image_count}, gt_lines={gt_count}"
        )

    return writer_folder, image_count


# Train AttentionHTR on one style-transfer writer folder

def train_attentionhtr_on_folder(
    experiment_name,
    writer_id,
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
        validation_mode = "separate_validation_split"

    attention_model_dir = PROJECT_ROOT / "AttentionHTR" / "model"
    create_lmdb_script = attention_model_dir / "create_lmdb_dataset.py"
    train_script = attention_model_dir / "train.py"

    if not attention_model_dir.exists():
        raise FileNotFoundError(f"AttentionHTR folder not found: {attention_model_dir}")

    if not create_lmdb_script.exists():
        raise FileNotFoundError(f"create_lmdb_dataset.py not found: {create_lmdb_script}")

    if not train_script.exists():
        raise FileNotFoundError(f"train.py not found: {train_script}")

    if not pretrained_model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_model_path}")

    train_gt_path = train_path / "gt.txt"
    val_gt_path = val_path / "gt.txt"

    if not train_gt_path.exists():
        raise FileNotFoundError(f"Train gt.txt not found: {train_gt_path}")

    if not val_gt_path.exists():
        raise FileNotFoundError(f"Validation gt.txt not found: {val_gt_path}")

    train_image_count = count_images(train_path)
    train_gt_count = count_gt_lines(train_gt_path)

    val_image_count = count_images(val_path)
    val_gt_count = count_gt_lines(val_gt_path)

    if train_image_count == 0:
        raise ValueError(f"No training images found in: {train_path}")

    if val_image_count == 0:
        raise ValueError(f"No validation images found in: {val_path}")

    if train_image_count != train_gt_count:
        raise ValueError(
            f"Train folder inconsistent:\n"
            f"{train_path}\n"
            f"images={train_image_count}, gt_lines={train_gt_count}"
        )

    if val_image_count != val_gt_count:
        raise ValueError(
            f"Validation folder inconsistent:\n"
            f"{val_path}\n"
            f"images={val_image_count}, gt_lines={val_gt_count}"
        )

    sample_name = f"train_{train_image_count}_val_{val_image_count}"
    writer_name = f"writer_{writer_id}"

    exp_name = f"{experiment_name}/{writer_name}/{sample_name}"

    train_lmdb = (
        PROJECT_ROOT
        / "lmdb_data"
        / experiment_name
        / writer_name
        / sample_name
        / "train_lmdb"
    )

    val_lmdb = (
        PROJECT_ROOT
        / "lmdb_data"
        / experiment_name
        / writer_name
        / sample_name
        / "val_lmdb"
    )

    model_folder = (
        PROJECT_ROOT
        / "saved_models"
        / experiment_name
        / writer_name
        / sample_name
    )

    best_model_path = model_folder / "best_accuracy.pth"

    iters_per_epoch = max(1, train_image_count // batch_size)
    num_iter = epochs * iters_per_epoch

    print("=" * 80)
    print("Style-transfer personalization training")
    print("=" * 80)
    print(f"experiment_name:  {experiment_name}")
    print(f"writer_id:        {writer_id}")
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
    print(f"model_folder:     {model_folder}")
    print("=" * 80)

    recreate_lmdb_folder(train_lmdb)
    recreate_lmdb_folder(val_lmdb)
    model_folder.mkdir(parents=True, exist_ok=True)

    create_train_lmdb_command = (
        f"python {quote_path(create_lmdb_script)} "
        f"--inputPath {quote_path(train_path)} "
        f"--gtFile {quote_path(train_gt_path)} "
        f"--outputPath {quote_path(train_lmdb)}"
    )

    create_val_lmdb_command = (
        f"python {quote_path(create_lmdb_script)} "
        f"--inputPath {quote_path(val_path)} "
        f"--gtFile {quote_path(val_gt_path)} "
        f"--outputPath {quote_path(val_lmdb)}"
    )

    train_command = (
        f"python {quote_path(train_script)} "
        f"--exp_name {exp_name} "
        f"--train_data {quote_path(train_lmdb)} "
        f"--valid_data {quote_path(val_lmdb)} "
        f"--select_data / "
        f"--batch_ratio 1 "
        f"--manualSeed 1 "
        f"--Transformation TPS "
        f"--FeatureExtraction ResNet "
        f"--SequenceModeling BiLSTM "
        f"--Prediction Attn "
        f"--sensitive "
        f"--num_iter {num_iter} "
        f"--batch_size {batch_size} "
        f"--saved_model {quote_path(pretrained_model_path)}"
    )

    try:
        print("Creating train LMDB.")
        run_command(create_train_lmdb_command, cwd=attention_model_dir)

        print("Creating validation LMDB.")
        run_command(create_val_lmdb_command, cwd=attention_model_dir)

        print("Starting style-transfer personalization training.")
        run_command(train_command, cwd=PROJECT_ROOT)

    finally:
        delete_lmdb_cache(train_lmdb, val_lmdb, keep_lmdb=keep_lmdb)

    if not best_model_path.exists():
        candidates = list(model_folder.rglob("best_accuracy.pth"))

        if not candidates:
            raise FileNotFoundError(
                f"No best_accuracy.pth found after training in:\n{model_folder}"
            )

        best_model_path = candidates[0]

    print("=" * 80)
    print("Style-transfer training finished")
    print("=" * 80)
    print(f"best_model_path: {best_model_path}")
    print("=" * 80)

    return best_model_path


# Run one style-transfer experiment

def run_style_transfer_personalization_experiment(
    experiment_name,
    writer_ids,
    generic_model_path,
    keep_models=KEEP_MODELS,
):
    if experiment_name not in SYNTHETIC_DATASETS:
        raise ValueError(f"Unknown style-transfer experiment: {experiment_name}")

    result_csv = SYNTHETIC_RESULTS_CSVS[experiment_name]
    rows = []

    print("\n" + "=" * 80)
    print(f"Running style-transfer experiment: {experiment_name}")
    print("=" * 80)
    print(f"dataset_root: {SYNTHETIC_DATASETS[experiment_name]}")
    print(f"result_csv:   {result_csv}")

    if experiment_name == "Big_Style_Transfer":
        print("validation:   progressive 70% train / 30% validation")
        print(f"begin:        {BIG_STYLE_TRANSFER_BEGIN_SAMPLES}")
        print(f"end:          {BIG_STYLE_TRANSFER_END_SAMPLES}")
        print(f"step:         {BIG_STYLE_TRANSFER_STEP_SAMPLES}")
    else:
        print("validation:   same as training")

    print("=" * 80)

    for writer_id in writer_ids:
        print("\n" + "=" * 80)
        print(f"{experiment_name} | writer_{writer_id}")
        print("=" * 80)

        source_train_path, available_train_samples = get_synthetic_writer_folder(
            experiment_name=experiment_name,
            writer_id=writer_id,
        )

        training_plans = []

        if experiment_name == "Big_Style_Transfer":
            sample_sizes = build_big_style_transfer_sample_sizes(
                available_samples=available_train_samples,
            )

            print(f"Progressive Big Style Transfer sample sizes: {sample_sizes}")

            for sample_size in sample_sizes:
                split_info = create_big_style_transfer_train_val_split(
                    writer_id=writer_id,
                    source_folder=source_train_path,
                    sample_size=sample_size,
                    val_ratio=BIG_STYLE_TRANSFER_VAL_RATIO,
                    seed=RANDOM_SEED,
                    recreate=RECREATE_BIG_STYLE_TRANSFER_SPLITS,
                )

                training_plans.append({
                    "train_path": split_info["train_path"],
                    "val_path": split_info["val_path"],
                    "sample_size": split_info["sample_size"],
                    "train_samples": split_info["train_samples"],
                    "val_samples": split_info["val_samples"],
                    "total_synthetic_samples": split_info["total_samples"],
                    "validation_mode": "progressive_70_train_30_val",
                })

        else:
            training_plans.append({
                "train_path": source_train_path,
                "val_path": None,
                "sample_size": available_train_samples,
                "train_samples": available_train_samples,
                "val_samples": available_train_samples,
                "total_synthetic_samples": available_train_samples,
                "validation_mode": "same_as_training",
            })

        for plan in training_plans:
            train_path = plan["train_path"]
            val_path = plan["val_path"]
            sample_size = plan["sample_size"]
            train_samples = plan["train_samples"]
            val_samples = plan["val_samples"]
            total_synthetic_samples = plan["total_synthetic_samples"]
            validation_mode = plan["validation_mode"]

            model_path = train_attentionhtr_on_folder(
                experiment_name=experiment_name,
                writer_id=writer_id,
                train_path=train_path,
                val_path=val_path,
                pretrained_model_path=generic_model_path,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
            )

            cer_value, _, test_path = evaluate_model_on_writer(
                writer_id=writer_id,
                model_path=model_path,
                test_samples=TEST_SAMPLES,
            )

            row = {
                "experiment": experiment_name,
                "writer_id": writer_id,
                "training_writer_id": writer_id,
                "validation_mode": validation_mode,
                "sample_size": sample_size,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "available_train_samples": available_train_samples,
                "total_synthetic_samples": total_synthetic_samples,
                "test_data": "normal_previous_test_data",
                "test_samples": TEST_SAMPLES,
                "cer": cer_value,
                "model_path": str(model_path),
                "source_train_path": str(source_train_path),
                "train_path": str(train_path),
                "val_path": str(val_path) if val_path is not None else str(train_path),
                "test_path": str(test_path),
            }

            append_result_and_save(
                path=result_csv,
                rows=rows,
                row=row,
            )

            print(
                f"Style-transfer result | experiment={experiment_name} | "
                f"writer_id={writer_id} | "
                f"validation={validation_mode} | "
                f"sample_size={sample_size} | "
                f"train={train_samples} | "
                f"val={val_samples} | "
                f"cer={cer_value:.4f}"
            )

            if not keep_models:
                model_dir_to_delete = (
                    PROJECT_ROOT
                    / "saved_models"
                    / experiment_name
                    / f"writer_{writer_id}"
                )

                if model_dir_to_delete.exists():
                    shutil.rmtree(model_dir_to_delete)
                    print(f"Deleted model folder: {model_dir_to_delete}")

    return {
        "experiment": experiment_name,
        "results": rows,
        "csv": result_csv,
    }


# Run selected style-transfer experiments

def run_selected_style_transfer_experiments(
    writer_ids,
    generic_model_path,
    run_one_shot=False,
    run_few_shot=False,
    run_big_style_transfer=False,
    keep_models=KEEP_MODELS,
):
    outputs = {}

    if run_one_shot:
        outputs["Style_Transfer_One_Shot"] = run_style_transfer_personalization_experiment(
            experiment_name="Style_Transfer_One_Shot",
            writer_ids=writer_ids,
            generic_model_path=generic_model_path,
            keep_models=keep_models,
        )

    if run_few_shot:
        outputs["Style_Transfer_Few_Shot"] = run_style_transfer_personalization_experiment(
            experiment_name="Style_Transfer_Few_Shot",
            writer_ids=writer_ids,
            generic_model_path=generic_model_path,
            keep_models=keep_models,
        )

    if run_big_style_transfer:
        outputs["Big_Style_Transfer"] = run_style_transfer_personalization_experiment(
            experiment_name="Big_Style_Transfer",
            writer_ids=writer_ids,
            generic_model_path=generic_model_path,
            keep_models=keep_models,
        )

    return outputs
