"""Mixed real/synthetic personalization experiment."""

import shutil
import sys
from pathlib import Path

from config import (
    PROJECT_ROOT,
    MIXED_REAL_SYNTHETIC_RESULTS_ROOT,
    MIXED_REAL_SYNTHETIC_RATIOS,
    LMDB_ROOT,
    SAVED_MODELS_ROOT,
    TEST_SAMPLES,
    BATCH_SIZE,
    EPOCHS,
    KEEP_MODELS,
    KEEP_LMDB,
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
from htr_personalization.style_transfer_preparation import (
    ratio_to_name,
    count_images,
    count_gt_lines,
    run_command,
    prepare_mixed_train_folder,
)


def train_attentionhtr_on_folder(
    writer_id,
    ratio_name,
    condition_name,
    train_path,
    pretrained_model_path,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_lmdb=KEEP_LMDB,
):
    train_path = Path(train_path)
    pretrained_model_path = Path(pretrained_model_path)

    writer_name = f"writer_{writer_id}"
    sample_name = train_path.name

    experiment_root_name = "Mixed_Real_Synthetic"

    exp_name = (
        f"{experiment_root_name}/"
        f"{ratio_name}/"
        f"{condition_name}/"
        f"{writer_name}/"
        f"{sample_name}"
    )

    train_lmdb = (
        LMDB_ROOT
        / experiment_root_name
        / ratio_name
        / condition_name
        / writer_name
        / sample_name
        / "train_lmdb"
    )

    val_lmdb = (
        LMDB_ROOT
        / experiment_root_name
        / ratio_name
        / condition_name
        / writer_name
        / sample_name
        / "val_lmdb"
    )

    expected_model_folder = (
        SAVED_MODELS_ROOT
        / experiment_root_name
        / ratio_name
        / condition_name
        / writer_name
        / sample_name
    )

    gt_path = train_path / "gt.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Train folder not found: {train_path}")

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
        raise ValueError(f"No images found in train folder: {train_path}")

    if image_count != gt_count:
        raise ValueError(
            f"Train folder inconsistent: images={image_count}, gt_lines={gt_count}"
        )

    iters_per_epoch = max(1, image_count // batch_size)
    num_iter = epochs * iters_per_epoch

    print("\n" + "=" * 80)
    print("Training mixed real/synthetic personalization")
    print("=" * 80)
    print(f"writer_id:       {writer_id}")
    print(f"ratio_name:      {ratio_name}")
    print(f"condition_name:  {condition_name}")
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

        print("Starting training...")
        run_command(train_command, cwd=PROJECT_ROOT)

    finally:
        delete_lmdb_cache(train_lmdb, val_lmdb, keep_lmdb=keep_lmdb)

    best_model_path = get_best_accuracy_model(expected_model_folder)

    print("\n" + "=" * 80)
    print("Mixed real/synthetic training finished")
    print("=" * 80)
    print(f"best_model_path: {best_model_path}")
    print("=" * 80)

    return best_model_path


# Results

def get_mixed_results_csv(mixed_ratio):
    ratio_name = ratio_to_name(mixed_ratio)

    return (
        MIXED_REAL_SYNTHETIC_RESULTS_ROOT
        / ratio_name
        / f"Mixed_Real_Synthetic_{ratio_name}.csv"
    )


def run_condition(
    writer_id,
    generic_model_path,
    ratio_name,
    mixed_ratio,
    condition_name,
    condition_info,
    global_info,
    result_csv,
    rows,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
):
    train_path = condition_info["train_path"]

    model_path = train_attentionhtr_on_folder(
        writer_id=writer_id,
        ratio_name=ratio_name,
        condition_name=condition_name,
        train_path=train_path,
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
        "experiment": "Mixed_Real_Synthetic",
        "condition": condition_name,
        "ratio_name": ratio_name,
        "writer_id": writer_id,
        "training_writer_id": writer_id,
        "real_ratio_requested": global_info["real_ratio_requested"],
        "synthetic_ratio_requested": global_info["synthetic_ratio_requested"],
        "real_ratio_actual_mixed": global_info["actual_real_ratio"],
        "synthetic_ratio_actual_mixed": global_info["actual_synthetic_ratio"],
        "real_available": global_info["real_available"],
        "synthetic_available": global_info["synthetic_available"],
        "selected_real_count": global_info["real_count"],
        "selected_synthetic_count": global_info["synthetic_count"],
        "condition_real_used": condition_info["real_used"],
        "condition_synthetic_used": condition_info["synthetic_used"],
        "condition_total_used": condition_info["total_used"],
        "test_samples": TEST_SAMPLES,
        "cer": cer_value,
        "model_path": str(model_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }

    append_result_and_save(
        path=result_csv,
        rows=rows,
        row=row,
    )

    print(
        f"Result | ratio={ratio_name} | condition={condition_name} | "
        f"writer_id={writer_id} | "
        f"real={condition_info['real_used']} | "
        f"synthetic={condition_info['synthetic_used']} | "
        f"total={condition_info['total_used']} | "
        f"cer={cer_value:.4f}"
    )

    delete_model_checkpoints(
        model_path=model_path,
        keep_models=keep_models,
    )

    return row


# Experiment runner

def run_single_mixed_experiment(
    writer_ids,
    generic_model_path,
    mixed_ratio,
    seed=1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
):
    ratio_name = ratio_to_name(mixed_ratio)
    result_csv = get_mixed_results_csv(mixed_ratio)

    rows = []

    print("\n" + "=" * 80)
    print(f"Running mixed real/synthetic experiment: Mixed_Real_Synthetic/{ratio_name}")
    print("=" * 80)
    print(f"mixed_ratio: {mixed_ratio}")
    print("One mixed condition is trained for each ratio.")
    print(f"result_csv:  {result_csv}")
    print("=" * 80)

    for writer_id in writer_ids:
        print("\n" + "#" * 80)
        print(f"Mixed_Real_Synthetic/{ratio_name} | writer_{writer_id}")
        print("#" * 80)

        dataset_info = prepare_mixed_train_folder(
            writer_id=writer_id,
            mixed_ratio=mixed_ratio,
            seed=seed,
            recreate=True,
        )

        run_condition(
            writer_id=writer_id,
            generic_model_path=generic_model_path,
            ratio_name=ratio_name,
            mixed_ratio=mixed_ratio,
            condition_name="mixed_real_synthetic",
            condition_info=dataset_info["mixed"],
            global_info=dataset_info,
            result_csv=result_csv,
            rows=rows,
            batch_size=batch_size,
            epochs=epochs,
            keep_models=keep_models,
        )

    return {
        "experiment": "Mixed_Real_Synthetic",
        "ratio_name": ratio_name,
        "mixed_ratio": mixed_ratio,
        "results": rows,
        "csv": result_csv,
    }


def run_mixed_real_synthetic_experiments(
    writer_ids,
    generic_model_path,
    mixed_ratios=MIXED_REAL_SYNTHETIC_RATIOS,
    seed=1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
):
    outputs = {}

    for mixed_ratio in mixed_ratios:
        ratio_name = ratio_to_name(mixed_ratio)

        outputs[ratio_name] = run_single_mixed_experiment(
            writer_ids=writer_ids,
            generic_model_path=generic_model_path,
            mixed_ratio=mixed_ratio,
            seed=seed,
            batch_size=batch_size,
            epochs=epochs,
            keep_models=keep_models,
        )

    return outputs


# Ordered real-then-synthetic experiment

