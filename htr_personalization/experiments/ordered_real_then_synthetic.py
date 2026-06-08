"""Ordered real-then-synthetic personalization experiment."""

import shutil
import sys
from pathlib import Path

from config import (
    PROJECT_ROOT,
    ORDERED_REAL_THEN_SYNTHETIC_RESULTS_CSV,
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
    ORDERED_BEGIN_SAMPLES,
    ORDERED_REAL_LIMIT,
    ORDERED_END_SAMPLES,
    ORDERED_STEP_SAMPLES,
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
    build_ordered_sample_sizes,
    calculate_ordered_counts,
    prepare_ordered_mixed_train_folder,
    count_images,
    count_gt_lines,
    run_command,
)


def train_attentionhtr_on_ordered_mixed_folder(
    writer_id,
    sample_size,
    train_path,
    pretrained_model_path,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_lmdb=KEEP_LMDB,
):
    train_path = Path(train_path)
    pretrained_model_path = Path(pretrained_model_path)

    writer_name = f"writer_{writer_id}"
    sample_name = f"samples_{sample_size}"

    experiment_root_name = "Ordered_Real_Then_Synthetic"

    exp_name = (
        f"{experiment_root_name}/"
        f"{writer_name}/"
        f"{sample_name}"
    )

    train_lmdb = (
        LMDB_ROOT
        / experiment_root_name
        / writer_name
        / sample_name
        / "train_lmdb"
    )

    val_lmdb = (
        LMDB_ROOT
        / experiment_root_name
        / writer_name
        / sample_name
        / "val_lmdb"
    )

    expected_model_folder = (
        SAVED_MODELS_ROOT
        / experiment_root_name
        / writer_name
        / sample_name
    )

    gt_path = train_path / "gt.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Ordered mixed train folder not found: {train_path}")

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
        raise ValueError(f"No images found in ordered mixed folder: {train_path}")

    if image_count != gt_count:
        raise ValueError(
            f"Ordered mixed folder inconsistent: images={image_count}, gt_lines={gt_count}"
        )

    iters_per_epoch = max(1, image_count // batch_size)
    num_iter = epochs * iters_per_epoch

    print("\n" + "=" * 80)
    print("Training ordered real-then-synthetic personalization")
    print("=" * 80)
    print(f"writer_id:       {writer_id}")
    print(f"sample_size:     {sample_size}")
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

        print("Starting ordered mixed training...")
        run_command(train_command, cwd=PROJECT_ROOT)

    finally:
        delete_lmdb_cache(train_lmdb, val_lmdb, keep_lmdb=keep_lmdb)

    best_model_path = get_best_accuracy_model(expected_model_folder)

    print("\n" + "=" * 80)
    print("Ordered real-then-synthetic training finished")
    print("=" * 80)
    print(f"best_model_path: {best_model_path}")
    print("=" * 80)

    return best_model_path


# Experiment runner

def run_ordered_real_then_synthetic_experiment(
    writer_ids,
    generic_model_path,
    begin_samples=ORDERED_BEGIN_SAMPLES,
    real_limit=ORDERED_REAL_LIMIT,
    end_samples=ORDERED_END_SAMPLES,
    step_samples=ORDERED_STEP_SAMPLES,
    seed=1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    keep_models=KEEP_MODELS,
):
    rows = []

    sample_sizes = build_ordered_sample_sizes(
        begin_samples=begin_samples,
        end_samples=end_samples,
        step_samples=step_samples,
    )

    print("\n" + "=" * 80)
    print("Running ordered real-then-synthetic experiment")
    print("=" * 80)
    print(f"sample_sizes: {sample_sizes}")
    print(f"real_limit:   {real_limit}")
    print(f"result_csv:   {ORDERED_REAL_THEN_SYNTHETIC_RESULTS_CSV}")
    print("=" * 80)

    for writer_id in writer_ids:
        print("\n" + "#" * 80)
        print(f"Ordered_Real_Then_Synthetic | writer_{writer_id}")
        print("#" * 80)

        for sample_size in sample_sizes:
            real_count, synthetic_count = calculate_ordered_counts(
                sample_size=sample_size,
                real_limit=real_limit,
            )

            print("\n" + "-" * 80)
            print(
                f"writer_{writer_id} | samples_{sample_size} | "
                f"real={real_count} | synthetic={synthetic_count}"
            )
            print("-" * 80)

            ordered_info = prepare_ordered_mixed_train_folder(
                writer_id=writer_id,
                sample_size=sample_size,
                seed=seed,
                recreate=True,
            )

            train_path = ordered_info["train_path"]

            model_path = train_attentionhtr_on_ordered_mixed_folder(
                writer_id=writer_id,
                sample_size=sample_size,
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
                "experiment": "Ordered_Real_Then_Synthetic",
                "writer_id": writer_id,
                "training_writer_id": writer_id,
                "sample_size": sample_size,
                "real_limit": real_limit,
                "real_used": ordered_info["real_used"],
                "synthetic_used": ordered_info["synthetic_used"],
                "total_used": ordered_info["total_used"],
                "real_available": ordered_info["real_available"],
                "synthetic_available": ordered_info["synthetic_available"],
                "test_samples": TEST_SAMPLES,
                "cer": cer_value,
                "model_path": str(model_path),
                "train_path": str(train_path),
                "test_path": str(test_path),
            }

            append_result_and_save(
                path=ORDERED_REAL_THEN_SYNTHETIC_RESULTS_CSV,
                rows=rows,
                row=row,
            )

            print(
                f"Ordered result | writer_id={writer_id} | "
                f"sample_size={sample_size} | "
                f"real={ordered_info['real_used']} | "
                f"synthetic={ordered_info['synthetic_used']} | "
                f"total={ordered_info['total_used']} | "
                f"cer={cer_value:.4f}"
            )

            delete_model_checkpoints(
                model_path=model_path,
                keep_models=keep_models,
            )

    return {
        "experiment": "Ordered_Real_Then_Synthetic",
        "results": rows,
        "csv": ORDERED_REAL_THEN_SYNTHETIC_RESULTS_CSV,
    }
